import asyncio
import json
import traceback
import redis
import uuid
from kafka import KafkaConsumer

from src.config import settings
from src.modules.llm import ResilientLLM
from src.modules.graph import FalkorGraph
from src.modules.vector import VectorDB
from src.utils.ingestion import DocumentIngestor
from src.utils.processing import recursive_summarize, split_text
from src.services.notification import NotificationService
from src.prompts import (
    CONTEXTUAL_SUMMARY_PROMPT, 
    CONTEXTUAL_HEADER_PROMPT, 
    GRAPH_EXTRACTION_SYSTEM, 
    GRAPH_EXTRACTION_USER
)
from src.logging_config import setup_logger

logger = setup_logger("WorkerEngine")

# --- INITIALIZATION ---
# 1. Standard Redis for static status checking (GET /status)
status_redis = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)

# 2. Notification Service for Real-time Streaming (Pub/Sub)
notifier = NotificationService()

# 3. Core Modules
llm = ResilientLLM()
graph_db = FalkorGraph(settings.FALKOR_URL)
vec_db = VectorDB(settings.QDRANT_URL)
ingestor = DocumentIngestor()

async def process_job(msg):
    """
    Full Pipeline:
    1. Notify Start
    2. Load File
    3. Generate Contextual Summary (Recursive)
    4. Extract Knowledge Graph (Local LLM)
    5. Index Vectors with Contextual Headers
    6. Notify Success/Failure
    """
    try:
        data = json.loads(msg.value.decode('utf-8'))
    except Exception as e:
        logger.error(f"Failed to decode Kafka message: {e}")
        return

    path = data.get('path')
    batch_id = data.get('batch')
    
    # Validation
    if not path or not batch_id:
        logger.error("Invalid message format: Missing path or batch_id")
        return

    status_key = f"job:{batch_id}:{path}"
    
    # --- STEP 0: NOTIFY START ---
    status_redis.set(status_key, "PROCESSING")
    await notifier.publish_update(batch_id, {
        "file": path,
        "status": "PROCESSING",
        "progress": "Started processing..."
    })
    logger.info(f"🚀 Started: {path}")

    file_hash = data.get('hash')
    if file_hash:
        current_status = status_redis.get(f"hash:{file_hash}")
        if current_status == b"COMPLETED" or current_status == "COMPLETED":
            logger.info(f"⏭️ Skipping already completed file: {path}")
            await notifier.publish_update(batch_id, {
                "file": path, "status": "SKIPPED", "progress": "Already processed."
            })
            return

    try:
        # --- STEP 1: LOAD ---
        raw_docs = ingestor.load_file(path)
        if not raw_docs: 
            raise ValueError(f"File could not be loaded or is empty: {path}")

        # Combine text and build Page Map
        full_text = ""
        page_map = [] # List of (start_index, end_index, page_number)
        
        for d in raw_docs:
            start_idx = len(full_text)
            page_content = d.page_content
            # Add separator only if not first, but for mapping simplest is just append
            # To match original "\n\n".join, we append "\n\n" to all except last? 
            # Actually easier: append "\n\n" to all, strip at end.
            
            # Using \n\n as separator
            separator = "\n\n"
            text_segment = page_content + separator
            full_text += text_segment
            
            end_idx = len(full_text)
            # LangChain usually puts page number in metadata['page'] (0-indexed or 1-indexed)
            p_num = d.metadata.get('page', 0) + 1 
            page_map.append((start_idx, end_idx, p_num))

        full_text = full_text.strip() # Remove trailing whitespace from last join

        # --- STEP 2: SUMMARY (Recursive) ---
        await notifier.publish_update(batch_id, {
            "file": path, "status": "PROCESSING", "progress": "Generating Document Summary..."
        })
        
        # We pass the ResilientLLM's local generator to save costs
        summary = await recursive_summarize(
            full_text, 
            llm.generate_local, 
            prompt_template=CONTEXTUAL_SUMMARY_PROMPT
        )

        # --- STEP 3 & 4: COMBINED GRAPH + VECTOR PROCESSING ---
        # Strategy: Process chunks sequentially to extract Graph Entities AND Index Vectors
        # allowing us to Link Chunks <-> Entities in the Graph.
        
        chunks = split_text(full_text) # Uses settings.CHUNK_SIZE (3000) and settings.CHUNK_OVERLAP (200)
        total_chunks = len(chunks)

        await notifier.publish_update(batch_id, {
            "file": path, "status": "PROCESSING", "progress": f"Processing {total_chunks} Chunks (Graph + Vector)..."
        })

        for i, chunk_text in enumerate(chunks):
            # Global Chunk Index
            # Must be UUID or Int for Qdrant
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{batch_id}_{i}"))
            
            await notifier.publish_update(batch_id, {
                "file": path, 
                "status": "PROCESSING", 
                "progress": f"Processing Chunk {i+1}/{total_chunks}..."
            })

            # A. GRAPH EXTRACTION
            try:
                graph_json = await llm.generate_local(
                    GRAPH_EXTRACTION_USER.format(text=chunk_text), 
                    system=GRAPH_EXTRACTION_SYSTEM, 
                    json_mode=True
                )
                
                # Check for code fences
                graph_json_clean = graph_json.replace("```json", "").replace("```", "").strip()
                
                entities_in_chunk = set()
                if graph_json_clean:
                    g_data = json.loads(graph_json_clean)
                    # Insert Relationships & Collect Entities
                    for r in g_data.get("relationships", []):
                        s, rel, o = r.get('source'), r.get('relation'), r.get('target')
                        if s and rel and o:
                            graph_db.insert_triple(s, rel, o)
                            entities_in_chunk.add(s)
                            entities_in_chunk.add(o)
                    
                    # LINK CHUNK -> ENTITIES
                    if entities_in_chunk:
                        graph_db.insert_chunk_link(chunk_id, list(entities_in_chunk), source=path)
                        logger.info(f"Linked Chunk {i} to {len(entities_in_chunk)} entities (Source: {path})")

            except Exception as e:
                logger.warning(f"Graph Error Chunk {i}: {e}")

            # B. VECTOR INDEXING
            # Generate Contextual Header
            try:
                header = await llm.generate_local(
                    CONTEXTUAL_HEADER_PROMPT.format(doc_summary=summary, chunk_text=chunk_text)
                )
                full_content = f"CONTEXT: {header}\n\nCONTENT: {chunk_text}"
                
                # Find Chunk's position in Full Text to resolve Page Number
                # We track search_start to handle duplicate phrases correctly (sequential order)
                if i == 0: search_start = 0
                
                chunk_start_index = full_text.find(chunk_text, search_start)
                if chunk_start_index != -1:
                    # Update search_start for next chunk (advance by 1 to allow overlapping matches)
                    search_start = chunk_start_index + 1
                    
                    # Resolve Page Number
                    chunk_page = 0
                    for start, end, p_num in page_map:
                        if start <= chunk_start_index < end:
                            chunk_page = p_num
                            break
                else:
                    chunk_page = 0 # Fallback
                    
                # Embed & Upsert
                vec = await llm.get_embedding(full_content)
                vec_db.upsert(
                    text=full_content, 
                    vector=vec, 
                    meta={
                        "source": path, 
                        "batch": batch_id,
                        "chunk_index": i,
                        "chunk_id": chunk_id,
                        "page_number": chunk_page 
                    },
                    id=chunk_id # Ensure Vector DB ID matches Graph Node ID
                )
            except Exception as e:
                 logger.error(f"Vector Error Chunk {i}: {e}")
        status_redis.set(status_key, "COMPLETED")
        
        # Update Deduplication Hash to COMPLETED
        file_hash = data.get('hash')
        if file_hash:
            status_redis.set(f"hash:{file_hash}", "COMPLETED")

        await notifier.publish_update(batch_id, {
            "file": path,
            "status": "COMPLETED",
            "progress": "Done"
        })
        logger.info(f"✅ Completed: {path}")

    except Exception as e:
        # --- ERROR HANDLING ---
        error_msg = str(e)
        stack_trace = traceback.format_exc()
        
        logger.error(f"❌ Failed {path}: {stack_trace}")
        
        status_redis.set(status_key, f"FAILED: {error_msg}")
        
        # Release Deduplication Lock so user can retry
        file_hash = data.get('hash')
        if file_hash:
            logger.info(f"Releasing deduplication lock for {file_hash}")
            status_redis.delete(f"hash:{file_hash}")
        
        # Notify User of Failure
        await notifier.publish_update(batch_id, {
            "file": path,
            "status": "FAILED",
            "error": error_msg
        })

def start():
    """
    Main Loop: Connects to Kafka and processes messages.
    """
    try:
        consumer = KafkaConsumer(
            'doc_ingest',
            bootstrap_servers=settings.KAFKA_BOOTSTRAP,
            group_id='enterprise_worker',
            auto_offset_reset='earliest',
            enable_auto_commit=False, # Manual commit to prevent duplicates
            value_deserializer=lambda x: x,
            max_poll_interval_ms=600000, # 10 minutes processing time
            session_timeout_ms=30000,
            heartbeat_interval_ms=10000
        )
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        logger.info(f"👷 Worker Listening on {settings.KAFKA_BOOTSTRAP}...")
        
        for msg in consumer:
            # We use an event loop to run the async process_job function
            loop.run_until_complete(process_job(msg))
            
            # Explicitly commit offset ensuring at-least-once (or exactly-once in practice here)
            consumer.commit()
            
    except Exception as e:
        logger.critical(f"Critical Worker Failure: {e}")
        import time
        time.sleep(5) 
        start()

if __name__ == "__main__":
    start()