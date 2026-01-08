from fastapi import FastAPI, UploadFile, Request, HTTPException
from fastapi.responses import StreamingResponse
from kafka import KafkaProducer
import uuid
import json
import shutil
import os

from src.config import settings
from src.modules.llm import ResilientLLM
from src.modules.vector import VectorDB
from src.modules.graph import FalkorGraph
from src.services.retrieval import RetrievalService
from src.services.notification import NotificationService
from src.logging_config import setup_logger

from fastapi.staticfiles import StaticFiles

# ... imports ...

logger = setup_logger("API")
app = FastAPI(title="Federated RAG")
# create temp directory for PDFs if not exists
os.makedirs("temp", exist_ok=True)
# Serve 'temp' directory for PDF viewing
app.mount("/files", StaticFiles(directory="temp"), name="files")

# 1. Dependency Injection (Init Services)
producer = KafkaProducer(bootstrap_servers=settings.KAFKA_BOOTSTRAP)
llm = ResilientLLM()
vec_db = VectorDB(settings.QDRANT_URL)
graph_db = FalkorGraph(settings.FALKOR_URL)

retrieval_service = RetrievalService(llm, vec_db, graph_db)
notification_service = NotificationService()

import hashlib
import redis

# Redis for deduplication not using NotificationService's connection
redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)

@app.post("/reset")
async def reset_system():
    # 1. Clear Vectors
    vec_db.clear()
    
    # 2. Clear Graph
    graph_db.reset_graph()
    
    # 3. Clear Redis Deduplication Keys
    keys = redis_client.keys("hash:*")
    if keys:
        redis_client.delete(*keys)

    # 4. Clear job statuses
    job_keys = redis_client.keys("job:*")
    if job_keys:
        redis_client.delete(*job_keys)

    # 5. Clear Temp Files
    shutil.rmtree("temp", ignore_errors=True)
    os.makedirs("temp", exist_ok=True)

    logger.info("♻️ System Reset Triggered")
    return {"message": "System Reset Successful (Vectors, Graph, Redis cache, Files cleared)."}

@app.post("/upload")
async def upload_files(files: list[UploadFile]):
    batch_id = str(uuid.uuid4())
    results = []
    
    for file in files:
        # 1. Calc MD5
        md5_hash = hashlib.md5()
        path = f"temp/{batch_id}_{file.filename}"
        
        # Stream save to avoid memory issues and calc hash
        with open(path, "wb") as f:
            while chunk := await file.read(8192):
                md5_hash.update(chunk)
                f.write(chunk)
        
        file_hash = md5_hash.hexdigest()
        
        # 2. Check Deduplication
        if redis_client.exists(f"hash:{file_hash}"):
            logger.info(f"Duplicate file detected: {file.filename} ({file_hash})")
            results.append({
                "file": file.filename,
                "status": "skipped",
                "message": "Duplicate file"
            })
            # Clean up temp
            os.remove(path)
            continue

        # 3. New File -> Mark Hash and Send to Kafka
        redis_client.set(f"hash:{file_hash}", "QUEUED") # Mark as In-Flight
        producer.send('doc_ingest', json.dumps({
            'path': path, 
            'batch': batch_id,
            'hash': file_hash
        }).encode('utf-8'))
        results.append({
            "file": file.filename,
            "status": "queued",
            "message": "Processing started"
        })
    
    return {
        "batch_id": batch_id, 
        "results": results,
        "stream_url": f"/stream/{batch_id}" 
    }

@app.get("/stream/{batch_id}")
async def stream_progress(batch_id: str, request: Request):
    """
    User Subscription Endpoint.
    Returns a real-time stream of events (SSE) as the worker processes files.
    """
    return StreamingResponse(
        notification_service.event_generator(batch_id, request),
        media_type="text/event-stream"
    )

@app.get("/documents")
async def list_documents():
    """List all available documents for context filtering."""
    if not os.path.exists("temp"):
        return {"documents": []}
    
    files = []
    for f in os.listdir("temp"):
        if os.path.isfile(os.path.join("temp", f)) and not f.startswith('.'):
            # Return full path as ID (since 'source' field in DB uses relative path 'temp/filename')
            files.append({"id": f"temp/{f}", "name": f})
            
    return {"documents": files}

@app.get("/query")
async def query(q: str, filter: str = None, mode: str = "hybrid"):
    """
    Thin wrapper around the Retrieval Service.
    mode: 'hybrid', 'vector', 'graph'
    """
    try:
        if filter == "None": filter = None 
        
        result = await retrieval_service.hybrid_search(q, file_filter=filter, mode=mode)
        
        # Log Debug Info
        debug_info = result.get('debug', {})
        if debug_info:
            logger.info(f"🐛 Debug [Query='{q}']: {json.dumps(debug_info, indent=2)}")
            
        return result
    except Exception as e:
        logger.error(f"API Query Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# -- Settings API --
from pydantic import BaseModel

class SettingsUpdate(BaseModel):
    provider: str
    api_key: str = None
    model: str = None
    endpoint: str = None
    api_version: str = None
    deployment: str = None
    embedding_deployment: str = None
    use_local_llm: bool = False

@app.get("/settings")
async def get_settings():
    """
    Returns current configuration (masked).
    """
    def mask(s): return f"{s[:3]}***{s[-2:]}" if s and len(s) > 5 else (s if s else "")

    return {
        "provider": llm.provider,
        "api_key": mask(settings.OPENAI_API_KEY) if llm.provider == "openai" else 
                   mask(settings.AZURE_OPENAI_API_KEY) if llm.provider == "azure" else
                   mask(settings.GEMINI_API_KEY) if llm.provider == "gemini" else "",
        "model": settings.BIG_MODEL if llm.provider == "openai" else settings.SMALL_MODEL, # Simply return relevant model
        "endpoint": settings.AZURE_OPENAI_ENDPOINT if llm.provider == "azure" else "",
        "api_version": settings.AZURE_OPENAI_API_VERSION if llm.provider == "azure" else "",
        "deployment": settings.AZURE_OPENAI_DEPLOYMENT_NAME if llm.provider == "azure" else "",
        "embedding_deployment": settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME if llm.provider == "azure" else "",
        "use_local_llm": settings.USE_LOCAL_LLM
    }

@app.post("/settings")
async def update_settings(config: SettingsUpdate):
    """
    Hot-swap LLM Provider.
    """
    try:
        # Update LLM Provider
        llm.reconfigure(
            provider=config.provider,
            api_key=config.api_key,
            model=config.model,
            endpoint=config.endpoint,
            api_version=config.api_version,
            deployment=config.deployment,
            embedding_deployment=config.embedding_deployment
        )

        # Update Global Settings
        settings.USE_LOCAL_LLM = config.use_local_llm
        
        logger.info(f"Settings Updated: Provider={config.provider}, LocalLLM={settings.USE_LOCAL_LLM}")
        return {"status": "updated", "provider": llm.provider, "use_local_llm": settings.USE_LOCAL_LLM}
    except Exception as e:
        logger.error(f"Settings Update Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))