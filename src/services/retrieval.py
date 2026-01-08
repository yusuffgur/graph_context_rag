import json
from tenacity import retry, stop_after_attempt, wait_exponential
from src.modules.llm import ResilientLLM
from src.modules.vector import VectorDB
from src.modules.graph import FalkorGraph
from src.logging_config import setup_logger

from flashrank import Ranker, RerankRequest

logger = setup_logger("RetrievalService")

class RetrievalService:
    def __init__(self, llm: ResilientLLM, vec_db: VectorDB, graph_db: FalkorGraph):
        self.llm = llm
        self.vec_db = vec_db
        self.graph_db = graph_db
        # Initialize Reranker (Small, fast model by default)
        self.ranker = Ranker()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=3))
    async def hybrid_search(self, query: str, file_filter: str = None, mode: str = "hybrid"):
        """
        Orchestrates the Federated Search with Query Expansion.
        mode: 'hybrid' (default), 'vector', 'graph'
        """
        try:
            # 0. Query Expansion (Smart Refinement)
            # If query is short (< 5 words), try to expand it contextually or make it a question
            refined_query = query
            if len(query.split()) < 5:
                refinement_res = await self.llm.generate_local(
                    f"The user provided a short, ambiguous search term for a RAG system. "
                    f"Expand this term into a broad question that asks for 'definitions, categories, or examples' of the term within the document. "
                    f"Avoid assuming a specific industry or domain. "
                    f"Term: '{query}'",
                    system="You are a semantic query expander. Return ONLY the broad question."
                )
                refined_query = refinement_res.strip().strip('"')
            
            target_entity = ""
            graph_points = []
            
            # --- PATH A: GRAPH SEARCH (Skip if mode='vector') ---
            if mode in ["hybrid", "graph"]:
                # Step A: Entity Extraction
                entity_res = await self.llm.generate_local(
                    f"Analyze the following query and extract the most relevant primary entity (Person, Organization, Project) OR Key Concept (Technical Term, Process) as a single string. If none, return 'None'. Query: '{refined_query}'",
                    system="You are a precise entity and concept extractor. Return ONLY the entity or concept name (e.g., 'Requirements Analysis', 'Security'). Do not explain."
                )
                target_entity = entity_res.strip().strip('"').strip("'")
                logger.info(f"🔍 Extracted Entity: '{target_entity}'")
                
                if target_entity.lower() == "none":
                    target_entity = ""
                
                # Step B: Get Neighbors & Related Chunks (Graph Retrieval)
                graph_context_str = ""
                if target_entity:
                    # 1. Get Neighbors (Relationships)
                    graph_data = self.graph_db.query_neighbors(target_entity)
                    logger.info(f"🕸️ Graph Neighbors for '{target_entity}': {graph_data}")
                    
                    # Format Graph Context for LLM
                    graph_context_str = self._format_graph_response(graph_data)
                    
                    # 2. Get Source Content (Chunks)
                    # Strategy: Get chunks for Target Entity + Chunks for Neighbors (1-hop expansion)
                    target_entities = [target_entity]
                    
                    # Add neighbors to search list
                    if graph_data:
                        for row in graph_data:
                            if len(row) >= 2:
                                neighbor_name = row[1]
                                target_entities.append(neighbor_name)
                    
                    # Deduplicate
                    target_entities = list(set(target_entities))
                    logger.info(f"🕸️ Expanded Graph Search Entities: {target_entities}")

                    try:
                        all_chunk_ids = []
                        for ent in target_entities:
                            c_ids = self.graph_db.get_chunks_for_entity(ent, file_filter=file_filter)
                            all_chunk_ids.extend(c_ids)
                        
                        # Deduplicate chunk IDs
                        all_chunk_ids = list(set(all_chunk_ids))
                        logger.info(f"🔗 Linked Chunk IDs (Total): {len(all_chunk_ids)}")
                       
                        if all_chunk_ids:
                            # Limit graph chunks (increased limit for expanded search)
                            top_chunk_ids = all_chunk_ids[:25] 
                            graph_points = self.vec_db.get_by_ids(top_chunk_ids)
                            logger.info(f"📄 Retrieved {len(graph_points)} Chunk Payloads from Vector DB via Graph")
                    except Exception as e:
                        logger.error(f"Error fetching graph chunks: {e}")

            # --- PATH B: VECTOR SEARCH (Skip if mode='graph') ---
            vec_results = []
            if mode in ["hybrid", "vector"]:
                # 1. Vector Search (Using Refined Query)
                q_vec = await self.llm.get_embedding(refined_query)
                # Increase limit for better recall
                vec_results = self.vec_db.search(q_vec, limit=20, file_filter=file_filter)
            
            # 2. Unified Candidate Pooling & Deduplication
            seen_ids = set()
            pass_through_docs = []
            
            # Add Vector results
            for r in vec_results:
                if r.id not in seen_ids:
                    seen_ids.add(r.id)
                    pass_through_docs.append({"id": r.id, "text": r.payload['text'], "meta": r.payload})

            # Add Graph results
            for r in graph_points:
                if r.id not in seen_ids:
                    seen_ids.add(r.id)
                    pass_through_docs.append({"id": r.id, "text": r.payload['text'], "meta": r.payload})
            
            logger.info(f"🔀 Unified Reranking Pool: {len(pass_through_docs)} documents (Mode: {mode})")

            # 3. Reranking (FlashRank)
            # Even if only 1 source, we rerank to get score
            if not pass_through_docs:
                logger.warning(f"⚠️ No documents found from {mode} search. Skipping Rerank.")
                reranked_results = []
            else:
                rerank_request = RerankRequest(query=refined_query, passages=pass_through_docs)
                reranked_results = self.ranker.rerank(rerank_request)
            
            # Select Top 8 Winners (Increased from 5 per user request)
            top_docs = reranked_results[:8]

            ctx_text = "\n".join([
                f"- {r['text']} (Src: {r['meta'].get('source')})" 
                for r in top_docs
            ])

            # 4. Synthesis
            final_prompt = f"""
            You are a helpful assistant. 
            Answer the user's query mostly based on the provided Context.
            
            - If the Context mentions the term, summarize its usage, examples, or categories found.
            - If the answer is NOT in the Context, say "I cannot find the answer in the provided documents."
            - Cite the source filename if possible.

            Original Query: {query}
            Refined Intent: {refined_query}

            [Graph Relationships]
            {graph_context_str if mode != 'vector' else 'N/A'}

            [Relevant Knowledge ({mode.upper()})]
            {ctx_text}
            
            Your Answer:
            """
            
            answer = await self.llm.generate_cloud(final_prompt)
            
            return {
                "answer": answer,
                # Return full metadata for UI to render
                "sources": [
                    {
                        "source": r['meta'].get('source'),
                        "text": r['text'],
                        "score": float(r.get('score', 0)), # FlashRank score cast to float
                        "chunk_index": r['meta'].get('chunk_index', 0),
                        "page_number": r['meta'].get('page_number', 1)
                    } 
                    for r in top_docs
                ],
                "graph_context": target_entity,
                "debug": {
                    "mode": mode,
                    "original_query": query,
                    "refined_query": refined_query,
                    "final_prompt": final_prompt,
                    "vector_candidates": len(vec_results),
                    "graph_candidates": len(graph_points),
                    "total_candidates": len(pass_through_docs),
                    "reranked_candidates": len(reranked_results),
                    "llm_provider": self.llm.provider
                }
            }

        except Exception as e:
            logger.error(f"Retrieval Logic Failed: {e}")
            raise e # Tenacity will catch this and retry

    def _format_graph_response(self, graph_data):
        """Helper to format nested list from RedisGraph."""
        # The 'query_neighbors' method in graph.py now extracts 'res[1]',
        # which is the list of records: [['WORKS_AT', 'Acme'], ...]
        
        lines = []
        if not graph_data or not isinstance(graph_data, list):
             return "No relationships found."

        try:
            for row in graph_data:
                # Ensure row is a list/tuple and has at least 2 elements
                if isinstance(row, (list, tuple)) and len(row) >= 2:
                    lines.append(f"- {row[0]} -> {row[1]}")
            
            if not lines:
                return "No meaningful descriptors found."
                
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"Graph formatting error: {e}")
            return "Error formatting graph context."
            headers = raw_data[0]
            rows = raw_data[1]
            
            if not rows:
                return "Entity found in graph, but has no direct connections."
            
            lines = []
            for row in rows:
                if len(row) >= 2:
                    rel_type = row[0]
                    target_name = row[1]
                    lines.append(f"- {rel_type} -> {target_name}")
            
            return "\n".join(lines) if lines else "No meaningful connections."
            
        except Exception as e:
            logger.error(f"Graph Parse Error: {e}")
            return str(raw_data) # Fallback to raw if logic fails