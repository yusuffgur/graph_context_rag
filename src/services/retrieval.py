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
                # Step A: Entity Extraction (Extract UP TO 3 entities)
                entity_res = await self.llm.generate_local(
                    f"Analyze the following query and extract up to 3 mmost relevant primary entities (Person, Organization, Project) OR Key Concepts. Return them as a comma-separated list. If none, return 'None'. Query: '{refined_query}'",
                    system="You are a precise entity extractor. Return ONLY a comma-separated list e.g., 'Project X, Security, John Doe'. No preamble."
                )
                raw_entities = entity_res.strip().strip('"').strip("'")
                
                if raw_entities.lower() == "none" or not raw_entities:
                    target_entities = []
                else:
                    target_entities = [e.strip() for e in raw_entities.split(",") if e.strip()]
                
                logger.info(f"🔍 Extracted Entities: {target_entities}")
                
                # Step B: Get Neighbors & Paths
                graph_context_lines = []
                if target_entities:
                    # 1. Get Neighbors (Relationships)
                    neighbors_data = self.graph_db.query_neighbors(target_entities)
                    # neighbors_data is [[Source, Rel, Target], ...]
                    
                    # 2. Get Paths between entities (if > 1 entity)
                    paths_data = self.graph_db.find_paths(target_entities)
                    
                    # Combine and Format
                    all_graph_data = neighbors_data + paths_data
                    # Deduplicate triples
                    unique_triples = set()
                    for row in all_graph_data:
                        if isinstance(row, (list, tuple)) and len(row) >= 3:
                            # (Source, Rel, Target)
                            t = (row[0], row[1], row[2])
                            unique_triples.add(t)
                            
                    graph_context_str = self._format_graph_response(list(unique_triples))
                    
                    # 3. Get Source Content (Chunks)
                    # Expand search to include neighbors found in the graph
                    expanded_entities = set(target_entities)
                    for s, r, t in unique_triples:
                         expanded_entities.add(s)
                         expanded_entities.add(t)
                    
                    expanded_entities_list = list(expanded_entities)
                    logger.info(f"🕸️ Expanded Graph Search Entities: {expanded_entities_list}")


                    try:
                        all_chunk_ids = []
                        # Batch query for chunks? Currently get_chunks_for_entity is loop-based
                        # We can optimize this later, or update get_chunks_for_entity to accept a list too.
                        # For now, let's just loop over the top X most relevant entities to avoid massive queries
                        # Limit to top 10 entities to keep it fast
                        search_candidates = expanded_entities_list[:10]
                        
                        for ent in search_candidates:
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
                "graph_context": target_entities if mode in ["hybrid", "graph"] else [],
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
        # Expecting list of tuples/lists: [(Source, Rel, Target), ...]
        
        lines = []
        if not graph_data or not isinstance(graph_data, list):
             return "No relationships found."

        try:
            for row in graph_data:
                # Ensure row is a list/tuple and has at least 3 elements [Source, Rel, Target]
                if isinstance(row, (list, tuple)) and len(row) >= 3:
                    lines.append(f"- {row[0]} -[{row[1]}]-> {row[2]}")
                # Backwards compatibility (if only 2 items)
                elif isinstance(row, (list, tuple)) and len(row) >= 2:
                    lines.append(f"- {row[0]} -> {row[1]}")
            
            if not lines:
                return "No meaningful descriptors found."
                
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"Graph formatting error: {e}")
            return "Error formatting graph context."