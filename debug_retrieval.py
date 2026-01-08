import asyncio
import logging
import json
from src.config import settings
from src.modules.llm import ResilientLLM
from src.modules.vector import VectorDB
from src.services.retrieval import RetrievalService
from src.modules.graph import FalkorGraph

# Setup minimal logging
logging.basicConfig(level=logging.ERROR) # Only errors to keep output clean
logger = logging.getLogger("DebugRetrieval")

async def test_retrieval(query: str):
    print(f"\n🔎 Testing Query: '{query}'")
    
    # Init Components
    llm = ResilientLLM()
    # Ensure context limit is set for local models if needed, though default is fine
    llm.context_limit = 4096 
    
    vec_db = VectorDB(settings.QDRANT_URL)
    graph_db = FalkorGraph(settings.FALKOR_URL)
    svc = RetrievalService(llm, vec_db, graph_db) 

    try:
        results = await svc.hybrid_search(query)
        
        # Extract Debug Info
        debug = results.get('debug', {})
        
        print("\n--- DEBUG INFO ---")
        print(f"Original: '{debug.get('original_query', 'N/A')}'")
        print(f"Refined : '{debug.get('refined_query', 'N/A')}'")
        print(f"Provider: {debug.get('llm_provider', 'N/A')}")
        print(f"Candidates: {debug.get('vector_candidates')} Vec -> {debug.get('reranked_candidates')} Reranked")
        
        print("\n--- ANSWER ---")
        print(results['answer'])
        
    except Exception as e:
        print(f"Error in hybrid search: {e}")

if __name__ == "__main__":
    asyncio.run(test_retrieval("type"))
