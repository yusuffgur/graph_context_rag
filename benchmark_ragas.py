import asyncio
import os
import json
import requests
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    faithfulness,
    answer_relevancy,
)
from src.config import settings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings, AzureChatOpenAI, AzureOpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from datetime import datetime

# 1. Configuration
API_URL = "http://127.0.0.1:8000"

# 2. Setup Wrappers (Using settings)
if settings.LLM_PROVIDER == "azure":
    openai_llm = AzureChatOpenAI(
        api_version=settings.AZURE_OPENAI_API_VERSION,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY,
        azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
    )
    openai_emb = AzureOpenAIEmbeddings(
        api_version=settings.AZURE_OPENAI_API_VERSION,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY,
        azure_deployment=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
    )
else:
    openai_llm = ChatOpenAI(model=settings.BIG_MODEL if hasattr(settings, "BIG_MODEL") else "gpt-4o", api_key=settings.OPENAI_API_KEY)
    openai_emb = OpenAIEmbeddings(model="text-embedding-3-small", api_key=settings.OPENAI_API_KEY)

wrapped_llm = LangchainLLMWrapper(openai_llm)
wrapped_embeddings = LangchainEmbeddingsWrapper(openai_emb)

async def run_system_benchmark():
    # 3. Load Questions
    print("📂 Loading benchmark/dataset.json...")
    try:
        with open("benchmark/dataset.json", "r") as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print("❌ benchmark/dataset.json not found! Please create it.")
        return

    # Data container for Ragas
    data_points = {
        "user_input": [],
        "response": [],
        "retrieved_contexts": [],
        "reference": [] 
    }

    # 4. Query System
    print(f"🚀 Querying Local Graph RAG for {len(raw_data)} questions...")
    
    for item in raw_data:
        q = item.get("question")
        gt = item.get("ground_truth")
        mode = "graph"
        
        print(f"   -> Asking: {q}")
        try:
            # Call your API
            # Graph retrieval + LLM generation can be slow, especially with local models.
            resp = requests.get(f"{API_URL}/query", params={"q": q,"mode": mode}, timeout=300)
            
            if resp.status_code == 200:
                data = resp.json()
                answer = data.get("answer", "")
                
                # RAGAS expects list of strings for contexts
                sources = [s.get("text", "") for s in data.get("sources", [])]
                
                data_points["user_input"].append(q)
                data_points["response"].append(answer)
                data_points["retrieved_contexts"].append(sources)
                data_points["reference"].append(gt) # Single string reference
            else:
                print(f"      ❌ Failed: {resp.status_code}")
                
        except Exception as e:
            print(f"      ❌ Error: {e}")

    if not data_points["user_input"]:
        print("❌ No successful queries to evaluate.")
        return

    # 5. Build Dataset & Evaluate
    dataset = Dataset.from_dict(data_points)
    
    print("\n⚖️  Running RAGAS Evaluation on YOUR System...")
    print("   (This uses LLM-as-a-Judge to score your Graph RAG vs Ground Truth)")
    
    # RunConfig to silence warnings or handle timeouts (optional)
    # n=1 to avoid "requested 3 got 1" warning if needed, but Ragas 0.4 handles defaults best.
    
    results = evaluate(
        dataset=dataset,
        metrics=[
            context_precision,
            faithfulness,
            answer_relevancy,
        ],
        llm=wrapped_llm,
        embeddings=wrapped_embeddings
    )

    print("\n✅ Evaluation Results:")
    print(results)
    
    # Save
    df = results.to_pandas()
    df.to_csv(f"benchmark/system_results_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)
    print("💾 Saved results to benchmark/system_results.csv")

if __name__ == "__main__":
    if "OPENAI_API_KEY" not in os.environ:
         pass # Handled by settings
    asyncio.run(run_system_benchmark())
