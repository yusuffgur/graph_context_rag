
import asyncio
import os
from src.config import settings
from src.modules.llm import ResilientLLM

# Mock the parts we need
async def test_extraction():
    llm = ResilientLLM()
    question = "What is the primary motivation for integrating security during the requirements analysis phase?"
    
    print(f"❓ Question: {question}")
    
    # Replicating the logic from retrieval.py
    refined_query = question 
    # (Skipping expansion for now as it usually just rephrases)
    
    prompt = f"Analyze the following query and extract the most relevant primary entity (Person, Organization, Project) OR Key Concept (Technical Term, Process) as a single string. If none, return 'None'. Query: '{refined_query}'"
    
    print(f"\n📤 Sending Prompt to {llm.provider}...")
    entity_res = await llm.generate_local(
        prompt,
        system="You are a precise entity and concept extractor. Return ONLY the entity or concept name (e.g., 'Requirements Analysis', 'Security'). Do not explain."
    )
    
    target_entity = entity_res.strip().strip('"').strip("'")
    print(f"\n🔍 Extracted Entity: '{target_entity}'")

if __name__ == "__main__":
    asyncio.run(test_extraction())
