import asyncio
from src.config import settings
from src.modules.llm import ResilientLLM
from src.utils.processing import recursive_summarize

CONTEXTUAL_SUMMARY_PROMPT = """
Summarize the following text securely:
{text}
"""

async def test_fallback():
    print(f"USE_LOCAL_LLM: {settings.USE_LOCAL_LLM}")
    
    llm = ResilientLLM()
    text = "This is a test " * 500 # Simulate medium text
    
    print("Testing recursive_summarize with fallback...")
    try:
        res = await recursive_summarize(text, llm.generate_local, CONTEXTUAL_SUMMARY_PROMPT)
        print(f"Result: {res[:100]}...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_fallback())
