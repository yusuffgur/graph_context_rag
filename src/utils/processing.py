from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import settings

def split_text(text, chunk_size=None, chunk_overlap=None):
    size = chunk_size or settings.CHUNK_SIZE
    overlap = chunk_overlap or settings.CHUNK_OVERLAP
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    return splitter.split_text(text)

async def recursive_summarize(text, llm_func, prompt_template):
    """Hierarchical summarization for large docs."""
    # If small enough, just summarize
    if len(text) < 12000:
        return await llm_func(prompt_template.format(text=text))
    
    # Else, split and conquer
    mid = len(text) // 2
    part1 = await recursive_summarize(text[:mid], llm_func, prompt_template)
    part2 = await recursive_summarize(text[mid:], llm_func, prompt_template)
    
    merge_prompt = f"Merge these two summaries:\n1. {part1}\n2. {part2}"
    return await llm_func(merge_prompt)