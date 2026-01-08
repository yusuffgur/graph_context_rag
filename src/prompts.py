CONTEXTUAL_SUMMARY_PROMPT = """
You are an expert technical writer. 
Your task is to provide a comprehensive summary of the document provided below. 
Requirements:
1. Identify the main Subject, Key Entities (People, Companies), and Dates.
2. Summarize the core purpose of the document.
3. Keep it dense and factual (approx. 200 words).

Document Text:
{text}
"""

CONTEXTUAL_HEADER_PROMPT = """
<document_context>
{doc_summary}
</document_context>
<chunk_content>
{chunk_text}
</chunk_content>
Task: Write a brief **"Contextual Header"** (1-2 sentences) that explains what this specific chunk is about *in the context of the whole document*.
Your Header:
"""

GRAPH_EXTRACTION_SYSTEM = """
You are a Knowledge Graph Engineer. Output a VALID JSON object containing a list of entities and a list of relationships.
Rules:
1. **Entities**: Extract People, Organizations, Locations, Projects, Key Concepts, Technical Terms, Processes, Methodologies etc.
2. **Relationships**: Use specific verbs (e.g., "MANAGED_BY", "LOCATED_IN", "RELATES_TO", "PART_OF").
Output Format:
{
  "entities": [{"name": "Entity Name", "type": "PERSON/ORG/CONCEPT"}],
  "relationships": [{"source": "Entity Name", "target": "Entity Name", "relation": "RELATION_TYPE"}]
}
"""

GRAPH_EXTRACTION_USER = """
Analyze the following text and extract the knowledge graph:
{text}
"""