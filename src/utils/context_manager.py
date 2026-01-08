import aiohttp
import re

DEFAULT_CONTEXT = 4096

# Regex to find context window in Ollama Modelfile if structured data is missing
CTX_PATTERN = re.compile(r'num_ctx\s+(\d+)')

async def get_ollama_context_window(base_url: str, model_name: str) -> int:
    """
    Probes the running Ollama instance to find the configured context window.
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{base_url}/api/show", json={"name": model_name}) as resp:
                if resp.status != 200:
                    print(f"⚠️ Could not fetch model info for {model_name}. Using default.")
                    return DEFAULT_CONTEXT
                
                data = await resp.json()
                
                # 1. Try structured details (newer Ollama versions)
                if "details" in data and "context_length" in data["details"]:
                    return int(data["details"]["context_length"])

                # 2. Parse Modelfile parameters
                if "parameters" in data:
                    match = CTX_PATTERN.search(data["parameters"])
                    if match:
                        return int(match.group(1))
                        
                # 3. Parse Modelfile raw text
                if "modelfile" in data:
                    match = CTX_PATTERN.search(data["modelfile"])
                    if match:
                        return int(match.group(1))

        print(f"ℹ️ Model {model_name} context not explicit. Using default {DEFAULT_CONTEXT}")
        return DEFAULT_CONTEXT
    except Exception as e:
        print(f"⚠️ Error probing Ollama: {e}")
        return DEFAULT_CONTEXT

def get_openai_context_window(model_name: str) -> int:
    """Safe lookup for OpenAI models."""
    specs = {
        "gpt-4o": 128000,
        "gpt-4-turbo": 128000,
        "gpt-4": 8192,
        "gpt-3.5-turbo": 16385,
    }
    # Return known limit or 128k for modern models, 4k for safety
    return specs.get(model_name, 4096)