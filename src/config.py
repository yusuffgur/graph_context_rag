import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OLLAMA_URL = os.getenv("OLLAMA_URL")
    QDRANT_URL = os.getenv("QDRANT_URL")
    FALKOR_URL = os.getenv("FALKOR_URL")
    REDIS_URL = os.getenv("REDIS_URL")
    KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # LLM Config
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower() # openai, gemini, ollama, azure
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    # Azure OpenAI Config
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
    AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")

    # Vector DB Config
    EMBEDDING_DIMENSION = 1536 if LLM_PROVIDER != "ollama" else 768

    SMALL_MODEL = "mistral"
    BIG_MODEL = "gpt-4o"

    # Processing Config
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 4000))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

    # Local Model Toggle
    USE_LOCAL_LLM = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"

settings = Settings()