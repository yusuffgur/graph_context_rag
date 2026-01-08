import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain_openai import ChatOpenAI, OpenAIEmbeddings, AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from src.config import settings
from src.utils.context_manager import get_ollama_context_window
from src.logging_config import setup_logger

logger = setup_logger(__name__)

class ResilientLLM:
    def __init__(self):
        self.context_limit = None
        self.reconfigure(settings.LLM_PROVIDER)

    def reconfigure(self, provider: str, **kwargs):
        """
        Re-initializes the LLM and Embedder based on new provider/settings.
        Update global settings if kwargs provided.
        """
        self.provider = provider.lower()
        logger.info(f"Configuring ResilientLLM with Provider: {self.provider.upper()}")

        # Update Settings (Runtime)
        if "api_key" in kwargs and kwargs["api_key"]:
            if self.provider == "openai": settings.OPENAI_API_KEY = kwargs["api_key"]
            elif self.provider == "gemini": settings.GEMINI_API_KEY = kwargs["api_key"]
            elif self.provider == "azure": settings.AZURE_OPENAI_API_KEY = kwargs["api_key"]
        
        if "model" in kwargs and kwargs["model"]:
            if self.provider == "openai": settings.BIG_MODEL = kwargs["model"]
            # Add other model overrides as needed

        if "endpoint" in kwargs and kwargs["endpoint"]:
             if self.provider == "azure": settings.AZURE_OPENAI_ENDPOINT = kwargs["endpoint"]

        if "api_version" in kwargs and kwargs["api_version"]:
             if self.provider == "azure": settings.AZURE_OPENAI_API_VERSION = kwargs["api_version"]

        if "deployment" in kwargs and kwargs["deployment"]:
             if self.provider == "azure": settings.AZURE_OPENAI_DEPLOYMENT_NAME = kwargs["deployment"]

        if "embedding_deployment" in kwargs and kwargs["embedding_deployment"]:
             if self.provider == "azure": settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = kwargs["embedding_deployment"]

        self._setup_clients()

    def _setup_clients(self):
        # -- Factory Logic --
        if self.provider == "gemini":
            self.cloud_llm = ChatGoogleGenerativeAI(
                model="gemma-3-27b-it", 
                temperature=0, 
                google_api_key=settings.GEMINI_API_KEY,
                convert_system_message_to_human=True
            )
            self.embedder = GoogleGenerativeAIEmbeddings(
                model="gemini-embedding-001",
                google_api_key=settings.GEMINI_API_KEY
            )
        elif self.provider == "azure":
            self.cloud_llm = AzureChatOpenAI(
                azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
                openai_api_version=settings.AZURE_OPENAI_API_VERSION,
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                api_key=settings.AZURE_OPENAI_API_KEY,
                temperature=0
            )
            self.embedder = AzureOpenAIEmbeddings(
                azure_deployment=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
                openai_api_version=settings.AZURE_OPENAI_API_VERSION,
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                api_key=settings.AZURE_OPENAI_API_KEY
            )
        elif self.provider == "ollama":
            self.cloud_llm = ChatOllama(
                base_url=settings.OLLAMA_URL, 
                model=settings.SMALL_MODEL, 
                temperature=0
            )
            self.embedder = OllamaEmbeddings(
                base_url=settings.OLLAMA_URL, 
                model=settings.SMALL_MODEL
            )
        else: # Default: OpenAI
            self.cloud_llm = ChatOpenAI(
                model=settings.BIG_MODEL, 
                temperature=0, 
                api_key=settings.OPENAI_API_KEY
            )
            self.embedder = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)

    async def init_context(self):
        if not self.context_limit:
            self.context_limit = await get_ollama_context_window(settings.OLLAMA_URL, settings.SMALL_MODEL)
            logger.info(f"Local Context Limit detected: {self.context_limit}")

    async def _check_local_model_exists(self, model_name: str) -> bool:
        """Checks if the specific model exists in Ollama's library."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{settings.OLLAMA_URL}/api/tags", timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        models = [m.get("name") for m in data.get("models", [])]
                        # Check availability
                        exists = any(model_name in m for m in models)
                        if not exists:
                            logger.warning(f"Local Model '{model_name}' NOT FOUND in Ollama. Available: {models}")
                        return exists
        except Exception as e:
            logger.warning(f"Failed to check local model availability: {e}")
            return False
        return False

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), retry=retry_if_exception_type(aiohttp.ClientError))
    async def generate_local(self, prompt: str, system: str = None, json_mode: bool = False):
        """
        Prioritizes Local LLM (Ollama) even if global provider is Cloud.
        Falls back to Cloud Provider only if Ollama fails.
        """
        # 1. Check Settings Toggle - Immediate Fallback if Disabled
        if not settings.USE_LOCAL_LLM:
            logger.info("Local LLM Disabled by Settings. Using Cloud Fallback.")
            return await self.generate_cloud(prompt, system or "You are a helpful assistant.", json_mode)

        try:
            # 2. Check Availability
            if not await self._check_local_model_exists(settings.SMALL_MODEL):
                 logger.warning(f"Model {settings.SMALL_MODEL} not available locally. Using Cloud Fallback.")
                 return await self.generate_cloud(prompt, system, json_mode)

            # 3. Try Ollama Generation
            if self.context_limit is None: await self.init_context()
            
            safe_prompt = prompt[:self.context_limit * 3] 
            payload = {
                "model": settings.SMALL_MODEL,
                "prompt": safe_prompt,
                "system": system if system else "",
                "stream": False,
                "options": {
                    "num_ctx": self.context_limit,
                    "temperature": 0.1
                }
            }
            if json_mode:
                payload["format"] = "json"
                
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{settings.OLLAMA_URL}/api/generate", json=payload, timeout=120) as resp:
                    if resp.status == 200:
                         data = await resp.json()
                         return data.get("response", "")
                    else:
                         raise Exception(f"Ollama Error: {resp.status} - {await resp.text()}")

        except Exception as e:
            logger.warning(f"Local LLM Failed ({e}). Falling back to Global Provider ({self.provider}).")
            return await self.generate_cloud(prompt, system or "You are a helpful assistant.")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def generate_cloud(self, prompt: str, system: str = "You are a helpful assistant.", json_mode: bool = False):
        """Synthesis using the selected Provider (OpenAI / Gemini / Ollama)."""
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=prompt)
        ]
        response = await self.cloud_llm.ainvoke(messages)
        return response.content

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    async def get_embedding(self, text):
        return await self.embedder.aembed_query(text)