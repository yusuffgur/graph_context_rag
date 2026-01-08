import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from src.logging_config import setup_logger

logger = setup_logger(__name__)

class DocumentIngestor:
    def load_file(self, path: str):
        if not os.path.exists(path): 
            logger.error(f"File not found: {path}")
            return ""
            
        try:
            ext = os.path.splitext(path)[1].lower()
            if ext == ".pdf":
                loader = PyPDFLoader(path)
            else:
                # Default to text loader for everything else
                loader = TextLoader(path, encoding="utf-8")
                
            docs = loader.load()
            return docs # Return full list of Document objects with metadata
        except Exception as e:
            logger.error(f"Failed to load file {path}: {e}")
            return []