import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class AppConfig(BaseSettings):
    CHUNK_SIZE: int = os.getenv("CHUNK_SIZE")
    CHUNK_OVERLAP: int = os.getenv("CHUNK_OVERLAP")
    TOP_K: int = os.getenv("TOP_K")

    CANDIDATE_K: int = int(os.getenv("RERANK_CANDIDATE_K"))
    RERANKER_MODEL_NAME: str = os.getenv("RERANKER_MODEL_NAME")

    EMBEDDING_MODEL_PROVIDER: str = os.getenv("EMBEDDING_MODEL_PROVIDER")
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME")

    CHROMA_PERSIST_DIRECTORY: str = os.getenv("CHROMA_PERSIST_DIRECTORY")
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME")

    LANCEDB_URI: str = os.getenv("LANCEDB_URI")
    LANCEDB_TABLE_NAME: str = os.getenv("LANCEDB_TABLE_NAME")

    CHAT_MODEL_PROVIDER: str = os.getenv("CHAT_MODEL_PROVIDER")
    CHAT_MODEL_NAME: str = os.getenv("CHAT_MODEL_NAME")
    CHAT_MODEL_TEMPERATURE: float = os.getenv("CHAT_MODEL_TEMPERATURE", "0.7")


app_config = AppConfig()
