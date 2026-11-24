from functools import lru_cache

from langchain_chroma import Chroma
from langchain_community.vectorstores import LanceDB

from naive_rag.config.setting import app_config
from naive_rag.model.model import get_embedding_model


@lru_cache(maxsize=1)
def get_vectore_store():
    return Chroma(
        collection_name=app_config.CHROMA_COLLECTION_NAME,
        persist_directory=app_config.CHROMA_PERSIST_DIRECTORY,
        embedding_function=get_embedding_model(),
        collection_configuration={"hnsw": {"space": "cosine"}},
    )  # You can specify an embedding function if needed


@lru_cache(maxsize=1)
def get_lancedb_vectore_store() -> LanceDB:
    return LanceDB(
        embedding=get_embedding_model(),
        uri=app_config.LANCEDB_URI,
        table_name=app_config.LANCEDB_TABLE_NAME,
        distance="cosine",
        mode="append",
    )
