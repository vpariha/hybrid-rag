from lancedb.rerankers import RRFReranker
from langchain_community.vectorstores import LanceDB
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
from sentence_transformers import CrossEncoder

from naive_rag.config.setting import app_config


class LanceDBHybridRetriever(BaseRetriever):
    vector_store: LanceDB
    embeddings: Embeddings
    cross_encoder: CrossEncoder = CrossEncoder(app_config.RERANKER_MODEL_NAME)
    k: int = 5
    candidate_k: int = 20

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(self, query: str) -> list[Document]:
        rows = (
            self.vector_store.get_table()
            .search(
                query_type="hybrid",
                vector_column_name="vector",
                fts_columns="text",
            )
            .vector(self.embeddings.embed_query(query))
            .text(query)
            .rerank(RRFReranker())
            .limit(self.candidate_k)
            .to_list()
        )

        documents = []
        for row in rows:
            metadata = {
                key: value
                for key, value in row.items()
                if key not in {"text", "vector", "_distance", "_relevance_score"}
            }
            documents.append(Document(page_content=row["text"], metadata=metadata))
        if not documents:
            return []

        scores = self.cross_encoder.predict(
            [[query, document.page_content] for document in documents]
        )
        ranked_documents = sorted(
            zip(scores, documents, strict=True),
            key=lambda item: float(item[0]),
            reverse=True,
        )
        return [document for _, document in ranked_documents[: self.k]]
