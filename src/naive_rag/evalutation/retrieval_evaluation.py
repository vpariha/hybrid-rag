import json
import math
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import LanceDB
from langchain_core.documents import Document

from naive_rag.model.model import get_embedding_model
from naive_rag.persistance.vector_store import get_lancedb_vectore_store
from naive_rag.retrival.hybrid_retriever import LanceDBHybridRetriever


def load_dataset(dataset_path: Path) -> list[dict[str, Any]]:
    with dataset_path.open(encoding="utf-8") as dataset_file:
        return json.load(dataset_file)


def get_chunk_ids_from_documents(documents: list[Document]) -> list[str]:
    chunk_ids: list[str] = []
    for document in documents:
        chunk_id = document.metadata.get("chunk_id")
        if not chunk_id:
            chunk_id = document.metadata.get("id")
        chunk_ids.append(chunk_id)
    return chunk_ids


def calculate_metrics(
    relevance_scores: dict[str, int], retrieved_ids: list[str], k: int
) -> dict[str, float]:

    unique_ids = list(dict.fromkeys(retrieved_ids))
    ranked_ids = unique_ids[:k]

    relevant_ids = {
        chunk_id for chunk_id, score in relevance_scores.items() if score >= 2
    }
    hits = sum(chunk_id in relevant_ids for chunk_id in ranked_ids)

    recall = hits / len(relevant_ids) if relevant_ids else 0.0

    precision = hits / k

    reciprocal_rank = next(
        (
            1.0 / rank
            for rank, chunk_id in enumerate(ranked_ids, start=1)
            if chunk_id in relevant_ids
        ),
        0.0,
    )

    actual_scores = [relevance_scores.get(chunk_id, 0) for chunk_id in ranked_ids]
    ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]

    dcg = _compute_dcg(actual_scores)
    ideal_dcg = _compute_dcg(ideal_scores)

    return {
        f"recall_at_{k}": recall,
        f"precision_at_{k}": precision,
        "mrr": reciprocal_rank,
        f"ndcg_at_{k}": dcg / ideal_dcg if ideal_dcg else 0.0,
    }


def _compute_dcg(scores: list[int]) -> float:
    total_dcg = 0.0
    for rank, score in enumerate(scores, start=1):
        gain = 2**score - 1
        discount = math.log2(rank + 1)
        total_dcg += gain / discount
    return total_dcg


def _average_metrics(
    per_query_metrics: list[dict[str, float]], query_count: int
) -> dict[str, float]:
    metric_names = per_query_metrics[0]
    return {
        metric: sum(query[metric] for query in per_query_metrics) / query_count
        for metric in metric_names
    }


def evaluate_dataset(
    dataset: list[dict[str, Any]], vector_store: LanceDB, k: int = 5
) -> dict[str, Any]:

    results: dict[str, list[dict[str, float]]] = {
        "similarity": [],
        "sparse": [],
        "mmr": [],
        "hybrid": [],
    }

    for item in dataset:
        query = item["query"]
        print(f"{query}" + "")
        relevance_scores = item["relevance_scores"]
        hybrid_retriever = LanceDBHybridRetriever(
            vector_store=vector_store,
            embeddings=get_embedding_model(),
            k=k,
        )

        similarity_ids = get_chunk_ids_from_documents(
            vector_store.similarity_search(query, k=k)
        )

        mmr_ids = get_chunk_ids_from_documents(
            vector_store.max_marginal_relevance_search(
                query, k=k, fetch_k=max(20, k * 4)
            )
        )
        bm25_rows = (
            vector_store.get_table()
            .search(query, query_type="fts", fts_columns="text")
            .limit(k)
            .to_list()
        )
        bm25_ids = [row.get("chunk_id") or row.get("id") for row in bm25_rows]
        results["sparse"].append(calculate_metrics(relevance_scores, bm25_ids, k))

        hybrid_ids = get_chunk_ids_from_documents(hybrid_retriever.invoke(query))

        results["hybrid"].append(calculate_metrics(relevance_scores, hybrid_ids, k))
        results["similarity"].append(
            calculate_metrics(relevance_scores, similarity_ids, k)
        )
        results["mmr"].append(calculate_metrics(relevance_scores, mmr_ids, k))

    return {
        "queries_evaluated": len(dataset),
        "k": k,
        "similarity": _average_metrics(results["similarity"], len(dataset)),
        "sparse": _average_metrics(results["sparse"], len(dataset)),
        "mmr": _average_metrics(results["mmr"], len(dataset)),
        "hybrid": _average_metrics(results["hybrid"], len(dataset)),
    }


def main() -> None:
    vector_store = get_lancedb_vectore_store()

    dataset = load_dataset(Path("evaluation.json"))
    metrics = evaluate_dataset(dataset, vector_store, k=5)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
