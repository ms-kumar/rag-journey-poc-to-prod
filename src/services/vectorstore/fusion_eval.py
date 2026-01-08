"""
Evaluation metrics for retrieval fusion.

Provides recall@k, precision@k, and uplift measurement for comparing
fusion methods against individual search strategies.
"""

import logging
from dataclasses import dataclass

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Metrics for retrieval evaluation."""

    recall_at_k: dict[int, float]
    precision_at_k: dict[int, float]
    mrr: float  # Mean Reciprocal Rank
    map: float  # Mean Average Precision
    ndcg: float  # Normalized Discounted Cumulative Gain
    total_relevant: int
    total_retrieved: int


@dataclass
class UpliftMetrics:
    """Uplift metrics comparing fusion to baselines."""

    fusion_recall: dict[int, float]
    baseline_recalls: dict[str, dict[int, float]]
    recall_uplift: dict[str, dict[int, float]]  # % improvement over baseline
    best_baseline_recall: dict[int, float]
    uplift_over_best: dict[int, float]


def calculate_recall_at_k(
    retrieved_docs: list[Document],
    relevant_doc_ids: set[str],
    k_values: list[int] | None = None,
) -> dict[int, float]:
    """
    Calculate recall@k for retrieved documents.

    Recall@k = (# relevant docs in top-k) / (# total relevant docs)

    Args:
        retrieved_docs: Retrieved documents in ranked order
        relevant_doc_ids: Set of relevant document IDs
        k_values: k values to compute (defaults to [1, 3, 5, 10, 20])

    Returns:
        Dict mapping k to recall@k
    """
    if k_values is None:
        k_values = [1, 3, 5, 10, 20]

    if not relevant_doc_ids:
        return dict.fromkeys(k_values, 0.0)

    recall = {}
    for k in k_values:
        top_k = retrieved_docs[:k]
        retrieved_ids = {_get_doc_id(doc) for doc in top_k}
        relevant_retrieved = retrieved_ids & relevant_doc_ids
        recall[k] = len(relevant_retrieved) / len(relevant_doc_ids)

    return recall


def calculate_precision_at_k(
    retrieved_docs: list[Document],
    relevant_doc_ids: set[str],
    k_values: list[int] | None = None,
) -> dict[int, float]:
    """
    Calculate precision@k for retrieved documents.

    Precision@k = (# relevant docs in top-k) / k

    Args:
        retrieved_docs: Retrieved documents in ranked order
        relevant_doc_ids: Set of relevant document IDs
        k_values: k values to compute

    Returns:
        Dict mapping k to precision@k
    """
    if k_values is None:
        k_values = [1, 3, 5, 10, 20]

    if not relevant_doc_ids:
        return dict.fromkeys(k_values, 0.0)

    precision = {}
    for k in k_values:
        top_k = retrieved_docs[:k]
        retrieved_ids = {_get_doc_id(doc) for doc in top_k}
        relevant_retrieved = retrieved_ids & relevant_doc_ids
        precision[k] = len(relevant_retrieved) / k if k > 0 else 0.0

    return precision


def calculate_mrr(
    retrieved_docs: list[Document],
    relevant_doc_ids: set[str],
) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).

    MRR = 1 / (rank of first relevant document)

    Args:
        retrieved_docs: Retrieved documents in ranked order
        relevant_doc_ids: Set of relevant document IDs

    Returns:
        MRR score
    """
    for rank, doc in enumerate(retrieved_docs, start=1):
        doc_id = _get_doc_id(doc)
        if doc_id in relevant_doc_ids:
            return 1.0 / rank
    return 0.0


def calculate_map(
    retrieved_docs: list[Document],
    relevant_doc_ids: set[str],
) -> float:
    """
    Calculate Mean Average Precision (MAP).

    MAP = (Σ P@k * rel(k)) / # relevant docs
    where P@k is precision at k and rel(k) is 1 if doc at k is relevant.

    Args:
        retrieved_docs: Retrieved documents in ranked order
        relevant_doc_ids: Set of relevant document IDs

    Returns:
        MAP score
    """
    if not relevant_doc_ids:
        return 0.0

    num_relevant = 0
    sum_precisions = 0.0

    for rank, doc in enumerate(retrieved_docs, start=1):
        doc_id = _get_doc_id(doc)
        if doc_id in relevant_doc_ids:
            num_relevant += 1
            precision_at_rank = num_relevant / rank
            sum_precisions += precision_at_rank

    return sum_precisions / len(relevant_doc_ids) if relevant_doc_ids else 0.0


def calculate_ndcg(
    retrieved_docs: list[Document],
    relevant_doc_ids: set[str],
    k: int | None = None,
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG@k).

    DCG = Σ rel(i) / log2(i + 1)
    NDCG = DCG / IDCG (ideal DCG)

    Args:
        retrieved_docs: Retrieved documents in ranked order
        relevant_doc_ids: Set of relevant document IDs
        k: Cutoff (None = use all docs)

    Returns:
        NDCG score
    """
    import math

    if not relevant_doc_ids:
        return 0.0

    if k is not None:
        retrieved_docs = retrieved_docs[:k]

    # Calculate DCG
    dcg = 0.0
    for rank, doc in enumerate(retrieved_docs, start=1):
        doc_id = _get_doc_id(doc)
        relevance = 1.0 if doc_id in relevant_doc_ids else 0.0
        dcg += relevance / math.log2(rank + 1)

    # Calculate ideal DCG (all relevant docs at top)
    num_relevant = min(len(relevant_doc_ids), len(retrieved_docs))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(num_relevant))

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(
    retrieved_docs: list[Document],
    relevant_doc_ids: set[str],
    k_values: list[int] | None = None,
) -> EvaluationMetrics:
    """
    Comprehensive evaluation of retrieval results.

    Args:
        retrieved_docs: Retrieved documents in ranked order
        relevant_doc_ids: Set of relevant document IDs
        k_values: k values for recall/precision

    Returns:
        EvaluationMetrics with all metrics
    """
    if k_values is None:
        k_values = [1, 3, 5, 10, 20]

    return EvaluationMetrics(
        recall_at_k=calculate_recall_at_k(retrieved_docs, relevant_doc_ids, k_values),
        precision_at_k=calculate_precision_at_k(retrieved_docs, relevant_doc_ids, k_values),
        mrr=calculate_mrr(retrieved_docs, relevant_doc_ids),
        map=calculate_map(retrieved_docs, relevant_doc_ids),
        ndcg=calculate_ndcg(retrieved_docs, relevant_doc_ids),
        total_relevant=len(relevant_doc_ids),
        total_retrieved=len(retrieved_docs),
    )


def calculate_uplift(
    fusion_docs: list[Document],
    baseline_results: dict[str, list[Document]],
    relevant_doc_ids: set[str],
    k_values: list[int] | None = None,
) -> UpliftMetrics:
    """
    Calculate recall uplift of fusion over baseline methods.

    Args:
        fusion_docs: Documents from fusion
        baseline_results: Dict of baseline search results
        relevant_doc_ids: Set of relevant document IDs
        k_values: k values for recall

    Returns:
        UpliftMetrics with uplift percentages

    Example:
        >>> uplift = calculate_uplift(
        ...     fusion_docs=fused_results,
        ...     baseline_results={"vector": vector_docs, "bm25": bm25_docs},
        ...     relevant_doc_ids=ground_truth,
        ... )
        >>> print(f"Uplift over vector@10: {uplift.recall_uplift['vector'][10]:.1f}%")
    """
    if k_values is None:
        k_values = [1, 3, 5, 10, 20]

    # Calculate fusion recall
    fusion_recall = calculate_recall_at_k(fusion_docs, relevant_doc_ids, k_values)

    # Calculate baseline recalls
    baseline_recalls = {}
    for method, docs in baseline_results.items():
        baseline_recalls[method] = calculate_recall_at_k(docs, relevant_doc_ids, k_values)

    # Calculate uplift percentages
    recall_uplift: dict[str, dict[int, float]] = {}
    for method, baseline_recall in baseline_recalls.items():
        recall_uplift[method] = {}
        for k in k_values:
            baseline_val = baseline_recall[k]
            fusion_val = fusion_recall[k]
            if baseline_val > 0:
                uplift_pct = ((fusion_val - baseline_val) / baseline_val) * 100.0
            else:
                uplift_pct = 100.0 if fusion_val > 0 else 0.0
            recall_uplift[method][k] = uplift_pct

    # Find best baseline at each k
    best_baseline_recall = {}
    for k in k_values:
        best_val = max(baseline_recalls[m][k] for m in baseline_recalls)
        best_baseline_recall[k] = best_val

    # Calculate uplift over best baseline
    uplift_over_best = {}
    for k in k_values:
        best_val = best_baseline_recall[k]
        fusion_val = fusion_recall[k]
        if best_val > 0:
            uplift_pct = ((fusion_val - best_val) / best_val) * 100.0
        else:
            uplift_pct = 100.0 if fusion_val > 0 else 0.0
        uplift_over_best[k] = uplift_pct

    return UpliftMetrics(
        fusion_recall=fusion_recall,
        baseline_recalls=baseline_recalls,
        recall_uplift=recall_uplift,
        best_baseline_recall=best_baseline_recall,
        uplift_over_best=uplift_over_best,
    )


# ─── Helper Functions ────────────────────────────────────────────────────────


def _get_doc_id(doc: Document) -> str:
    """Get unique document identifier."""
    if "chunk_id" in doc.metadata:
        return str(doc.metadata["chunk_id"])
    return doc.page_content[:100]
