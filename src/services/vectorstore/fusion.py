"""
Fusion methods for combining multiple retrieval results.

Supports Reciprocal Rank Fusion (RRF) and weighted score fusion
for combining results from vector, BM25, sparse, and hybrid searches.
"""

import logging
from collections import defaultdict

from langchain_core.documents import Document

from src.schemas.services.vectorstore import FusionConfig

logger = logging.getLogger(__name__)


class FusionResult:
    """Result from fusion of multiple searches."""

    def __init__(
        self,
        documents: list[Document],
        fusion_scores: dict[str, float],
        component_ranks: dict[str, dict[str, int]],
        method: str,
    ):
        """
        Initialize fusion result.

        Args:
            documents: Fused and ranked documents
            fusion_scores: Final fusion scores for each doc
            component_ranks: Ranks from each search component
            method: Fusion method used
        """
        self.documents = documents
        self.fusion_scores = fusion_scores
        self.component_ranks = component_ranks
        self.method = method

    def get_top_k(self, k: int) -> list[Document]:
        """Get top k documents."""
        return self.documents[:k]


def reciprocal_rank_fusion(
    results: dict[str, list[Document]],
    k: int = 60,
    tie_break_strategy: str = "score",
) -> FusionResult:
    """
    Combine results using Reciprocal Rank Fusion (RRF).

    RRF formula: score(d) = Σ 1/(k + rank(d))
    where rank(d) is the rank of document d in each result list.

    Args:
        results: Dict mapping search type to ranked documents
        k: RRF constant (default 60)
        tie_break_strategy: How to break ties ("score", "rank", "stable")

    Returns:
        FusionResult with combined rankings

    Example:
        >>> results = {
        ...     "vector": [doc1, doc2, doc3],
        ...     "bm25": [doc2, doc1, doc4],
        ...     "sparse": [doc1, doc3, doc2]
        ... }
        >>> fused = reciprocal_rank_fusion(results, k=60)
        >>> top_docs = fused.get_top_k(5)
    """
    # Track document scores and ranks
    doc_scores: dict[str, float] = defaultdict(float)
    doc_ranks: dict[str, dict[str, int]] = defaultdict(dict)
    doc_objects: dict[str, Document] = {}

    # Compute RRF scores
    for search_type, docs in results.items():
        for rank, doc in enumerate(docs, start=1):
            doc_id = _get_doc_id(doc)
            rrf_score = 1.0 / (k + rank)
            doc_scores[doc_id] += rrf_score
            doc_ranks[doc_id][search_type] = rank
            doc_objects[doc_id] = doc

    # Sort by RRF score (descending)
    sorted_doc_ids = sorted(
        doc_scores.keys(),
        key=lambda doc_id: (
            doc_scores[doc_id],
            _tie_breaker(doc_id, doc_objects, doc_ranks, tie_break_strategy),
        ),
        reverse=True,
    )

    # Build result with copies to avoid mutating originals
    fused_docs = []
    for doc_id in sorted_doc_ids:
        original_doc = doc_objects[doc_id]
        # Create new document with fusion metadata
        fused_doc = Document(
            page_content=original_doc.page_content,
            metadata={
                **original_doc.metadata,
                "fusion_score": doc_scores[doc_id],
                "fusion_method": "rrf",
                "component_ranks": doc_ranks[doc_id],
            },
        )
        fused_docs.append(fused_doc)

    return FusionResult(
        documents=fused_docs,
        fusion_scores=dict(doc_scores),
        component_ranks=dict(doc_ranks),
        method="rrf",
    )


def weighted_fusion(
    results: dict[str, list[Document]],
    weights: dict[str, float] | None = None,
    normalize_scores: bool = True,
    tie_break_strategy: str = "score",
) -> FusionResult:
    """
    Combine results using weighted score fusion.

    Normalizes scores from each search type and combines them
    using configurable weights.

    Args:
        results: Dict mapping search type to ranked documents
        weights: Weights for each search type (defaults to equal)
        normalize_scores: Normalize scores to [0, 1] before fusion
        tie_break_strategy: How to break ties ("score", "rank", "stable")

    Returns:
        FusionResult with combined rankings

    Example:
        >>> results = {
        ...     "vector": [doc1, doc2, doc3],
        ...     "bm25": [doc2, doc1, doc4],
        ... }
        >>> weights = {"vector": 0.7, "bm25": 0.3}
        >>> fused = weighted_fusion(results, weights=weights)
    """
    # Default to equal weights
    if weights is None:
        weights = {search_type: 1.0 / len(results) for search_type in results}

    # Validate weights
    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 0.01:
        logger.warning(f"Weights sum to {total_weight}, normalizing to 1.0")
        weights = {k: v / total_weight for k, v in weights.items()}

    # Track document scores and ranks
    doc_scores: dict[str, float] = defaultdict(float)
    doc_component_scores: dict[str, dict[str, float]] = defaultdict(dict)
    doc_ranks: dict[str, dict[str, int]] = defaultdict(dict)
    doc_objects: dict[str, Document] = {}

    # Normalize and combine scores
    for search_type, docs in results.items():
        weight = weights.get(search_type, 0.0)
        if weight == 0.0:
            continue

        # Extract scores
        scores = [doc.metadata.get("score", 0.0) for doc in docs]

        # Normalize if requested
        if normalize_scores and scores:
            scores = _normalize_scores_minmax(scores)

        # Add weighted scores
        for rank, (doc, score) in enumerate(zip(docs, scores, strict=True), start=1):
            doc_id = _get_doc_id(doc)
            weighted_score = score * weight
            doc_scores[doc_id] += weighted_score
            doc_component_scores[doc_id][search_type] = score
            doc_ranks[doc_id][search_type] = rank
            doc_objects[doc_id] = doc

    # Sort by weighted score (descending)
    sorted_doc_ids = sorted(
        doc_scores.keys(),
        key=lambda doc_id: (
            doc_scores[doc_id],
            _tie_breaker(doc_id, doc_objects, doc_ranks, tie_break_strategy),
        ),
        reverse=True,
    )

    # Build result with copies to avoid mutating originals
    fused_docs = []
    for doc_id in sorted_doc_ids:
        original_doc = doc_objects[doc_id]
        # Create new document with fusion metadata
        fused_doc = Document(
            page_content=original_doc.page_content,
            metadata={
                **original_doc.metadata,
                "fusion_score": doc_scores[doc_id],
                "fusion_method": "weighted",
                "component_scores": doc_component_scores[doc_id],
                "component_ranks": doc_ranks[doc_id],
                "fusion_weights": weights,
            },
        )
        fused_docs.append(fused_doc)

    return FusionResult(
        documents=fused_docs,
        fusion_scores=dict(doc_scores),
        component_ranks=dict(doc_ranks),
        method="weighted",
    )


def fuse_results(
    results: dict[str, list[Document]],
    config: FusionConfig | None = None,
) -> FusionResult:
    """
    Fuse multiple search results using configured method.

    Args:
        results: Dict mapping search type to ranked documents
        config: Fusion configuration

    Returns:
        FusionResult with combined rankings

    Example:
        >>> config = FusionConfig(method="rrf", rrf_k=60)
        >>> fused = fuse_results(results, config)
    """
    if config is None:
        config = FusionConfig()

    if config.method == "rrf":
        return reciprocal_rank_fusion(
            results,
            k=config.rrf_k,
            tie_break_strategy=config.tie_break_strategy,
        )
    if config.method == "weighted":
        return weighted_fusion(
            results,
            weights=config.weights,
            normalize_scores=config.normalize_scores,
            tie_break_strategy=config.tie_break_strategy,
        )
    raise ValueError(f"Unknown fusion method: {config.method}")


# ─── Helper Functions ────────────────────────────────────────────────────────


def _get_doc_id(doc: Document) -> str:
    """Get unique document identifier."""
    # Use chunk_id if available, otherwise hash content
    if "chunk_id" in doc.metadata:
        return str(doc.metadata["chunk_id"])
    # Use first 100 chars of content as ID
    return doc.page_content[:100]


def _normalize_scores_minmax(scores: list[float]) -> list[float]:
    """Normalize scores to [0, 1] using min-max normalization."""
    if not scores:
        return scores

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [1.0] * len(scores)

    return [(s - min_score) / (max_score - min_score) for s in scores]


def _tie_breaker(
    doc_id: str,
    doc_objects: dict[str, Document],
    doc_ranks: dict[str, dict[str, int]],
    strategy: str,
) -> tuple[float, ...] | tuple[str, ...]:
    """
    Compute tie-breaking value for sorting.

    Returns tuple for lexicographic sorting (lower is better).

    Args:
        doc_id: Document identifier
        doc_objects: All document objects
        doc_ranks: Ranks from each component
        strategy: Tie-breaking strategy

    Returns:
        Tuple of values for tie-breaking
    """
    if strategy == "score":
        # Use original score (negated for descending sort)
        doc = doc_objects[doc_id]
        original_score = doc.metadata.get("score", 0.0)
        return (-original_score,)

    if strategy == "rank":
        # Use average rank across components (lower is better)
        ranks = list(doc_ranks[doc_id].values())
        avg_rank = sum(ranks) / len(ranks) if ranks else float("inf")
        return (avg_rank,)

    if strategy == "stable":
        # Use document ID for stable sorting
        return (doc_id,)

    raise ValueError(f"Unknown tie-break strategy: {strategy}")
