"""
Dense retrieval metrics and utilities.

Provides tools for:
- Score normalization across different search types
- Latency percentile tracking (p50, p95, p99)
- Index snapshot and persistence
- Retrieval quality metrics
"""

import statistics
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievalMetrics:
    """Track retrieval performance metrics."""

    # Latency tracking (in seconds)
    latencies: list[float] = field(default_factory=list)

    # Score statistics
    scores: list[float] = field(default_factory=list)

    # Query counts
    total_queries: int = 0
    cache_hits: int = 0

    # Per-search-type metrics
    metrics_by_type: dict[str, "RetrievalMetrics"] = field(default_factory=dict)

    def record_query(
        self,
        latency: float,
        scores: list[float],
        search_type: str = "vector",
        cache_hit: bool = False,
    ) -> None:
        """
        Record a single query's metrics.

        Args:
            latency: Query latency in seconds
            scores: List of similarity scores for retrieved documents
            search_type: Type of search (vector, bm25, hybrid)
            cache_hit: Whether this was a cache hit
        """
        self.total_queries += 1
        self.latencies.append(latency)
        self.scores.extend(scores)

        if cache_hit:
            self.cache_hits += 1

        # Track per-type metrics
        if search_type not in self.metrics_by_type:
            self.metrics_by_type[search_type] = RetrievalMetrics()

        type_metrics = self.metrics_by_type[search_type]
        type_metrics.total_queries += 1
        type_metrics.latencies.append(latency)
        type_metrics.scores.extend(scores)
        if cache_hit:
            type_metrics.cache_hits += 1

    def get_latency_percentiles(self) -> dict[str, float]:
        """
        Calculate latency percentiles (p50, p90, p95, p99).

        Returns:
            Dict with percentile values in milliseconds
        """
        if not self.latencies:
            return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}

        sorted_latencies = sorted(self.latencies)
        n = len(sorted_latencies)

        def percentile(p: float) -> float:
            """Calculate percentile value."""
            k = (n - 1) * p
            f = int(k)
            c = f + 1
            if c >= n:
                return sorted_latencies[-1] * 1000  # Convert to ms
            d0 = sorted_latencies[f] * 1000
            d1 = sorted_latencies[c] * 1000
            return d0 + (d1 - d0) * (k - f)

        return {
            "p50": percentile(0.50),
            "p90": percentile(0.90),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
            "mean": statistics.mean(self.latencies) * 1000,
            "min": min(self.latencies) * 1000,
            "max": max(self.latencies) * 1000,
        }

    def get_score_statistics(self) -> dict[str, float]:
        """
        Calculate score statistics.

        Returns:
            Dict with score statistics
        """
        if not self.scores:
            return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}

        return {
            "mean": statistics.mean(self.scores),
            "median": statistics.median(self.scores),
            "min": min(self.scores),
            "max": max(self.scores),
            "std": statistics.stdev(self.scores) if len(self.scores) > 1 else 0.0,
        }

    def get_cache_hit_rate(self) -> float:
        """
        Calculate cache hit rate.

        Returns:
            Cache hit rate as percentage (0-100)
        """
        if self.total_queries == 0:
            return 0.0
        return (self.cache_hits / self.total_queries) * 100

    def get_summary(self) -> dict[str, Any]:
        """
        Get comprehensive metrics summary.

        Returns:
            Dict with all metrics
        """
        summary = {
            "total_queries": self.total_queries,
            "cache_hit_rate": self.get_cache_hit_rate(),
            "latency": self.get_latency_percentiles(),
            "scores": self.get_score_statistics(),
        }

        # Add per-type summaries
        if self.metrics_by_type:
            by_type: dict[str, dict[str, Any]] = {}
            for search_type, metrics in self.metrics_by_type.items():
                by_type[search_type] = {
                    "total_queries": metrics.total_queries,
                    "cache_hit_rate": metrics.get_cache_hit_rate(),
                    "latency": metrics.get_latency_percentiles(),
                    "scores": metrics.get_score_statistics(),
                }
            summary["by_search_type"] = by_type

        return summary

    def reset(self) -> None:
        """Reset all metrics."""
        self.latencies.clear()
        self.scores.clear()
        self.total_queries = 0
        self.cache_hits = 0
        self.metrics_by_type.clear()

    def __repr__(self) -> str:
        """String representation."""
        latency_p = self.get_latency_percentiles()
        return (
            f"RetrievalMetrics(queries={self.total_queries}, "
            f"p50={latency_p['p50']:.2f}ms, p95={latency_p['p95']:.2f}ms, "
            f"cache_hit_rate={self.get_cache_hit_rate():.1f}%)"
        )


def normalize_scores(
    scores: list[float],
    method: str = "minmax",
    score_range: tuple[float, float] | None = None,
) -> list[float]:
    """
    Normalize similarity scores to [0, 1] range.

    Different search types (vector, BM25, hybrid) return scores in different
    ranges. Normalization enables fair comparison and score-based filtering.

    Args:
        scores: List of raw scores to normalize
        method: Normalization method:
            - "minmax": Scale to [0, 1] using min-max scaling
            - "zscore": Standardize using z-score
            - "sigmoid": Apply sigmoid function (for unbounded scores)
        score_range: Optional (min, max) range for the scores.
            If provided, uses this instead of computing from data.

    Returns:
        List of normalized scores in [0, 1] range

    Example:
        >>> scores = [0.8, 0.6, 0.9, 0.7]
        >>> normalize_scores(scores, method="minmax")
        [0.666, 0.0, 1.0, 0.333]

        >>> # BM25 scores (unbounded)
        >>> bm25_scores = [15.3, 8.2, 12.1, 6.5]
        >>> normalize_scores(bm25_scores, method="sigmoid")
        [0.999, 0.997, 0.999, 0.998]
    """
    if not scores:
        return []

    if len(scores) == 1:
        return [1.0]  # Single score is always max

    if method == "minmax":
        if score_range:
            min_score, max_score = score_range
        else:
            min_score = min(scores)
            max_score = max(scores)

        if max_score == min_score:
            return [1.0] * len(scores)

        return [(s - min_score) / (max_score - min_score) for s in scores]

    if method == "zscore":
        mean = statistics.mean(scores)
        std = statistics.stdev(scores) if len(scores) > 1 else 1.0

        if std == 0:
            return [0.5] * len(scores)

        # Z-score then map to [0, 1] using sigmoid
        z_scores = [(s - mean) / std for s in scores]
        return [1 / (1 + pow(2.71828, -z)) for z in z_scores]

    if method == "sigmoid":
        # Apply sigmoid: 1 / (1 + e^(-x))
        # Works well for unbounded scores like BM25
        return [1 / (1 + pow(2.71828, -s)) for s in scores]

    raise ValueError(
        f"Unknown normalization method: {method}. Choose from: minmax, zscore, sigmoid"
    )


class RetrievalTimer:
    """
    Context manager for timing retrieval operations.

    Example:
        >>> metrics = RetrievalMetrics()
        >>> with RetrievalTimer(metrics, scores=[0.9, 0.8], search_type="vector"):
        ...     results = vectorstore.similarity_search(query, k=5)
    """

    def __init__(
        self,
        metrics: RetrievalMetrics,
        scores: list[float] | None = None,
        search_type: str = "vector",
        cache_hit: bool = False,
    ):
        """
        Initialize timer.

        Args:
            metrics: RetrievalMetrics instance to record to
            scores: Similarity scores (can be set in __exit__ if not known yet)
            search_type: Type of search operation
            cache_hit: Whether this was a cache hit
        """
        self.metrics = metrics
        self.scores = scores or []
        self.search_type = search_type
        self.cache_hit = cache_hit
        self.start_time: float = 0.0

    def __enter__(self) -> "RetrievalTimer":
        """Start timing."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop timing and record metrics."""
        latency = time.time() - self.start_time
        self.metrics.record_query(
            latency=latency,
            scores=self.scores,
            search_type=self.search_type,
            cache_hit=self.cache_hit,
        )

    def set_scores(self, scores: list[float]) -> None:
        """Set scores after retrieval (if not known at __enter__)."""
        self.scores = scores


def calculate_mrr(relevant_ranks: list[int]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).

    MRR measures how well the retrieval system ranks relevant documents.
    Higher is better (max 1.0 when relevant doc is always rank 1).

    Args:
        relevant_ranks: List of ranks (1-indexed) where relevant docs appear.
            Use 0 if no relevant doc found for that query.

    Returns:
        MRR score between 0 and 1

    Example:
        >>> # Relevant docs at positions 1, 3, and not found (0)
        >>> calculate_mrr([1, 3, 0])
        0.444  # (1/1 + 1/3 + 0) / 3
    """
    if not relevant_ranks:
        return 0.0

    reciprocal_ranks = [1 / r if r > 0 else 0.0 for r in relevant_ranks]
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def calculate_recall_at_k(retrieved: set[str], relevant: set[str], k: int) -> float:
    """
    Calculate Recall@k metric.

    Recall@k = (# relevant docs in top-k) / (# relevant docs total)

    Args:
        retrieved: Set of retrieved document IDs (top-k)
        relevant: Set of all relevant document IDs
        k: Cutoff rank

    Returns:
        Recall@k score between 0 and 1

    Example:
        >>> retrieved = {"doc1", "doc2", "doc5"}  # top-3
        >>> relevant = {"doc1", "doc3", "doc5", "doc7"}  # 4 relevant
        >>> calculate_recall_at_k(retrieved, relevant, k=3)
        0.5  # Found 2 out of 4 relevant docs
    """
    if not relevant:
        return 1.0  # Perfect recall when there are no relevant docs

    retrieved_k = set(list(retrieved)[:k])
    relevant_in_k = retrieved_k & relevant
    return len(relevant_in_k) / len(relevant)


def calculate_precision_at_k(retrieved: set[str], relevant: set[str], k: int) -> float:
    """
    Calculate Precision@k metric.

    Precision@k = (# relevant docs in top-k) / k

    Args:
        retrieved: Set of retrieved document IDs (top-k)
        relevant: Set of all relevant document IDs
        k: Cutoff rank

    Returns:
        Precision@k score between 0 and 1

    Example:
        >>> retrieved = {"doc1", "doc2", "doc5"}  # top-3
        >>> relevant = {"doc1", "doc3", "doc5", "doc7"}  # 4 relevant
        >>> calculate_precision_at_k(retrieved, relevant, k=3)
        0.666  # 2 out of 3 retrieved are relevant
    """
    if k == 0:
        return 0.0

    retrieved_k = set(list(retrieved)[:k])
    relevant_in_k = retrieved_k & relevant
    return len(relevant_in_k) / k
