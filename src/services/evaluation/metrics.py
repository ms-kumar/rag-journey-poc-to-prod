"""
Comprehensive metrics for RAG evaluation.

Provides retrieval metrics (precision, recall, MRR, NDCG) and
generation metrics (faithfulness, relevance, answer quality).
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types of evaluation metrics."""

    # Retrieval metrics
    PRECISION_AT_K = "precision@k"
    RECALL_AT_K = "recall@k"
    MRR = "mrr"  # Mean Reciprocal Rank
    NDCG_AT_K = "ndcg@k"  # Normalized Discounted Cumulative Gain
    MAP = "map"  # Mean Average Precision

    # Generation metrics
    FAITHFULNESS = "faithfulness"  # Answer is grounded in retrieved docs
    RELEVANCE = "relevance"  # Answer addresses the query
    ANSWER_QUALITY = "answer_quality"  # Overall answer quality

    # End-to-end metrics
    LATENCY_P50 = "latency_p50"
    LATENCY_P95 = "latency_p95"
    LATENCY_P99 = "latency_p99"


@dataclass
class MetricResult:
    """Result from a single metric calculation."""

    metric_type: MetricType
    value: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.metric_type.value}: {self.value:.4f}"


@dataclass
class RAGMetrics:
    """Container for all RAG evaluation metrics."""

    # Retrieval metrics
    precision_at_k: dict[int, float] = field(default_factory=dict)
    recall_at_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg_at_k: dict[int, float] = field(default_factory=dict)
    mean_average_precision: float = 0.0

    # Generation metrics
    faithfulness: float = 0.0
    relevance: float = 0.0
    answer_quality: float = 0.0

    # Performance metrics
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0

    # Metadata
    num_queries: int = 0
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            "retrieval": {
                "precision@k": self.precision_at_k,
                "recall@k": self.recall_at_k,
                "mrr": self.mrr,
                "ndcg@k": self.ndcg_at_k,
                "map": self.mean_average_precision,
            },
            "generation": {
                "faithfulness": self.faithfulness,
                "relevance": self.relevance,
                "answer_quality": self.answer_quality,
            },
            "performance": {
                "latency_p50_ms": self.latency_p50,
                "latency_p95_ms": self.latency_p95,
                "latency_p99_ms": self.latency_p99,
            },
            "metadata": {
                "num_queries": self.num_queries,
                "timestamp": self.timestamp,
            },
        }

    def get_summary(self) -> str:
        """Get human-readable summary of metrics."""
        lines = [
            "=== RAG Evaluation Metrics ===",
            f"\nRetrieval Metrics (n={self.num_queries}):",
            f"  Precision@5: {self.precision_at_k.get(5, 0.0):.3f}",
            f"  Recall@10: {self.recall_at_k.get(10, 0.0):.3f}",
            f"  MRR: {self.mrr:.3f}",
            f"  NDCG@10: {self.ndcg_at_k.get(10, 0.0):.3f}",
            f"  MAP: {self.mean_average_precision:.3f}",
            "\nGeneration Metrics:",
            f"  Faithfulness: {self.faithfulness:.3f}",
            f"  Relevance: {self.relevance:.3f}",
            f"  Answer Quality: {self.answer_quality:.3f}",
            "\nPerformance:",
            f"  Latency P50: {self.latency_p50:.1f}ms",
            f"  Latency P95: {self.latency_p95:.1f}ms",
            f"  Latency P99: {self.latency_p99:.1f}ms",
        ]
        return "\n".join(lines)


class MetricsCalculator:
    """Calculate various RAG evaluation metrics."""

    @staticmethod
    def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
        """
        Calculate Precision@k.

        Args:
            retrieved_ids: List of retrieved document IDs (in order)
            relevant_ids: Set of relevant document IDs
            k: Number of top results to consider

        Returns:
            Precision@k score (0-1)
        """
        if not retrieved_ids or k == 0:
            return 0.0

        top_k = retrieved_ids[:k]
        relevant_retrieved = sum(1 for doc_id in top_k if doc_id in relevant_ids)
        return relevant_retrieved / k

    @staticmethod
    def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
        """
        Calculate Recall@k.

        Args:
            retrieved_ids: List of retrieved document IDs (in order)
            relevant_ids: Set of relevant document IDs
            k: Number of top results to consider

        Returns:
            Recall@k score (0-1)
        """
        if not relevant_ids:
            return 0.0
        if not retrieved_ids:
            return 0.0

        top_k = retrieved_ids[:k]
        relevant_retrieved = sum(1 for doc_id in top_k if doc_id in relevant_ids)
        return relevant_retrieved / len(relevant_ids)

    @staticmethod
    def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).

        Args:
            retrieved_ids: List of retrieved document IDs (in order)
            relevant_ids: Set of relevant document IDs

        Returns:
            MRR score (0-1)
        """
        for i, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_ids:
                return 1.0 / i
        return 0.0

    @staticmethod
    def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@k).

        Binary relevance version where documents are either relevant (1) or not (0).

        Args:
            retrieved_ids: List of retrieved document IDs (in order)
            relevant_ids: Set of relevant document IDs
            k: Number of top results to consider

        Returns:
            NDCG@k score (0-1)
        """
        import math

        if not retrieved_ids or not relevant_ids or k == 0:
            return 0.0

        # Calculate DCG@k
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k], start=1):
            relevance = 1 if doc_id in relevant_ids else 0
            dcg += relevance / math.log2(i + 1)

        # Calculate ideal DCG@k
        num_relevant = min(len(relevant_ids), k)
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, num_relevant + 1))

        if idcg == 0:
            return 0.0

        return dcg / idcg

    @staticmethod
    def average_precision(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
        """
        Calculate Average Precision (AP).

        Args:
            retrieved_ids: List of retrieved document IDs (in order)
            relevant_ids: Set of relevant document IDs

        Returns:
            Average Precision score (0-1)
        """
        if not relevant_ids:
            return 0.0

        num_relevant = 0
        sum_precisions = 0.0

        for i, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_ids:
                num_relevant += 1
                precision_at_i = num_relevant / i
                sum_precisions += precision_at_i

        if num_relevant == 0:
            return 0.0

        return sum_precisions / len(relevant_ids)

    @staticmethod
    def calculate_latency_percentiles(latencies: list[float]) -> dict[str, float]:
        """
        Calculate latency percentiles.

        Args:
            latencies: List of latency values in milliseconds

        Returns:
            Dict with p50, p95, p99 percentiles
        """
        if not latencies:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}

        import statistics

        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)

        def percentile(p: float) -> float:
            k = (n - 1) * p
            f = int(k)
            c = f + 1
            if c >= n:
                return sorted_latencies[-1]
            d0 = sorted_latencies[f]
            d1 = sorted_latencies[c]
            return d0 + (d1 - d0) * (k - f)

        return {
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
            "mean": statistics.mean(latencies),
        }
