"""Reranker service schemas."""

from typing import Any

from pydantic import BaseModel, Field


class RerankResult(BaseModel):
    """Result from reranking operation."""

    documents: list[Any] = Field(description="Reranked documents")
    scores: list[float] = Field(description="Relevance scores for each document")
    original_ranks: list[int] = Field(description="Original ranking positions")
    execution_time: float = Field(description="Time taken for reranking in seconds")
    model_used: str = Field(description="Model identifier used for reranking")
    fallback_used: bool = Field(default=False, description="Whether fallback was used")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class PrecisionMetrics(BaseModel):
    """Precision metrics for reranking evaluation."""

    precision_at_k: dict[int, float] = Field(description="Precision at various k values")
    total_relevant: int = Field(description="Total number of relevant documents")
    total_retrieved: int = Field(description="Total number of retrieved documents")
    improvement_over_baseline: dict[int, float] = Field(
        default_factory=dict, description="Percentage improvement over baseline precision at k"
    )


class ComparisonResult(BaseModel):
    """Result from comparing baseline vs reranked retrieval."""

    baseline_metrics: PrecisionMetrics = Field(description="Metrics for baseline retrieval")
    reranked_metrics: PrecisionMetrics = Field(description="Metrics after reranking")
    improvement: dict[int, float] = Field(description="Improvement percentage at each k value")
    statistical_significance: dict[int, bool] = Field(
        default_factory=dict, description="Statistical significance at each k value"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class BenchmarkConfig(BaseModel):
    """Configuration for reranker benchmarking."""

    k_values: list[int] = Field(default=[1, 3, 5, 10], description="K values for evaluation")
    num_queries: int = Field(default=100, description="Number of queries to test")
    timeout_per_query: float = Field(default=10.0, description="Timeout per query in seconds")
    include_latency: bool = Field(default=True, description="Include latency measurements")
    include_precision: bool = Field(default=True, description="Include precision metrics")
    include_recall: bool = Field(default=True, description="Include recall metrics")
