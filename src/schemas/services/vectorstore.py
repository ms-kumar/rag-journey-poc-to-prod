"""Vectorstore service schemas."""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class FusionConfig(BaseModel):
    """Configuration for retrieval fusion."""

    method: Literal["rrf", "weighted", "max"] = Field(
        default="rrf", description="Fusion method to use"
    )
    rrf_k: int = Field(default=60, description="RRF constant k parameter")
    weights: dict[str, float] | None = Field(
        default=None, description="Weights for each search type (must sum to 1.0)"
    )
    normalize_scores: bool = Field(default=True, description="Normalize scores before fusion")
    tie_break_strategy: Literal["score", "rank", "stable"] = Field(
        default="score", description="Strategy for breaking ties"
    )

    @field_validator("weights")
    @classmethod
    def validate_weights(cls, v: dict[str, float] | None) -> dict[str, float] | None:
        """Validate weights sum to 1.0."""
        if v is not None and abs(sum(v.values()) - 1.0) > 0.001:
            raise ValueError("Weights must sum to 1.0")
        return v


class FusionResult(BaseModel):
    """Result from fusion operation."""

    documents: list[Any] = Field(description="Fused and ranked documents")
    fusion_scores: list[float] = Field(description="Final fusion scores")
    component_ranks: dict[str, list[int]] = Field(description="Original ranks from each component")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class RetrievalMetrics(BaseModel):
    """Metrics for retrieval operations."""

    latencies: list[float] = Field(default_factory=list, description="Query latencies in seconds")
    scores: list[float] = Field(default_factory=list, description="Retrieval scores (flattened)")
    total_queries: int = Field(default=0, description="Total number of queries")
    cache_hits: int = Field(default=0, description="Number of cache hits")
    metrics_by_type: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="Metrics grouped by query type"
    )


class IndexMapping(BaseModel):
    """Configuration for a single field index."""

    field_name: str = Field(description="Name of the field to index")
    field_type: Literal["keyword", "integer", "float", "text", "datetime", "bool", "geo"] = Field(
        description="Type of the field"
    )
    tokenizer: Any = Field(default=None, description="Tokenizer type for text fields")
    min_token_len: int | None = Field(default=None, description="Minimum token length")
    max_token_len: int | None = Field(default=None, description="Maximum token length")
    lowercase: bool | None = Field(default=None, description="Convert to lowercase")
    range: bool = Field(default=False, description="Enable range queries for numeric fields")
    lookup: bool = Field(default=True, description="Enable exact match lookups")


class EvaluationMetrics(BaseModel):
    """Metrics for retrieval evaluation."""

    recall_at_k: dict[int, float] = Field(description="Recall at various k values")
    precision_at_k: dict[int, float] = Field(description="Precision at various k values")
    mrr: float = Field(description="Mean Reciprocal Rank")
    map: float = Field(description="Mean Average Precision")
    ndcg: float = Field(description="Normalized Discounted Cumulative Gain")
    total_relevant: int = Field(description="Total number of relevant documents")
    total_retrieved: int = Field(description="Total number of retrieved documents")


class UpliftMetrics(BaseModel):
    """Uplift metrics comparing fusion to baselines."""

    fusion_recall: dict[int, float] = Field(description="Recall for fusion method")
    baseline_recalls: dict[str, dict[int, float]] = Field(description="Recall for each baseline")
    recall_uplift: dict[str, dict[int, float]] = Field(
        description="Percentage improvement over each baseline"
    )
    best_baseline_recall: dict[int, float] = Field(description="Best baseline recall at each k")
    uplift_over_best: dict[int, float] = Field(description="Uplift over best baseline at each k")
