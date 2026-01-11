"""Cost tracking schemas."""

from pydantic import BaseModel, Field, computed_field


class ModelMetrics(BaseModel):
    """Metrics for a specific model."""

    model_name: str = Field(..., description="Name of the model")
    total_requests: int = Field(0, description="Total number of requests")
    total_cost: float = Field(0.0, description="Total cost in USD")
    total_tokens: int = Field(0, description="Total tokens processed")
    total_latency: float = Field(0.0, description="Total latency in seconds")
    errors: int = Field(0, description="Number of errors")
    quality_scores: list[float] = Field(default_factory=list, description="Quality scores")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def avg_latency(self) -> float:
        """Average latency per request in ms."""
        if self.total_requests == 0:
            return 0.0
        return (self.total_latency / self.total_requests) * 1000

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cost_per_1k(self) -> float:
        """Cost per 1000 queries."""
        if self.total_requests == 0:
            return 0.0
        return (self.total_cost / self.total_requests) * 1000

    @computed_field  # type: ignore[prop-decorator]
    @property
    def avg_quality(self) -> float:
        """Average quality score."""
        if not self.quality_scores:
            return 0.0
        return sum(self.quality_scores) / len(self.quality_scores)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def error_rate(self) -> float:
        """Error rate as percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.errors / self.total_requests) * 100

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cost_per_token(self) -> float:
        """Cost per token."""
        if self.total_tokens == 0:
            return 0.0
        return self.total_cost / self.total_tokens


class CostReport(BaseModel):
    """Cost report for all models."""

    total_cost: float = Field(..., description="Total cost across all models")
    total_requests: int = Field(..., description="Total requests across all models")
    models: dict[str, ModelMetrics] = Field(..., description="Metrics per model")


class ScalingDecision(BaseModel):
    """Autoscaling decision."""

    action: str = Field(..., description="Scaling action: scale_up, scale_down, maintain")
    current_tier: str = Field(..., description="Current model tier")
    recommended_tier: str = Field(..., description="Recommended model tier")
    reason: str = Field(..., description="Reason for the decision")
    load_factor: float = Field(..., description="Current load factor")


class AutoscalingPolicy(BaseModel):
    """Policy for autoscaling decisions."""

    scale_up_threshold: float = Field(0.8, description="Load threshold to scale up", ge=0.0, le=1.0)
    scale_down_threshold: float = Field(
        0.3, description="Load threshold to scale down", ge=0.0, le=1.0
    )
    cooldown_period: int = Field(300, description="Cooldown period in seconds", ge=0)
    min_tier: str = Field("small", description="Minimum allowed tier")
    max_tier: str = Field("large", description="Maximum allowed tier")


class LoadMetrics(BaseModel):
    """Current load metrics for autoscaling."""

    current_load: float = Field(..., description="Current load as a fraction", ge=0.0)
    request_rate: float = Field(..., description="Requests per second", ge=0.0)
    avg_latency: float = Field(..., description="Average latency in ms", ge=0.0)
    error_rate: float = Field(..., description="Error rate as percentage", ge=0.0, le=100.0)


class ModelCandidate(BaseModel):
    """Model candidate for selection."""

    model_name: str = Field(..., description="Name of the model")
    tier: str = Field(..., description="Model tier: small, medium, large")
    cost_per_1k: float = Field(..., description="Cost per 1000 tokens", ge=0.0)
    avg_latency_ms: float = Field(..., description="Average latency in ms", ge=0.0)
    quality_score: float = Field(..., description="Quality score", ge=0.0, le=1.0)
    context_window: int = Field(..., description="Context window size", ge=1)
    supports_streaming: bool = Field(False, description="Whether streaming is supported")
