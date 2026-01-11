"""Performance monitoring schemas."""

from pydantic import BaseModel, Field


class PerformanceMetrics(BaseModel):
    """Performance metrics for a specific operation."""

    operation: str = Field(..., description="Name of the operation")
    duration_ms: float = Field(..., description="Duration in milliseconds", ge=0.0)
    timestamp: str = Field(..., description="Timestamp in ISO format")
    success: bool = Field(True, description="Whether the operation succeeded")
    metadata: dict[str, str | int | float] | None = Field(None, description="Additional metadata")


class PerformanceReport(BaseModel):
    """Performance report for all operations."""

    total_operations: int = Field(..., description="Total number of operations", ge=0)
    avg_duration_ms: float = Field(..., description="Average duration in ms", ge=0.0)
    p50_ms: float = Field(..., description="50th percentile duration", ge=0.0)
    p95_ms: float = Field(..., description="95th percentile duration", ge=0.0)
    p99_ms: float = Field(..., description="99th percentile duration", ge=0.0)
    operations: dict[str, "OperationStats"] = Field(..., description="Stats per operation")


class OperationStats(BaseModel):
    """Statistics for a specific operation type."""

    count: int = Field(..., description="Number of operations", ge=0)
    avg_duration_ms: float = Field(..., description="Average duration", ge=0.0)
    min_duration_ms: float = Field(..., description="Minimum duration", ge=0.0)
    max_duration_ms: float = Field(..., description="Maximum duration", ge=0.0)
    success_rate: float = Field(..., description="Success rate percentage", ge=0.0, le=100.0)
