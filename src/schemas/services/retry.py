"""Retry and resilience schemas."""

from pydantic import BaseModel, Field


class RetryConfig(BaseModel):
    """Configuration for retry behavior."""

    max_retries: int = Field(3, description="Maximum number of retry attempts", ge=0)
    initial_delay: float = Field(1.0, description="Initial delay in seconds", gt=0.0)
    max_delay: float = Field(60.0, description="Maximum delay in seconds", gt=0.0)
    exponential_base: float = Field(2.0, description="Exponential backoff base", gt=1.0)
    jitter: bool = Field(True, description="Whether to add jitter to delays")
    retryable_status_codes: list[int] = Field(
        default_factory=lambda: [429, 500, 502, 503, 504],
        description="HTTP status codes to retry on",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "max_retries": 3,
                "initial_delay": 1.0,
                "max_delay": 60.0,
                "exponential_base": 2.0,
                "jitter": True,
                "retryable_status_codes": [429, 500, 502, 503, 504],
            }
        }
