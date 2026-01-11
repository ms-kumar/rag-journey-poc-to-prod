"""Cache service schemas."""

from pydantic import BaseModel, Field


class CacheEntry(BaseModel):
    """Cache entry model."""

    key: str = Field(..., description="Cache key")
    value: str | dict | list = Field(..., description="Cached value")
    ttl: int | None = Field(None, description="Time to live in seconds", ge=0)


class CacheStats(BaseModel):
    """Cache statistics."""

    hits: int = Field(0, description="Number of cache hits", ge=0)
    misses: int = Field(0, description="Number of cache misses", ge=0)
    size: int = Field(0, description="Current cache size", ge=0)
    hit_rate: float = Field(0.0, description="Cache hit rate percentage", ge=0.0, le=100.0)


class SemanticCacheResult(BaseModel):
    """Result from semantic cache lookup."""

    found: bool = Field(..., description="Whether a match was found")
    similarity: float | None = Field(None, description="Similarity score", ge=0.0, le=1.0)
    cached_value: str | None = Field(None, description="Cached value if found")
    cache_key: str | None = Field(None, description="Cache key that was matched")
