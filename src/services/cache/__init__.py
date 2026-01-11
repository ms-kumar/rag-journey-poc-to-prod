"""
Caching services with Redis, semantic caching, and monitoring.

Similar to Mother of AI's cache architecture with unified clients.
"""

from src.services.cache.client import CacheClient
from src.services.cache.factory import (
    create_cache_metrics,
    make_cache_client,
    make_redis_client,
    make_semantic_cache_client,
)
from src.services.cache.metrics import CacheMetrics, CacheStats
from src.services.cache.semantic_cache import SemanticCacheClient

__all__ = [
    # Unified clients (Mother of AI pattern)
    "CacheClient",
    "SemanticCacheClient",
    # Factory functions
    "make_cache_client",
    "make_redis_client",
    "make_semantic_cache_client",
    # Metrics
    "CacheMetrics",
    "CacheStats",
    # Legacy factory
    "create_cache_metrics",
]
