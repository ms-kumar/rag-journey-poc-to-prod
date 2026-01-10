"""
Factory for creating cache instances from application settings.
"""

from typing import TYPE_CHECKING

from src.services.cache.metrics import CacheMetrics, StalenessConfig
from src.services.cache.redis_client import RedisCache, RedisCacheConfig
from src.services.cache.semantic_cache import SemanticCache, SemanticCacheConfig
from src.services.embeddings.client import EmbedClient

if TYPE_CHECKING:
    from src.config import Settings


def create_redis_cache(settings: "Settings") -> RedisCache:
    """
    Create Redis cache from application settings.

    Args:
        settings: Application settings

    Returns:
        Configured Redis cache instance
    """
    config = RedisCacheConfig.from_settings(settings)
    return RedisCache(config)


def create_semantic_cache(
    settings: "Settings",
    embed_client: EmbedClient | None = None,
) -> SemanticCache:
    """
    Create semantic cache from application settings.

    Args:
        settings: Application settings
        embed_client: Optional embedding client (creates new one if not provided)

    Returns:
        Configured semantic cache instance
    """
    # Create embedding client if not provided
    if embed_client is None:
        embed_client = EmbedClient(
            dim=settings.cache.semantic_embedding_dim,
            normalize=True,
        )

    # Create Redis cache
    redis_cache = create_redis_cache(settings)

    # Create semantic cache config
    config = SemanticCacheConfig.from_settings(settings)

    return SemanticCache(
        embed_client=embed_client,
        config=config,
        redis_cache=redis_cache,
    )


def create_cache_metrics(
    settings: "Settings",
    cache: RedisCache | None = None,
) -> CacheMetrics:
    """
    Create cache metrics from application settings.

    Args:
        settings: Application settings
        cache: Optional Redis cache instance (creates new one if not provided)

    Returns:
        Configured cache metrics instance
    """
    # Create staleness config
    staleness_config = StalenessConfig.from_settings(settings)

    return CacheMetrics(
        staleness_config=staleness_config,
    )
