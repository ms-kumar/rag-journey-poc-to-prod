"""
Factory for creating cache instances from application settings.
"""

import logging
from typing import TYPE_CHECKING

import redis

from src.exceptions import CacheException, RedisConnectionError
from src.services.cache.client import CacheClient
from src.services.cache.metrics import CacheMetrics
from src.services.cache.semantic_cache import SemanticCacheClient
from src.services.embeddings.client import EmbedClient

if TYPE_CHECKING:
    from src.config import CacheSettings, RedisSettings

logger = logging.getLogger(__name__)


def make_redis_client(redis_settings: "RedisSettings") -> redis.Redis:
    """
    Create Redis client with connection pooling.

    Args:
        redis_settings: Redis settings
    Returns:
        Connected Redis client instance

    Raises:
        RedisConnectionError: If connection fails
    """

    try:
        client = redis.Redis(
            host=redis_settings.host,
            port=redis_settings.port,
            password=redis_settings.password if redis_settings.password else None,
            db=redis_settings.db,
            decode_responses=redis_settings.decode_responses,
            socket_timeout=redis_settings.socket_timeout,
            socket_connect_timeout=redis_settings.socket_timeout,
            max_connections=redis_settings.max_connections,
            retry_on_timeout=True,
            retry_on_error=[redis.ConnectionError, redis.TimeoutError],
        )

        # Test connection
        client.ping()
        logger.info(f"Connected to Redis at {redis_settings.host}:{redis_settings.port}")
        return client

    except redis.ConnectionError as e:
        logger.error(f"Failed to connect to Redis: {e}")
        raise RedisConnectionError(
            f"Failed to connect to Redis at {redis_settings.host}:{redis_settings.port}"
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error creating Redis client: {e}")
        raise CacheException(f"Failed to create Redis client: {e}") from e


def make_cache_client(settings: "CacheSettings") -> CacheClient | None:
    """
    Create unified cache client (exact match + semantic caching).

    Gracefully handles Redis connection failures by returning None.

    Args:
        settings: Application settings

    Returns:
        CacheClient instance or None if Redis unavailable
    """
    try:
        redis_client = make_redis_client(settings.redis)
        cache_client = CacheClient(
            redis_client=redis_client,
            ttl_hours=settings.default_ttl // 3600,  # Convert seconds to hours
            key_prefix=settings.key_prefix,
        )
        logger.info("Cache client created successfully")
        return cache_client

    except (RedisConnectionError, CacheException) as e:
        logger.warning(f"Cache client unavailable: {e}. Continuing without cache.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error creating cache client: {e}")
        return None


def make_semantic_cache_client(
    settings: "CacheSettings",
    cache_client: CacheClient | None = None,
    embed_client: EmbedClient | None = None,
) -> SemanticCacheClient | None:
    """
    Create semantic cache client for similarity-based caching.
    Gracefully handles missing dependencies with fallback to None.

    Args:
        settings: Application settings
        cache_client: Optional CacheClient (creates new one if not provided)
        embed_client: Optional embedding client (creates new one if not provided)

    Returns:
        SemanticCacheClient instance or None if dependencies unavailable
    """
    try:
        # Create cache client if not provided
        if cache_client is None:
            cache_client = make_cache_client(settings)
            if cache_client is None:
                logger.warning("Cannot create semantic cache: CacheClient unavailable")
                return None

        # Create embedding client if not provided
        if embed_client is None:
            embed_client = EmbedClient(
                dim=settings.semantic_embedding_dim,
                normalize=True,
            )

        # Create semantic cache client
        semantic_cache = SemanticCacheClient(
            cache_client=cache_client,
            embed_client=embed_client,
            similarity_threshold=settings.semantic_similarity_threshold,
            embedding_dim=settings.semantic_embedding_dim,
            max_candidates=settings.semantic_max_candidates,
        )

        logger.info("Semantic cache client created successfully")
        return semantic_cache

    except Exception as e:
        logger.error(f"Failed to create semantic cache client: {e}")
        return None


def create_cache_metrics(settings: "CacheSettings") -> CacheMetrics:
    """
    Create cache metrics from application settings.

    Args:
        settings: Application settings

    Returns:
        Configured cache metrics instance
    """

    return CacheMetrics(staleness_config=settings)
