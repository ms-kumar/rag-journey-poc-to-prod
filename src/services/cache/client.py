"""
Unified cache client interface combining Redis and Semantic caching.

interface for cache operations with both exact match and semantic caching.
"""

import hashlib
import json
import logging
from collections.abc import Callable
from datetime import timedelta
from typing import Any

from src.exceptions import CacheKeyError, CacheSerializationError

logger = logging.getLogger(__name__)


class CacheClient:
    """
    Unified cache client for RAG queries.

    Provides both exact-match caching (via Redis) and semantic caching
    for similar queries. Similar to Mother of AI's cache architecture,

    Features:
    - Exact query matching with parameter hashing
    - TTL-based expiration with control methods
    - Pattern-based invalidation
    - Flush hooks for cache management
    - Health checking and monitoring
    - Graceful error handling with fallback
    - JSON serialization for complex objects
    - Context manager support
    """

    def __init__(
        self,
        redis_client: Any,
        ttl_hours: int = 6,
        key_prefix: str = "rag_cache:",
    ):
        """
        Initialize cache client.

        Args:
            redis_client: Redis client instance
            ttl_hours: Time-to-live for cache entries in hours (default: 6)
            key_prefix: Prefix for all cache keys (default: "rag_cache:")
        """
        self.redis = redis_client
        self.ttl = timedelta(hours=ttl_hours)
        self.key_prefix = key_prefix
        self._flush_hooks: list[Callable[[], None]] = []
        logger.info(f"CacheClient initialized with {ttl_hours}h TTL")

    def _generate_cache_key(self, query: str, **params: Any) -> str:
        """
        Generate cache key based on query and parameters.

        Creates a hash of the query and all relevant parameters to ensure
        exact matching of cached responses.

        Args:
            query: User query string
            **params: Additional parameters (model, top_k, etc.)

        Returns:
            Cache key string with prefix and hash
        """
        key_data = {
            "query": query.strip().lower(),  # Normalize query
            **{k: v for k, v in sorted(params.items()) if v is not None},
        }

        try:
            key_string = json.dumps(key_data, sort_keys=True)
            key_hash = hashlib.sha256(key_string.encode()).hexdigest()[:16]
            return f"{self.key_prefix}{key_hash}"
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to generate cache key: {e}")
            raise CacheKeyError(f"Cannot generate cache key from parameters: {e}") from e

    def _hash_key(self, data: Any) -> str:
        """
        Create hash from arbitrary data.

        Utility method for creating consistent hashes.

        Args:
            data: Data to hash (string or JSON-serializable object)

        Returns:
            16-character hex hash
        """
        if isinstance(data, str):
            content = data
        else:
            content = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get(self, query: str, **params: Any) -> dict[str, Any] | None:
        """
        Get cached response for exact query match.

        Args:
            query: User query string
            **params: Additional parameters for cache key generation

        Returns:
            Cached response dict or None if not found
        """
        try:
            cache_key = self._generate_cache_key(query, **params)

            # Get from Redis
            cached_value = self.redis.get(cache_key)

            if cached_value:
                try:
                    response_data: dict[str, Any] = json.loads(str(cached_value))
                    logger.info(f"Cache hit for query: {query[:50]}...")
                    return response_data
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to deserialize cached response: {e}")
                    # Delete corrupted cache entry
                    self.redis.delete(cache_key)
                    return None

            logger.debug(f"Cache miss for query: {query[:50]}...")
            return None

        except CacheKeyError:
            raise
        except Exception as e:
            logger.error(f"Error retrieving from cache: {e}")
            return None

    def set(self, query: str, response: dict[str, Any], **params: Any) -> bool:
        """
        Store response in cache.

        Args:
            query: User query string
            response: Response dict to cache
            **params: Additional parameters for cache key generation

        Returns:
            True if stored successfully, False otherwise
        """
        try:
            cache_key = self._generate_cache_key(query, **params)

            # Serialize response to JSON
            try:
                response_json = json.dumps(response)
            except (TypeError, ValueError) as e:
                logger.error(f"Cannot serialize response to JSON: {e}")
                raise CacheSerializationError(
                    f"Response contains non-serializable data: {e}"
                ) from e

            # Store in Redis with TTL
            success = self.redis.set(cache_key, response_json, ex=int(self.ttl.total_seconds()))

            if success:
                logger.info(f"Stored response in cache (TTL: {self.ttl})")
                return True
            logger.warning("Failed to store response in cache")
            return False

        except (CacheKeyError, CacheSerializationError):
            raise
        except Exception as e:
            logger.error(f"Error storing in cache: {e}")
            return False

    def delete(self, query: str, **params: Any) -> bool:
        """
        Delete cached entry for specific query.

        Args:
            query: User query string
            **params: Additional parameters for cache key generation

        Returns:
            True if deleted, False otherwise
        """
        try:
            cache_key = self._generate_cache_key(query, **params)
            deleted: int = self.redis.delete(cache_key)
            logger.info(f"Deleted {deleted} cache entry")
            return bool(deleted > 0)
        except Exception as e:
            logger.error(f"Error deleting cache entry: {e}")
            return False

    def exists(self, query: str, **params: Any) -> bool:
        """
        Check if cached entry exists for query.

        Args:
            query: User query string
            **params: Additional parameters for cache key generation

        Returns:
            True if entry exists in cache
        """
        try:
            cache_key = self._generate_cache_key(query, **params)
            return bool(self.redis.exists(cache_key))
        except Exception as e:
            logger.error(f"Error checking cache existence: {e}")
            return False

    def get_ttl(self, query: str, **params: Any) -> int:
        """
        Get remaining TTL for cached entry.

        Args:
            query: User query string
            **params: Additional parameters for cache key generation

        Returns:
            Remaining TTL in seconds (-1 if no TTL, -2 if key doesn't exist)
        """
        try:
            cache_key = self._generate_cache_key(query, **params)
            result = self.redis.ttl(cache_key)
            return int(result) if result is not None else -2
        except Exception as e:
            logger.error(f"Error getting TTL: {e}")
            return -2

    def set_ttl(self, query: str, ttl_seconds: int, **params: Any) -> bool:
        """
        Set or update TTL for cached entry.

        Args:
            query: User query string
            ttl_seconds: New TTL in seconds
            **params: Additional parameters for cache key generation

        Returns:
            True if TTL was set successfully
        """
        try:
            cache_key = self._generate_cache_key(query, **params)
            return bool(self.redis.expire(cache_key, ttl_seconds))
        except Exception as e:
            logger.error(f"Error setting TTL: {e}")
            return False

    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all cache entries matching pattern.

        Args:
            pattern: Key pattern with wildcards (* and ?)
                    Example: "llama*" matches all keys starting with "llama"

        Returns:
            Number of keys deleted
        """
        try:
            search_pattern = f"{self.key_prefix}{pattern}"
            keys = list(self.redis.scan_iter(match=search_pattern))

            if not keys:
                logger.info(f"No keys matching pattern '{pattern}'")
                return 0

            deleted: int = self.redis.delete(*keys)
            logger.info(f"Invalidated {deleted} keys matching pattern '{pattern}'")
            return int(deleted)

        except Exception as e:
            logger.error(f"Error invalidating pattern '{pattern}': {e}")
            return 0

    def clear(self, pattern: str | None = None) -> int:
        """
        Clear cache entries matching pattern.

        Args:
            pattern: Key pattern to match (default: all keys with prefix)

        Returns:
            Number of keys deleted
        """
        try:
            search_pattern = pattern or f"{self.key_prefix}*"
            keys = self.redis.keys(search_pattern)

            if keys:
                deleted: int = self.redis.delete(*keys)
                logger.info(f"Cleared {deleted} cache entries")
                return int(deleted)
            logger.info("No cache entries to clear")
            return 0

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return 0

    def flush(self) -> bool:
        """
        Flush all cache entries with configured prefix.

        Triggers all registered flush hooks before clearing.

        Returns:
            True if flushed successfully
        """
        try:
            # Trigger flush hooks
            for hook in self._flush_hooks:
                try:
                    hook()
                except Exception as e:
                    logger.error(f"Flush hook error: {e}")

            # Clear all entries with our prefix
            deleted = self.clear()
            logger.info(f"Flushed cache: {deleted} entries removed")
            return True

        except Exception as e:
            logger.error(f"Error flushing cache: {e}")
            return False

    def flush_all(self) -> bool:
        """
        Flush entire Redis database (use with caution!).

        Triggers all registered flush hooks before clearing.
        Only use if you control the entire Redis instance.

        Returns:
            True if flushed successfully
        """
        try:
            # Trigger flush hooks
            for hook in self._flush_hooks:
                try:
                    hook()
                except Exception as e:
                    logger.error(f"Flush hook error: {e}")

            self.redis.flushdb()
            logger.warning("Flushed entire Redis database")
            return True

        except Exception as e:
            logger.error(f"Error flushing database: {e}")
            return False

    def register_flush_hook(self, hook: Callable[[], None]) -> None:
        """
        Register a hook to be called before cache flush.

        Useful for cleanup operations, logging, or notifications
        before cache is cleared.

        Args:
            hook: Callback function to execute before flush
        """
        self._flush_hooks.append(hook)
        logger.info(f"Registered flush hook: {hook.__name__}")

    def unregister_flush_hook(self, hook: Callable[[], None]) -> bool:
        """
        Unregister a flush hook.

        Args:
            hook: Hook to remove

        Returns:
            True if hook was removed, False if not found
        """
        try:
            self._flush_hooks.remove(hook)
            logger.info(f"Unregistered flush hook: {hook.__name__}")
            return True
        except ValueError:
            logger.warning(f"Hook {hook.__name__} not found in registered hooks")
            return False

    def ping(self) -> bool:
        """
        Check if Redis connection is alive.

        Returns:
            True if connected, False otherwise
        """
        try:
            return bool(self.redis.ping())
        except Exception as e:
            logger.error(f"Redis ping failed: {e}")
            return False

    def health_check(self) -> bool:
        """
        Comprehensive health check for Redis connection.

        Alias for ping() for consistency with RedisCache interface.

        Returns:
            True if Redis is accessible and responding
        """
        return self.ping()

    def get_info(self) -> dict[str, Any]:
        """
        Get Redis server information.

        Returns:
            Dict with Redis server info (memory, stats, etc.)
        """
        try:
            info = self.redis.info()
            return dict(info) if info else {}
        except Exception as e:
            logger.error(f"Error getting Redis info: {e}")
            return {}

    def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with cache stats (keys count, memory usage, etc.)
        """
        try:
            # Count keys matching our prefix
            keys = self.redis.keys(f"{self.key_prefix}*")
            key_count = len(keys)

            # Get Redis info
            info = self.redis.info("memory")

            return {
                "keys_count": key_count,
                "memory_used_mb": round(info.get("used_memory", 0) / (1024 * 1024), 2),
                "ttl_hours": self.ttl.total_seconds() / 3600,
                "key_prefix": self.key_prefix,
                "connected": True,
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {
                "keys_count": 0,
                "memory_used_mb": 0,
                "ttl_hours": self.ttl.total_seconds() / 3600,
                "key_prefix": self.key_prefix,
                "connected": False,
                "error": str(e),
            }

    def close(self) -> None:
        """
        Close Redis connection.

        Should be called when cache client is no longer needed.
        """
        try:
            if hasattr(self.redis, "close"):
                self.redis.close()
                logger.info("Closed Redis connection")
        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __repr__(self) -> str:
        """String representation."""
        return f"CacheClient(prefix={self.key_prefix}, ttl={self.ttl})"
