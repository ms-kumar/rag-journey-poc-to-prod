"""
Redis-based cache client with TTL, invalidation, and flush hooks.
"""

import hashlib
import json
import logging
from collections.abc import Callable
from typing import Any

import redis
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RedisCacheConfig(BaseModel):
    """Configuration for Redis cache."""

    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, description="Redis port")
    db: int = Field(default=0, description="Redis database number")
    password: str | None = Field(default=None, description="Redis password")
    default_ttl: int = Field(default=3600, description="Default TTL in seconds (1 hour)")
    key_prefix: str = Field(default="rag:", description="Prefix for all cache keys")
    max_connections: int = Field(default=10, description="Max Redis connections in pool")
    socket_timeout: int = Field(default=5, description="Socket timeout in seconds")
    decode_responses: bool = Field(default=True, description="Decode responses to strings")

    @classmethod
    def from_settings(cls, settings: Any) -> "RedisCacheConfig":
        """Create config from application settings."""
        cache_settings = settings.cache
        return cls(
            host=cache_settings.redis_host,
            port=cache_settings.redis_port,
            db=cache_settings.redis_db,
            password=cache_settings.redis_password,
            default_ttl=cache_settings.default_ttl,
            key_prefix=cache_settings.key_prefix,
            max_connections=cache_settings.redis_max_connections,
            socket_timeout=cache_settings.redis_socket_timeout,
            decode_responses=cache_settings.redis_decode_responses,
        )


class RedisCache:
    """
    Redis-based cache with TTL, invalidation, and flush hooks.

    Features:
    - Automatic TTL (time-to-live) for cached entries
    - Key invalidation patterns
    - Flush hooks for cache clearing
    - Connection pooling
    - JSON serialization
    """

    def __init__(self, config: RedisCacheConfig | None = None):
        """
        Initialize Redis cache.

        Args:
            config: Redis cache configuration
        """
        self.config = config or RedisCacheConfig()
        self._client: redis.Redis | None = None
        self._flush_hooks: list[Callable[[], None]] = []

    def _ensure_connected(self) -> redis.Redis:
        """Ensure Redis connection is established."""
        if self._client is None:
            try:
                self._client = redis.Redis(
                    host=self.config.host,
                    port=self.config.port,
                    db=self.config.db,
                    password=self.config.password,
                    max_connections=self.config.max_connections,
                    socket_timeout=self.config.socket_timeout,
                    decode_responses=self.config.decode_responses,
                )
                # Test connection
                self._client.ping()
                logger.info(
                    f"Connected to Redis at {self.config.host}:{self.config.port}/{self.config.db}"
                )
            except redis.ConnectionError as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise
        return self._client

    def _make_key(self, key: str) -> str:
        """Create prefixed cache key."""
        return f"{self.config.key_prefix}{key}"

    def _hash_key(self, data: Any) -> str:
        """Create hash key from data."""
        if isinstance(data, str):
            content = data
        else:
            content = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get(self, key: str) -> Any | None:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        try:
            client = self._ensure_connected()
            cache_key = self._make_key(key)
            value = client.get(cache_key)

            if value is None:
                return None

            # Try to deserialize JSON
            try:
                return json.loads(str(value))
            except (json.JSONDecodeError, TypeError):
                return value

        except redis.RedisError as e:
            logger.warning(f"Redis get error for key '{key}': {e}")
            return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
        nx: bool = False,
        xx: bool = False,
    ) -> bool:
        """
        Set value in cache with TTL.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None = use default)
            nx: Only set if key doesn't exist
            xx: Only set if key exists

        Returns:
            True if set successfully
        """
        try:
            client = self._ensure_connected()
            cache_key = self._make_key(key)

            # Serialize value
            if isinstance(value, (dict, list, tuple)):
                serialized = json.dumps(value)
            else:
                serialized = str(value)

            # Set with TTL
            ttl_seconds = ttl if ttl is not None else self.config.default_ttl

            result = client.set(
                cache_key,
                serialized,
                ex=ttl_seconds,
                nx=nx,
                xx=xx,
            )

            return bool(result)

        except redis.RedisError as e:
            logger.warning(f"Redis set error for key '{key}': {e}")
            return False

    def delete(self, key: str) -> bool:
        """
        Delete key from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted
        """
        try:
            client = self._ensure_connected()
            cache_key = self._make_key(key)
            return bool(client.delete(cache_key))
        except redis.RedisError as e:
            logger.warning(f"Redis delete error for key '{key}': {e}")
            return False

    def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if key exists
        """
        try:
            client = self._ensure_connected()
            cache_key = self._make_key(key)
            return bool(client.exists(cache_key))
        except redis.RedisError as e:
            logger.warning(f"Redis exists error for key '{key}': {e}")
            return False

    def ttl(self, key: str) -> int:
        """
        Get remaining TTL for key.

        Args:
            key: Cache key

        Returns:
            Remaining TTL in seconds (-1 if no TTL, -2 if key doesn't exist)
        """
        try:
            client = self._ensure_connected()
            cache_key = self._make_key(key)
            result = client.ttl(cache_key)
            return int(result) if result is not None else -2  # type: ignore[arg-type]
        except redis.RedisError as e:
            logger.warning(f"Redis TTL error for key '{key}': {e}")
            return -2

    def expire(self, key: str, ttl: int) -> bool:
        """
        Set expiration time for key.

        Args:
            key: Cache key
            ttl: Time-to-live in seconds

        Returns:
            True if expiration set successfully
        """
        try:
            client = self._ensure_connected()
            cache_key = self._make_key(key)
            return bool(client.expire(cache_key, ttl))
        except redis.RedisError as e:
            logger.warning(f"Redis expire error for key '{key}': {e}")
            return False

    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all keys matching pattern.

        Args:
            pattern: Key pattern (supports * and ? wildcards)

        Returns:
            Number of keys deleted
        """
        try:
            client = self._ensure_connected()
            pattern_key = self._make_key(pattern)

            # Find matching keys
            keys = list(client.scan_iter(match=pattern_key))

            if not keys:
                return 0

            # Delete all matching keys
            result = client.delete(*keys)
            return int(result) if result is not None else 0  # type: ignore[arg-type]

        except redis.RedisError as e:
            logger.warning(f"Redis invalidate_pattern error for pattern '{pattern}': {e}")
            return 0

    def flush(self) -> bool:
        """
        Flush all cache entries with the configured prefix.

        Returns:
            True if flushed successfully
        """
        try:
            # Trigger flush hooks before clearing cache
            for hook in self._flush_hooks:
                try:
                    hook()
                except Exception as e:
                    logger.error(f"Flush hook error: {e}")

            client = self._ensure_connected()
            pattern_key = self._make_key("*")

            # Find all keys with our prefix
            keys = list(client.scan_iter(match=pattern_key))

            if keys:
                client.delete(*keys)
                logger.info(f"Flushed {len(keys)} cache entries")

            return True

        except redis.RedisError as e:
            logger.error(f"Redis flush error: {e}")
            return False

    def flush_all(self) -> bool:
        """
        Flush entire Redis database (use with caution!).

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

            client = self._ensure_connected()
            client.flushdb()
            logger.warning("Flushed entire Redis database")
            return True

        except redis.RedisError as e:
            logger.error(f"Redis flush_all error: {e}")
            return False

    def register_flush_hook(self, hook: Callable[[], None]) -> None:
        """
        Register a hook to be called before cache flush.

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
            True if hook was removed
        """
        try:
            self._flush_hooks.remove(hook)
            logger.info(f"Unregistered flush hook: {hook.__name__}")
            return True
        except ValueError:
            return False

    def get_info(self) -> dict[str, Any]:
        """
        Get Redis server info.

        Returns:
            Redis server information
        """
        try:
            client = self._ensure_connected()
            info = client.info()
            return dict(info) if info else {}  # type: ignore[arg-type]
        except redis.RedisError as e:
            logger.error(f"Redis info error: {e}")
            return {}

    def health_check(self) -> bool:
        """
        Check if Redis is healthy.

        Returns:
            True if Redis is accessible
        """
        try:
            client = self._ensure_connected()
            result = client.ping()
            return bool(result) if result is not None else False
        except redis.RedisError:
            return False

    def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            self._client.close()
            self._client = None
            logger.info("Closed Redis connection")

    def __enter__(self):
        """Context manager entry."""
        self._ensure_connected()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __repr__(self) -> str:
        """String representation."""
        return f"RedisCache(host={self.config.host}, port={self.config.port}, db={self.config.db})"
