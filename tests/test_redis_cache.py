"""
Tests for CacheClient (unified Redis cache client).
"""

import json
from unittest.mock import MagicMock

import pytest

from src.services.cache.client import CacheClient


class TestCacheClient:
    """Tests for CacheClient."""

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        redis_instance = MagicMock()
        redis_instance.ping.return_value = True
        redis_instance.get.return_value = None
        redis_instance.set.return_value = True
        redis_instance.delete.return_value = 1
        redis_instance.keys.return_value = []
        redis_instance.exists.return_value = False
        redis_instance.ttl.return_value = -1
        redis_instance.expire.return_value = True
        redis_instance.scan_iter.return_value = iter([])
        redis_instance.info.return_value = {"used_memory": 1024}
        return redis_instance

    @pytest.fixture
    def cache(self, mock_redis):
        """Create cache instance with mocked Redis."""
        return CacheClient(
            redis_client=mock_redis, ttl_hours=1, key_prefix="test:"
        )

    def test_initialization(self, mock_redis):
        """Test cache initialization."""
        cache = CacheClient(redis_client=mock_redis, ttl_hours=6, key_prefix="rag:")

        assert cache.redis == mock_redis
        assert cache.ttl.total_seconds() == 6 * 3600
        assert cache.key_prefix == "rag:"

    def test_generate_cache_key(self, cache):
        """Test cache key generation."""
        key1 = cache._generate_cache_key("test query", model="llama", top_k=5)
        key2 = cache._generate_cache_key("test query", model="llama", top_k=5)
        key3 = cache._generate_cache_key("test query", model="llama", top_k=10)

        # Same parameters should generate same key
        assert key1 == key2
        # Different parameters should generate different key
        assert key1 != key3
        # Key should have prefix
        assert key1.startswith("test:")

    def test_hash_key(self, cache):
        """Test key hashing."""
        hash1 = cache._hash_key("test data")
        hash2 = cache._hash_key("test data")
        hash3 = cache._hash_key("different data")

        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 16

    def test_get_cache_hit(self, cache, mock_redis):
        """Test cache hit."""
        mock_redis.get.return_value = json.dumps({"answer": "test"})

        result = cache.get("test query", model="llama")

        assert result == {"answer": "test"}

    def test_get_cache_miss(self, cache, mock_redis):
        """Test cache miss."""
        mock_redis.get.return_value = None

        result = cache.get("test query", model="llama")

        assert result is None

    def test_set_cache(self, cache, mock_redis):
        """Test setting cache."""
        mock_redis.set.return_value = True

        result = cache.set("test query", {"answer": "test"}, model="llama")

        assert result is True
        mock_redis.set.assert_called_once()

    def test_delete(self, cache, mock_redis):
        """Test deleting cache entry."""
        mock_redis.delete.return_value = 1

        result = cache.delete("test query", model="llama")

        assert result is True
        mock_redis.delete.assert_called_once()

    def test_exists(self, cache, mock_redis):
        """Test checking if entry exists."""
        mock_redis.exists.return_value = True

        result = cache.exists("test query", model="llama")

        assert result is True

    def test_get_ttl(self, cache, mock_redis):
        """Test getting TTL."""
        mock_redis.ttl.return_value = 300

        ttl = cache.get_ttl("test query", model="llama")

        assert ttl == 300

    def test_set_ttl(self, cache, mock_redis):
        """Test setting TTL."""
        mock_redis.expire.return_value = True

        result = cache.set_ttl("test query", 600, model="llama")

        assert result is True

    def test_invalidate_pattern(self, cache, mock_redis):
        """Test pattern invalidation."""
        mock_redis.scan_iter.return_value = iter(["test:key1", "test:key2"])
        mock_redis.delete.return_value = 2

        count = cache.invalidate_pattern("key*")

        assert count == 2

    def test_clear(self, cache, mock_redis):
        """Test clearing cache."""
        mock_redis.keys.return_value = ["test:key1", "test:key2"]
        mock_redis.delete.return_value = 2

        count = cache.clear()

        assert count == 2

    def test_ping(self, cache, mock_redis):
        """Test ping."""
        mock_redis.ping.return_value = True

        result = cache.ping()

        assert result is True

    def test_health_check(self, cache, mock_redis):
        """Test health check."""
        mock_redis.ping.return_value = True

        result = cache.health_check()

        assert result is True

    def test_get_stats(self, cache, mock_redis):
        """Test getting stats."""
        mock_redis.keys.return_value = ["test:key1", "test:key2"]
        mock_redis.info.return_value = {"used_memory": 2048}

        stats = cache.get_stats()

        assert stats["keys_count"] == 2
        assert stats["memory_used_mb"] >= 0
        assert stats["connected"] is True

    def test_context_manager(self, cache, mock_redis):
        """Test context manager support."""
        with cache as c:
            assert c is cache

    def test_flush(self, cache, mock_redis):
        """Test flushing cache."""
        mock_redis.keys.return_value = ["test:key1"]
        mock_redis.delete.return_value = 1

        result = cache.flush()

        assert result is True

    def test_flush_hooks(self, cache):
        """Test flush hooks."""
        hook_called = []

        def test_hook():
            hook_called.append(True)

        cache.register_flush_hook(test_hook)
        cache.flush()

        assert len(hook_called) == 1

    def test_repr(self, cache):
        """Test string representation."""
        repr_str = repr(cache)

        assert "CacheClient" in repr_str
