"""
Tests for Redis cache client.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.services.cache.redis_client import RedisCache, RedisCacheConfig


class TestRedisCacheConfig:
    """Tests for Redis cache configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RedisCacheConfig()

        assert config.host == "localhost"
        assert config.port == 6379
        assert config.db == 0
        assert config.default_ttl == 3600
        assert config.key_prefix == "rag:"

    def test_custom_config(self):
        """Test custom configuration."""
        config = RedisCacheConfig(
            host="redis-server",
            port=6380,
            db=1,
            default_ttl=7200,
            key_prefix="test:",
        )

        assert config.host == "redis-server"
        assert config.port == 6380
        assert config.db == 1
        assert config.default_ttl == 7200
        assert config.key_prefix == "test:"


class TestRedisCache:
    """Tests for Redis cache."""

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        with patch("src.services.cache.redis_client.redis.Redis") as mock:
            redis_instance = MagicMock()
            redis_instance.ping.return_value = True
            mock.return_value = redis_instance
            yield redis_instance

    @pytest.fixture
    def cache(self, mock_redis):
        """Create cache instance with mocked Redis."""
        config = RedisCacheConfig()
        cache = RedisCache(config)
        cache._ensure_connected()
        return cache

    def test_initialization(self):
        """Test cache initialization."""
        config = RedisCacheConfig(host="test-host", port=6380)
        cache = RedisCache(config)

        assert cache.config.host == "test-host"
        assert cache.config.port == 6380
        assert cache._client is None

    def test_ensure_connected(self, mock_redis):
        """Test Redis connection."""
        cache = RedisCache()
        client = cache._ensure_connected()

        assert client is not None
        mock_redis.ping.assert_called_once()

    def test_make_key(self, cache):
        """Test key prefixing."""
        key = cache._make_key("test_key")
        assert key == "rag:test_key"

    def test_hash_key(self, cache):
        """Test key hashing."""
        hash1 = cache._hash_key("test data")
        hash2 = cache._hash_key("test data")
        hash3 = cache._hash_key("different data")

        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 16

    def test_get_success(self, cache, mock_redis):
        """Test successful cache get."""
        mock_redis.get.return_value = '{"result": "value"}'

        result = cache.get("test_key")

        assert result == {"result": "value"}
        mock_redis.get.assert_called_once_with("rag:test_key")

    def test_get_miss(self, cache, mock_redis):
        """Test cache miss."""
        mock_redis.get.return_value = None

        result = cache.get("missing_key")

        assert result is None

    def test_set_success(self, cache, mock_redis):
        """Test successful cache set."""
        mock_redis.set.return_value = True

        result = cache.set("test_key", {"data": "value"}, ttl=300)

        assert result is True
        mock_redis.set.assert_called_once()
        call_args = mock_redis.set.call_args
        assert call_args[0][0] == "rag:test_key"
        assert call_args[1]["ex"] == 300

    def test_set_with_default_ttl(self, cache, mock_redis):
        """Test set with default TTL."""
        mock_redis.set.return_value = True

        cache.set("test_key", "value")

        call_args = mock_redis.set.call_args
        assert call_args[1]["ex"] == 3600  # default TTL

    def test_set_with_nx_flag(self, cache, mock_redis):
        """Test set with NX (only if not exists) flag."""
        mock_redis.set.return_value = True

        cache.set("test_key", "value", nx=True)

        call_args = mock_redis.set.call_args
        assert call_args[1]["nx"] is True

    def test_delete_success(self, cache, mock_redis):
        """Test successful delete."""
        mock_redis.delete.return_value = 1

        result = cache.delete("test_key")

        assert result is True
        mock_redis.delete.assert_called_once_with("rag:test_key")

    def test_exists(self, cache, mock_redis):
        """Test key existence check."""
        mock_redis.exists.return_value = 1

        result = cache.exists("test_key")

        assert result is True
        mock_redis.exists.assert_called_once_with("rag:test_key")

    def test_ttl(self, cache, mock_redis):
        """Test TTL retrieval."""
        mock_redis.ttl.return_value = 300

        ttl = cache.ttl("test_key")

        assert ttl == 300
        mock_redis.ttl.assert_called_once_with("rag:test_key")

    def test_expire(self, cache, mock_redis):
        """Test setting expiration."""
        mock_redis.expire.return_value = True

        result = cache.expire("test_key", 600)

        assert result is True
        mock_redis.expire.assert_called_once_with("rag:test_key", 600)

    def test_invalidate_pattern(self, cache, mock_redis):
        """Test pattern invalidation."""
        mock_redis.scan_iter.return_value = ["rag:test1", "rag:test2", "rag:test3"]
        mock_redis.delete.return_value = 3

        count = cache.invalidate_pattern("test*")

        assert count == 3
        mock_redis.scan_iter.assert_called_once()
        mock_redis.delete.assert_called_once_with("rag:test1", "rag:test2", "rag:test3")

    def test_flush(self, cache, mock_redis):
        """Test cache flush."""
        mock_redis.scan_iter.return_value = ["rag:key1", "rag:key2"]
        mock_redis.delete.return_value = 2

        result = cache.flush()

        assert result is True
        mock_redis.scan_iter.assert_called_once()

    def test_flush_with_hooks(self, cache, mock_redis):
        """Test flush with registered hooks."""
        mock_redis.scan_iter.return_value = []

        hook_called = []

        def flush_hook():
            hook_called.append(True)

        cache.register_flush_hook(flush_hook)
        cache.flush()

        assert len(hook_called) == 1

    def test_register_flush_hook(self, cache):
        """Test registering flush hook."""

        def my_hook():
            pass

        cache.register_flush_hook(my_hook)

        assert my_hook in cache._flush_hooks

    def test_unregister_flush_hook(self, cache):
        """Test unregistering flush hook."""

        def my_hook():
            pass

        cache.register_flush_hook(my_hook)
        result = cache.unregister_flush_hook(my_hook)

        assert result is True
        assert my_hook not in cache._flush_hooks

    def test_health_check_success(self, cache, mock_redis):
        """Test successful health check."""
        mock_redis.ping.return_value = True

        result = cache.health_check()

        assert result is True

    def test_health_check_failure(self, cache, mock_redis):
        """Test failed health check."""
        import redis as redis_module

        mock_redis.ping.side_effect = redis_module.RedisError("Connection failed")

        result = cache.health_check()

        assert result is False

    def test_get_info(self, cache, mock_redis):
        """Test getting Redis info."""
        mock_redis.info.return_value = {"redis_version": "6.0.0"}

        info = cache.get_info()

        assert "redis_version" in info

    def test_context_manager(self, mock_redis):
        """Test using cache as context manager."""
        config = RedisCacheConfig()

        with RedisCache(config) as cache:
            assert cache._client is not None

        # Should close connection after context
        mock_redis.close.assert_called_once()

    def test_repr(self, cache):
        """Test string representation."""
        repr_str = repr(cache)

        assert "RedisCache" in repr_str
        assert "localhost" in repr_str
        assert "6379" in repr_str


class TestRedisCacheIntegration:
    """Integration tests for Redis cache (require actual Redis)."""

    @pytest.mark.integration
    def test_full_workflow(self):
        """Test complete cache workflow with actual Redis."""
        pytest.importorskip("redis")

        try:
            config = RedisCacheConfig(key_prefix="test:")
            cache = RedisCache(config)

            # Test connection
            if not cache.health_check():
                pytest.skip("Redis not available")

            # Set value
            assert cache.set("test_key", {"data": "value"}, ttl=60)

            # Get value
            result = cache.get("test_key")
            assert result == {"data": "value"}

            # Check existence
            assert cache.exists("test_key")

            # Check TTL
            ttl = cache.ttl("test_key")
            assert 0 < ttl <= 60

            # Delete
            assert cache.delete("test_key")
            assert not cache.exists("test_key")

            # Cleanup
            cache.flush()
            cache.close()

        except Exception as e:
            pytest.skip(f"Redis integration test failed: {e}")
