"""
Tests for semantic cache client.
"""

import json
from unittest.mock import MagicMock

import pytest

from src.services.cache.client import CacheClient
from src.services.cache.semantic_cache import SemanticCacheClient
from src.services.embeddings.client import EmbedClient


class TestSemanticCacheClient:
    """Tests for SemanticCacheClient."""

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        redis_instance = MagicMock()
        redis_instance.ping.return_value = True
        redis_instance.get.return_value = None
        redis_instance.set.return_value = True
        redis_instance.delete.return_value = 1
        redis_instance.keys.return_value = []
        return redis_instance

    @pytest.fixture
    def mock_embed_client(self):
        """Mock embedding client."""
        embed_client = MagicMock(spec=EmbedClient)
        # Return a simple embedding
        embed_client.embed.return_value = [[0.1] * 384]
        return embed_client

    @pytest.fixture
    def cache_client(self, mock_redis):
        """Create cache client."""
        return CacheClient(
            redis_client=mock_redis, ttl_hours=1, key_prefix="semantic:"
        )

    @pytest.fixture
    def semantic_cache(self, cache_client, mock_embed_client):
        """Create semantic cache instance."""
        return SemanticCacheClient(
            cache_client=cache_client,
            embed_client=mock_embed_client,
            similarity_threshold=0.95,
            embedding_dim=384,
            max_candidates=100,
        )

    def test_initialization(self, semantic_cache, cache_client, mock_embed_client):
        """Test semantic cache initialization."""
        assert semantic_cache.cache == cache_client
        assert semantic_cache.embed_client == mock_embed_client
        assert semantic_cache.similarity_threshold == 0.95
        assert semantic_cache.embedding_dim == 384
        assert semantic_cache.max_candidates == 100

    def test_get_cache_miss(self, semantic_cache, mock_redis):
        """Test cache miss when no index."""
        mock_redis.get.return_value = None

        result = semantic_cache.get("test query")

        assert result is None

    def test_set_cache(self, semantic_cache, mock_redis):
        """Test setting cache."""
        result = semantic_cache.set("test query", {"answer": "test"})

        assert result is True
        # Should have set embedding and value
        assert mock_redis.set.call_count >= 2

    def test_clear(self, semantic_cache):
        """Test clearing cache."""
        count = semantic_cache.clear()

        assert isinstance(count, int)

    def test_flush(self, semantic_cache):
        """Test flushing cache."""
        result = semantic_cache.flush()

        assert result is True

    def test_ping(self, semantic_cache, mock_redis):
        """Test ping."""
        mock_redis.ping.return_value = True

        result = semantic_cache.ping()

        assert result is True

    def test_health_check(self, semantic_cache, mock_redis):
        """Test health check."""
        mock_redis.ping.return_value = True

        result = semantic_cache.health_check()

        assert result is True

    def test_get_stats(self, semantic_cache, mock_redis):
        """Test getting stats."""
        mock_redis.get.return_value = json.dumps([])
        mock_redis.ping.return_value = True

        stats = semantic_cache.get_stats()

        assert "total_entries" in stats
        assert "similarity_threshold" in stats
        assert "embedding_dim" in stats
        assert stats["connected"] is True

    def test_context_manager(self, semantic_cache):
        """Test context manager support."""
        with semantic_cache as sc:
            assert sc is semantic_cache

    def test_repr(self, semantic_cache):
        """Test string representation."""
        repr_str = repr(semantic_cache)

        assert "SemanticCacheClient" in repr_str
        assert "0.95" in repr_str
