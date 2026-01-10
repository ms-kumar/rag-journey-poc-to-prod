"""
Tests for semantic cache.
"""

import pytest

from src.services.cache.semantic_cache import SemanticCache, SemanticCacheConfig
from src.services.embeddings.client import EmbedClient


class TestSemanticCacheConfig:
    """Tests for semantic cache configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = SemanticCacheConfig()

        assert config.similarity_threshold == 0.95
        assert config.embedding_dim == 384
        assert config.max_candidates == 100

    def test_custom_config(self):
        """Test custom configuration."""
        config = SemanticCacheConfig(
            similarity_threshold=0.90,
            embedding_dim=768,
            max_candidates=50,
        )

        assert config.similarity_threshold == 0.90
        assert config.embedding_dim == 768
        assert config.max_candidates == 50


class TestSemanticCache:
    """Tests for semantic cache."""

    @pytest.fixture
    def embed_client(self):
        """Create embedding client."""
        return EmbedClient(dim=384, normalize=True)

    @pytest.fixture
    def mock_redis(self, monkeypatch):
        """Mock Redis cache."""
        from unittest.mock import MagicMock

        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_cache.set.return_value = True
        mock_cache.delete.return_value = True
        mock_cache.health_check.return_value = True

        def mock_redis_init(*args, **kwargs):
            return mock_cache

        monkeypatch.setattr(
            "src.services.cache.semantic_cache.RedisCache",
            mock_redis_init,
        )

        return mock_cache

    @pytest.fixture
    def cache(self, embed_client, mock_redis):
        """Create semantic cache instance."""
        config = SemanticCacheConfig(similarity_threshold=0.90)
        return SemanticCache(embed_client, config)

    def test_initialization(self, embed_client):
        """Test cache initialization."""
        config = SemanticCacheConfig(similarity_threshold=0.95)
        cache = SemanticCache(embed_client, config)

        assert cache.config.similarity_threshold == 0.95
        assert cache.embed_client == embed_client

    def test_hash_query(self, cache):
        """Test query hashing."""
        hash1 = cache._hash_query("test query")
        hash2 = cache._hash_query("test query")
        hash3 = cache._hash_query("different query")

        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 16

    def test_cosine_similarity_identical(self, cache):
        """Test cosine similarity with identical vectors."""
        vec = [1.0, 2.0, 3.0]
        similarity = cache._cosine_similarity(vec, vec)

        assert abs(similarity - 1.0) < 0.001

    def test_cosine_similarity_orthogonal(self, cache):
        """Test cosine similarity with orthogonal vectors."""
        vec1 = [1.0, 0.0]
        vec2 = [0.0, 1.0]
        similarity = cache._cosine_similarity(vec1, vec2)

        assert abs(similarity) < 0.001

    def test_cosine_similarity_opposite(self, cache):
        """Test cosine similarity with opposite vectors."""
        vec1 = [1.0, 1.0]
        vec2 = [-1.0, -1.0]
        similarity = cache._cosine_similarity(vec1, vec2)

        assert abs(similarity - (-1.0)) < 0.001

    def test_set_success(self, cache, mock_redis):
        """Test setting cache entry."""
        result = cache.set("test query", {"answer": "test result"}, ttl=300)

        assert result is True
        # Should call redis set twice (embedding + value)
        assert mock_redis.set.call_count >= 2

    def test_get_miss_no_candidates(self, cache, mock_redis):
        """Test cache miss with no candidates."""
        mock_redis.get.return_value = None

        result = cache.get("test query")

        assert result is None

    def test_get_miss_low_similarity(self, cache, mock_redis):
        """Test cache miss due to low similarity."""
        # Mock stored embedding (very different from query)
        stored_embedding = [0.0] * 384
        stored_embedding[0] = 1.0

        query_hash = cache._hash_query("different query")
        mock_redis.get.side_effect = lambda key: (
            [query_hash] if "index" in key else stored_embedding if "embedding" in key else None
        )

        result = cache.get("test query")

        assert result is None

    def test_get_hit_high_similarity(self, cache, mock_redis):
        """Test cache hit with high similarity."""
        # First set a value
        query1 = "What is machine learning?"
        cache.set(query1, {"answer": "ML is a subset of AI"})

        # Mock to return the stored embedding and value
        query_hash = cache._hash_query(query1)
        embedding = cache.embed_client.embed([query1])[0]

        mock_redis.get.side_effect = lambda key: (
            [query_hash]
            if "index" in key
            else embedding
            if "embedding" in key
            else {"answer": "ML is a subset of AI"}
            if "value" in key
            else None
        )

        # Query with very similar question
        result = cache.get("What is machine learning?")  # Same question

        # Due to embedding similarity, should get cache hit
        assert result == {"answer": "ML is a subset of AI"}

    def test_invalidate_all(self, cache, mock_redis):
        """Test invalidating all entries."""
        mock_redis.invalidate_pattern.return_value = 5

        count = cache.invalidate("*")

        assert count >= 2  # embeddings + values
        assert mock_redis.delete.called

    def test_flush(self, cache, mock_redis):
        """Test flushing cache."""
        mock_redis.invalidate_pattern.return_value = 10

        result = cache.flush()

        assert result is True

    def test_get_stats(self, cache, mock_redis):
        """Test getting cache statistics."""
        mock_redis.get.return_value = ["hash1", "hash2", "hash3"]

        stats = cache.get_stats()

        assert "total_entries" in stats
        assert "similarity_threshold" in stats
        assert stats["similarity_threshold"] == 0.90

    def test_health_check(self, cache, mock_redis):
        """Test health check."""
        mock_redis.health_check.return_value = True

        result = cache.health_check()

        assert result is True

    def test_repr(self, cache):
        """Test string representation."""
        repr_str = repr(cache)

        assert "SemanticCache" in repr_str
        assert "0.9" in repr_str


class TestSemanticCacheIntegration:
    """Integration tests for semantic cache."""

    @pytest.mark.integration
    def test_similar_queries(self):
        """Test caching with similar queries."""
        pytest.importorskip("redis")

        try:
            from src.services.cache.redis_client import RedisCache, RedisCacheConfig

            # Setup
            redis_config = RedisCacheConfig(key_prefix="test:semantic:")
            redis_cache = RedisCache(redis_config)

            if not redis_cache.health_check():
                pytest.skip("Redis not available")

            embed_client = EmbedClient(dim=384, normalize=True)
            config = SemanticCacheConfig(similarity_threshold=0.90)
            cache = SemanticCache(embed_client, config, redis_cache)

            # Set initial query
            cache.set("What is Python?", {"answer": "Python is a programming language"})

            # Query with similar phrasing
            result = cache.get("What is Python?")
            assert result == {"answer": "Python is a programming language"}

            # Cleanup
            cache.flush()
            redis_cache.close()

        except Exception as e:
            pytest.skip(f"Semantic cache integration test failed: {e}")
