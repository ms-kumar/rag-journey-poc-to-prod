"""
Tests for embedding cache functionality.
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.services.embeddings.cache import EmbeddingCache


class TestEmbeddingCache:
    """Test embedding cache basic operations."""

    def test_initialization(self):
        """Test cache initialization."""
        cache = EmbeddingCache(max_size=100, model_identifier="test-model")
        assert len(cache) == 0
        assert cache.max_size == 100
        assert cache.model_identifier == "test-model"

    def test_put_and_get(self):
        """Test basic put and get operations."""
        cache = EmbeddingCache(max_size=10)
        embedding = [1.0, 2.0, 3.0]

        cache.put("hello", embedding)
        retrieved = cache.get("hello")

        assert retrieved == embedding
        assert cache.stats["hits"] == 1
        assert cache.stats["misses"] == 0

    def test_get_miss(self):
        """Test cache miss."""
        cache = EmbeddingCache(max_size=10)

        result = cache.get("nonexistent")

        assert result is None
        assert cache.stats["hits"] == 0
        assert cache.stats["misses"] == 1

    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = EmbeddingCache(max_size=3)

        # Fill cache
        cache.put("a", [1.0])
        cache.put("b", [2.0])
        cache.put("c", [3.0])

        # Access 'a' to make it recently used
        cache.get("a")

        # Add new item, should evict 'b' (least recently used)
        cache.put("d", [4.0])

        assert cache.get("a") is not None
        assert cache.get("b") is None  # Evicted
        assert cache.get("c") is not None
        assert cache.get("d") is not None

    def test_batch_operations(self):
        """Test batch put and get."""
        cache = EmbeddingCache(max_size=10)

        texts = ["hello", "world", "test"]
        embeddings = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

        cache.put_batch(texts, embeddings)

        retrieved = cache.get_batch(texts)

        assert len(retrieved) == 3
        assert retrieved["hello"] == [1.0, 2.0]
        assert retrieved["world"] == [3.0, 4.0]
        assert retrieved["test"] == [5.0, 6.0]

    def test_batch_partial_hit(self):
        """Test batch get with partial cache hits."""
        cache = EmbeddingCache(max_size=10)

        cache.put("hello", [1.0, 2.0])
        cache.put("world", [3.0, 4.0])

        retrieved = cache.get_batch(["hello", "world", "missing"])

        assert len(retrieved) == 2
        assert "hello" in retrieved
        assert "world" in retrieved
        assert "missing" not in retrieved

    def test_clear(self):
        """Test cache clearing."""
        cache = EmbeddingCache(max_size=10)

        cache.put("hello", [1.0])
        cache.put("world", [2.0])
        assert len(cache) == 2

        cache.clear()

        assert len(cache) == 0
        assert cache.stats["hits"] == 0
        assert cache.stats["misses"] == 0

    def test_stats(self):
        """Test cache statistics."""
        cache = EmbeddingCache(max_size=10)

        # Generate some hits and misses
        cache.put("a", [1.0])
        cache.get("a")  # Hit
        cache.get("a")  # Hit
        cache.get("b")  # Miss
        cache.get("c")  # Miss

        stats = cache.stats

        assert stats["size"] == 1
        assert stats["max_size"] == 10
        assert stats["hits"] == 2
        assert stats["misses"] == 2
        assert stats["hit_rate"] == 0.5

    def test_different_models_different_keys(self):
        """Test that different model identifiers produce different cache keys."""
        cache1 = EmbeddingCache(max_size=10, model_identifier="model-1")
        cache2 = EmbeddingCache(max_size=10, model_identifier="model-2")

        cache1.put("hello", [1.0, 2.0])
        cache2.put("hello", [3.0, 4.0])

        # Each cache should have its own entry
        assert cache1.get("hello") == [1.0, 2.0]
        assert cache2.get("hello") == [3.0, 4.0]

    def test_empty_batch(self):
        """Test batch operations with empty sequences."""
        cache = EmbeddingCache(max_size=10)

        cache.put_batch([], [])
        retrieved = cache.get_batch([])

        assert len(retrieved) == 0

    def test_batch_size_mismatch(self):
        """Test that batch put raises error on size mismatch."""
        cache = EmbeddingCache(max_size=10)

        with pytest.raises(ValueError, match="same length"):
            cache.put_batch(["a", "b"], [[1.0]])


class TestEmbeddingCachePersistence:
    """Test disk persistence functionality."""

    def test_save_to_disk(self):
        """Test saving cache to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = EmbeddingCache(max_size=10, cache_dir=tmpdir, model_identifier="test")

            cache.put("hello", [1.0, 2.0, 3.0])
            cache.put("world", [4.0, 5.0, 6.0])

            cache.save_to_disk()

            # Check file exists
            cache_file = Path(tmpdir) / "embedding_cache_test.json"
            assert cache_file.exists()

            # Check content
            with cache_file.open("r", encoding="utf-8") as f:
                data = json.load(f)

            assert data["model_identifier"] == "test"
            assert len(data["cache"]) == 2

    def test_load_from_disk(self):
        """Test loading cache from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save cache
            cache1 = EmbeddingCache(max_size=10, cache_dir=tmpdir, model_identifier="test")
            cache1.put("hello", [1.0, 2.0, 3.0])
            cache1.put("world", [4.0, 5.0, 6.0])
            cache1.save_to_disk()

            # Create new cache instance - should load from disk
            cache2 = EmbeddingCache(max_size=10, cache_dir=tmpdir, model_identifier="test")

            assert len(cache2) == 2
            assert cache2.get("hello") == [1.0, 2.0, 3.0]
            assert cache2.get("world") == [4.0, 5.0, 6.0]

    def test_no_disk_cache(self):
        """Test that cache works without disk persistence."""
        cache = EmbeddingCache(max_size=10, cache_dir=None)

        cache.put("hello", [1.0])
        cache.save_to_disk()  # Should not raise error

        assert cache.get("hello") == [1.0]

    def test_repr(self):
        """Test cache string representation."""
        cache = EmbeddingCache(max_size=100)
        cache.put("test", [1.0])
        cache.get("test")  # Hit
        cache.get("miss")  # Miss

        repr_str = repr(cache)

        assert "1/100" in repr_str
        assert "50.00%" in repr_str
