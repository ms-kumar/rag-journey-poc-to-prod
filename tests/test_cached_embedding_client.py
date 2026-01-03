"""
Tests for cached embedding client.
"""

import tempfile

from src.services.embeddings.cached_client import CachedEmbeddingClient
from src.services.embeddings.client import EmbedClient


class MockEmbeddingProvider:
    """Mock embedding provider for testing."""

    def __init__(self, dim=64):
        self.dim = dim
        self.model_name = "mock-model"
        self.call_count = 0
        self.last_batch_size = 0

    def embed(self, texts):
        """Generate mock embeddings."""
        self.call_count += 1
        self.last_batch_size = len(texts)
        # Simple deterministic embeddings based on text length
        return [[float(len(text)) / 100.0] * self.dim for text in texts]

    @property
    def dimension(self):
        return self.dim


class TestCachedEmbeddingClient:
    """Test cached embedding client."""

    def test_initialization(self):
        """Test client initialization."""
        provider = MockEmbeddingProvider()
        client = CachedEmbeddingClient(provider, cache_enabled=True)

        assert client.cache_enabled
        assert client.provider == provider

    def test_embed_without_cache(self):
        """Test embedding without caching."""
        provider = MockEmbeddingProvider(dim=32)
        client = CachedEmbeddingClient(provider, cache_enabled=False)

        texts = ["hello", "world"]
        embeddings = client.embed(texts)

        assert len(embeddings) == 2
        assert len(embeddings[0]) == 32
        assert provider.call_count == 1

    def test_embed_with_cache_hit(self):
        """Test that cached embeddings are reused."""
        provider = MockEmbeddingProvider()
        client = CachedEmbeddingClient(provider, cache_enabled=True)

        # First call - should compute
        texts = ["hello", "world"]
        embeddings1 = client.embed(texts)
        assert provider.call_count == 1

        # Second call - should use cache
        embeddings2 = client.embed(texts)
        assert provider.call_count == 1  # No new computation
        assert embeddings1 == embeddings2

        # Check cache stats
        stats = client.cache_stats
        assert stats["hits"] == 2  # Both texts cached
        assert stats["size"] == 2

    def test_embed_partial_cache_hit(self):
        """Test embedding with partial cache hits."""
        provider = MockEmbeddingProvider()
        client = CachedEmbeddingClient(provider, cache_enabled=True, batch_size=10)

        # First batch
        client.embed(["hello", "world"])
        assert provider.call_count == 1

        # Second batch with one new text
        embeddings = client.embed(["hello", "world", "new"])
        assert provider.call_count == 2  # Only compute "new"
        assert len(embeddings) == 3

        # Check that only one text was computed
        assert provider.last_batch_size == 1

    def test_batch_processing(self):
        """Test that large batches are split correctly."""
        provider = MockEmbeddingProvider()
        client = CachedEmbeddingClient(provider, cache_enabled=True, batch_size=2)

        # 5 texts with batch size 2 should result in 3 calls (2+2+1)
        # when computing from scratch
        texts = [f"text{i}" for i in range(5)]
        embeddings = client.embed(texts)

        assert len(embeddings) == 5
        # First call computes all 5 in batches: 2+2+1 = 3 batches
        # But since they're all cache misses, they get batched for computation
        # The _compute_in_batches method will make 3 calls
        assert provider.call_count == 3

    def test_batch_processing_with_cache(self):
        """Test batch processing respects cache."""
        provider = MockEmbeddingProvider()
        client = CachedEmbeddingClient(provider, cache_enabled=True, batch_size=2)

        # First: compute 5 texts in batches
        texts = [f"text{i}" for i in range(5)]
        client.embed(texts)
        first_call_count = provider.call_count

        # Second: re-request same texts, should use cache
        client.embed(texts)
        assert provider.call_count == first_call_count  # No new calls

    def test_result_order_preserved(self):
        """Test that result order matches input order."""
        provider = MockEmbeddingProvider()
        client = CachedEmbeddingClient(provider, cache_enabled=True)

        # Populate cache with some texts
        client.embed(["a", "b", "c"])

        # Request in different order with mix of cached and new
        texts = ["c", "new", "a", "b"]
        embeddings = client.embed(texts)

        # Check that embeddings correspond to correct texts
        assert len(embeddings) == 4
        # "c" and "new" and "a" and "b" should each have correct embedding
        # (based on text length in mock provider)
        assert embeddings[0] == embeddings[2]  # "c" and "a" have same length
        assert embeddings[2] == embeddings[3]  # "a" and "b" have same length

    def test_empty_texts(self):
        """Test embedding empty text list."""
        provider = MockEmbeddingProvider()
        client = CachedEmbeddingClient(provider, cache_enabled=True)

        embeddings = client.embed([])

        assert embeddings == []
        assert provider.call_count == 0

    def test_clear_cache(self):
        """Test cache clearing."""
        provider = MockEmbeddingProvider()
        client = CachedEmbeddingClient(provider, cache_enabled=True)

        # Populate cache
        client.embed(["hello", "world"])
        assert client.cache_stats["size"] == 2

        # Clear cache
        client.clear_cache()

        assert client.cache_stats["size"] == 0

        # Re-embedding should recompute
        initial_count = provider.call_count
        client.embed(["hello"])
        assert provider.call_count == initial_count + 1

    def test_dimension_property(self):
        """Test dimension property."""
        provider = MockEmbeddingProvider(dim=128)
        client = CachedEmbeddingClient(provider)

        assert client.dimension == 128

    def test_dimension_fallback(self):
        """Test dimension property fallback."""

        class ProviderWithoutDim:
            model_name = "test"

            def embed(self, texts):
                return [[1.0, 2.0, 3.0]] * len(texts)

        provider = ProviderWithoutDim()
        client = CachedEmbeddingClient(provider)

        # Should compute dimension from actual embedding
        assert client.dimension == 3

    def test_cache_stats(self):
        """Test cache statistics reporting."""
        provider = MockEmbeddingProvider()
        client = CachedEmbeddingClient(provider, cache_enabled=True, cache_max_size=100)

        # Generate some activity
        client.embed(["a", "b", "c"])
        client.embed(["a", "b"])  # Hits
        client.embed(["d"])  # Miss

        stats = client.cache_stats

        assert stats["size"] == 4
        assert stats["max_size"] == 100
        assert stats["hits"] > 0
        assert stats["misses"] > 0

    def test_repr(self):
        """Test string representation."""
        provider = MockEmbeddingProvider()
        client = CachedEmbeddingClient(provider)

        repr_str = repr(client)

        assert "CachedEmbeddingClient" in repr_str
        assert "MockEmbeddingProvider" in repr_str


class TestCachedClientWithRealProvider:
    """Test cached client with real embedding provider."""

    def test_with_hash_provider(self):
        """Test caching with hash-based embeddings."""
        provider = EmbedClient(model_name="test-hash", dim=64)
        client = CachedEmbeddingClient(provider, cache_enabled=True)

        texts = ["hello world", "test embedding", "cached results"]

        # First call
        embeddings1 = client.embed(texts)
        assert len(embeddings1) == 3
        assert len(embeddings1[0]) == 64

        # Second call should use cache and produce identical results
        embeddings2 = client.embed(texts)
        assert embeddings1 == embeddings2

        # Cache should have all texts
        assert client.cache_stats["size"] == 3
        assert client.cache_stats["hits"] == 3

    def test_save_and_load_cache(self):
        """Test saving and loading cache from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = EmbedClient(dim=32)

            # Create client and populate cache
            client1 = CachedEmbeddingClient(
                provider, cache_enabled=True, cache_dir=tmpdir, cache_max_size=100
            )

            texts = ["test1", "test2", "test3"]
            embeddings1 = client1.embed(texts)
            client1.save_cache()

            # Create new client - should load from disk
            client2 = CachedEmbeddingClient(
                provider, cache_enabled=True, cache_dir=tmpdir, cache_max_size=100
            )

            # Should have cached embeddings
            assert client2.cache_stats["size"] == 3

            # Should produce same embeddings without recomputation
            embeddings2 = client2.embed(texts)
            assert embeddings1 == embeddings2
