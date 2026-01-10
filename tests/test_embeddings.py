"""
Unit tests for embedding services.
"""

import math

import pytest

from src.exceptions import EmbeddingDimensionMismatch
from src.services.embeddings.adapter import LangChainEmbeddingsAdapter
from src.services.embeddings.client import EmbedClient


class TestEmbedClient:
    """Test EmbedClient with various boundary conditions."""

    def test_initialization_valid(self):
        """Test valid initialization."""
        client = EmbedClient(model_name="test-model", dim=64, normalize=True)
        assert client.model_name == "test-model"
        assert client.dim == 64
        assert client.normalize is True

    def test_initialization_custom_dim(self):
        """Test initialization with custom dimension."""
        client = EmbedClient(dim=128)
        assert client.dim == 128

    def test_initialization_invalid_dim_zero(self):
        """Test initialization with zero dimension raises error."""
        with pytest.raises(EmbeddingDimensionMismatch, match="dim must be a positive integer"):
            EmbedClient(dim=0)

    def test_initialization_invalid_dim_negative(self):
        """Test initialization with negative dimension raises error."""
        with pytest.raises(EmbeddingDimensionMismatch, match="dim must be a positive integer"):
            EmbedClient(dim=-10)

    def test_embed_single_text(self):
        """Test embedding a single text."""
        client = EmbedClient(dim=64, normalize=True)
        embeddings = client.embed(["hello world"])

        assert len(embeddings) == 1
        assert len(embeddings[0]) == 64
        # All values should be floats
        assert all(isinstance(x, float) for x in embeddings[0])

    def test_embed_multiple_texts(self):
        """Test embedding multiple texts."""
        client = EmbedClient(dim=32, normalize=True)
        texts = ["hello", "world", "test"]
        embeddings = client.embed(texts)

        assert len(embeddings) == 3
        assert all(len(emb) == 32 for emb in embeddings)

    def test_embed_empty_list(self):
        """Test embedding empty list."""
        client = EmbedClient(dim=64)
        embeddings = client.embed([])

        assert embeddings == []

    def test_embed_empty_string(self):
        """Test embedding empty string."""
        client = EmbedClient(dim=64)
        embeddings = client.embed([""])

        assert len(embeddings) == 1
        assert len(embeddings[0]) == 64

    def test_deterministic_embeddings(self):
        """Test that same text produces same embeddings."""
        client = EmbedClient(dim=64, normalize=True)
        text = "deterministic test"

        emb1 = client.embed([text])[0]
        emb2 = client.embed([text])[0]

        # Should be identical
        assert emb1 == emb2

    def test_different_texts_different_embeddings(self):
        """Test that different texts produce different embeddings."""
        client = EmbedClient(dim=64, normalize=True)

        emb1 = client.embed(["hello"])[0]
        emb2 = client.embed(["world"])[0]

        # Should be different
        assert emb1 != emb2

    def test_normalized_embeddings(self):
        """Test that normalized embeddings have unit norm."""
        client = EmbedClient(dim=64, normalize=True)
        embeddings = client.embed(["test text"])

        # Calculate L2 norm
        vec = embeddings[0]
        norm = math.sqrt(sum(x * x for x in vec))

        # Should be approximately 1.0
        assert abs(norm - 1.0) < 1e-6

    def test_unnormalized_embeddings(self):
        """Test embeddings without normalization."""
        client = EmbedClient(dim=64, normalize=False)
        embeddings = client.embed(["test text"])

        vec = embeddings[0]
        norm = math.sqrt(sum(x * x for x in vec))

        # Should not necessarily be 1.0
        # Just verify it's a reasonable value
        assert norm > 0

    def test_embedding_values_in_range(self):
        """Test that embedding values are in expected range."""
        client = EmbedClient(dim=64, normalize=False)
        embeddings = client.embed(["test"])

        # Without normalization, values should be in [-1, 1]
        vec = embeddings[0]
        assert all(-1.0 <= x <= 1.0 for x in vec)

    def test_embed_none_value(self):
        """Test embedding with None value."""
        client = EmbedClient(dim=32)
        embeddings = client.embed([None])

        # Should handle None as empty string
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 32

    def test_embed_mixed_valid_and_none(self):
        """Test embedding with mix of valid texts and None."""
        client = EmbedClient(dim=32)
        texts = ["hello", None, "world"]
        embeddings = client.embed(texts)

        assert len(embeddings) == 3
        assert all(len(emb) == 32 for emb in embeddings)

    def test_unicode_text_embedding(self):
        """Test embedding with unicode characters."""
        client = EmbedClient(dim=64)
        texts = ["Hello 世界", "🌍 emoji", "café"]
        embeddings = client.embed(texts)

        assert len(embeddings) == 3
        assert all(len(emb) == 64 for emb in embeddings)

    def test_very_long_text(self):
        """Test embedding very long text."""
        client = EmbedClient(dim=64)
        long_text = "word " * 10000
        embeddings = client.embed([long_text])

        assert len(embeddings) == 1
        assert len(embeddings[0]) == 64

    def test_special_characters(self):
        """Test embedding with special characters."""
        client = EmbedClient(dim=32)
        texts = ["!@#$%^&*()", "\n\t\r", "line1\nline2"]
        embeddings = client.embed(texts)

        assert len(embeddings) == 3
        assert all(len(emb) == 32 for emb in embeddings)

    def test_dimension_one(self):
        """Test with dimension of 1."""
        client = EmbedClient(dim=1, normalize=True)
        embeddings = client.embed(["test"])

        assert len(embeddings[0]) == 1
        # With normalization, single dimension should be ±1
        assert abs(abs(embeddings[0][0]) - 1.0) < 1e-6

    def test_large_dimension(self):
        """Test with large dimension."""
        client = EmbedClient(dim=1024)
        embeddings = client.embed(["test"])

        assert len(embeddings[0]) == 1024

    def test_whitespace_only_text(self):
        """Test embedding whitespace-only text."""
        client = EmbedClient(dim=32)
        embeddings = client.embed(["   \t\n   "])

        assert len(embeddings) == 1
        assert len(embeddings[0]) == 32


class TestLangChainEmbeddingsAdapter:
    """Test LangChain embeddings adapter."""

    def test_adapter_initialization(self):
        """Test adapter initialization."""
        embed_client = EmbedClient(dim=64)
        adapter = LangChainEmbeddingsAdapter(embed_client)

        assert adapter._client == embed_client

    def test_embed_documents(self):
        """Test embed_documents method."""
        embed_client = EmbedClient(dim=32)
        adapter = LangChainEmbeddingsAdapter(embed_client)

        docs = ["doc1", "doc2", "doc3"]
        embeddings = adapter.embed_documents(docs)

        assert len(embeddings) == 3
        assert all(len(emb) == 32 for emb in embeddings)

    def test_embed_query(self):
        """Test embed_query method."""
        embed_client = EmbedClient(dim=32)
        adapter = LangChainEmbeddingsAdapter(embed_client)

        query = "test query"
        embedding = adapter.embed_query(query)

        assert len(embedding) == 32
        assert all(isinstance(x, float) for x in embedding)

    def test_adapter_consistency_with_client(self):
        """Test that adapter produces same results as client."""
        embed_client = EmbedClient(dim=64)
        adapter = LangChainEmbeddingsAdapter(embed_client)

        text = "consistency test"

        # Direct client call
        client_emb = embed_client.embed([text])[0]

        # Via adapter
        adapter_emb = adapter.embed_query(text)

        # Should be identical
        assert client_emb == adapter_emb

    def test_adapter_empty_documents(self):
        """Test adapter with empty document list."""
        embed_client = EmbedClient(dim=32)
        adapter = LangChainEmbeddingsAdapter(embed_client)

        embeddings = adapter.embed_documents([])
        assert embeddings == []

    def test_adapter_empty_query(self):
        """Test adapter with empty query."""
        embed_client = EmbedClient(dim=32)
        adapter = LangChainEmbeddingsAdapter(embed_client)

        embedding = adapter.embed_query("")
        assert len(embedding) == 32
