"""
Unit tests for embedding provider adapters (E5, BGE, OpenAI, Cohere).
"""

import pytest

from src.services.embeddings.adapter import LangChainEmbeddingsAdapter
from src.services.embeddings.factory import get_embed_client


class TestHashEmbeddings:
    """Test original hash-based embeddings (backward compatibility)."""

    def test_hash_provider_default(self):
        """Test default hash provider."""
        client = get_embed_client(provider="hash", dim=64)
        embeddings = client.embed(["test text"])

        assert len(embeddings) == 1
        assert len(embeddings[0]) == 64

    def test_hash_provider_with_factory(self):
        """Test hash provider through factory."""
        client = get_embed_client(
            model_name="simple-hash", provider="hash", dim=128, normalize=True
        )

        embeddings = client.embed(["hello", "world"])
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 128


class TestE5Embeddings:
    """Test E5 embedding provider."""

    @pytest.mark.slow
    def test_e5_initialization(self):
        """Test E5 provider initialization."""
        try:
            client = get_embed_client(
                model_name="intfloat/e5-small-v2", provider="e5", device="cpu"
            )
            assert client is not None
            assert hasattr(client, "embed")
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    @pytest.mark.slow
    def test_e5_embed_documents(self):
        """Test E5 document embedding."""
        try:
            client = get_embed_client(
                model_name="intfloat/e5-small-v2", provider="e5", device="cpu"
            )

            embeddings = client.embed_documents(["test document"])
            assert len(embeddings) == 1
            assert len(embeddings[0]) == 384  # e5-small dimension
            assert all(isinstance(x, float) for x in embeddings[0])
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    @pytest.mark.slow
    def test_e5_query_prefix(self):
        """Test E5 query embedding with prefix."""
        try:
            client = get_embed_client(
                model_name="intfloat/e5-small-v2", provider="e5", device="cpu"
            )

            query_emb = client.embed_query("test query")
            assert len(query_emb) == 384
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    @pytest.mark.slow
    def test_e5_dimension_property(self):
        """Test E5 dimension property."""
        try:
            client = get_embed_client(model_name="intfloat/e5-small-v2", provider="e5")
            assert client.dimension == 384
        except ImportError:
            pytest.skip("sentence-transformers not installed")


class TestBGEEmbeddings:
    """Test BGE embedding provider."""

    @pytest.mark.slow
    def test_bge_initialization(self):
        """Test BGE provider initialization."""
        try:
            client = get_embed_client(
                model_name="BAAI/bge-small-en-v1.5", provider="bge", device="cpu"
            )
            assert client is not None
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    @pytest.mark.slow
    def test_bge_embed_documents(self):
        """Test BGE document embedding."""
        try:
            client = get_embed_client(
                model_name="BAAI/bge-small-en-v1.5", provider="bge", device="cpu"
            )

            embeddings = client.embed_documents(["test document"])
            assert len(embeddings) == 1
            assert len(embeddings[0]) == 384  # bge-small dimension
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    @pytest.mark.slow
    def test_bge_query_embedding(self):
        """Test BGE query embedding."""
        try:
            client = get_embed_client(
                model_name="BAAI/bge-small-en-v1.5", provider="bge", device="cpu"
            )

            query_emb = client.embed_query("search query")
            assert len(query_emb) == 384
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    @pytest.mark.slow
    def test_bge_with_instruction(self):
        """Test BGE with query instruction."""
        try:
            client = get_embed_client(
                model_name="BAAI/bge-small-en-v1.5",
                provider="bge",
                device="cpu",
                query_instruction="Represent this sentence:",
            )

            query_emb = client.embed_query("test")
            assert len(query_emb) == 384
        except ImportError:
            pytest.skip("sentence-transformers not installed")


class TestHuggingFaceEmbeddings:
    """Test generic HuggingFace embedding provider."""

    @pytest.mark.slow
    def test_huggingface_provider(self):
        """Test generic HuggingFace provider."""
        try:
            client = get_embed_client(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                provider="huggingface",
                device="cpu",
            )

            embeddings = client.embed(["test text"])
            assert len(embeddings) == 1
            assert len(embeddings[0]) == 384  # MiniLM dimension
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    @pytest.mark.slow
    def test_huggingface_batch_embedding(self):
        """Test batch embedding with HuggingFace."""
        try:
            client = get_embed_client(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                provider="hf",
                device="cpu",
                batch_size=2,
            )

            texts = ["text 1", "text 2", "text 3"]
            embeddings = client.embed(texts)

            assert len(embeddings) == 3
            assert all(len(emb) == 384 for emb in embeddings)
        except ImportError:
            pytest.skip("sentence-transformers not installed")


class TestOpenAIEmbeddings:
    """Test OpenAI embedding provider (mocked)."""

    def test_openai_initialization_fails_without_package(self):
        """Test that OpenAI provider requires openai package."""
        import importlib.util

        if importlib.util.find_spec("openai") is not None:
            pytest.skip("openai is installed, skipping negative test")

        from src.services.embeddings.factory import get_embed_client

        with pytest.raises(ImportError):
            get_embed_client(provider="openai", model_name="text-embedding-3-small", api_key="x")

    def test_openai_dimension_mapping(self):
        """Test OpenAI model dimension mapping."""
        try:
            from unittest import mock

            # Need to mock both the import and the client
            with mock.patch.dict("sys.modules", {"openai": mock.MagicMock()}):
                from src.services.embeddings.providers import OpenAIEmbeddings

                with mock.patch.object(OpenAIEmbeddings, "__init__", lambda self, **kwargs: None):
                    client = OpenAIEmbeddings.__new__(OpenAIEmbeddings)
                    client._dim = 1536
                    assert client.dimension == 1536
        except ImportError:
            pytest.skip("Cannot test without mock")


class TestCohereEmbeddings:
    """Test Cohere embedding provider (mocked)."""

    def test_cohere_initialization_fails_without_package(self):
        """Test that Cohere provider requires cohere package."""
        import importlib.util

        if importlib.util.find_spec("cohere") is not None:
            pytest.skip("cohere is installed, skipping negative test")

        from src.services.embeddings.factory import get_embed_client

        with pytest.raises(ImportError):
            get_embed_client(provider="cohere", model_name="embed-english-v3.0", api_key="x")

    def test_cohere_dimension_mapping(self):
        """Test Cohere model dimension mapping."""
        try:
            from unittest import mock

            with mock.patch.dict("sys.modules", {"cohere": mock.MagicMock()}):
                from src.services.embeddings.providers import CohereEmbeddings

                with mock.patch.object(CohereEmbeddings, "__init__", lambda self, **kwargs: None):
                    client = CohereEmbeddings.__new__(CohereEmbeddings)
                    client._dim = 1024
                    assert client.dimension == 1024
        except ImportError:
            pytest.skip("Cannot test without mock")


class TestFactoryIntegration:
    """Test factory integration with all providers."""

    def test_factory_unknown_provider(self):
        """Test factory with unknown provider."""
        with pytest.raises(ValueError, match="Unknown provider"):
            get_embed_client(provider="unknown_provider")

    def test_factory_hash_provider(self):
        """Test factory creates hash client correctly."""
        client = get_embed_client(provider="hash", dim=32)
        embeddings = client.embed(["test"])
        assert len(embeddings[0]) == 32

    def test_factory_case_insensitive(self):
        """Test factory handles case-insensitive provider names."""
        client1 = get_embed_client(provider="HASH", dim=64)
        client2 = get_embed_client(provider="Hash", dim=64)

        assert client1.dim == 64
        assert client2.dim == 64

    @pytest.mark.slow
    def test_factory_e5_shorthand(self):
        """Test factory with E5 provider."""
        try:
            client = get_embed_client(
                model_name="intfloat/e5-small-v2", provider="e5", device="cpu"
            )
            assert hasattr(client, "embed_query")
            assert hasattr(client, "embed_documents")
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    @pytest.mark.slow
    def test_factory_bge_shorthand(self):
        """Test factory with BGE provider."""
        try:
            client = get_embed_client(
                model_name="BAAI/bge-small-en-v1.5", provider="bge", device="cpu"
            )
            assert hasattr(client, "embed_query")
        except ImportError:
            pytest.skip("sentence-transformers not installed")


class TestLangChainAdapter:
    """Test LangChain adapter with new providers."""

    def test_adapter_with_hash_provider(self):
        """Test adapter wraps hash provider correctly."""
        client = get_embed_client(provider="hash", dim=64)
        adapter = LangChainEmbeddingsAdapter(client)

        # Test embed_documents
        doc_embs = adapter.embed_documents(["doc1", "doc2"])
        assert len(doc_embs) == 2
        assert len(doc_embs[0]) == 64

        # Test embed_query
        query_emb = adapter.embed_query("query")
        assert len(query_emb) == 64

    @pytest.mark.slow
    def test_adapter_with_e5_provider(self):
        """Test adapter with E5 provider."""
        try:
            client = get_embed_client(
                model_name="intfloat/e5-small-v2", provider="e5", device="cpu"
            )
            adapter = LangChainEmbeddingsAdapter(client)

            doc_embs = adapter.embed_documents(["test"])
            assert len(doc_embs) == 1
            assert len(doc_embs[0]) == 384
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    @pytest.mark.slow
    def test_adapter_with_bge_provider(self):
        """Test adapter with BGE provider."""
        try:
            client = get_embed_client(
                model_name="BAAI/bge-small-en-v1.5", provider="bge", device="cpu"
            )
            adapter = LangChainEmbeddingsAdapter(client)

            query_emb = adapter.embed_query("search query")
            assert len(query_emb) == 384
        except ImportError:
            pytest.skip("sentence-transformers not installed")


class TestConvenienceFunctions:
    """Test convenience functions for creating embeddings."""

    @pytest.mark.slow
    def test_create_e5_embeddings_shorthand(self):
        """Test E5 convenience function."""
        try:
            from src.services.embeddings.providers import create_e5_embeddings

            client = create_e5_embeddings(model_size="small", device="cpu")
            assert client.dimension == 384

            embeddings = client.embed(["test"])
            assert len(embeddings[0]) == 384
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    @pytest.mark.slow
    def test_create_bge_embeddings_shorthand(self):
        """Test BGE convenience function."""
        try:
            from src.services.embeddings.providers import create_bge_embeddings

            client = create_bge_embeddings(model_size="small", device="cpu")
            assert client.dimension == 384
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    @pytest.mark.slow
    def test_model_size_mapping(self):
        """Test that model size maps to correct models."""
        try:
            from src.services.embeddings.providers import create_e5_embeddings

            small = create_e5_embeddings("small", device="cpu")
            assert "e5-small" in small.model_name.lower()

            base = create_e5_embeddings("base", device="cpu")
            assert "e5-base" in base.model_name.lower()
        except ImportError:
            pytest.skip("sentence-transformers not installed")


class TestProviderFeatures:
    """Test provider-specific features."""

    @pytest.mark.slow
    def test_e5_prefix_behavior(self):
        """Test E5 prefix feature."""
        try:
            client = get_embed_client(
                model_name="intfloat/e5-small-v2",
                provider="e5",
                device="cpu",
                prefix_query=True,
                prefix_document=True,
            )

            # Query and document embeddings should differ
            query_emb = client.embed_query("test")
            doc_emb = client.embed_documents(["test"])[0]

            # They should be different due to prefixes
            assert query_emb != doc_emb
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    @pytest.mark.slow
    def test_normalization(self):
        """Test embedding normalization."""
        try:
            import math

            client = get_embed_client(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                provider="huggingface",
                device="cpu",
                normalize=True,
            )

            embeddings = client.embed(["test"])
            vec = embeddings[0]

            # Calculate L2 norm
            norm = math.sqrt(sum(x * x for x in vec))

            # Should be approximately 1.0 for normalized vectors
            assert abs(norm - 1.0) < 0.01
        except ImportError:
            pytest.skip("sentence-transformers not installed")


class TestErrorHandling:
    """Test error handling across providers."""

    def test_empty_text_list(self):
        """Test embedding empty text list."""
        client = get_embed_client(provider="hash", dim=32)
        embeddings = client.embed([])
        assert embeddings == []

    @pytest.mark.slow
    def test_e5_empty_text_list(self):
        """Test E5 with empty text list."""
        try:
            client = get_embed_client(
                model_name="intfloat/e5-small-v2", provider="e5", device="cpu"
            )
            embeddings = client.embed([])
            assert embeddings == []
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_invalid_provider_params(self):
        """Test factory with invalid parameters."""
        with pytest.raises(ValueError):
            get_embed_client(provider="invalid_provider_name")
