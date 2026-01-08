"""
Tests for sparse vector search functionality.

Tests cover:
- Sparse vector storage
- Sparse search
- Sparse search with metrics
- Integration with SPLADE encoder
"""

import pytest


class TestSparseVectorStorage:
    """Test sparse vector storage in vectorstore."""

    def test_add_texts_with_sparse_vectors(self, vectorstore_with_sparse):
        """Test adding texts with sparse vectors."""
        vs = vectorstore_with_sparse

        # Add texts (should compute both dense and sparse)
        ids = vs.add_texts(
            texts=["machine learning", "deep learning"],
            metadatas=[{"source": "doc1"}, {"source": "doc2"}],
        )

        assert len(ids) == 2

    def test_sparse_encoder_called_on_add(self, vectorstore_with_sparse, mock_sparse_encoder):
        """Test that sparse encoder is called when adding texts."""
        vs = vectorstore_with_sparse

        # Reset call count
        mock_sparse_encoder.encode_call_count = 0

        vs.add_texts(texts=["machine learning", "neural networks"])

        # Should have called encode_documents once
        assert mock_sparse_encoder.encode_call_count == 1

    def test_add_texts_without_sparse_encoder(self, vectorstore_no_sparse):
        """Test adding texts without sparse encoder (should work normally)."""
        vs = vectorstore_no_sparse

        ids = vs.add_texts(texts=["machine learning", "deep learning"])
        assert len(ids) == 2


class TestSparseSearch:
    """Test sparse vector search."""

    def test_sparse_search_basic(self, vectorstore_with_data_sparse):
        """Test basic sparse search."""
        vs = vectorstore_with_data_sparse

        docs = vs.sparse_search("machine learning", k=2)

        assert len(docs) <= 2
        for doc in docs:
            assert doc.metadata.get("search_type") == "sparse"
            assert "score" in doc.metadata

    def test_sparse_search_with_filter(self, vectorstore_with_data_sparse):
        """Test sparse search with metadata filter."""
        vs = vectorstore_with_data_sparse

        docs = vs.sparse_search(
            "neural networks",
            k=5,
            filter_dict={"source": "doc1"},
        )

        # Should only return docs with source=doc1
        for doc in docs:
            assert doc.metadata.get("source") == "doc1"

    def test_sparse_search_k_parameter(self, vectorstore_with_data_sparse):
        """Test that k parameter limits results."""
        vs = vectorstore_with_data_sparse

        docs = vs.sparse_search("learning", k=3)
        assert len(docs) <= 3

    def test_sparse_search_requires_enabled(self, vectorstore_no_sparse):
        """Test that sparse search fails if not enabled."""
        vs = vectorstore_no_sparse

        with pytest.raises(ValueError, match="Sparse search not enabled"):
            vs.sparse_search("machine learning")

    def test_sparse_search_requires_encoder(self, vectorstore_sparse_no_encoder):
        """Test that sparse search fails without encoder."""
        vs = vectorstore_sparse_no_encoder

        with pytest.raises(ValueError, match="No sparse encoder provided"):
            vs.sparse_search("machine learning")


class TestSparseSearchWithMetrics:
    """Test sparse search with metrics tracking."""

    def test_sparse_search_with_metrics_tracking(self, vectorstore_with_data_sparse_metrics):
        """Test that sparse search tracks metrics."""
        vs = vectorstore_with_data_sparse_metrics

        # Reset metrics
        vs.reset_metrics()

        # Perform search
        _ = vs.sparse_search_with_metrics("machine learning", k=5)

        # Check metrics were tracked
        metrics = vs.get_retrieval_metrics()
        assert metrics is not None
        assert "sparse" in metrics["by_search_type"]
        assert metrics["by_search_type"]["sparse"]["total_queries"] == 1

    def test_sparse_search_score_normalization(self, vectorstore_with_data_sparse_metrics):
        """Test sparse search with score normalization."""
        vs = vectorstore_with_data_sparse_metrics

        docs = vs.sparse_search_with_metrics(
            "machine learning",
            k=5,
            normalize_scores=True,
        )

        # All scores should be normalized to [0, 1]
        for doc in docs:
            score = doc.metadata.get("score", 0)
            assert 0.0 <= score <= 1.0
            assert doc.metadata.get("score_normalized") is True

    def test_sparse_search_without_normalization(self, vectorstore_with_data_sparse_metrics):
        """Test sparse search without score normalization."""
        vs = vectorstore_with_data_sparse_metrics

        docs = vs.sparse_search_with_metrics(
            "machine learning",
            k=5,
            normalize_scores=False,
        )

        # Scores should not be marked as normalized
        for doc in docs:
            assert doc.metadata.get("score_normalized") is not True

    def test_multiple_sparse_searches_track_separately(self, vectorstore_with_data_sparse_metrics):
        """Test that multiple sparse searches track correctly."""
        vs = vectorstore_with_data_sparse_metrics

        vs.reset_metrics()

        vs.sparse_search_with_metrics("machine learning", k=5)
        vs.sparse_search_with_metrics("deep learning", k=3)
        vs.sparse_search_with_metrics("neural networks", k=10)

        metrics = vs.get_retrieval_metrics()
        assert metrics["by_search_type"]["sparse"]["total_queries"] == 3


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_sparse_encoder():
    """Mock SPLADE encoder for testing."""

    class MockSparseEncoder:
        """Mock sparse encoder."""

        def __init__(self):
            self.encode_call_count = 0

        def encode_query(self, query: str) -> dict[int, float]:
            """Mock encode_query."""
            self.encode_call_count += 1
            # Return mock sparse vector
            return {
                100: 0.8,
                200: 0.6,
                300: 0.4,
                400: 0.2,
            }

        def encode_documents(self, texts: list[str]) -> list[dict[int, float]]:
            """Mock encode_documents."""
            self.encode_call_count += 1
            # Return mock sparse vectors
            return [{100: 0.8, 200: 0.6, 300: 0.4} for _ in texts]

    return MockSparseEncoder()


@pytest.fixture
def vectorstore_with_sparse(sample_embeddings, mock_sparse_encoder):
    """Vectorstore with sparse vectors enabled."""
    from src.services.vectorstore.client import (
        QdrantVectorStoreClient,
        VectorStoreConfig,
    )

    config = VectorStoreConfig(
        collection_name="test_sparse",
        vector_size=64,
        enable_sparse=True,
        sparse_vector_name="sparse",
    )
    return QdrantVectorStoreClient(
        embeddings=sample_embeddings,
        config=config,
        sparse_encoder=mock_sparse_encoder,
    )


@pytest.fixture
def vectorstore_no_sparse(sample_embeddings):
    """Vectorstore without sparse vectors."""
    from src.services.vectorstore.client import (
        QdrantVectorStoreClient,
        VectorStoreConfig,
    )

    config = VectorStoreConfig(
        collection_name="test_no_sparse",
        vector_size=64,
        enable_sparse=False,
    )
    return QdrantVectorStoreClient(
        embeddings=sample_embeddings,
        config=config,
    )


@pytest.fixture
def vectorstore_sparse_no_encoder(sample_embeddings):
    """Vectorstore with sparse enabled but no encoder."""
    from src.services.vectorstore.client import (
        QdrantVectorStoreClient,
        VectorStoreConfig,
    )

    config = VectorStoreConfig(
        collection_name="test_sparse_no_encoder",
        vector_size=64,
        enable_sparse=True,
    )
    return QdrantVectorStoreClient(
        embeddings=sample_embeddings,
        config=config,
        sparse_encoder=None,
    )


@pytest.fixture
def vectorstore_with_data_sparse(vectorstore_with_sparse):
    """Vectorstore with sparse vectors and sample data."""
    vs = vectorstore_with_sparse

    # Add sample data
    vs.add_texts(
        texts=[
            "machine learning is a subset of AI",
            "deep learning uses neural networks",
            "neural networks have multiple layers",
            "reinforcement learning learns from rewards",
        ],
        metadatas=[
            {"source": "doc1", "category": "ML"},
            {"source": "doc2", "category": "DL"},
            {"source": "doc1", "category": "DL"},
            {"source": "doc3", "category": "RL"},
        ],
    )

    return vs


@pytest.fixture
def vectorstore_with_data_sparse_metrics(sample_embeddings, mock_sparse_encoder):
    """Vectorstore with sparse vectors, data, and metrics enabled."""
    from src.services.vectorstore.client import (
        QdrantVectorStoreClient,
        VectorStoreConfig,
    )

    config = VectorStoreConfig(
        collection_name="test_sparse_metrics",
        vector_size=64,
        enable_sparse=True,
        sparse_vector_name="sparse",
        enable_metrics=True,
        normalize_scores=False,
    )

    vs = QdrantVectorStoreClient(
        embeddings=sample_embeddings,
        config=config,
        sparse_encoder=mock_sparse_encoder,
    )

    # Add sample data
    vs.add_texts(
        texts=[
            "machine learning algorithms",
            "deep learning models",
            "neural network architectures",
            "reinforcement learning agents",
        ],
        metadatas=[
            {"source": "doc1"},
            {"source": "doc2"},
            {"source": "doc1"},
            {"source": "doc3"},
        ],
    )

    return vs


@pytest.fixture
def sample_embeddings():
    """Simple mock embeddings for testing."""

    class MockEmbeddings:
        """Mock embeddings client."""

        def embed_documents(self, texts):
            """Return random vectors."""
            import random

            return [[random.random() for _ in range(64)] for _ in texts]

        def embed_query(self, text):
            """Return random vector."""
            import random

            return [random.random() for _ in range(64)]

    return MockEmbeddings()
