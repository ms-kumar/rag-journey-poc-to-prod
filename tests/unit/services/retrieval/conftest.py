"""Pytest configuration for retrieval tests."""

from unittest.mock import Mock

import pytest


@pytest.fixture
def sample_query_vector():
    """Provide a sample query vector."""
    import numpy as np

    return np.random.rand(384).tolist()


@pytest.fixture
def sample_search_results():
    """Provide sample search results."""
    return [
        {
            "id": "doc1",
            "score": 0.95,
            "text": "Python is a programming language.",
            "metadata": {"source": "test"},
        },
        {
            "id": "doc2",
            "score": 0.87,
            "text": "Machine learning uses data.",
            "metadata": {"source": "test"},
        },
    ]


@pytest.fixture
def mock_vector_store():
    """Provide a mock vector store."""
    store = Mock()
    store.search = Mock(return_value=[])
    store.add_documents = Mock(return_value=True)
    store.delete = Mock(return_value=True)
    return store


@pytest.fixture
def retrieval_config():
    """Provide retrieval configuration."""
    return {
        "top_k": 5,
        "score_threshold": 0.7,
        "rerank": True,
        "use_hybrid_search": False,
    }
