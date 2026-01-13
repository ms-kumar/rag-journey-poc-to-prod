"""Pytest configuration for embeddings tests."""

from unittest.mock import Mock

import numpy as np
import pytest


@pytest.fixture
def sample_embedding():
    """Provide a sample embedding vector."""
    return np.random.rand(384).tolist()


@pytest.fixture
def sample_embeddings_batch():
    """Provide a batch of sample embeddings."""
    return [np.random.rand(384).tolist() for _ in range(5)]


@pytest.fixture
def mock_embedding_model():
    """Provide a mock embedding model."""
    model = Mock()
    model.encode = Mock(return_value=np.random.rand(384))
    model.encode_batch = Mock(return_value=np.random.rand(5, 384))
    return model


@pytest.fixture
def embedding_config():
    """Provide embedding configuration."""
    return {
        "model_name": "all-MiniLM-L6-v2",
        "dimension": 384,
        "batch_size": 32,
        "normalize": True,
    }
