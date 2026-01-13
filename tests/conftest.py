"""Root pytest configuration and shared fixtures for all tests.

This file contains global fixtures and configuration that apply to all tests.
Module-specific fixtures should be placed in their respective conftest.py files.
"""

import os
import sys
from pathlib import Path

import pytest

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


# =============================================================================
# Global Test Configuration
# =============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, isolated)")
    config.addinivalue_line("markers", "integration: Integration tests (slower, with dependencies)")
    config.addinivalue_line("markers", "slow: Slow-running tests")
    config.addinivalue_line("markers", "asyncio: Async tests")
    config.addinivalue_line("markers", "canary: Canary tests for CI quick validation")
    config.addinivalue_line("markers", "agent: Agent-related tests")
    config.addinivalue_line("markers", "cache: Cache-related tests")
    config.addinivalue_line("markers", "embeddings: Embedding-related tests")
    config.addinivalue_line("markers", "guardrails: Guardrails and safety tests")
    config.addinivalue_line("markers", "retrieval: Retrieval and search tests")


# =============================================================================
# Global Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment variables before each test."""
    original_env = os.environ.copy()
    yield
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture(autouse=True)
def reset_global_registry():
    """Reset global agent registry before each test."""
    try:
        import src.services.agent.tools.registry as registry_module

        registry_module._global_registry = None
        yield
        registry_module._global_registry = None
    except ImportError:
        yield


@pytest.fixture(autouse=True)
def reset_global_tracker():
    """Reset global metrics tracker before each test."""
    try:
        import src.services.agent.metrics.tracker as tracker_module

        tracker_module._global_tracker = None
        yield
        tracker_module._global_tracker = None
    except ImportError:
        yield


@pytest.fixture
def temp_data_dir(tmp_path) -> Path:
    """Provide a temporary directory for test data."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


@pytest.fixture
def sample_text() -> str:
    """Provide sample text for testing."""
    return """
    Machine learning is a subset of artificial intelligence that focuses on
    building systems that can learn from data. It uses statistical techniques
    to give computers the ability to learn without being explicitly programmed.
    """


@pytest.fixture
def sample_documents() -> list[dict]:
    """Provide sample documents for testing."""
    return [
        {
            "id": "doc1",
            "text": "Python is a high-level programming language.",
            "metadata": {"source": "test", "category": "programming"},
        },
        {
            "id": "doc2",
            "text": "Machine learning uses algorithms to learn from data.",
            "metadata": {"source": "test", "category": "ml"},
        },
        {
            "id": "doc3",
            "text": "FastAPI is a modern web framework for building APIs.",
            "metadata": {"source": "test", "category": "web"},
        },
    ]


# =============================================================================
# Test Helpers
# =============================================================================


@pytest.fixture
def mock_env_vars(monkeypatch) -> dict:
    """Provide mock environment variables for testing."""
    env_vars = {
        "OPENAI_API_KEY": "test-key-123",
        "REDIS_URL": "redis://localhost:6379",
        "QDRANT_URL": "http://localhost:6333",
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    return env_vars


# =============================================================================
# Async Test Support
# =============================================================================


@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    import asyncio

    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
