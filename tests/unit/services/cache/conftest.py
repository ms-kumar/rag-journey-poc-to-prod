"""Pytest configuration for cache tests."""

from unittest.mock import AsyncMock, Mock

import pytest


@pytest.fixture
def mock_redis_client():
    """Provide a mock Redis client."""
    client = Mock()
    client.get = AsyncMock(return_value=None)
    client.set = AsyncMock(return_value=True)
    client.delete = AsyncMock(return_value=1)
    client.exists = AsyncMock(return_value=False)
    client.ping = AsyncMock(return_value=True)
    return client


@pytest.fixture
def sample_cache_key():
    """Provide a sample cache key."""
    return "test:cache:key:123"


@pytest.fixture
def sample_cache_value():
    """Provide a sample cache value."""
    return {"data": "cached_content", "timestamp": 1234567890}


@pytest.fixture
def cache_config():
    """Provide cache configuration."""
    return {
        "ttl": 3600,
        "max_size": 1000,
        "eviction_policy": "lru",
    }
