"""Mock factories for creating test doubles."""

from typing import Any
from unittest.mock import AsyncMock, Mock


class MockFactory:
    """Factory for creating common mocks."""

    @staticmethod
    def create_llm_client(
        response: str = "Test response",
        streaming: bool = False,
    ) -> Mock:
        """Create a mock LLM client."""
        client = Mock()
        if streaming:
            client.stream = AsyncMock(return_value=[response])
        else:
            client.complete = AsyncMock(return_value={"content": response})
        return client

    @staticmethod
    def create_embedding_client(dimension: int = 384) -> Mock:
        """Create a mock embedding client."""
        import numpy as np

        client = Mock()
        client.embed = AsyncMock(return_value=np.random.rand(dimension).tolist())
        client.embed_batch = AsyncMock(return_value=np.random.rand(5, dimension).tolist())
        return client

    @staticmethod
    def create_cache_client() -> Mock:
        """Create a mock cache client."""
        client = Mock()
        client.get = AsyncMock(return_value=None)
        client.set = AsyncMock(return_value=True)
        client.delete = AsyncMock(return_value=True)
        client.exists = AsyncMock(return_value=False)
        return client

    @staticmethod
    def create_vector_store() -> Mock:
        """Create a mock vector store."""
        store = Mock()
        store.search = AsyncMock(return_value=[])
        store.add = AsyncMock(return_value=True)
        store.delete = AsyncMock(return_value=True)
        store.upsert = AsyncMock(return_value=True)
        return store

    @staticmethod
    def create_reranker() -> Mock:
        """Create a mock reranker."""
        reranker = Mock()
        reranker.rerank = Mock(return_value=[])
        return reranker


class AsyncMockFactory:
    """Factory for creating async mocks."""

    @staticmethod
    async def async_return(value: Any):
        """Helper to create async return value."""
        return value

    @staticmethod
    def create_async_iterator(items: list[Any]):
        """Create an async iterator mock."""

        async def async_gen():
            for item in items:
                yield item

        return async_gen()


class ContextManagerMock:
    """Mock for context managers."""

    def __init__(self, return_value: Any = None):
        self.return_value = return_value
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self.return_value

    def __exit__(self, *args):
        self.exited = True
        return False

    async def __aenter__(self):
        self.entered = True
        return self.return_value

    async def __aexit__(self, *args):
        self.exited = True
        return False
