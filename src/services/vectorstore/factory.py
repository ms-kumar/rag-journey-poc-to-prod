"""
Factory for creating vectorstore clients from application settings.
"""

import logging
from typing import TYPE_CHECKING

from src.services.vectorstore.client import QdrantVectorStoreClient

if TYPE_CHECKING:
    from src.config import VectorStoreSettings

logger = logging.getLogger(__name__)


def get_vectorstore_client(
    embeddings,
    settings: "VectorStoreSettings",
) -> QdrantVectorStoreClient:
    """
    Factory helper to create a Qdrant-backed vectorstore client.

    Args:
      embeddings: LangChain-compatible embeddings object (must implement embed_documents/embed_query).
      settings: VectorStore settings (url, api_key, collection_name, etc.)

    Returns:
        Configured QdrantVectorStoreClient instance
    """
    logger.info(f"Created QdrantVectorStoreClient with collection={settings.collection_name}")
    return QdrantVectorStoreClient(embeddings=embeddings, config=settings)


def make_vectorstore_client(
    settings: "VectorStoreSettings",
    embeddings,
    vector_size: int,
    retry_config=None,
) -> QdrantVectorStoreClient:
    """
    Create vectorstore client from application settings.

    Args:
        settings: Vectorstore settings
        embeddings: LangChain-compatible embeddings object
        vector_size: Dimension of the embedding vectors
        retry_config: Optional retry configuration

    Returns:
        Configured QdrantVectorStoreClient instance
    """
    logger.info(f"Vectorstore client created with collection={settings.collection_name}")
    return QdrantVectorStoreClient(
        embeddings=embeddings, config=settings, retry_config=retry_config
    )


def create_from_settings(
    settings: "VectorStoreSettings",
    embeddings,
    vector_size: int | None = None,
    **overrides,
) -> QdrantVectorStoreClient:
    """
    Create vectorstore client from application settings with optional overrides.

    Deprecated: Use make_vectorstore_client() instead.
    """
    return QdrantVectorStoreClient(embeddings=embeddings, config=settings)
