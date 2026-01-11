"""
Factory for creating vectorstore clients from application settings.
"""

import logging
from typing import TYPE_CHECKING

from src.services.vectorstore.client import QdrantVectorStoreClient, VectorStoreConfig

if TYPE_CHECKING:
    from src.config import VectorStoreSettings

logger = logging.getLogger(__name__)


def get_vectorstore_client(
    embeddings,
    qdrant_url: str | None = None,
    api_key: str | None = None,
    collection_name: str = "default",
    prefer_grpc: bool = True,
    distance: str = "Cosine",
    vector_size: int = 64,
    enable_bm25: bool = False,
) -> QdrantVectorStoreClient:
    """
    Factory helper to create a Qdrant-backed vectorstore client.

    Args:
      embeddings: LangChain-compatible embeddings object (must implement embed_documents/embed_query).
      qdrant_url: URL of Qdrant (e.g. "http://localhost:6333"). If None, default client connection is used.
      api_key: optional API key for hosted Qdrant.
      collection_name: Qdrant collection name to use.
      prefer_grpc: whether to prefer gRPC transport.
      distance: distance metric used when creating the collection (Cosine/Dot/Euclid).
      vector_size: dimension of the embedding vectors.
      enable_bm25: whether to enable BM25 indexing on page_content for hybrid search.
    """
    config = VectorStoreConfig(
        qdrant_url=qdrant_url,
        api_key=api_key,
        prefer_grpc=prefer_grpc,
        collection_name=collection_name,
        distance=distance,
        vector_size=vector_size,
        enable_bm25=enable_bm25,
    )
    logger.info(f"Created QdrantVectorStoreClient with collection={collection_name}")
    return QdrantVectorStoreClient(embeddings=embeddings, config=config)


def make_vectorstore_client(
    settings: "VectorStoreSettings",
    embeddings,
    vector_size: int,
) -> QdrantVectorStoreClient:
    """
    Create vectorstore client from application settings.

    Args:
        settings: Vectorstore settings
        embeddings: LangChain-compatible embeddings object
        vector_size: Dimension of the embedding vectors

    Returns:
        Configured QdrantVectorStoreClient instance
    """
    config = VectorStoreConfig(
        qdrant_url=settings.url,
        api_key=settings.api_key,
        prefer_grpc=settings.prefer_grpc,
        collection_name=settings.collection_name,
        distance="Cosine",
        vector_size=vector_size,
        enable_bm25=settings.enable_bm25,
    )
    logger.info(f"Vectorstore client created with collection={settings.collection_name}")
    return QdrantVectorStoreClient(embeddings=embeddings, config=config)


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
    config = VectorStoreConfig.from_settings(
        settings,
        vector_size=vector_size or 64,  # Default dimension
        **overrides,
    )
    return QdrantVectorStoreClient(embeddings=embeddings, config=config)
