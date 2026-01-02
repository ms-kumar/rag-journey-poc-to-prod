from .client import QdrantVectorStoreClient, VectorStoreConfig
from typing import Optional


def get_vectorstore_client(
    embeddings,
    qdrant_url: Optional[str] = None,
    api_key: Optional[str] = None,
    collection_name: str = "default",
    prefer_grpc: bool = True,
    distance: str = "Cosine",
    vector_size: int = 64,
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
    """
    config = VectorStoreConfig(
        qdrant_url=qdrant_url,
        api_key=api_key,
        prefer_grpc=prefer_grpc,
        collection_name=collection_name,
        distance=distance,
        vector_size=vector_size,
    )
    return QdrantVectorStoreClient(embeddings=embeddings, config=config)
