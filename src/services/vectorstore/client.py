from typing import Iterable, List, Optional, Sequence
from dataclasses import dataclass

try:
    from langchain.vectorstores import Qdrant
    from qdrant_client import QdrantClient
except Exception:  # pragma: no cover - dependency check
    Qdrant = None  # type: ignore
    QdrantClient = None  # type: ignore


@dataclass
class VectorStoreConfig:
    qdrant_url: Optional[str] = None
    api_key: Optional[str] = None
    prefer_grpc: bool = True
    collection_name: str = "default"
    distance: str = "Cosine"  # or "Dot", "Euclid"


class DependencyMissingError(RuntimeError):
    pass


class QdrantVectorStoreClient:
    """
    Thin wrapper around LangChain's Qdrant vectorstore.

    - Requires an `embeddings` object compatible with LangChain (has `embed_documents` / `embed_query`).
    - Uses `qdrant-client.QdrantClient` for connection.
    """

    def __init__(self, embeddings, config: VectorStoreConfig):
        if Qdrant is None or QdrantClient is None:
            raise DependencyMissingError(
                "langchain and qdrant-client are required. "
                "Install with: pip install 'langchain' 'qdrant-client'"
            )
        if embeddings is None:
            raise ValueError("An embeddings object (LangChain-compatible) is required")

        self.embeddings = embeddings
        self.config = config

        # Create Qdrant client. If no URL provided, use default local connection.
        client_kwargs = {}
        if config.qdrant_url:
            client_kwargs["url"] = config.qdrant_url
        if config.api_key:
            client_kwargs["api_key"] = config.api_key
        client_kwargs["prefer_grpc"] = config.prefer_grpc

        self.qdrant_client = QdrantClient(**client_kwargs)
        # Create / connect the LangChain Qdrant vectorstore instance
        # Passing the embeddings object so add_texts/from_texts can compute embeddings.
        self.vs = Qdrant(
            client=self.qdrant_client,
            collection_name=config.collection_name,
            embeddings=self.embeddings,
            prefer_grpc=config.prefer_grpc,
        )

    def add_texts(
        self,
        texts: Sequence[str],
        metadatas: Optional[Sequence[Optional[dict]]] = None,
        ids: Optional[Sequence[str]] = None,
    ):
        """
        Adds texts to the Qdrant collection. Uses the embeddings object provided at construction.
        """
        return self.vs.add_texts(texts=texts, metadatas=metadatas, ids=ids)

    def similarity_search(self, query: str, k: int = 5):
        """
        Performs similarity search for a text query (embeds query using embeddings.embed_query).
        Returns a list of LangChain Document objects (or whatever the Qdrant store returns).
        """
        return self.vs.similarity_search(query, k=k)

    def similarity_search_by_vector(self, vector: Sequence[float], k: int = 5):
        """
        Performs similarity search using a raw vector.
        """
        return self.vs.similarity_search_by_vector(vector, k=k)

    def get_vector(self, id: str) -> Optional[List[float]]:
        """
        Retrieve the stored vector for a point id using the underlying qdrant client.

        Note: qdrant-client API differs across versions; we attempt a couple of common calls.
        Returns None if the point/vector is not found.
        """
        try:
            # qdrant-client v1.x: get_point(collection_name, id)
            resp = self.qdrant_client.get_point(collection_name=self.config.collection_name, id=id)
            # resp may be a PointStruct; attempt to read vector attribute
            if hasattr(resp, "vector"):
                return list(resp.vector) if resp.vector is not None else None
            # If resp is a dict-like
            if isinstance(resp, dict) and "vector" in resp:
                return resp["vector"]
        except Exception:
            pass

        try:
            # newer qdrant-client sometimes exposes retrieve or points API
            resp = self.qdrant_client.retrieve(collection_name=self.config.collection_name, point_id=id)
            if isinstance(resp, dict) and "vector" in resp:
                return resp["vector"]
            if hasattr(resp, "vector"):
                return list(resp.vector) if resp.vector is not None else None
        except Exception:
            pass

        return None
