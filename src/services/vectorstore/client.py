from collections.abc import Sequence
from dataclasses import dataclass
from uuid import uuid4

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


@dataclass
class VectorStoreConfig:
    qdrant_url: str | None = None
    api_key: str | None = None
    prefer_grpc: bool = True
    collection_name: str = "default"
    distance: str = "Cosine"  # or "Dot", "Euclid"
    vector_size: int = 64  # Default embedding dimension


class DependencyMissingError(RuntimeError):
    pass


class QdrantVectorStoreClient:
    """
    Vectorstore client using qdrant-client directly (not deprecated LangChain wrapper).

    - Requires an `embeddings` object compatible with LangChain (has `embed_documents` / `embed_query`).
    - Uses `qdrant-client.QdrantClient` for connection.
    """

    def __init__(self, embeddings, config: VectorStoreConfig):
        if QdrantClient is None:
            raise DependencyMissingError(
                "qdrant-client is required. Install with: pip install 'qdrant-client'"
            )
        if embeddings is None:
            raise ValueError("An embeddings object (LangChain-compatible) is required")

        self.embeddings = embeddings
        self.config = config

        # Create Qdrant client
        client_kwargs = {}
        if config.qdrant_url:
            client_kwargs["url"] = config.qdrant_url
        if config.api_key:
            client_kwargs["api_key"] = config.api_key
        client_kwargs["prefer_grpc"] = str(config.prefer_grpc)

        self.qdrant_client = QdrantClient(**client_kwargs)  # type: ignore[arg-type]

        # Ensure collection exists
        self._ensure_collection_exists()

    def _ensure_collection_exists(self) -> None:
        """Create collection if it doesn't exist, or recreate if dimensions mismatch."""
        distance_map = {
            "Cosine": Distance.COSINE,
            "Dot": Distance.DOT,
            "Euclid": Distance.EUCLID,
        }

        try:
            collection_info = self.qdrant_client.get_collection(self.config.collection_name)
            # Check if vector size matches
            existing_size = collection_info.config.params.vectors.size  # type: ignore[union-attr]
            if existing_size != self.config.vector_size:
                # Dimensions mismatch - delete and recreate
                self.qdrant_client.delete_collection(self.config.collection_name)
                self.qdrant_client.create_collection(
                    collection_name=self.config.collection_name,
                    vectors_config=VectorParams(
                        size=self.config.vector_size,
                        distance=distance_map.get(self.config.distance, Distance.COSINE),
                    ),
                )
        except Exception:
            # Collection doesn't exist, create it
            self.qdrant_client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(
                    size=self.config.vector_size,
                    distance=distance_map.get(self.config.distance, Distance.COSINE),
                ),
            )

    def add_texts(
        self,
        texts: Sequence[str],
        metadatas: Sequence[dict | None] | None = None,
        ids: Sequence[str] | None = None,
    ) -> list[str]:
        """
        Adds texts to the Qdrant collection. Computes embeddings and stores them.
        """
        if not texts:
            return []

        # Generate embeddings
        vectors = self.embeddings.embed_documents(list(texts))

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid4()) for _ in texts]

        # Build points
        points = []
        for i, (text, vector, point_id) in enumerate(zip(texts, vectors, ids, strict=True)):
            payload = {"page_content": text}
            if metadatas and i < len(metadatas) and metadatas[i]:
                payload.update(metadatas[i])  # type: ignore[arg-type]
            points.append(PointStruct(id=point_id, vector=vector, payload=payload))

        # Upsert to Qdrant
        self.qdrant_client.upsert(
            collection_name=self.config.collection_name,
            points=points,
        )

        return list(ids)

    def similarity_search(self, query: str, k: int = 5) -> list[Document]:
        """
        Performs similarity search for a text query.
        Returns a list of LangChain Document objects.
        """
        query_vector = self.embeddings.embed_query(query)

        results = self.qdrant_client.query_points(
            collection_name=self.config.collection_name,
            query=query_vector,
            limit=k,
        )

        documents = []
        for result in results.points:
            page_content = result.payload.get("page_content", "") if result.payload else ""
            metadata = (
                {k: v for k, v in result.payload.items() if k != "page_content"}
                if result.payload
                else {}
            )
            metadata["score"] = result.score
            documents.append(Document(page_content=page_content, metadata=metadata))

        return documents

    def similarity_search_by_vector(self, vector: Sequence[float], k: int = 5) -> list[Document]:
        """
        Performs similarity search using a raw vector.
        """
        results = self.qdrant_client.query_points(
            collection_name=self.config.collection_name,
            query=list(vector),
            limit=k,
        )

        documents = []
        for result in results.points:
            page_content = result.payload.get("page_content", "") if result.payload else ""
            metadata = (
                {k: v for k, v in result.payload.items() if k != "page_content"}
                if result.payload
                else {}
            )
            metadata["score"] = result.score
            documents.append(Document(page_content=page_content, metadata=metadata))

        return documents

    def get_vector(self, id: str) -> list[float] | None:
        """
        Retrieve the stored vector for a point id.
        Returns None if the point/vector is not found.
        """
        try:
            results = self.qdrant_client.retrieve(
                collection_name=self.config.collection_name,
                ids=[id],
                with_vectors=True,
            )
            if results and len(results) > 0:
                return list(results[0].vector) if results[0].vector else None  # type: ignore[arg-type]
        except Exception:  # nosec B110
            pass
        return None
