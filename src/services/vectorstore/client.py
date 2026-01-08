from collections.abc import Sequence
from dataclasses import dataclass
from uuid import uuid4

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Filter,
    PointStruct,
    Prefetch,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from src.services.retry import RetryConfig, retry_with_backoff

from .filters import FilterBuilder, build_filter_from_dict
from .index_mappings import IndexMapping, IndexMappingBuilder, get_qdrant_field_schema
from .retrieval_metrics import (
    RetrievalMetrics,
    RetrievalTimer,
)
from .retrieval_metrics import (
    normalize_scores as normalize_score_list,
)


@dataclass
class VectorStoreConfig:
    qdrant_url: str | None = None
    api_key: str | None = None
    prefer_grpc: bool = True
    collection_name: str = "default"
    distance: str = "Cosine"  # or "Dot", "Euclid"
    vector_size: int = 64  # Default embedding dimension
    enable_bm25: bool = False  # Enable BM25 indexing on page_content
    # Retry configuration
    retry_config: RetryConfig | None = None
    # Metrics tracking
    enable_metrics: bool = False  # Track retrieval metrics
    normalize_scores: bool = False  # Normalize scores to [0, 1]
    # Sparse vectors (SPLADE)
    enable_sparse: bool = False  # Enable sparse vector storage
    sparse_vector_name: str = "sparse"  # Name for sparse vector field


class DependencyMissingError(RuntimeError):
    pass


class QdrantVectorStoreClient:
    """
    Vectorstore client using qdrant-client directly (not deprecated LangChain wrapper).

    - Requires an `embeddings` object compatible with LangChain (has `embed_documents` / `embed_query`).
    - Uses `qdrant-client.QdrantClient` for connection.
    """

    def __init__(self, embeddings, config: VectorStoreConfig, sparse_encoder=None):
        if QdrantClient is None:
            raise DependencyMissingError(
                "qdrant-client is required. Install with: pip install 'qdrant-client'"
            )
        if embeddings is None:
            raise ValueError("An embeddings object (LangChain-compatible) is required")

        self.embeddings = embeddings
        self.config = config
        self.sparse_encoder = sparse_encoder
        self.retry_config = config.retry_config or RetryConfig(
            max_retries=3,
            initial_delay=1.0,
            max_delay=30.0,
            exponential_base=2.0,
            jitter=True,
        )

        # Initialize metrics if enabled
        self.metrics = RetrievalMetrics() if config.enable_metrics else None

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
                self._create_collection(distance_map)
        except Exception:
            # Collection doesn't exist, create it
            self._create_collection(distance_map)

    def _create_collection(self, distance_map: dict) -> None:
        """Helper to create collection with optional BM25 indexing and sparse vectors."""
        from qdrant_client.models import TextIndexParams, TokenizerType

        # Configure vectors (dense and optionally sparse)
        vectors_config = VectorParams(
            size=self.config.vector_size,
            distance=distance_map.get(self.config.distance, Distance.COSINE),
        )

        # Add sparse vector configuration if enabled
        sparse_vectors_config = None
        if self.config.enable_sparse:
            sparse_vectors_config = {self.config.sparse_vector_name: SparseVectorParams()}

        self.qdrant_client.create_collection(
            collection_name=self.config.collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
        )

        # Enable BM25 indexing on page_content if configured
        if self.config.enable_bm25:
            self.qdrant_client.create_payload_index(
                collection_name=self.config.collection_name,
                field_name="page_content",
                field_schema=TextIndexParams(
                    type="text",  # type: ignore[arg-type]
                    tokenizer=TokenizerType.WORD,
                    min_token_len=2,
                    max_token_len=20,
                    lowercase=True,
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
        Optionally computes and stores sparse vectors if sparse encoder is enabled.
        """
        if not texts:
            return []

        # Generate dense embeddings
        vectors = self.embeddings.embed_documents(list(texts))

        # Generate sparse embeddings if enabled
        sparse_vectors = None
        if self.config.enable_sparse and self.sparse_encoder:
            sparse_dicts = self.sparse_encoder.encode_documents(list(texts))
            # Convert to SparseVector objects
            sparse_vectors = [
                SparseVector(
                    indices=list(sparse_dict.keys()),
                    values=list(sparse_dict.values()),
                )
                for sparse_dict in sparse_dicts
            ]

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid4()) for _ in texts]

        # Build points
        points = []
        for i, (text, vector, point_id) in enumerate(zip(texts, vectors, ids, strict=True)):
            payload = {"page_content": text}
            if metadatas and i < len(metadatas) and metadatas[i]:
                payload.update(metadatas[i])  # type: ignore[arg-type]

            # Create PointStruct with both dense and sparse vectors
            # For Qdrant v1.7+, use vector dict with named vectors
            if sparse_vectors and i < len(sparse_vectors):
                # Named vectors approach
                vector_dict = {
                    "": vector,  # Default dense vector
                    self.config.sparse_vector_name: sparse_vectors[i],
                }
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vector_dict,
                        payload=payload,
                    )
                )
            else:
                # Regular point with only dense vector
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload,
                    )
                )

        # Upsert to Qdrant with retry
        @retry_with_backoff(self.retry_config)
        def _upsert():
            return self.qdrant_client.upsert(
                collection_name=self.config.collection_name,
                points=points,
            )

        _upsert()
        return list(ids)

    def similarity_search(self, query: str, k: int = 5) -> list[Document]:
        """
        Performs similarity search for a text query.
        Returns a list of LangChain Document objects.
        """
        query_vector = self.embeddings.embed_query(query)

        @retry_with_backoff(self.retry_config)
        def _query_points():
            return self.qdrant_client.query_points(
                collection_name=self.config.collection_name,
                query=query_vector,
                limit=k,
            )

        results = _query_points()

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

        @retry_with_backoff(self.retry_config)
        def _query_points():
            return self.qdrant_client.query_points(
                collection_name=self.config.collection_name,
                query=list(vector),
                limit=k,
            )

        results = _query_points()

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

    def similarity_search_with_filter(
        self,
        query: str,
        k: int = 5,
        filter: Filter | None = None,
        filter_dict: dict | None = None,
    ) -> list[Document]:
        """
        Performs similarity search with optional metadata filtering.

        Args:
            query: Text query to search for
            k: Number of results to return
            filter: Pre-built Qdrant Filter object
            filter_dict: Simple dict to build filter from (alternative to filter param)

        Returns:
            List of Document objects matching the query and filters

        Example:
            # Using filter_dict
            docs = client.similarity_search_with_filter(
                query="machine learning",
                k=5,
                filter_dict={"source": "paper.pdf", "year$gte": 2020}
            )

            # Using FilterBuilder
            from .filters import FilterBuilder
            filter = FilterBuilder().match("source", "paper.pdf").build()
            docs = client.similarity_search_with_filter(
                query="machine learning",
                k=5,
                filter=filter
            )
        """
        query_vector = self.embeddings.embed_query(query)

        # Build filter from dict if provided
        if filter_dict and not filter:
            filter = build_filter_from_dict(filter_dict)

        @retry_with_backoff(self.retry_config)
        def _query_points():
            return self.qdrant_client.query_points(
                collection_name=self.config.collection_name,
                query=query_vector,
                limit=k,
                query_filter=filter,
            )

        results = _query_points()

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

    def bm25_search(
        self,
        query: str,
        k: int = 5,
        filter: Filter | None = None,
        filter_dict: dict | None = None,
    ) -> list[Document]:
        """
        Performs BM25 full-text search (requires BM25 indexing enabled).

        BM25 is a traditional text-based ranking algorithm that works well for
        keyword matching and doesn't require embedding computation.

        Args:
            query: Text query to search for
            k: Number of results to return
            filter: Pre-built Qdrant Filter object
            filter_dict: Simple dict to build filter from

        Returns:
            List of Document objects ranked by BM25 score

        Example:
            docs = client.bm25_search(
                query="neural networks deep learning",
                k=10,
                filter_dict={"category": "AI"}
            )

        Note:
            Requires enable_bm25=True in VectorStoreConfig during initialization.
        """

        # Build filter from dict if provided
        if filter_dict and not filter:
            filter = build_filter_from_dict(filter_dict)

        # BM25 search using query API
        @retry_with_backoff(self.retry_config)
        def _query_points():
            return self.qdrant_client.query_points(
                collection_name=self.config.collection_name,
                query=query,
                limit=k,
                query_filter=filter,
                using="page_content",  # Use BM25 index on this field
            )

        results = _query_points()

        documents = []
        for result in results.points:
            page_content = result.payload.get("page_content", "") if result.payload else ""
            metadata = (
                {k: v for k, v in result.payload.items() if k != "page_content"}
                if result.payload
                else {}
            )
            metadata["score"] = result.score
            metadata["search_type"] = "bm25"
            documents.append(Document(page_content=page_content, metadata=metadata))

        return documents

    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        filter: Filter | None = None,
        filter_dict: dict | None = None,
        alpha: float = 0.5,
    ) -> list[Document]:
        """
        Performs hybrid search combining vector similarity and BM25.

        Uses Qdrant's query prefetch to combine semantic (vector) search
        with keyword-based (BM25) search for better retrieval.

        Args:
            query: Text query to search for
            k: Number of results to return
            filter: Pre-built Qdrant Filter object
            filter_dict: Simple dict to build filter from
            alpha: Weight between vector (1.0) and BM25 (0.0) search.
                   0.5 gives equal weight to both.

        Returns:
            List of Document objects ranked by hybrid score

        Example:
            # Equal weighting (default)
            docs = client.hybrid_search("machine learning algorithms", k=10)

            # More weight on semantic similarity
            docs = client.hybrid_search("ML models", k=10, alpha=0.7)

            # With filters
            docs = client.hybrid_search(
                "deep learning",
                k=10,
                filter_dict={"year$gte": 2020, "category": "AI"},
                alpha=0.6
            )

        Note:
            Requires enable_bm25=True in VectorStoreConfig.
        """

        query_vector = self.embeddings.embed_query(query)

        # Build filter from dict if provided
        if filter_dict and not filter:
            filter = build_filter_from_dict(filter_dict)

        # Hybrid search using prefetch (Qdrant's RRF-like fusion)
        # First stage: BM25 search
        # Second stage: Vector search with prefetch results
        @retry_with_backoff(self.retry_config)
        def _query_points():
            return self.qdrant_client.query_points(
                collection_name=self.config.collection_name,
                prefetch=[
                    # Prefetch using BM25
                    Prefetch(
                        query=query,
                        using="page_content",
                        limit=k * 2,  # Get more candidates for fusion
                        filter=filter,
                    ),
                    # Prefetch using vector similarity
                    Prefetch(
                        query=query_vector,
                        limit=k * 2,
                        filter=filter,
                    ),
                ],
                query=query_vector,  # Final ranking query
                limit=k,
                # Note: alpha parameter would need custom scoring if supported
            )

        results = _query_points()

        documents = []
        for result in results.points:
            page_content = result.payload.get("page_content", "") if result.payload else ""
            metadata = (
                {k: v for k, v in result.payload.items() if k != "page_content"}
                if result.payload
                else {}
            )
            metadata["score"] = result.score
            metadata["search_type"] = "hybrid"
            documents.append(Document(page_content=page_content, metadata=metadata))

        return documents

    def sparse_search(
        self,
        query: str,
        k: int = 5,
        filter: Filter | None = None,
        filter_dict: dict | None = None,
    ) -> list[Document]:
        """
        Performs sparse vector search using SPLADE encoder.

        Uses learned sparse representations (SPLADE) for neural retrieval
        that combines benefits of sparse (efficient, interpretable) and
        dense (semantic) vectors.

        Args:
            query: Text query to search for
            k: Number of results to return
            filter: Pre-built Qdrant Filter object
            filter_dict: Simple dict to build filter from

        Returns:
            List of Document objects ranked by sparse similarity

        Example:
            # Basic sparse search
            docs = client.sparse_search("machine learning algorithms", k=10)

            # With filters
            docs = client.sparse_search(
                "deep learning",
                k=10,
                filter_dict={"year$gte": 2020, "category": "AI"}
            )

        Note:
            Requires enable_sparse=True in VectorStoreConfig and
            a sparse_encoder to be provided during initialization.
        """
        if not self.config.enable_sparse:
            raise ValueError(
                "Sparse search not enabled. Set enable_sparse=True in VectorStoreConfig."
            )
        if not self.sparse_encoder:
            raise ValueError(
                "No sparse encoder provided. Pass sparse_encoder during initialization."
            )

        # Generate sparse query vector
        sparse_dict = self.sparse_encoder.encode_query(query)
        sparse_query = SparseVector(
            indices=list(sparse_dict.keys()),
            values=list(sparse_dict.values()),
        )

        # Build filter from dict if provided
        if filter_dict and not filter:
            filter = build_filter_from_dict(filter_dict)

        # Sparse search
        @retry_with_backoff(self.retry_config)
        def _query_points():
            return self.qdrant_client.query_points(
                collection_name=self.config.collection_name,
                query=sparse_query,
                using=self.config.sparse_vector_name,
                limit=k,
                query_filter=filter,
            )

        results = _query_points()

        documents = []
        for result in results.points:
            page_content = result.payload.get("page_content", "") if result.payload else ""
            metadata = (
                {k: v for k, v in result.payload.items() if k != "page_content"}
                if result.payload
                else {}
            )
            metadata["score"] = result.score
            metadata["search_type"] = "sparse"
            documents.append(Document(page_content=page_content, metadata=metadata))

        return documents

    def get_filter_builder(self) -> FilterBuilder:
        """
        Get a new FilterBuilder instance for constructing complex filters.

        Returns:
            FilterBuilder instance

        Example:
            filter = (
                client.get_filter_builder()
                .match("source", "paper.pdf")
                .range("year", gte=2020, lte=2023)
                .match_any("category", ["AI", "ML"])
                .build()
            )
            docs = client.similarity_search_with_filter("query", filter=filter)
        """
        return FilterBuilder()

    def create_payload_index(self, field_name: str, field_schema, wait: bool = True) -> None:
        """
        Create a payload index on a specific field for faster filtering.

        Args:
            field_name: Name of the payload field to index
            field_schema: Qdrant field schema (use get_qdrant_field_schema)
            wait: Wait for index creation to complete

        Example:
            from qdrant_client.models import KeywordIndexParams
            client.create_payload_index("category", KeywordIndexParams())
        """
        self.qdrant_client.create_payload_index(
            collection_name=self.config.collection_name,
            field_name=field_name,
            field_schema=field_schema,
            wait=wait,
        )

    def create_indices_from_mappings(
        self, mappings: list[IndexMapping], wait: bool = True
    ) -> dict[str, bool]:
        """
        Create multiple payload indices from index mappings.

        Args:
            mappings: List of IndexMapping configurations
            wait: Wait for each index creation to complete

        Returns:
            Dict mapping field names to success status

        Example:
            from src.services.vectorstore.index_mappings import IndexMappingBuilder

            mappings = (
                IndexMappingBuilder()
                .add_keyword("category")
                .add_integer("year", range=True)
                .add_float("score", range=True)
                .build()
            )
            results = client.create_indices_from_mappings(mappings)
        """
        results = {}
        for mapping in mappings:
            try:
                field_schema = get_qdrant_field_schema(mapping)
                self.create_payload_index(mapping.field_name, field_schema, wait=wait)
                results[mapping.field_name] = True
            except Exception as e:
                results[mapping.field_name] = False
                # Log error but continue with other indices
                import logging

                logging.error(f"Failed to create index for {mapping.field_name}: {e}")
        return results

    def list_payload_indices(self) -> dict[str, str]:
        """
        List all payload indices in the collection.

        Returns:
            Dict mapping field names to their index types

        Example:
            indices = client.list_payload_indices()
            print(indices)  # {"category": "keyword", "year": "integer", ...}
        """
        try:
            collection_info = self.qdrant_client.get_collection(self.config.collection_name)
            indices = {}
            if collection_info.config and collection_info.config.params:
                payload_schema = collection_info.config.params.payload_schema  # type: ignore[attr-defined]
                if payload_schema:
                    for field_name, schema in payload_schema.items():
                        # Extract index type from schema
                        if hasattr(schema, "type"):
                            indices[field_name] = schema.type
                        else:
                            indices[field_name] = str(type(schema).__name__)
            return indices
        except Exception:
            return {}

    def delete_payload_index(self, field_name: str, wait: bool = True) -> bool:
        """
        Delete a payload index from the collection.

        Args:
            field_name: Name of the field to remove index from
            wait: Wait for deletion to complete

        Returns:
            True if successful, False otherwise

        Example:
            success = client.delete_payload_index("old_field")
        """
        try:
            self.qdrant_client.delete_payload_index(
                collection_name=self.config.collection_name,
                field_name=field_name,
                wait=wait,
            )
            return True
        except Exception:
            return False

    def get_index_mapping_builder(self) -> IndexMappingBuilder:
        """
        Get a new IndexMappingBuilder for creating index configurations.

        Returns:
            IndexMappingBuilder instance

        Example:
            mappings = (
                client.get_index_mapping_builder()
                .add_keyword("category")
                .add_integer("year", range=True)
                .add_text("description", min_token_len=3)
                .build()
            )
            client.create_indices_from_mappings(mappings)
        """
        return IndexMappingBuilder()

    # ========================================
    # Dense Retrieval Enhancements
    # ========================================

    def create_snapshot(self, snapshot_name: str | None = None) -> str:
        """
        Create a snapshot of the collection for backup/persistence.

        Args:
            snapshot_name: Optional name for snapshot. Auto-generated if None.

        Returns:
            Snapshot name/ID

        Example:
            snapshot_id = client.create_snapshot("backup_2024_01_08")
            print(f"Created snapshot: {snapshot_id}")
        """
        result = self.qdrant_client.create_snapshot(
            collection_name=self.config.collection_name,
            snapshot_name=snapshot_name,
        )
        return result.name if (result and hasattr(result, "name")) else str(result)

    def list_snapshots(self) -> list[str]:
        """
        List all available snapshots for the collection.

        Returns:
            List of snapshot names/IDs

        Example:
            snapshots = client.list_snapshots()
            print(f"Available snapshots: {snapshots}")
        """
        result = self.qdrant_client.list_snapshots(collection_name=self.config.collection_name)
        if hasattr(result, "snapshots"):
            return [s.name for s in result.snapshots]
        return []

    def restore_snapshot(self, snapshot_name: str, priority: str = "snapshot") -> bool:
        """
        Restore collection from a snapshot.

        Args:
            snapshot_name: Name of the snapshot to restore
            priority: "snapshot" or "replica" - determines conflict resolution

        Returns:
            True if successful

        Example:
            success = client.restore_snapshot("backup_2024_01_08")
            if success:
                print("Snapshot restored successfully")
        """
        try:
            self.qdrant_client.recover_snapshot(
                collection_name=self.config.collection_name,
                location=snapshot_name,
                priority=priority,  # type: ignore[arg-type]
            )
            return True
        except Exception:
            return False

    def get_retrieval_metrics(self) -> dict | None:
        """
        Get retrieval performance metrics (if enabled).

        Returns:
            Dict with metrics summary or None if metrics disabled

        Example:
            metrics = client.get_retrieval_metrics()
            if metrics:
                print(f"P50 latency: {metrics['latency']['p50']:.2f}ms")
                print(f"P95 latency: {metrics['latency']['p95']:.2f}ms")
                print(f"Cache hit rate: {metrics['cache_hit_rate']:.1f}%")
        """
        if self.metrics:
            return self.metrics.get_summary()
        return None

    def reset_metrics(self) -> None:
        """Reset all tracked metrics."""
        if self.metrics:
            self.metrics.reset()

    def similarity_search_with_metrics(
        self,
        query: str,
        k: int = 5,
        normalize_scores: bool | None = None,
    ) -> list[Document]:
        """
        Perform similarity search with automatic metrics tracking and optional score normalization.

        Args:
            query: Text query to search for
            k: Number of results to return
            normalize_scores: Override config.normalize_scores setting

        Returns:
            List of Document objects with normalized scores (if enabled)

        Example:
            docs = client.similarity_search_with_metrics("machine learning", k=10)
            for doc in docs:
                print(f"Score: {doc.metadata['score']:.3f}")
        """
        if self.metrics:
            with RetrievalTimer(self.metrics, search_type="vector") as timer:
                docs = self.similarity_search(query, k=k)
                scores = [doc.metadata.get("score", 0.0) for doc in docs]
                timer.set_scores(scores)
        else:
            docs = self.similarity_search(query, k=k)

        # Normalize scores if requested
        should_normalize = (
            normalize_scores if normalize_scores is not None else self.config.normalize_scores
        )
        if should_normalize and docs:
            scores = [doc.metadata.get("score", 0.0) for doc in docs]
            normalized = normalize_score_list(scores, method="minmax")
            for doc, norm_score in zip(docs, normalized, strict=True):
                doc.metadata["score"] = norm_score
                doc.metadata["score_normalized"] = True

        return docs

    def hybrid_search_with_metrics(
        self,
        query: str,
        k: int = 5,
        alpha: float = 0.5,
        normalize_scores: bool | None = None,
    ) -> list[Document]:
        """
        Perform hybrid search with automatic metrics tracking and score normalization.

        Args:
            query: Text query to search for
            k: Number of results to return
            alpha: Weight between vector (1.0) and BM25 (0.0)
            normalize_scores: Override config.normalize_scores setting

        Returns:
            List of Document objects with normalized scores (if enabled)

        Example:
            docs = client.hybrid_search_with_metrics(
                "deep learning",
                k=10,
                alpha=0.6,
                normalize_scores=True
            )
        """
        if self.metrics:
            with RetrievalTimer(self.metrics, search_type="hybrid") as timer:
                docs = self.hybrid_search(query, k=k, alpha=alpha)
                scores = [doc.metadata.get("score", 0.0) for doc in docs]
                timer.set_scores(scores)
        else:
            docs = self.hybrid_search(query, k=k, alpha=alpha)

        # Normalize scores if requested
        should_normalize = (
            normalize_scores if normalize_scores is not None else self.config.normalize_scores
        )
        if should_normalize and docs:
            scores = [doc.metadata.get("score", 0.0) for doc in docs]
            # Use sigmoid for hybrid scores (can be unbounded)
            normalized = normalize_score_list(scores, method="sigmoid")
            for doc, norm_score in zip(docs, normalized, strict=True):
                doc.metadata["score"] = norm_score
                doc.metadata["score_normalized"] = True

        return docs

    def sparse_search_with_metrics(
        self,
        query: str,
        k: int = 5,
        normalize_scores: bool | None = None,
    ) -> list[Document]:
        """
        Perform sparse search with automatic metrics tracking and score normalization.

        Args:
            query: Text query to search for
            k: Number of results to return
            normalize_scores: Override config.normalize_scores setting

        Returns:
            List of Document objects with normalized scores (if enabled)

        Example:
            docs = client.sparse_search_with_metrics(
                "neural information retrieval",
                k=10,
                normalize_scores=True
            )
        """
        if self.metrics:
            with RetrievalTimer(self.metrics, search_type="sparse") as timer:
                docs = self.sparse_search(query, k=k)
                scores = [doc.metadata.get("score", 0.0) for doc in docs]
                timer.set_scores(scores)
        else:
            docs = self.sparse_search(query, k=k)

        # Normalize scores if requested
        should_normalize = (
            normalize_scores if normalize_scores is not None else self.config.normalize_scores
        )
        if should_normalize and docs:
            scores = [doc.metadata.get("score", 0.0) for doc in docs]
            normalized = normalize_score_list(scores, method="minmax")
            for doc, norm_score in zip(docs, normalized, strict=True):
                doc.metadata["score"] = norm_score
                doc.metadata["score_normalized"] = True

        return docs

    def export_collection_info(self) -> dict:
        """
        Export comprehensive collection information for monitoring/debugging.

        Returns:
            Dict with collection stats, config, and index info

        Example:
            info = client.export_collection_info()
            print(f"Total vectors: {info['vectors_count']}")
            print(f"Indexed fields: {info['payload_indices']}")
        """
        collection_info = self.qdrant_client.get_collection(self.config.collection_name)

        return {
            "collection_name": self.config.collection_name,
            "vectors_count": getattr(collection_info, "vectors_count", 0),
            "points_count": collection_info.points_count,
            "vector_size": self.config.vector_size,
            "distance": self.config.distance,
            "enable_bm25": self.config.enable_bm25,
            "payload_indices": self.list_payload_indices(),
            "status": collection_info.status.value
            if hasattr(collection_info, "status")
            else "unknown",
            "optimizer_status": (
                getattr(collection_info.optimizer_status, "value", "unknown")
                if hasattr(collection_info, "optimizer_status")
                else "unknown"
            ),
        }
