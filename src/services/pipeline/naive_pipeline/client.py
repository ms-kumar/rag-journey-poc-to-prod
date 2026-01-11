from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal
from uuid import uuid4

from src.services.chunking.factory import get_chunking_client
from src.services.embeddings.factory import (
    get_embed_client,
    get_langchain_embeddings_adapter,
)
from src.services.generation.factory import get_generator
from src.services.ingestion.factory import get_ingestion_client
from src.services.reranker.factory import get_reranker
from src.services.vectorstore.factory import get_vectorstore_client


@dataclass
class NaivePipelineConfig:
    """Configuration for the NaivePipeline."""

    ingestion_dir: str = "./data"
    chunk_size: int = 512
    embed_dim: int = 64
    qdrant_url: str | None = None
    collection_name: str = "naive_collection"
    generator_model: str = "gpt2"
    generator_device: int | None = None
    # Re-ranker configuration
    enable_reranker: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_batch_size: int = 32
    reranker_timeout: float = 30.0
    reranker_top_k: int | None = None  # None = rerank all candidates


class NaivePipeline:
    """
    Simple RAG orchestrator that:
      1. Ingests documents (local .txt files)
      2. Chunks them into smaller pieces
      3. Computes embeddings for each chunk
      4. Stores vectors in a vectorstore (Qdrant)
      5. Performs retrieval + generation (RAG)
    """

    def __init__(self, config: NaivePipelineConfig | None = None):
        self.config = config or NaivePipelineConfig()
        self._setup_components()

    def _setup_components(self) -> None:
        """Initialize all pipeline components."""
        # Ingestion client
        self.ingest_client = get_ingestion_client(directory=self.config.ingestion_dir)

        # Chunker
        self.chunker = get_chunking_client(chunk_size=self.config.chunk_size)

        # Embeddings (wrapped for LangChain compatibility)
        self.embed_client = get_embed_client(dim=self.config.embed_dim)
        self.lc_embeddings = get_langchain_embeddings_adapter(self.embed_client)

        # Vectorstore (Qdrant via LangChain wrapper)
        from src.config import VectorStoreSettings

        vectorstore_settings = VectorStoreSettings(
            url=self.config.qdrant_url,
            collection_name=self.config.collection_name,
            vector_size=self.config.embed_dim,
        )
        self.vectorstore = get_vectorstore_client(
            embeddings=self.lc_embeddings,
            settings=vectorstore_settings,
        )

        # Generator (HuggingFace pipeline or fallback)
        self.generator = get_generator(
            model_name=self.config.generator_model,
            device=self.config.generator_device,
        )

        # Re-ranker (optional)
        self.reranker = None
        if self.config.enable_reranker:
            # Convert int device to string if needed
            device = (
                str(self.config.generator_device)
                if self.config.generator_device is not None
                else None
            )
            self.reranker = get_reranker(
                model_name=self.config.reranker_model,
                batch_size=self.config.reranker_batch_size,
                timeout_seconds=self.config.reranker_timeout,
                device=device,
            )

    def ingest_and_index(self) -> int:
        """
        Ingest documents from the configured directory, chunk, embed, and store them.

        Returns:
            Number of chunks indexed.
        """
        docs = self.ingest_client.ingest()
        total_chunks = 0

        for doc_idx, doc in enumerate(docs):
            chunks = self.chunker.chunk(doc)
            if not chunks:
                continue

            # Generate IDs and metadata for each chunk
            ids = [str(uuid4()) for _ in chunks]
            metadatas = [
                {
                    "source": f"doc-{doc_idx}",
                    "chunk_index": j,
                    "chunk_id": f"doc-{doc_idx}-chunk-{j}",
                }
                for j in range(len(chunks))
            ]

            # Add to vectorstore (embeddings computed internally)
            self.vectorstore.add_texts(texts=chunks, metadatas=metadatas, ids=ids)
            total_chunks += len(chunks)

        return total_chunks

    def retrieve(
        self,
        query: str,
        k: int = 5,
        search_type: Literal["vector", "bm25", "hybrid", "sparse"] = "vector",
        filters: dict | None = None,
        hybrid_alpha: float = 0.5,
        enable_reranking: bool | None = None,
    ) -> list:
        """
        Retrieve top-k similar documents for the given query.

        Args:
            query: The search query.
            k: Number of results to return.
            search_type: Type of search ("vector", "bm25", "hybrid", or "sparse")
            filters: Optional metadata filters (dict format)
            hybrid_alpha: Weight for hybrid search (0.0=BM25, 1.0=vector)
            enable_reranking: Override global reranking setting for this query

        Returns:
            List of retrieved documents (LangChain Document objects or similar).
        """
        # Determine if we should use reranking
        use_reranking = (
            enable_reranking
            if enable_reranking is not None
            else (self.config.enable_reranker and self.reranker is not None)
        )

        # If reranking is enabled, retrieve more candidates than requested
        retrieval_k = k
        if use_reranking:
            # Get more candidates for better reranking (typically 2-3x more)
            retrieval_k = min(k * 3, 100)  # Cap at 100 to avoid excessive retrieval

        # Perform initial retrieval
        if search_type == "bm25":
            documents = self.vectorstore.bm25_search(query, k=retrieval_k, filter_dict=filters)
        elif search_type == "hybrid":
            documents = self.vectorstore.hybrid_search(
                query, k=retrieval_k, filter_dict=filters, alpha=hybrid_alpha
            )
        elif search_type == "sparse":
            documents = self.vectorstore.sparse_search(query, k=retrieval_k, filter_dict=filters)
        else:  # vector (default)
            if filters:
                documents = self.vectorstore.similarity_search_with_filter(
                    query, k=retrieval_k, filter_dict=filters
                )
            else:
                documents = self.vectorstore.similarity_search(query, k=retrieval_k)

        # Apply reranking if enabled
        if use_reranking and self.reranker and documents:
            rerank_result = self.reranker.rerank(
                query=query, documents=documents, top_k=self.config.reranker_top_k or k
            )

            # Add reranking metadata to documents
            for i, doc in enumerate(rerank_result.documents):
                if doc.metadata is None:
                    doc.metadata = {}
                doc.metadata["reranked"] = True
                doc.metadata["original_rank"] = rerank_result.original_ranks[i]
                doc.metadata["rerank_score"] = (
                    rerank_result.scores[i] if rerank_result.scores else None
                )
                doc.metadata["rerank_model"] = rerank_result.model_used
                doc.metadata["rerank_fallback"] = rerank_result.fallback_used

            return rerank_result.documents[:k]  # Return exactly k documents

        return documents[:k]  # Return exactly k documents without reranking

    def generate(self, prompt: str, retrieved_docs: Sequence | None = None) -> str:
        """
        Generate a response using the prompt and optionally retrieved context.

        Args:
            prompt: The user's prompt.
            retrieved_docs: Optional list of retrieved documents to use as context.

        Returns:
            Generated text response.
        """
        context = self._build_context(retrieved_docs)
        rag_prompt = self._build_rag_prompt(prompt, context)
        return self.generator.generate(rag_prompt)  # type: ignore[no-any-return]

    def query(self, prompt: str, top_k: int = 5) -> str:
        """
        End-to-end RAG: retrieve relevant docs and generate a response.

        Args:
            prompt: The user's query.
            top_k: Number of documents to retrieve.

        Returns:
            Generated response with retrieved context.
        """
        retrieved = self.retrieve(prompt, k=top_k)
        return self.generate(prompt, retrieved_docs=retrieved)

    def _build_context(self, docs: Sequence | None) -> str:
        """Extract text content from retrieved documents."""
        if not docs:
            return ""

        content_list = []
        for doc in docs:
            if hasattr(doc, "page_content"):
                content_list.append(doc.page_content)
            elif isinstance(doc, str):
                content_list.append(doc)
            elif isinstance(doc, dict) and "text" in doc:
                content_list.append(doc["text"])

        return "\n\n".join(content_list[:3])

    def _build_rag_prompt(self, prompt: str, context: str) -> str:
        """Construct the RAG prompt with context."""
        if not context:
            return prompt

        return f"""Use the following context to answer the question.

Context:
{context}

Question: {prompt}

Answer:"""
