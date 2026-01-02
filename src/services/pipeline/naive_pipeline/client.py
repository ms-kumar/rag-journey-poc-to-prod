from typing import List, Optional, Sequence
from dataclasses import dataclass

from src.services.ingestion.factory import get_ingestion_client
from src.services.chunking.factory import get_chunking_client
from src.services.embeddings.factory import get_embed_client, get_langchain_embeddings_adapter
from src.services.vectorstore.factory import get_vectorstore_client
from src.services.generation.factory import get_generator


@dataclass
class NaivePipelineConfig:
    """Configuration for the NaivePipeline."""

    ingestion_dir: str = "./data"
    chunk_size: int = 200
    embed_dim: int = 64
    qdrant_url: Optional[str] = None
    collection_name: str = "naive_collection"
    generator_model: str = "gpt2"
    generator_device: Optional[int] = None


class NaivePipeline:
    """
    Simple RAG orchestrator that:
      1. Ingests documents (local .txt files)
      2. Chunks them into smaller pieces
      3. Computes embeddings for each chunk
      4. Stores vectors in a vectorstore (Qdrant)
      5. Performs retrieval + generation (RAG)
    """

    def __init__(self, config: Optional[NaivePipelineConfig] = None):
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
        self.vectorstore = get_vectorstore_client(
            embeddings=self.lc_embeddings,
            qdrant_url=self.config.qdrant_url,
            collection_name=self.config.collection_name,
        )

        # Generator (HuggingFace pipeline or fallback)
        self.generator = get_generator(
            model_name=self.config.generator_model,
            device=self.config.generator_device,
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
            ids = [f"doc-{doc_idx}-chunk-{j}" for j in range(len(chunks))]
            metadatas = [{"source": f"doc-{doc_idx}", "chunk_index": j} for j in range(len(chunks))]

            # Add to vectorstore (embeddings computed internally)
            self.vectorstore.add_texts(texts=chunks, metadatas=metadatas, ids=ids)
            total_chunks += len(chunks)

        return total_chunks

    def retrieve(self, query: str, k: int = 5) -> List:
        """
        Retrieve top-k similar documents for the given query.

        Args:
            query: The search query.
            k: Number of results to return.

        Returns:
            List of retrieved documents (LangChain Document objects or similar).
        """
        return self.vectorstore.similarity_search(query, k=k)

    def generate(self, prompt: str, retrieved_docs: Optional[Sequence] = None) -> str:
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
        return self.generator.generate(rag_prompt)

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

    def _build_context(self, docs: Optional[Sequence]) -> str:
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
