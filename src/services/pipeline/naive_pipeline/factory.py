from typing import List, Sequence, Optional, Dict, Any
from dataclasses import dataclass

# Try to import project services; if missing use minimal fallbacks.
try:
    from src.services.ingestion.factory import get_ingestion_client
except Exception:
    get_ingestion_client = None

try:
    from src.services.chunking.factory import get_chunker
except Exception:
    get_chunker = None

try:
    from src.services.embeddings.factory import get_embed_client, get_langchain_embeddings_adapter
except Exception:
    get_embed_client = None
    get_langchain_embeddings_adapter = None

try:
    from src.services.vectorstore.client import get_vectorstore_client, VectorStoreConfig
except Exception:
    get_vectorstore_client = None
    VectorStoreConfig = None

try:
    from src.services.generation.factory import get_generator
except Exception:
    from src.services.generation.client import get_generator  # fallback if you placed it there


@dataclass
class NaivePipelineConfig:
    ingestion_dir: str = "./data"
    chunk_size: int = 200
    embed_dim: int = 64
    qdrant_url: Optional[str] = None
    collection_name: str = "naive_collection"
    generator_model: str = "gpt2"
    generator_device: Optional[int] = None

class NaivePipeline:
    """
    Simple orchestrator that:
      1. ingests documents (local .txt)
      2. chunks them
      3. obtains embeddings
      4. stores vectors in a vectorstore
      5. can perform retrieval + generation (retrieval-augmented generation)
    """

    def __init__(self, config: Optional[NaivePipelineConfig] = None):
        self.config = config or NaivePipelineConfig()

        # ingestion
        self.ingest_client = get_ingestion_client(directory=self.config.ingestion_dir)

        # chunker
        self.chunker = get_chunker(chunk_size=self.config.chunk_size)

        # embeddings
        if get_embed_client:
            self.embed_client = get_embed_client(dim=self.config.embed_dim)
            # if LangChain adapter factory exists, wrap to be compatible with LC
            if get_langchain_embeddings_adapter:
                self.lc_embeddings = get_langchain_embeddings_adapter(self.embed_client)
            else:
                self.lc_embeddings = self.embed_client
        else:
            # fallback: very small deterministic embedder
            from src.services.embeddings.factory import EmbedClient as _EmbedClient  # may raise
            self.embed_client = _EmbedClient()
            self.lc_embeddings = self.embed_client

        # vectorstore (Qdrant via LangChain wrapper)
        cfg = VectorStoreConfig(
            qdrant_url=self.config.qdrant_url,
            collection_name=self.config.collection_name
        )
        self.vs = get_vectorstore_client(embeddings=self.lc_embeddings, qdrant_url=self.config.qdrant_url, collection_name=self.config.collection_name)
        # generator
        self.generator = get_generator(model_name=self.config.generator_model, device=self.config.generator_device)

    def ingest_and_index(self) -> int:
        docs = self.ingest_client.ingest()
        count = 0
        for i, doc in enumerate(docs):
            chunks = self.chunker.chunk(doc)
            # embed chunk batch
            if hasattr(self.lc_embeddings, "embed_documents"):
                vectors = self.lc_embeddings.embed_documents(chunks)
            else:
                vectors = self.lc_embeddings.embed(chunks)  # type: ignore
            ids = [f"doc-{i}-chunk-{j}" for j in range(len(chunks))]
            metas = [{"source": f"doc-{i}"} for _ in chunks]
            try:
                self.vs.add_texts(texts=chunks, metadatas=metas, ids=ids)
            except Exception:
                # try lower-level add
                for _id, vec, chunk, meta in zip(ids, vectors, chunks, metas):
                    try:
                        self.vs.add_vector(_id, vec)
                    except Exception:
                        pass
            count += len(chunks)
        return count

    def retrieve(self, query: str, k: int = 5):
        # Use vectorstore similarity search by query text (LangChain Qdrant supports this)
        if hasattr(self.vs, "similarity_search"):
            return self.vs.similarity_search(query, k=k)
        # fallback: no retrieval
        return []

    def generate(self, prompt: str, retrieved_docs: Optional[Sequence[str]] = None) -> str:
        # Simple RAG prompt: include top retrieved passages
        context = ""
        if retrieved_docs:
            # retrieved_docs may be LangChain Document objects with .page_content or .metadata
            content_list = []
            for r in retrieved_docs:
                if hasattr(r, "page_content"):
                    content_list.append(r.page_content)
                elif isinstance(r, str):
                    content_list.append(r)
                elif isinstance(r, dict) and "text" in r:
                    content_list.append(r["text"])
            context = "\n\n".join(content_list[:3])
        rag_prompt = prompt + "\n\nContext:\n" + context if context else prompt
        return self.generator.generate(rag_prompt)