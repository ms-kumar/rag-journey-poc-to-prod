from typing import List, Sequence


class LangChainEmbeddingsAdapter:
    """
    Adapter that exposes the minimal LangChain Embeddings API expected by
    `langchain.vectorstores.Qdrant`:

      - embed_documents(texts: Sequence[str]) -> List[List[float]]
      - embed_query(text: str) -> List[float]

    It wraps an existing embedding client that implements one of:
      - `embed(texts: Sequence[str]) -> List[List[float]]` (your current EmbedClient)
      - `embed_documents(...)` / `embed_query(...)` (already LangChain-compatible)

    The adapter performs simple detection and delegates accordingly.
    """

    def __init__(self, embed_client, batch_size: int = 32):
        self._client = embed_client
        self.batch_size = int(batch_size)

    def _call_embed(self, texts: Sequence[str]) -> List[List[float]]:
        # Preferred names in order: embed_documents, embed, embed_texts
        if hasattr(self._client, "embed_documents"):
            return self._client.embed_documents(texts)
        if hasattr(self._client, "embed"):
            return self._client.embed(list(texts))
        if hasattr(self._client, "embed_texts"):
            return self._client.embed_texts(list(texts))
        raise AttributeError(
            "Wrapped embed client must implement one of: "
            "'embed_documents', 'embed', or 'embed_texts'"
        )

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        """
        LangChain expects this method name for embedding a batch of documents.
        """
        return self._call_embed(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        LangChain expects this method for embedding a single query. We reuse the
        batch endpoint and return the first vector.
        """
        vecs = self._call_embed([text])
        return vecs[0] if vecs else []

    # Optional: convenience alias for LangChain's duck-typing
    def __call__(self, texts: Sequence[str]) -> List[List[float]]:
        return self.embed_documents(texts)
