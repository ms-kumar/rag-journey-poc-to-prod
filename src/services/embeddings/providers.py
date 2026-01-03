"""
Provider-specific embedding adapters for E5, BGE, OpenAI, and other services.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence


class BaseEmbeddingProvider(ABC):
    """Base class for embedding providers."""

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embeddings for a sequence of texts."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass


class HuggingFaceEmbeddings(BaseEmbeddingProvider):
    """
    Embedding provider using HuggingFace sentence-transformers.
    Supports E5, BGE, and other transformer models.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
        normalize: bool = True,
        batch_size: int = 32,
    ):
        """
        Initialize HuggingFace embeddings.

        Args:
            model_name: Model identifier (e.g., 'intfloat/e5-base-v2', 'BAAI/bge-base-en-v1.5')
            device: Device to use ('cpu', 'cuda', or None for auto-detect)
            normalize: Whether to L2-normalize embeddings
            batch_size: Batch size for encoding
        """
        self.model_name = model_name
        self.normalize = normalize
        self.batch_size = batch_size
        self._model = None
        self._dim = None

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name, device=device)
            self._dim = self._model.get_sentence_embedding_dimension()

        except ImportError:
            raise ImportError(
                "sentence-transformers is required for HuggingFace embeddings. "
                "Install with: pip install sentence-transformers"
            )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embeddings using sentence-transformers."""
        if not texts:
            return []

        # Convert to list if needed
        texts_list = list(texts)

        # Encode texts
        embeddings = self._model.encode(  # type: ignore[union-attr]
            texts_list,
            normalize_embeddings=self.normalize,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # Convert numpy arrays to lists
        return [emb.tolist() for emb in embeddings]

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self._dim  # type: ignore[return-value]


class E5Embeddings(HuggingFaceEmbeddings):
    """
    E5 embeddings from Microsoft.

    Available models:
    - intfloat/e5-small-v2 (384 dim)
    - intfloat/e5-base-v2 (768 dim)
    - intfloat/e5-large-v2 (1024 dim)
    - intfloat/multilingual-e5-small (384 dim)
    - intfloat/multilingual-e5-base (768 dim)
    - intfloat/multilingual-e5-large (1024 dim)
    """

    def __init__(
        self,
        model_name: str = "intfloat/e5-base-v2",
        device: str | None = None,
        prefix_query: bool = True,
        prefix_document: bool = True,
    ):
        """
        Initialize E5 embeddings.

        Args:
            model_name: E5 model name
            device: Device to use
            prefix_query: Whether to add 'query: ' prefix to queries
            prefix_document: Whether to add 'passage: ' prefix to documents
        """
        super().__init__(model_name=model_name, device=device, normalize=True)
        self.prefix_query = prefix_query
        self.prefix_document = prefix_document

    def embed(self, texts: Sequence[str], is_query: bool = False) -> list[list[float]]:
        """
        Embed texts with optional prefixes for E5 models.

        E5 models perform better with prefixes:
        - Queries: 'query: <text>'
        - Documents: 'passage: <text>'
        """
        if not texts:
            return []

        # Add prefixes if enabled
        if is_query and self.prefix_query:
            texts = [f"query: {text}" for text in texts]
        elif not is_query and self.prefix_document:
            texts = [f"passage: {text}" for text in texts]

        return super().embed(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query with query prefix."""
        return self.embed([text], is_query=True)[0]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed documents with passage prefix."""
        return self.embed(texts, is_query=False)


class BGEEmbeddings(HuggingFaceEmbeddings):
    """
    BGE (BAAI General Embedding) embeddings.

    Available models:
    - BAAI/bge-small-en-v1.5 (384 dim)
    - BAAI/bge-base-en-v1.5 (768 dim)
    - BAAI/bge-large-en-v1.5 (1024 dim)
    - BAAI/bge-m3 (1024 dim, multilingual)
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        device: str | None = None,
        query_instruction: str | None = None,
    ):
        """
        Initialize BGE embeddings.

        Args:
            model_name: BGE model name
            device: Device to use
            query_instruction: Instruction to prepend to queries
                             (e.g., "Represent this sentence for searching relevant passages:")
        """
        super().__init__(model_name=model_name, device=device, normalize=True)
        self.query_instruction = query_instruction

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query with optional instruction."""
        if self.query_instruction:
            text = f"{self.query_instruction} {text}"
        return super().embed([text])[0]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed documents (no instruction needed)."""
        return super().embed(texts)


class OpenAIEmbeddings(BaseEmbeddingProvider):
    """
    OpenAI embeddings provider.

    Available models:
    - text-embedding-3-small (1536 dim, $0.02/1M tokens)
    - text-embedding-3-large (3072 dim, $0.13/1M tokens)
    - text-embedding-ada-002 (1536 dim, legacy)
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        batch_size: int = 100,
    ):
        """
        Initialize OpenAI embeddings.

        Args:
            model: OpenAI model name
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            batch_size: Batch size for API calls
        """
        self.model = model
        self.batch_size = batch_size
        self._client = None
        self._dim = None

        try:
            from openai import OpenAI

            self._client = OpenAI(api_key=api_key)

            # Determine dimension based on model
            if "text-embedding-3-large" in model:
                self._dim = 3072
            elif "text-embedding-3-small" in model or "ada-002" in model:
                self._dim = 1536
            else:
                self._dim = 1536  # default

        except ImportError:
            raise ImportError(
                "openai is required for OpenAI embeddings. Install with: pip install openai"
            )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embeddings using OpenAI API."""
        if not texts:
            return []

        texts_list = list(texts)
        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts_list), self.batch_size):
            batch = texts_list[i : i + self.batch_size]

            response = self._client.embeddings.create(input=batch, model=self.model)  # type: ignore[union-attr]

            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self._dim  # type: ignore[return-value]


class CohereEmbeddings(BaseEmbeddingProvider):
    """
    Cohere embeddings provider.

    Available models:
    - embed-english-v3.0 (1024 dim)
    - embed-multilingual-v3.0 (1024 dim)
    - embed-english-light-v3.0 (384 dim)
    - embed-multilingual-light-v3.0 (384 dim)
    """

    def __init__(
        self,
        model: str = "embed-english-v3.0",
        api_key: str | None = None,
        input_type: str = "search_document",
        batch_size: int = 96,
    ):
        """
        Initialize Cohere embeddings.

        Args:
            model: Cohere model name
            api_key: Cohere API key (or set COHERE_API_KEY env var)
            input_type: Type of input ('search_document', 'search_query', 'classification', 'clustering')
            batch_size: Batch size for API calls
        """
        self.model = model
        self.input_type = input_type
        self.batch_size = batch_size
        self._client = None
        self._dim = None

        try:
            import cohere

            self._client = cohere.Client(api_key=api_key)

            # Determine dimension based on model
            if "light" in model:
                self._dim = 384
            else:
                self._dim = 1024

        except ImportError:
            raise ImportError(
                "cohere is required for Cohere embeddings. Install with: pip install cohere"
            )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embeddings using Cohere API."""
        if not texts:
            return []

        texts_list = list(texts)
        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts_list), self.batch_size):
            batch = texts_list[i : i + self.batch_size]

            response = self._client.embed(texts=batch, model=self.model, input_type=self.input_type)  # type: ignore[union-attr]

            all_embeddings.extend(response.embeddings)

        return all_embeddings

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self._dim  # type: ignore[return-value]


# Convenience aliases
def create_e5_embeddings(model_size: str = "base", device: str | None = None) -> E5Embeddings:
    """
    Create E5 embeddings with size shorthand.

    Args:
        model_size: 'small', 'base', or 'large'
        device: Device to use
    """
    model_map = {
        "small": "intfloat/e5-small-v2",
        "base": "intfloat/e5-base-v2",
        "large": "intfloat/e5-large-v2",
    }
    model_name = model_map.get(model_size, model_size)
    return E5Embeddings(model_name=model_name, device=device)


def create_bge_embeddings(model_size: str = "base", device: str | None = None) -> BGEEmbeddings:
    """
    Create BGE embeddings with size shorthand.

    Args:
        model_size: 'small', 'base', or 'large'
        device: Device to use
    """
    model_map = {
        "small": "BAAI/bge-small-en-v1.5",
        "base": "BAAI/bge-base-en-v1.5",
        "large": "BAAI/bge-large-en-v1.5",
    }
    model_name = model_map.get(model_size, model_size)
    return BGEEmbeddings(model_name=model_name, device=device)
