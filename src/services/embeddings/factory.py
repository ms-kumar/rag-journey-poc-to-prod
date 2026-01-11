"""
Factory for creating embedding clients from application settings.
"""

import logging
from typing import TYPE_CHECKING, Any

from src.exceptions import EmbeddingProviderError
from src.services.embeddings.adapter import LangChainEmbeddingsAdapter
from src.services.embeddings.cached_client import CachedEmbeddingClient
from src.services.embeddings.client import EmbedClient

if TYPE_CHECKING:
    from src.config import EmbeddingSettings

logger = logging.getLogger(__name__)


def get_embed_client(
    model_name: str = "simple-hash",
    dim: int = 64,
    normalize: bool = True,
    provider: str = "hash",
    device: str | None = None,
    api_key: str | None = None,
    cache_enabled: bool = True,
    cache_max_size: int = 10000,
    cache_dir: str | None = ".cache/embeddings",
    batch_size: int = 32,
    **kwargs,
):
    """
    Create an embedding client based on provider with caching support.

    Args:
        model_name: Model name or identifier
        dim: Embedding dimension (for hash provider)
        normalize: Whether to normalize embeddings
        provider: Provider type ('hash', 'e5', 'bge', 'huggingface', 'openai', 'cohere')
        device: Device for local models ('cpu', 'cuda', or None for auto)
        api_key: API key for external providers
        cache_enabled: Whether to enable embedding caching
        cache_max_size: Maximum number of embeddings to cache
        cache_dir: Directory for disk cache (None to disable persistence)
        batch_size: Batch size for encoding
        **kwargs: Additional provider-specific arguments

    Returns:
        CachedEmbeddingClient wrapping the base provider
    """
    provider = provider.lower()

    # Create base provider
    base_provider: Any

    if provider == "hash":
        base_provider = EmbedClient(model_name=model_name, dim=dim, normalize=normalize)

    elif provider == "e5":
        from .providers import E5Embeddings

        base_provider = E5Embeddings(model_name=model_name, device=device, **kwargs)

    elif provider == "bge":
        from .providers import BGEEmbeddings

        base_provider = BGEEmbeddings(model_name=model_name, device=device, **kwargs)

    elif provider == "huggingface" or provider == "hf":
        from .providers import HuggingFaceEmbeddings

        base_provider = HuggingFaceEmbeddings(
            model_name=model_name,
            device=device,
            normalize=normalize,
            batch_size=batch_size,
            **kwargs,
        )

    elif provider == "openai":
        from .providers import OpenAIEmbeddings

        # Use 'model' from kwargs if present, otherwise use model_name
        model = kwargs.pop("model", model_name)
        base_provider = OpenAIEmbeddings(model=model, api_key=api_key, **kwargs)

    elif provider == "cohere":
        from .providers import CohereEmbeddings

        # Use 'model' from kwargs if present, otherwise use model_name
        model = kwargs.pop("model", model_name)
        base_provider = CohereEmbeddings(model=model, api_key=api_key, **kwargs)

    else:
        raise EmbeddingProviderError(
            f"Unknown provider: {provider}. Supported: hash, e5, bge, huggingface, openai, cohere"
        )

    # Wrap with caching layer
    cached_client = CachedEmbeddingClient(
        provider=base_provider,
        cache_enabled=cache_enabled,
        cache_max_size=cache_max_size,
        cache_dir=cache_dir,
        batch_size=batch_size,
    )
    logger.info(f"Created embedding client with provider={provider}, cache_enabled={cache_enabled}")
    return cached_client


def make_embedding_client(settings: "EmbeddingSettings") -> CachedEmbeddingClient:
    """
    Create embedding client from application settings.

    Args:
        settings: Embedding settings

    Returns:
        Configured embedding client with caching
    """
    return get_embed_client(  # type: ignore[no-any-return]
        model_name=settings.model,
        dim=settings.dim,
        normalize=settings.normalize,
        provider=settings.provider,
        device=settings.device,
        api_key=settings.api_key,
        cache_enabled=settings.cache_enabled,
        cache_max_size=settings.cache_max_size,
        cache_dir=settings.cache_dir,
        batch_size=settings.batch_size,
    )


def create_from_settings(settings: "EmbeddingSettings", **overrides):
    """
    Create embedding client from application settings with optional overrides.

    Deprecated: Use make_embedding_client() instead.
    """
    return get_embed_client(
        model_name=overrides.get("model_name", settings.model),
        dim=overrides.get("dim", settings.dim),
        normalize=overrides.get("normalize", settings.normalize),
        provider=overrides.get("provider", settings.provider),
        device=overrides.get("device", settings.device),
        api_key=overrides.get("api_key", settings.api_key),
        cache_enabled=overrides.get("cache_enabled", settings.cache_enabled),
        cache_max_size=overrides.get("cache_max_size", settings.cache_max_size),
        cache_dir=overrides.get("cache_dir", settings.cache_dir),
        batch_size=overrides.get("batch_size", settings.batch_size),
    )


def get_langchain_embeddings_adapter(
    embed_client, batch_size: int = 32
) -> LangChainEmbeddingsAdapter:
    """Wrap any embedding client with LangChain adapter."""
    return LangChainEmbeddingsAdapter(embed_client, batch_size=batch_size)
