"""
Cached embedding client with batch processing support.

Wraps any embedding provider with caching and efficient batch operations.
"""

import logging
from collections.abc import Sequence
from typing import Any

from src.services.embeddings.cache import EmbeddingCache

logger = logging.getLogger(__name__)


class CachedEmbeddingClient:
    """
    Wrapper for embedding clients that adds caching and batch processing.

    Features:
    - Automatic caching of computed embeddings
    - Batch processing with cache-aware splitting
    - Only computes embeddings for uncached texts
    - Maintains result order
    """

    def __init__(
        self,
        provider: Any,
        cache_enabled: bool = True,
        cache_max_size: int = 10000,
        cache_dir: str | None = None,
        batch_size: int = 32,
    ):
        """
        Initialize cached embedding client.

        Args:
            provider: Embedding provider (must implement embed() method)
            cache_enabled: Whether to enable caching
            cache_max_size: Maximum cache size (number of embeddings)
            cache_dir: Directory for disk cache (None to disable disk persistence)
            batch_size: Batch size for encoding uncached texts
        """
        self.provider = provider
        self.cache_enabled = cache_enabled
        self.batch_size = batch_size

        # Initialize cache
        model_id = getattr(provider, "model_name", "unknown")
        self.cache = EmbeddingCache(
            max_size=cache_max_size, cache_dir=cache_dir, model_identifier=model_id
        )

        logger.info(f"Initialized CachedEmbeddingClient with {model_id}, cache={cache_enabled}")

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """
        Generate embeddings with caching.

        Args:
            texts: Sequence of input texts

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Fast path: no caching
        if not self.cache_enabled:
            return self.provider.embed(texts)  # type: ignore[no-any-return]

        # Convert to list for indexing
        texts_list = list(texts)
        n = len(texts_list)

        # Try to get from cache
        cached = self.cache.get_batch(texts_list)

        # Identify which texts need computation
        to_compute = []
        to_compute_indices = []

        for i, text in enumerate(texts_list):
            if text not in cached:
                to_compute.append(text)
                to_compute_indices.append(i)

        # Compute uncached embeddings in batches
        newly_computed = []
        if to_compute:
            logger.debug(
                f"Cache miss: {len(to_compute)}/{n} texts need computation "
                f"(hit_rate={self.cache.stats['hit_rate']:.2%})"
            )

            # Process in batches if provider supports it
            newly_computed = self._compute_in_batches(to_compute)

            # Store in cache
            self.cache.put_batch(to_compute, newly_computed)

        # Reconstruct results in original order
        results: list[list[float]] = []
        computed_idx = 0

        for _i, text in enumerate(texts_list):
            if text in cached:
                results.append(cached[text])
            elif computed_idx < len(newly_computed):
                results.append(newly_computed[computed_idx])
                computed_idx += 1

        return results

    def _compute_in_batches(self, texts: list[str]) -> list[list[float]]:
        """
        Compute embeddings in batches.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings
        """
        if not texts:
            return []

        # If batch size not specified or provider doesn't batch well, compute all at once
        if self.batch_size <= 0 or len(texts) <= self.batch_size:
            return self.provider.embed(texts)  # type: ignore[no-any-return]

        # Process in batches
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_embeddings = self.provider.embed(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self.cache.clear()
        logger.info("Embedding cache cleared")

    def save_cache(self) -> None:
        """Save cache to disk."""
        self.cache.save_to_disk()

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query text (delegates to provider if available).

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        if hasattr(self.provider, "embed_query"):
            # Use provider's specific method if available
            return self.provider.embed_query(text)  # type: ignore[no-any-return]
        # Fallback to regular embed
        return self.embed([text])[0]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """
        Embed multiple documents (delegates to provider if available).

        Args:
            texts: Sequence of document texts to embed

        Returns:
            List of embedding vectors
        """
        if hasattr(self.provider, "embed_documents"):
            # Use provider's specific method if available
            return self.provider.embed_documents(texts)  # type: ignore[no-any-return]
        # Fallback to regular embed
        return self.embed(texts)

    @property
    def cache_stats(self) -> dict:
        """Get cache statistics."""
        return self.cache.stats

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        # Try to get from provider
        if hasattr(self.provider, "dimension"):
            return self.provider.dimension  # type: ignore[no-any-return]
        if hasattr(self.provider, "dim"):
            return self.provider.dim  # type: ignore[no-any-return]
        # Fallback: compute one embedding and check its size
        test_embedding = self.provider.embed(["test"])[0]
        return len(test_embedding)

    @property
    def dim(self) -> int:
        """Return embedding dimension (backward compatibility)."""
        return self.dimension

    def __repr__(self) -> str:
        return f"CachedEmbeddingClient(provider={type(self.provider).__name__}, cache={self.cache})"
