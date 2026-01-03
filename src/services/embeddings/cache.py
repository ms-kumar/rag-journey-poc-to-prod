"""
Embedding cache for storing and retrieving computed embeddings.

Supports in-memory caching with LRU eviction and optional disk persistence.
"""

import hashlib
import json
import logging
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """
    LRU cache for embeddings with optional disk persistence.

    Features:
    - In-memory LRU cache with configurable max size
    - Optional disk persistence (JSON or pickle)
    - Cache key based on text hash + model identifier
    - Batch operations for efficient lookup/storage
    """

    def __init__(
        self,
        max_size: int = 10000,
        cache_dir: str | Path | None = None,
        model_identifier: str = "default",
    ):
        """
        Initialize embedding cache.

        Args:
            max_size: Maximum number of embeddings to cache in memory
            cache_dir: Directory for disk cache persistence (None to disable)
            model_identifier: Identifier for the model (included in cache keys)
        """
        self.max_size = max_size
        self.model_identifier = model_identifier
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # In-memory LRU cache
        self._cache: OrderedDict[str, list[float]] = OrderedDict()

        # Stats
        self._hits = 0
        self._misses = 0

        # Load from disk if enabled
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

    def _compute_key(self, text: str) -> str:
        """Compute cache key for text."""
        # Include model identifier to avoid conflicts between different models
        content = f"{self.model_identifier}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, text: str) -> list[float] | None:
        """
        Retrieve embedding from cache.

        Args:
            text: Input text

        Returns:
            Embedding vector if found, None otherwise
        """
        key = self._compute_key(text)

        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]

        self._misses += 1
        return None

    def get_batch(self, texts: Sequence[str]) -> dict[str, list[float]]:
        """
        Retrieve multiple embeddings from cache.

        Args:
            texts: Sequence of input texts

        Returns:
            Dictionary mapping cache keys to embeddings (only cached items)
        """
        result = {}
        for text in texts:
            embedding = self.get(text)
            if embedding is not None:
                result[text] = embedding
        return result

    def put(self, text: str, embedding: list[float]) -> None:
        """
        Store embedding in cache.

        Args:
            text: Input text
            embedding: Embedding vector
        """
        key = self._compute_key(text)

        # Remove oldest if at capacity
        if len(self._cache) >= self.max_size and key not in self._cache:
            self._cache.popitem(last=False)

        # Add/update cache
        self._cache[key] = embedding
        self._cache.move_to_end(key)

    def put_batch(self, texts: Sequence[str], embeddings: Sequence[list[float]]) -> None:
        """
        Store multiple embeddings in cache.

        Args:
            texts: Sequence of input texts
            embeddings: Corresponding embedding vectors
        """
        if len(texts) != len(embeddings):
            raise ValueError("texts and embeddings must have same length")

        for text, embedding in zip(texts, embeddings, strict=True):
            self.put(text, embedding)

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def save_to_disk(self) -> None:
        """Save cache to disk (if cache_dir is configured)."""
        if not self.cache_dir:
            return

        cache_file = self.cache_dir / f"embedding_cache_{self.model_identifier}.json"

        try:
            # Convert OrderedDict to regular dict for JSON serialization
            cache_data = {
                "model_identifier": self.model_identifier,
                "max_size": self.max_size,
                "cache": dict(self._cache),
                "hits": self._hits,
                "misses": self._misses,
            }

            with cache_file.open("w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2)

            logger.info(f"Saved {len(self._cache)} embeddings to {cache_file}")

        except Exception as e:
            logger.error(f"Failed to save cache to disk: {e}")

    def _load_from_disk(self) -> None:
        """Load cache from disk (if available)."""
        if not self.cache_dir:
            return

        cache_file = self.cache_dir / f"embedding_cache_{self.model_identifier}.json"

        if not cache_file.exists():
            return

        try:
            with cache_file.open("r", encoding="utf-8") as f:
                cache_data = json.load(f)

            # Restore cache
            self._cache = OrderedDict(cache_data.get("cache", {}))
            self._hits = cache_data.get("hits", 0)
            self._misses = cache_data.get("misses", 0)

            logger.info(f"Loaded {len(self._cache)} embeddings from {cache_file}")

        except Exception as e:
            logger.error(f"Failed to load cache from disk: {e}")

    @property
    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
        }

    def __len__(self) -> int:
        """Return current cache size."""
        return len(self._cache)

    def __repr__(self) -> str:
        stats = self.stats
        return (
            f"EmbeddingCache(size={stats['size']}/{stats['max_size']}, "
            f"hit_rate={stats['hit_rate']:.2%})"
        )
