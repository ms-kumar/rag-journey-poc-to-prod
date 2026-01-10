"""
Semantic cache for caching based on semantic similarity.
"""

import hashlib
import json
import logging
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from src.services.cache.redis_client import RedisCache, RedisCacheConfig
from src.services.embeddings.client import EmbedClient

logger = logging.getLogger(__name__)


class SemanticCacheConfig(BaseModel):
    """Configuration for semantic cache."""

    similarity_threshold: float = Field(
        default=0.95,
        description="Cosine similarity threshold for cache hits (0.0-1.0)",
    )
    embedding_dim: int = Field(default=384, description="Embedding dimension")
    max_candidates: int = Field(
        default=100,
        description="Maximum number of candidates to check for similarity",
    )
    redis_config: RedisCacheConfig = Field(
        default_factory=RedisCacheConfig,
        description="Redis cache configuration",
    )

    @classmethod
    def from_settings(cls, settings: Any) -> "SemanticCacheConfig":
        """Create config from application settings."""
        cache_settings = settings.cache
        return cls(
            similarity_threshold=cache_settings.semantic_similarity_threshold,
            embedding_dim=cache_settings.semantic_embedding_dim,
            max_candidates=cache_settings.semantic_max_candidates,
            redis_config=RedisCacheConfig.from_settings(settings),
        )


class SemanticCache:
    """
    Semantic cache that retrieves cached results based on semantic similarity.

    Instead of exact key matching, this cache compares query embeddings
    and returns cached results if similarity exceeds threshold.

    Features:
    - Semantic similarity matching using embeddings
    - Configurable similarity threshold
    - Efficient candidate selection
    - Backed by Redis for persistence
    """

    def __init__(
        self,
        embed_client: EmbedClient,
        config: SemanticCacheConfig | None = None,
        redis_cache: RedisCache | None = None,
    ):
        """
        Initialize semantic cache.

        Args:
            embed_client: Embedding client for generating query embeddings
            config: Semantic cache configuration
            redis_cache: Redis cache instance (or creates new one)
        """
        self.config = config or SemanticCacheConfig()
        self.embed_client = embed_client

        # Initialize Redis cache for storing embeddings and values
        self.redis_cache = redis_cache or RedisCache(self.config.redis_config)

        # Keys
        self._embedding_key_prefix = "semantic:embedding:"
        self._value_key_prefix = "semantic:value:"
        self._index_key = "semantic:index"

    def _make_embedding_key(self, query_hash: str) -> str:
        """Create embedding key."""
        return f"{self._embedding_key_prefix}{query_hash}"

    def _make_value_key(self, query_hash: str) -> str:
        """Create value key."""
        return f"{self._value_key_prefix}{query_hash}"

    def _hash_query(self, query: str) -> str:
        """Create hash of query."""
        return hashlib.sha256(query.encode()).hexdigest()[:16]

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score (0.0-1.0)
        """
        arr1 = np.array(vec1)
        arr2 = np.array(vec2)

        # Normalize vectors
        norm1 = np.linalg.norm(arr1)
        norm2 = np.linalg.norm(arr2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Calculate cosine similarity
        return float(np.dot(arr1, arr2) / (norm1 * norm2))

    def get(self, query: str) -> Any | None:
        """
        Get cached value for semantically similar query.

        Args:
            query: Query string

        Returns:
            Cached value if similar query found, None otherwise
        """
        try:
            # Generate embedding for query
            query_embedding = self.embed_client.embed([query])[0]

            # Get candidate hashes from index
            candidate_hashes_raw = self.redis_cache.get(self._index_key)
            if not candidate_hashes_raw:
                return None

            candidate_hashes = (
                candidate_hashes_raw
                if isinstance(candidate_hashes_raw, list)
                else json.loads(candidate_hashes_raw)
            )

            # Limit candidates
            candidates_to_check = candidate_hashes[: self.config.max_candidates]

            # Check similarity with each candidate
            best_similarity = 0.0
            best_hash = None

            for candidate_hash in candidates_to_check:
                # Get cached embedding
                embedding_key = self._make_embedding_key(candidate_hash)
                cached_embedding = self.redis_cache.get(embedding_key)

                if not cached_embedding:
                    continue

                # Calculate similarity
                similarity = self._cosine_similarity(query_embedding, cached_embedding)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_hash = candidate_hash

            # Check if best match exceeds threshold
            if best_similarity >= self.config.similarity_threshold and best_hash:
                # Get cached value
                value_key = self._make_value_key(best_hash)
                cached_value = self.redis_cache.get(value_key)

                if cached_value is not None:
                    logger.debug(
                        f"Semantic cache HIT (similarity={best_similarity:.3f}) for query: {query[:50]}"
                    )
                    return cached_value

            logger.debug(
                f"Semantic cache MISS (best_similarity={best_similarity:.3f}) for query: {query[:50]}"
            )
            return None

        except Exception as e:
            logger.error(f"Error in semantic cache get: {e}")
            return None

    def set(self, query: str, value: Any, ttl: int | None = None) -> bool:
        """
        Cache value with semantic embedding.

        Args:
            query: Query string
            value: Value to cache
            ttl: Time-to-live in seconds

        Returns:
            True if cached successfully
        """
        try:
            # Generate embedding
            query_embedding = self.embed_client.embed([query])[0]

            # Create hash for this query
            query_hash = self._hash_query(query)

            # Store embedding
            embedding_key = self._make_embedding_key(query_hash)
            self.redis_cache.set(embedding_key, query_embedding, ttl=ttl)

            # Store value
            value_key = self._make_value_key(query_hash)
            self.redis_cache.set(value_key, value, ttl=ttl)

            # Update index (list of all hashes)
            current_index = self.redis_cache.get(self._index_key) or []
            if not isinstance(current_index, list):
                current_index = json.loads(current_index) if current_index else []

            # Add to front of index (most recent first)
            if query_hash not in current_index:
                current_index.insert(0, query_hash)
                # Keep index size manageable
                if len(current_index) > self.config.max_candidates * 2:
                    current_index = current_index[: self.config.max_candidates * 2]

                self.redis_cache.set(self._index_key, current_index, ttl=None)

            logger.debug(f"Semantic cache SET for query: {query[:50]}")
            return True

        except Exception as e:
            logger.error(f"Error in semantic cache set: {e}")
            return False

    def invalidate(self, pattern: str = "*") -> int:
        """
        Invalidate semantic cache entries matching pattern.

        Args:
            pattern: Pattern for keys to invalidate

        Returns:
            Number of entries invalidated
        """
        count = 0

        # Invalidate embeddings
        count += self.redis_cache.invalidate_pattern(f"{self._embedding_key_prefix}{pattern}")

        # Invalidate values
        count += self.redis_cache.invalidate_pattern(f"{self._value_key_prefix}{pattern}")

        # Clear index if pattern matches all
        if pattern == "*":
            self.redis_cache.delete(self._index_key)
            count += 1

        logger.info(f"Invalidated {count} semantic cache entries")
        return count

    def flush(self) -> bool:
        """
        Flush all semantic cache entries.

        Returns:
            True if flushed successfully
        """
        try:
            # Clear all semantic cache keys
            self.invalidate("*")
            logger.info("Flushed semantic cache")
            return True
        except Exception as e:
            logger.error(f"Error flushing semantic cache: {e}")
            return False

    def get_stats(self) -> dict[str, Any]:
        """
        Get semantic cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        try:
            index = self.redis_cache.get(self._index_key) or []
            if not isinstance(index, list):
                index = json.loads(index) if index else []

            return {
                "total_entries": len(index),
                "max_candidates": self.config.max_candidates,
                "similarity_threshold": self.config.similarity_threshold,
                "embedding_dim": self.config.embedding_dim,
            }
        except Exception as e:
            logger.error(f"Error getting semantic cache stats: {e}")
            return {}

    def health_check(self) -> bool:
        """
        Check if semantic cache is healthy.

        Returns:
            True if operational
        """
        return self.redis_cache.health_check()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SemanticCache(threshold={self.config.similarity_threshold}, "
            f"dim={self.config.embedding_dim})"
        )
