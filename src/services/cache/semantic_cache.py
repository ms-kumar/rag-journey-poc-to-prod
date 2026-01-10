"""
Semantic cache client for caching based on semantic similarity.

Similar to CacheClient but uses embedding similarity instead of exact matching.
"""

import hashlib
import json
import logging
from typing import Any

import numpy as np

from src.services.cache.client import CacheClient
from src.services.embeddings.client import EmbedClient

logger = logging.getLogger(__name__)


class SemanticCacheClient:
    """
    Semantic cache client that retrieves cached results based on semantic similarity.

    Instead of exact key matching, this cache compares query embeddings
    and returns cached results if similarity exceeds threshold.

    Features:
    - Semantic similarity matching using embeddings
    - Configurable similarity threshold
    - Efficient candidate selection
    - Backed by CacheClient for persistence
    - Context manager support
    """

    def __init__(
        self,
        cache_client: CacheClient,
        embed_client: EmbedClient,
        similarity_threshold: float = 0.95,
        embedding_dim: int = 384,
        max_candidates: int = 100,
    ):
        """
        Initialize semantic cache client.

        Args:
            cache_client: CacheClient instance for storage
            embed_client: Embedding client for generating query embeddings
            similarity_threshold: Cosine similarity threshold (0.0-1.0, default: 0.95)
            embedding_dim: Embedding dimension (default: 384)
            max_candidates: Max candidates to check for similarity (default: 100)
        """
        self.cache = cache_client
        self.embed_client = embed_client
        self.similarity_threshold = similarity_threshold
        self.embedding_dim = embedding_dim
        self.max_candidates = max_candidates

        # Key prefixes for organization
        self._embedding_prefix = "semantic:emb:"
        self._value_prefix = "semantic:val:"
        self._index_key = "semantic:index"

        logger.info(
            f"SemanticCacheClient initialized (threshold={similarity_threshold}, "
            f"dim={embedding_dim})"
        )

    def _make_embedding_key(self, query_hash: str) -> str:
        """Create embedding storage key."""
        return f"{self._embedding_prefix}{query_hash}"

    def _make_value_key(self, query_hash: str) -> str:
        """Create value storage key."""
        return f"{self._value_prefix}{query_hash}"

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

    def get(self, query: str) -> dict[str, Any] | None:
        """
        Get cached value for semantically similar query.

        Args:
            query: Query string

        Returns:
            Cached response dict if similar query found, None otherwise
        """
        try:
            # Generate embedding for query
            query_embedding = self.embed_client.embed([query])[0]

            # Get candidate hashes from index
            candidate_hashes_raw = self.cache.get(self._index_key)
            if not candidate_hashes_raw:
                logger.debug("Semantic cache: No index found")
                return None

            candidate_hashes = (
                candidate_hashes_raw
                if isinstance(candidate_hashes_raw, list)
                else json.loads(str(candidate_hashes_raw))
            )

            # Limit candidates for performance
            candidates_to_check = candidate_hashes[: self.max_candidates]

            # Find best matching candidate
            best_similarity = 0.0
            best_hash = None

            for candidate_hash in candidates_to_check:
                # Get cached embedding
                embedding_key = self._make_embedding_key(candidate_hash)
                cached_embedding_raw = self.cache.get(embedding_key)

                if not cached_embedding_raw:
                    continue

                # Parse embedding (stored as JSON list)
                try:
                    cached_embedding: list[float] = (
                        cached_embedding_raw
                        if isinstance(cached_embedding_raw, list)
                        else json.loads(str(cached_embedding_raw))
                    )
                except (json.JSONDecodeError, TypeError, ValueError):
                    logger.warning(f"Invalid embedding format for {candidate_hash}")
                    continue

                # Calculate similarity
                similarity = self._cosine_similarity(query_embedding, cached_embedding)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_hash = candidate_hash

            # Check if best match exceeds threshold
            if best_similarity >= self.similarity_threshold and best_hash:
                # Get cached value
                value_key = self._make_value_key(best_hash)
                cached_value = self.cache.get(value_key)

                if cached_value is not None:
                    logger.info(
                        f"Semantic cache HIT (similarity={best_similarity:.3f}) "
                        f"for query: {query[:50]}..."
                    )
                    return cached_value

            logger.debug(f"Semantic cache MISS (best={best_similarity:.3f}) for: {query[:50]}...")
            return None

        except Exception as e:
            logger.error(f"Error in semantic cache get: {e}")
            return None

    def set(self, query: str, response: dict[str, Any]) -> bool:
        """
        Cache response with semantic embedding.

        Args:
            query: Query string
            response: Response dict to cache

        Returns:
            True if cached successfully
        """
        try:
            # Generate embedding
            query_embedding = self.embed_client.embed([query])[0]

            # Create hash for this query
            query_hash = self._hash_query(query)

            # Store embedding (direct key, no need for params)
            embedding_key = self._make_embedding_key(query_hash)
            self.cache.redis.set(embedding_key, json.dumps(query_embedding))

            # Store value
            value_key = self._make_value_key(query_hash)
            self.cache.redis.set(value_key, json.dumps(response))

            # Update index (list of all hashes)
            current_index_raw = self.cache.get(self._index_key)
            current_index = []

            if current_index_raw:
                current_index = (
                    current_index_raw
                    if isinstance(current_index_raw, list)
                    else json.loads(str(current_index_raw))
                )

            # Add to front of index (most recent first)
            if query_hash not in current_index:
                current_index.insert(0, query_hash)

                # Keep index size manageable
                max_index_size = self.max_candidates * 2
                if len(current_index) > max_index_size:
                    current_index = current_index[:max_index_size]

                self.cache.redis.set(self._index_key, json.dumps(current_index))

            logger.info(f"Semantic cache SET for query: {query[:50]}...")
            return True

        except Exception as e:
            logger.error(f"Error in semantic cache set: {e}")
            return False

    def clear(self, pattern: str = "*") -> int:
        """
        Clear semantic cache entries matching pattern.

        Args:
            pattern: Pattern for keys to clear (default: "*" for all)

        Returns:
            Number of entries cleared
        """
        count = 0

        try:
            # Clear embeddings
            count += self.cache.invalidate_pattern(f"{self._embedding_prefix}{pattern}")

            # Clear values
            count += self.cache.invalidate_pattern(f"{self._value_prefix}{pattern}")

            # Clear index if pattern matches all
            if pattern == "*":
                self.cache.delete(self._index_key)
                count += 1

            logger.info(f"Cleared {count} semantic cache entries")
            return count

        except Exception as e:
            logger.error(f"Error clearing semantic cache: {e}")
            return 0

    def flush(self) -> bool:
        """
        Flush all semantic cache entries.

        Returns:
            True if flushed successfully
        """
        try:
            self.clear("*")
            logger.info("Flushed semantic cache")
            return True
        except Exception as e:
            logger.error(f"Error flushing semantic cache: {e}")
            return False

    def ping(self) -> bool:
        """
        Check if semantic cache is operational.

        Returns:
            True if operational
        """
        return self.cache.ping()

    def health_check(self) -> bool:
        """
        Check if semantic cache is healthy.

        Alias for ping() for consistency.

        Returns:
            True if operational
        """
        return self.ping()

    def get_stats(self) -> dict[str, Any]:
        """
        Get semantic cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        try:
            index_raw = self.cache.get(self._index_key)
            index = []

            if index_raw:
                index = index_raw if isinstance(index_raw, list) else json.loads(str(index_raw))

            return {
                "total_entries": len(index),
                "max_candidates": self.max_candidates,
                "similarity_threshold": self.similarity_threshold,
                "embedding_dim": self.embedding_dim,
                "connected": self.cache.ping(),
            }
        except Exception as e:
            logger.error(f"Error getting semantic cache stats: {e}")
            return {
                "total_entries": 0,
                "max_candidates": self.max_candidates,
                "similarity_threshold": self.similarity_threshold,
                "embedding_dim": self.embedding_dim,
                "connected": False,
                "error": str(e),
            }

    def close(self) -> None:
        """
        Close underlying cache connection.

        Should be called when semantic cache client is no longer needed.
        """
        try:
            self.cache.close()
            logger.info("Closed semantic cache connection")
        except Exception as e:
            logger.error(f"Error closing semantic cache: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SemanticCacheClient(threshold={self.similarity_threshold}, "
            f"dim={self.embedding_dim}, max_candidates={self.max_candidates})"
        )
