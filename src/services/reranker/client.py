"""
Cross-encoder re-ranker for improving retrieval precision.

Provides:
- Model loading & batch scoring
- Candidate reordering
- Precision@k evaluation
- Timeouts & fallbacks

"""

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Callable

import torch
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from src.exceptions import (
    RerankerModelError,
    RerankerTimeoutError,
)
from src.services.retry import RetryConfig, retry_with_backoff

logger = logging.getLogger(__name__)


@dataclass
class RerankerConfig:
    """Configuration for cross-encoder re-ranker."""

    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    max_length: int = 512
    batch_size: int = 32
    timeout_seconds: float = 30.0
    device: str | None = None
    # Fallback options
    fallback_enabled: bool = True
    fallback_strategy: str = "original_order"  # "original_order" or "score_descending"
    # Performance settings
    use_fp16: bool = True

    def __post_init__(self):
        """Auto-detect device if not specified."""
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def from_settings(cls, settings: Any) -> "RerankerConfig":
        """Create config from application settings."""
        reranker_settings = settings.reranker
        return cls(
            model_name=reranker_settings.model_name,
            max_length=reranker_settings.max_length,
            batch_size=reranker_settings.batch_size,
            timeout_seconds=reranker_settings.timeout_seconds,
            device=reranker_settings.device,
            fallback_enabled=reranker_settings.fallback_enabled,
            fallback_strategy=reranker_settings.fallback_strategy,
            use_fp16=reranker_settings.use_fp16,
        )


@dataclass
class RerankResult:
    """Result from re-ranking operation."""

    documents: list[Document]
    scores: list[float]
    original_ranks: list[int]
    execution_time: float
    model_used: str
    fallback_used: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PrecisionMetrics:
    """Precision@k evaluation metrics."""

    precision_at_k: dict[int, float]
    total_relevant: int
    total_retrieved: int
    improvement_over_baseline: dict[int, float] = field(default_factory=dict)


class CrossEncoderReranker:
    """Cross-encoder re-ranker for improving retrieval precision."""

    def __init__(self, config: RerankerConfig | None = None):
        self.config = config or RerankerConfig()
        self.retry_config = RetryConfig(max_retries=2, initial_delay=1.0)
        self.model: CrossEncoder | None = None
        self.model_loaded = False

        # Load model on initialization
        self._load_model()

    def _load_model(self) -> None:
        """Load the cross-encoder model with error handling."""
        try:
            logger.info(f"Loading cross-encoder model: {self.config.model_name}")
            start_time = time.time()

            self.model = CrossEncoder(
                self.config.model_name, max_length=self.config.max_length, device=self.config.device
            )

            # Enable FP16 if requested and available
            if self.config.use_fp16 and self.config.device != "cpu":
                try:
                    self.model.model.half()
                    logger.info("Enabled FP16 optimization")
                except Exception as e:
                    logger.warning(f"Could not enable FP16: {e}")

            load_time = time.time() - start_time
            self.model_loaded = True
            logger.info(
                f"Cross-encoder model loaded successfully in {load_time:.2f}s "
                f"on device: {self.config.device}"
            )

        except Exception as e:
            logger.error(f"Failed to load cross-encoder model: {e}")
            self.model_loaded = False
            raise

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int | None = None,
        return_scores: bool = True,
    ) -> RerankResult:
        """Re-rank documents using cross-encoder scoring.

        Args:
            query: Search query
            documents: List of candidate documents to re-rank
            top_k: Number of top documents to return (None = all)
            return_scores: Whether to include relevance scores

        Returns:
            RerankResult with re-ranked documents
        """
        start_time = time.time()

        if not documents:
            return RerankResult(
                documents=[],
                scores=[],
                original_ranks=[],
                execution_time=0.0,
                model_used=self.config.model_name,
                fallback_used=False,
            )

        # Store original ranking
        original_ranks = list(range(len(documents)))

        try:
            # Attempt re-ranking with timeout
            scores = self._score_documents_with_timeout(query, documents)

            # Sort by scores (descending)
            scored_docs = list(zip(documents, scores, original_ranks, strict=True))
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            # Extract results
            reranked_docs = [doc for doc, _, _ in scored_docs]
            reranked_scores = [score for _, score, _ in scored_docs]
            new_original_ranks = [rank for _, _, rank in scored_docs]

            # Apply top_k limit if specified
            if top_k is not None:
                reranked_docs = reranked_docs[:top_k]
                reranked_scores = reranked_scores[:top_k]
                new_original_ranks = new_original_ranks[:top_k]

            execution_time = time.time() - start_time

            return RerankResult(
                documents=reranked_docs,
                scores=reranked_scores if return_scores else [],
                original_ranks=new_original_ranks,
                execution_time=execution_time,
                model_used=self.config.model_name,
                fallback_used=False,
            )

        except Exception as e:
            logger.warning(f"Re-ranking failed: {e}. Using fallback strategy.")
            return self._apply_fallback(documents, original_ranks, start_time)

    def _score_documents_with_timeout(self, query: str, documents: list[Document]) -> list[float]:
        """Score documents with timeout protection."""

        @retry_with_backoff(self.retry_config)
        def _score_batch() -> list[float]:
            if not self.model_loaded or self.model is None:
                raise RerankerModelError("Cross-encoder model not loaded")

            # Prepare query-document pairs
            pairs = [[query, doc.page_content] for doc in documents]

            # Score in batches to manage memory
            scores: list[float] = []
            for i in range(0, len(pairs), self.config.batch_size):
                batch = pairs[i : i + self.config.batch_size]

                # Apply timeout to each batch
                batch_scores = self._score_batch_with_timeout(batch)
                scores.extend(batch_scores)

            return scores

        # Cast the decorated function to ensure proper typing
        typed_score_batch: Callable[[], list[float]] = cast(
            "Callable[[], list[float]]", _score_batch
        )
        return typed_score_batch()

    def _score_batch_with_timeout(self, batch: list[list[str]]) -> list[float]:
        """Score a batch of query-document pairs with timeout."""
        if not self.model:
            raise RerankerModelError("Model not available")

        # Use a simple timeout approach
        start_time = time.time()

        try:
            scores = self.model.predict(batch)

            # Check if we exceeded timeout
            if time.time() - start_time > self.config.timeout_seconds:
                raise RerankerTimeoutError(
                    f"Batch scoring exceeded timeout ({self.config.timeout_seconds}s)"
                )

            # Convert to list of floats with proper type annotation
            result: list[float] = (
                scores.tolist() if hasattr(scores, "tolist") else cast("list[float]", list(scores))
            )
            return result

        except Exception as e:
            if time.time() - start_time > self.config.timeout_seconds:
                raise TimeoutError(f"Batch scoring timed out: {e}")
            raise

    def _apply_fallback(
        self, documents: list[Document], original_ranks: list[int], start_time: float
    ) -> RerankResult:
        """Apply fallback strategy when re-ranking fails."""

        if not self.config.fallback_enabled:
            raise RuntimeError("Re-ranking failed and fallback is disabled")

        logger.info(f"Applying fallback strategy: {self.config.fallback_strategy}")

        execution_time = time.time() - start_time

        if self.config.fallback_strategy == "original_order":
            return RerankResult(
                documents=documents,
                scores=[],
                original_ranks=original_ranks,
                execution_time=execution_time,
                model_used=f"{self.config.model_name} (fallback)",
                fallback_used=True,
            )

        if self.config.fallback_strategy == "score_descending":
            # Sort by existing scores if available
            scored_docs = []
            for i, doc in enumerate(documents):
                score = doc.metadata.get("score", 0.0) if doc.metadata else 0.0
                scored_docs.append((doc, score, i))

            scored_docs.sort(key=lambda x: x[1], reverse=True)

            return RerankResult(
                documents=[doc for doc, _, _ in scored_docs],
                scores=[score for _, score, _ in scored_docs],
                original_ranks=[rank for _, _, rank in scored_docs],
                execution_time=execution_time,
                model_used=f"{self.config.model_name} (fallback)",
                fallback_used=True,
            )

        # Default to original order
        return RerankResult(
            documents=documents,
            scores=[],
            original_ranks=original_ranks,
            execution_time=execution_time,
            model_used=f"{self.config.model_name} (fallback)",
            fallback_used=True,
        )

    def batch_rerank(
        self, queries: list[str], document_lists: list[list[Document]], top_k: int | None = None
    ) -> list[RerankResult]:
        """Re-rank multiple query-document sets in batch."""

        results = []
        for query, documents in zip(queries, document_lists, strict=True):
            result = self.rerank(query, documents, top_k=top_k)
            results.append(result)

        return results

    def evaluate_precision_at_k(
        self,
        query: str,
        documents: list[Document],
        relevant_doc_ids: set[str],
        baseline_docs: list[Document] | None = None,
        k_values: list[int] | None = None,
    ) -> PrecisionMetrics:
        """Evaluate precision@k improvement from re-ranking.

        Args:
            query: Search query
            documents: Documents to re-rank
            relevant_doc_ids: Set of relevant document IDs
            baseline_docs: Original ranking for comparison
            k_values: k values to evaluate (default: [1, 3, 5, 10])

        Returns:
            PrecisionMetrics with evaluation results
        """
        if k_values is None:
            k_values = [1, 3, 5, 10]

        # Re-rank documents
        rerank_result = self.rerank(query, documents)
        reranked_docs = rerank_result.documents

        # Calculate precision@k for re-ranked results
        precision_at_k = {}
        for k in k_values:
            if k > len(reranked_docs):
                k = len(reranked_docs)

            top_k_docs = reranked_docs[:k]
            relevant_in_top_k = 0

            for doc in top_k_docs:
                doc_id = self._extract_doc_id(doc)
                if doc_id in relevant_doc_ids:
                    relevant_in_top_k += 1

            precision_at_k[k] = relevant_in_top_k / k if k > 0 else 0.0

        # Calculate improvement over baseline if provided
        improvement = {}
        if baseline_docs:
            for k in k_values:
                if k > len(baseline_docs):
                    k = len(baseline_docs)

                baseline_top_k = baseline_docs[:k]
                baseline_relevant = 0

                for doc in baseline_top_k:
                    doc_id = self._extract_doc_id(doc)
                    if doc_id in relevant_doc_ids:
                        baseline_relevant += 1

                baseline_precision = baseline_relevant / k if k > 0 else 0.0
                improvement[k] = precision_at_k[k] - baseline_precision

        return PrecisionMetrics(
            precision_at_k=precision_at_k,
            total_relevant=len(relevant_doc_ids),
            total_retrieved=len(documents),
            improvement_over_baseline=improvement,
        )

    def _extract_doc_id(self, doc: Document) -> str:
        """Extract document ID from Document object."""
        if doc.metadata:
            # Try different ID field names
            for id_field in ["id", "doc_id", "chunk_id", "source"]:
                if id_field in doc.metadata:
                    return str(doc.metadata[id_field])

        # Fallback to hash of content
        return str(hash(doc.page_content))

    def health_check(self) -> dict[str, Any]:
        """Check reranker health status."""
        return {
            "model_loaded": self.model_loaded,
            "model_name": self.config.model_name,
            "device": self.config.device,
            "batch_size": self.config.batch_size,
            "timeout_seconds": self.config.timeout_seconds,
            "fallback_enabled": self.config.fallback_enabled,
            "cuda_available": torch.cuda.is_available(),
            "torch_version": torch.__version__,
        }
