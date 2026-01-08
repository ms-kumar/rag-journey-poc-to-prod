"""
Evaluation utilities for cross-encoder re-ranking.

Provides precision@k metrics, benchmarking tools, and comparison utilities.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document

from .client import CrossEncoderReranker, PrecisionMetrics

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Result from comparing baseline vs reranked results."""

    baseline_metrics: PrecisionMetrics
    reranked_metrics: PrecisionMetrics
    improvement: dict[int, float]
    statistical_significance: dict[int, bool] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkConfig:
    """Configuration for re-ranking benchmarks."""

    k_values: list[int] = field(default_factory=lambda: [1, 3, 5, 10, 20])
    num_queries: int = 100
    timeout_per_query: float = 30.0
    include_latency: bool = True
    include_precision: bool = True
    include_recall: bool = False


class RerankingEvaluator:
    """Evaluator for cross-encoder re-ranking performance."""

    def __init__(self, reranker: CrossEncoderReranker):
        self.reranker = reranker

    def compare_rankings(
        self,
        query: str,
        baseline_docs: list[Document],
        relevant_doc_ids: set[str],
        k_values: list[int] | None = None,
    ) -> ComparisonResult:
        """Compare baseline vs re-ranked results.

        Args:
            query: Search query
            baseline_docs: Original ranking from retrieval
            relevant_doc_ids: Set of relevant document IDs for evaluation
            k_values: k values to evaluate

        Returns:
            ComparisonResult with metrics comparison
        """
        if k_values is None:
            k_values = [1, 3, 5, 10]

        # Evaluate baseline precision
        baseline_metrics = self._calculate_precision_metrics(
            baseline_docs, relevant_doc_ids, k_values
        )

        # Evaluate re-ranked precision
        reranked_metrics = self.reranker.evaluate_precision_at_k(
            query=query,
            documents=baseline_docs,
            relevant_doc_ids=relevant_doc_ids,
            k_values=k_values,
        )

        # Calculate improvement
        improvement = {}
        for k in k_values:
            baseline_p = baseline_metrics.precision_at_k.get(k, 0.0)
            reranked_p = reranked_metrics.precision_at_k.get(k, 0.0)
            improvement[k] = reranked_p - baseline_p

        return ComparisonResult(
            baseline_metrics=baseline_metrics,
            reranked_metrics=reranked_metrics,
            improvement=improvement,
        )

    def benchmark_multiple_queries(
        self,
        queries: list[str],
        document_lists: list[list[Document]],
        relevant_doc_sets: list[set[str]],
        config: BenchmarkConfig | None = None,
    ) -> dict[str, Any]:
        """Benchmark re-ranking across multiple queries.

        Args:
            queries: List of test queries
            document_lists: List of document lists (one per query)
            relevant_doc_sets: List of relevant document ID sets (one per query)
            config: Benchmark configuration

        Returns:
            Comprehensive benchmark results
        """
        if config is None:
            config = BenchmarkConfig()

        if not (len(queries) == len(document_lists) == len(relevant_doc_sets)):
            raise ValueError("Queries, document lists, and relevant sets must have same length")

        results = []
        total_baseline_precision = dict.fromkeys(config.k_values, 0.0)
        total_reranked_precision = dict.fromkeys(config.k_values, 0.0)
        total_latency = 0.0

        logger.info(f"Starting benchmark with {len(queries)} queries")

        for i, (query, docs, relevant_ids) in enumerate(
            zip(queries, document_lists, relevant_doc_sets, strict=True)
        ):
            try:
                comparison = self.compare_rankings(
                    query=query,
                    baseline_docs=docs,
                    relevant_doc_ids=relevant_ids,
                    k_values=config.k_values,
                )

                # Re-rank for latency measurement if requested
                if config.include_latency:
                    rerank_result = self.reranker.rerank(query, docs)
                    total_latency += rerank_result.execution_time

                # Accumulate precision metrics
                for k in config.k_values:
                    total_baseline_precision[k] += comparison.baseline_metrics.precision_at_k.get(
                        k, 0.0
                    )
                    total_reranked_precision[k] += comparison.reranked_metrics.precision_at_k.get(
                        k, 0.0
                    )

                results.append(comparison)

                if (i + 1) % 10 == 0:
                    logger.info(f"Completed {i + 1}/{len(queries)} queries")

            except Exception as e:
                logger.warning(f"Failed to process query {i}: {e}")
                continue

        # Calculate averages
        num_successful = len(results)
        avg_baseline_precision = {
            k: v / num_successful for k, v in total_baseline_precision.items()
        }
        avg_reranked_precision = {
            k: v / num_successful for k, v in total_reranked_precision.items()
        }
        avg_improvement = {
            k: avg_reranked_precision[k] - avg_baseline_precision[k] for k in config.k_values
        }
        avg_latency = total_latency / num_successful if config.include_latency else 0.0

        return {
            "config": config,
            "num_queries": len(queries),
            "num_successful": num_successful,
            "individual_results": results,
            "avg_baseline_precision": avg_baseline_precision,
            "avg_reranked_precision": avg_reranked_precision,
            "avg_improvement": avg_improvement,
            "avg_latency_ms": avg_latency * 1000 if config.include_latency else None,
            "reranker_info": self.reranker.health_check(),
        }

    def _calculate_precision_metrics(
        self, documents: list[Document], relevant_doc_ids: set[str], k_values: list[int]
    ) -> PrecisionMetrics:
        """Calculate precision@k for a document ranking."""
        precision_at_k = {}

        for k in k_values:
            if k > len(documents):
                k = len(documents)

            top_k_docs = documents[:k]
            relevant_in_top_k = 0

            for doc in top_k_docs:
                doc_id = self._extract_doc_id(doc)
                if doc_id in relevant_doc_ids:
                    relevant_in_top_k += 1

            precision_at_k[k] = relevant_in_top_k / k if k > 0 else 0.0

        return PrecisionMetrics(
            precision_at_k=precision_at_k,
            total_relevant=len(relevant_doc_ids),
            total_retrieved=len(documents),
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


def print_benchmark_results(results: dict[str, Any]) -> None:
    """Print formatted benchmark results."""
    print("=" * 80)
    print("CROSS-ENCODER RE-RANKING BENCHMARK RESULTS")
    print("=" * 80)

    print(f"Queries processed: {results['num_successful']}/{results['num_queries']}")

    if results.get("avg_latency_ms"):
        print(f"Average latency: {results['avg_latency_ms']:.2f}ms per query")

    print("\nPRECISION@K RESULTS:")
    print("-" * 40)
    print("K\tBaseline\tReranked\tImprovement")
    print("-" * 40)

    for k in sorted(results["avg_baseline_precision"].keys()):
        baseline = results["avg_baseline_precision"][k]
        reranked = results["avg_reranked_precision"][k]
        improvement = results["avg_improvement"][k]

        print(f"{k}\t{baseline:.3f}\t\t{reranked:.3f}\t\t{improvement:+.3f}")

    print("\nMODEL INFO:")
    print("-" * 20)
    reranker_info = results["reranker_info"]
    print(f"Model: {reranker_info['model_name']}")
    print(f"Device: {reranker_info['device']}")
    print(f"Batch size: {reranker_info['batch_size']}")
    print(f"Model loaded: {reranker_info['model_loaded']}")

    print("=" * 80)
