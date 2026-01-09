"""
Main evaluation harness for RAG system.

Orchestrates evaluation runs, metric calculation, and result reporting.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from .dataset import EvalDataset, EvalExample
from .metrics import MetricsCalculator, RAGMetrics

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Result from a single evaluation run."""

    metrics: RAGMetrics
    passed: bool
    failed_checks: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    duration_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "metrics": self.metrics.to_dict(),
            "passed": self.passed,
            "failed_checks": self.failed_checks,
            "timestamp": self.timestamp,
            "duration_seconds": self.duration_seconds,
            "metadata": self.metadata,
        }


@dataclass
class ThresholdConfig:
    """Configuration for metric thresholds."""

    # Retrieval thresholds
    min_precision_at_5: float = 0.6
    min_recall_at_10: float = 0.7
    min_mrr: float = 0.5
    min_ndcg_at_10: float = 0.65
    min_map: float = 0.6

    # Generation thresholds
    min_faithfulness: float = 0.8
    min_relevance: float = 0.75
    min_answer_quality: float = 0.7

    # Performance thresholds
    max_latency_p95: float = 2000.0  # milliseconds

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "retrieval": {
                "min_precision@5": self.min_precision_at_5,
                "min_recall@10": self.min_recall_at_10,
                "min_mrr": self.min_mrr,
                "min_ndcg@10": self.min_ndcg_at_10,
                "min_map": self.min_map,
            },
            "generation": {
                "min_faithfulness": self.min_faithfulness,
                "min_relevance": self.min_relevance,
                "min_answer_quality": self.min_answer_quality,
            },
            "performance": {
                "max_latency_p95_ms": self.max_latency_p95,
            },
        }


class EvaluationHarness:
    """
    Main evaluation harness for RAG system.

    Orchestrates evaluation runs with metrics calculation and threshold checking.
    """

    def __init__(
        self,
        retrieval_function: Any,  # Function that takes query and returns documents
        generation_function: Any | None = None,  # Function that takes query + docs and returns answer
        thresholds: ThresholdConfig | None = None,
    ):
        """
        Initialize evaluation harness.

        Args:
            retrieval_function: Function to retrieve documents for a query
            generation_function: Optional function to generate answers
            thresholds: Optional threshold configuration
        """
        self.retrieval_function = retrieval_function
        self.generation_function = generation_function
        self.thresholds = thresholds or ThresholdConfig()
        self.calculator = MetricsCalculator()

    def evaluate(
        self,
        dataset: EvalDataset,
        k_values: list[int] | None = None,
        include_generation: bool = False,
    ) -> EvalResult:
        """
        Run evaluation on a dataset.

        Args:
            dataset: Evaluation dataset
            k_values: List of k values for precision@k and recall@k
            include_generation: Whether to evaluate generation metrics

        Returns:
            EvalResult with metrics and pass/fail status
        """
        k_values = k_values or [1, 3, 5, 10, 20]
        start_time = time.perf_counter()

        logger.info(f"Starting evaluation on {len(dataset)} examples...")

        # Initialize metric accumulators
        precision_sums = {k: 0.0 for k in k_values}
        recall_sums = {k: 0.0 for k in k_values}
        ndcg_sums = {k: 0.0 for k in k_values}
        mrr_sum = 0.0
        ap_sum = 0.0
        latencies = []

        faithfulness_scores = []
        relevance_scores = []
        quality_scores = []

        # Evaluate each example
        for i, example in enumerate(dataset.examples):
            logger.debug(f"Evaluating example {i+1}/{len(dataset)}: {example.query}")

            # Time the retrieval
            query_start = time.perf_counter()
            retrieved_docs = self.retrieval_function(example.query)
            query_latency = (time.perf_counter() - query_start) * 1000  # Convert to ms
            latencies.append(query_latency)

            # Extract document IDs
            retrieved_ids = [
                doc.metadata.get("id", f"doc_{i}") for i, doc in enumerate(retrieved_docs)
            ]
            relevant_ids = set(example.relevant_doc_ids)

            # Calculate retrieval metrics
            for k in k_values:
                precision = self.calculator.precision_at_k(retrieved_ids, relevant_ids, k)
                recall = self.calculator.recall_at_k(retrieved_ids, relevant_ids, k)
                ndcg = self.calculator.ndcg_at_k(retrieved_ids, relevant_ids, k)

                precision_sums[k] += precision
                recall_sums[k] += recall
                ndcg_sums[k] += ndcg

            # Calculate MRR and MAP
            mrr = self.calculator.mean_reciprocal_rank(retrieved_ids, relevant_ids)
            ap = self.calculator.average_precision(retrieved_ids, relevant_ids)
            mrr_sum += mrr
            ap_sum += ap

            # Evaluate generation if requested
            if include_generation and self.generation_function:
                try:
                    generated_answer = self.generation_function(
                        example.query, retrieved_docs
                    )

                    # Calculate generation metrics (simplified - in production use LLM-as-judge)
                    faithfulness = self._evaluate_faithfulness(
                        generated_answer, retrieved_docs
                    )
                    relevance = self._evaluate_relevance(
                        example.query, generated_answer
                    )
                    quality = self._evaluate_quality(generated_answer)

                    faithfulness_scores.append(faithfulness)
                    relevance_scores.append(relevance)
                    quality_scores.append(quality)
                except Exception as e:
                    logger.warning(f"Generation evaluation failed for query: {e}")

        # Aggregate metrics
        num_examples = len(dataset)
        metrics = RAGMetrics(
            precision_at_k={k: precision_sums[k] / num_examples for k in k_values},
            recall_at_k={k: recall_sums[k] / num_examples for k in k_values},
            mrr=mrr_sum / num_examples,
            ndcg_at_k={k: ndcg_sums[k] / num_examples for k in k_values},
            mean_average_precision=ap_sum / num_examples,
            faithfulness=sum(faithfulness_scores) / len(faithfulness_scores)
            if faithfulness_scores
            else 0.0,
            relevance=sum(relevance_scores) / len(relevance_scores)
            if relevance_scores
            else 0.0,
            answer_quality=sum(quality_scores) / len(quality_scores)
            if quality_scores
            else 0.0,
            num_queries=num_examples,
            timestamp=datetime.now().isoformat(),
        )

        # Add latency metrics
        latency_percentiles = self.calculator.calculate_latency_percentiles(latencies)
        metrics.latency_p50 = latency_percentiles["p50"]
        metrics.latency_p95 = latency_percentiles["p95"]
        metrics.latency_p99 = latency_percentiles["p99"]

        # Check thresholds
        passed, failed_checks = self._check_thresholds(metrics)

        duration = time.perf_counter() - start_time
        result = EvalResult(
            metrics=metrics,
            passed=passed,
            failed_checks=failed_checks,
            duration_seconds=duration,
            metadata={
                "dataset_name": dataset.name,
                "dataset_size": len(dataset),
                "k_values": k_values,
                "include_generation": include_generation,
            },
        )

        logger.info(f"Evaluation completed in {duration:.2f}s")
        logger.info(f"Result: {'PASSED' if passed else 'FAILED'}")
        if failed_checks:
            logger.warning(f"Failed checks: {', '.join(failed_checks)}")

        return result

    def _check_thresholds(self, metrics: RAGMetrics) -> tuple[bool, list[str]]:
        """
        Check if metrics meet thresholds.

        Args:
            metrics: Calculated metrics

        Returns:
            Tuple of (passed, list of failed checks)
        """
        failed_checks = []

        # Check retrieval thresholds
        if metrics.precision_at_k.get(5, 0.0) < self.thresholds.min_precision_at_5:
            failed_checks.append(
                f"Precision@5 {metrics.precision_at_k.get(5, 0.0):.3f} "
                f"< {self.thresholds.min_precision_at_5}"
            )

        if metrics.recall_at_k.get(10, 0.0) < self.thresholds.min_recall_at_10:
            failed_checks.append(
                f"Recall@10 {metrics.recall_at_k.get(10, 0.0):.3f} "
                f"< {self.thresholds.min_recall_at_10}"
            )

        if metrics.mrr < self.thresholds.min_mrr:
            failed_checks.append(
                f"MRR {metrics.mrr:.3f} < {self.thresholds.min_mrr}"
            )

        if metrics.ndcg_at_k.get(10, 0.0) < self.thresholds.min_ndcg_at_10:
            failed_checks.append(
                f"NDCG@10 {metrics.ndcg_at_k.get(10, 0.0):.3f} "
                f"< {self.thresholds.min_ndcg_at_10}"
            )

        if metrics.mean_average_precision < self.thresholds.min_map:
            failed_checks.append(
                f"MAP {metrics.mean_average_precision:.3f} < {self.thresholds.min_map}"
            )

        # Check generation thresholds
        if metrics.faithfulness > 0 and metrics.faithfulness < self.thresholds.min_faithfulness:
            failed_checks.append(
                f"Faithfulness {metrics.faithfulness:.3f} "
                f"< {self.thresholds.min_faithfulness}"
            )

        if metrics.relevance > 0 and metrics.relevance < self.thresholds.min_relevance:
            failed_checks.append(
                f"Relevance {metrics.relevance:.3f} < {self.thresholds.min_relevance}"
            )

        if metrics.answer_quality > 0 and metrics.answer_quality < self.thresholds.min_answer_quality:
            failed_checks.append(
                f"Answer Quality {metrics.answer_quality:.3f} "
                f"< {self.thresholds.min_answer_quality}"
            )

        # Check performance thresholds
        if metrics.latency_p95 > self.thresholds.max_latency_p95:
            failed_checks.append(
                f"Latency P95 {metrics.latency_p95:.1f}ms "
                f"> {self.thresholds.max_latency_p95}ms"
            )

        passed = len(failed_checks) == 0
        return passed, failed_checks

    def _evaluate_faithfulness(
        self, answer: str, context_docs: list[Document]
    ) -> float:
        """
        Evaluate if answer is faithful to the context.

        This is a simplified implementation. In production, use:
        - LLM-as-judge (GPT-4, Claude)
        - NLI models (entailment checking)
        - Fact verification models

        Args:
            answer: Generated answer
            context_docs: Retrieved context documents

        Returns:
            Faithfulness score (0-1)
        """
        # Placeholder: Check if answer contains content from context
        context_text = " ".join([doc.page_content for doc in context_docs])
        answer_words = set(answer.lower().split())
        context_words = set(context_text.lower().split())

        if not answer_words:
            return 0.0

        overlap = len(answer_words & context_words) / len(answer_words)
        return min(overlap * 1.2, 1.0)  # Scale up slightly

    def _evaluate_relevance(self, query: str, answer: str) -> float:
        """
        Evaluate if answer is relevant to the query.

        Simplified implementation. In production, use semantic similarity.

        Args:
            query: Original query
            answer: Generated answer

        Returns:
            Relevance score (0-1)
        """
        # Placeholder: Simple keyword overlap
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())

        if not query_words or not answer_words:
            return 0.0

        overlap = len(query_words & answer_words) / len(query_words)
        return min(overlap * 1.5, 1.0)  # Scale up

    def _evaluate_quality(self, answer: str) -> float:
        """
        Evaluate overall answer quality.

        Simplified implementation. In production, use:
        - LLM-as-judge for quality assessment
        - Readability metrics
        - Coherence models

        Args:
            answer: Generated answer

        Returns:
            Quality score (0-1)
        """
        # Placeholder: Basic heuristics
        if not answer or len(answer) < 10:
            return 0.0

        # Check for reasonable length (not too short or too long)
        word_count = len(answer.split())
        if word_count < 5:
            return 0.3
        elif word_count < 20:
            return 0.6
        elif word_count < 200:
            return 0.85
        else:
            return 0.7  # Very long answers might be verbose

    def save_results(self, result: EvalResult, filepath: Path | str) -> None:
        """
        Save evaluation results to file.

        Args:
            result: Evaluation result to save
            filepath: Path to save results
        """
        import json

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.info(f"Saved evaluation results to {filepath}")

    def compare_results(
        self, baseline_result: EvalResult, current_result: EvalResult
    ) -> dict[str, Any]:
        """
        Compare two evaluation results.

        Args:
            baseline_result: Baseline evaluation result
            current_result: Current evaluation result

        Returns:
            Comparison dictionary with deltas
        """
        baseline = baseline_result.metrics
        current = current_result.metrics

        comparison = {
            "retrieval": {
                "precision@5": {
                    "baseline": baseline.precision_at_k.get(5, 0.0),
                    "current": current.precision_at_k.get(5, 0.0),
                    "delta": current.precision_at_k.get(5, 0.0)
                    - baseline.precision_at_k.get(5, 0.0),
                },
                "recall@10": {
                    "baseline": baseline.recall_at_k.get(10, 0.0),
                    "current": current.recall_at_k.get(10, 0.0),
                    "delta": current.recall_at_k.get(10, 0.0)
                    - baseline.recall_at_k.get(10, 0.0),
                },
                "mrr": {
                    "baseline": baseline.mrr,
                    "current": current.mrr,
                    "delta": current.mrr - baseline.mrr,
                },
            },
            "performance": {
                "latency_p95": {
                    "baseline": baseline.latency_p95,
                    "current": current.latency_p95,
                    "delta": current.latency_p95 - baseline.latency_p95,
                },
            },
            "status": {
                "baseline_passed": baseline_result.passed,
                "current_passed": current_result.passed,
                "regression": baseline_result.passed and not current_result.passed,
            },
        }

        return comparison
