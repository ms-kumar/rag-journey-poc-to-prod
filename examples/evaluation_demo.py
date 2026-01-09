"""
Demo script for RAG evaluation harness.

Demonstrates how to use the evaluation framework.
"""

import logging
from pathlib import Path

from langchain_core.documents import Document

from src.services.evaluation import (
    EvalDataset,
    EvaluationHarness,
    RAGMetrics,
    ThresholdConfig,
)
from src.services.evaluation.dataset import DatasetBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def mock_retrieval_function(query: str, k: int = 10):
    """
    Mock retrieval function for demonstration.

    In production, this would be your actual vectorstore.similarity_search().
    """
    # Simulate different retrieval results based on query
    docs = {
        "What is RAG?": [
            Document(page_content="RAG is...", metadata={"id": "doc_rag_basics"}),
            Document(page_content="Vector search...", metadata={"id": "doc_vector"}),
            Document(page_content="Embeddings...", metadata={"id": "doc_embeddings"}),
        ],
        "How to use embeddings?": [
            Document(
                page_content="Embeddings are...", metadata={"id": "doc_embeddings"}
            ),
            Document(
                page_content="Sentence transformers...", metadata={"id": "doc_st"}
            ),
        ],
        "What is BM25?": [
            Document(page_content="BM25 is...", metadata={"id": "doc_bm25"}),
            Document(page_content="Sparse retrieval...", metadata={"id": "doc_sparse"}),
        ],
    }

    # Return docs for the query, or generic docs if not found
    return docs.get(
        query,
        [
            Document(page_content=f"Result for {query}", metadata={"id": f"doc_{i}"})
            for i in range(k)
        ],
    )


def demo_metrics():
    """Demonstrate metric calculations."""
    print("\n" + "=" * 60)
    print("📊 DEMO: Metric Calculations")
    print("=" * 60)

    from src.services.evaluation.metrics import MetricsCalculator

    calc = MetricsCalculator()

    # Example data
    retrieved_ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant_ids = {"doc1", "doc3", "doc5"}

    print("\nRetrieved documents:", retrieved_ids)
    print("Relevant documents:", relevant_ids)

    # Calculate metrics
    precision = calc.precision_at_k(retrieved_ids, relevant_ids, k=5)
    recall = calc.recall_at_k(retrieved_ids, relevant_ids, k=5)
    mrr = calc.mean_reciprocal_rank(retrieved_ids, relevant_ids)
    ndcg = calc.ndcg_at_k(retrieved_ids, relevant_ids, k=5)
    ap = calc.average_precision(retrieved_ids, relevant_ids)

    print(f"\nMetrics:")
    print(f"  Precision@5: {precision:.3f} (3 relevant out of 5)")
    print(f"  Recall@5: {recall:.3f} (3 retrieved out of 3 total)")
    print(f"  MRR: {mrr:.3f} (first relevant at position 1)")
    print(f"  NDCG@5: {ndcg:.3f} (ranking quality)")
    print(f"  AP: {ap:.3f} (average precision)")


def demo_dataset():
    """Demonstrate dataset creation and management."""
    print("\n" + "=" * 60)
    print("📝 DEMO: Dataset Management")
    print("=" * 60)

    # Create dataset
    builder = DatasetBuilder(name="demo_dataset", description="Demo evaluation dataset")

    # Add examples
    builder.dataset.add_example(
        query="What is RAG?",
        relevant_doc_ids=["doc_rag_basics", "doc_rag_intro"],
        expected_answer="RAG is Retrieval-Augmented Generation",
        metadata={"category": "basics", "difficulty": "easy"},
    )

    builder.dataset.add_example(
        query="How to implement embeddings?",
        relevant_doc_ids=["doc_embeddings", "doc_implementation"],
        metadata={"category": "implementation", "difficulty": "medium"},
    )

    builder.dataset.add_example(
        query="Optimize RAG performance",
        relevant_doc_ids=["doc_performance", "doc_optimization"],
        metadata={"category": "optimization", "difficulty": "hard"},
    )

    dataset = builder.build()

    print(f"\nDataset created: {dataset.name}")
    print(f"Description: {dataset.description}")
    print(f"Number of examples: {len(dataset)}")

    # Show statistics
    stats = dataset.get_statistics()
    print(f"\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Save dataset (to temp location)
    temp_path = Path("data/eval/demo_dataset.json")
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save(temp_path)
    print(f"\n✅ Dataset saved to: {temp_path}")

    return dataset


def demo_evaluation(dataset: EvalDataset):
    """Demonstrate running evaluation."""
    print("\n" + "=" * 60)
    print("🎯 DEMO: Running Evaluation")
    print("=" * 60)

    # Configure thresholds
    thresholds = ThresholdConfig(
        min_precision_at_5=0.5,
        min_recall_at_10=0.6,
        min_mrr=0.4,
        max_latency_p95=1000.0,
    )

    print("\nThresholds:")
    for key, value in thresholds.to_dict().items():
        print(f"  {key}: {value}")

    # Create harness
    harness = EvaluationHarness(
        retrieval_function=mock_retrieval_function, thresholds=thresholds
    )

    # Run evaluation
    print(f"\n🔄 Evaluating {len(dataset)} queries...")
    result = harness.evaluate(dataset, k_values=[1, 3, 5, 10])

    # Display results
    print(f"\n" + "=" * 60)
    print(result.metrics.get_summary())
    print("=" * 60)

    if result.passed:
        print("\n✅ EVALUATION PASSED - All thresholds met!")
    else:
        print("\n❌ EVALUATION FAILED - Some thresholds not met:")
        for check in result.failed_checks:
            print(f"  • {check}")

    # Save results
    results_path = Path("results/demo_eval_result.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    harness.save_results(result, results_path)
    print(f"\n💾 Results saved to: {results_path}")

    return result


def demo_comparison():
    """Demonstrate result comparison."""
    print("\n" + "=" * 60)
    print("🔄 DEMO: Result Comparison")
    print("=" * 60)

    # Create two mock results for comparison
    from src.services.evaluation.harness import EvalResult

    baseline_metrics = RAGMetrics(
        precision_at_k={5: 0.6, 10: 0.5},
        recall_at_k={10: 0.7},
        mrr=0.65,
        latency_p95=1200.0,
        num_queries=10,
    )

    current_metrics = RAGMetrics(
        precision_at_k={5: 0.7, 10: 0.6},
        recall_at_k={10: 0.75},
        mrr=0.70,
        latency_p95=1000.0,
        num_queries=10,
    )

    baseline_result = EvalResult(
        metrics=baseline_metrics, passed=True, duration_seconds=5.0
    )

    current_result = EvalResult(
        metrics=current_metrics, passed=True, duration_seconds=4.5
    )

    # Compare
    harness = EvaluationHarness(
        retrieval_function=mock_retrieval_function, thresholds=ThresholdConfig()
    )

    comparison = harness.compare_results(baseline_result, current_result)

    print("\nComparison:")
    print(f"  Precision@5: {comparison['retrieval']['precision@5']['baseline']:.3f} → "
          f"{comparison['retrieval']['precision@5']['current']:.3f} "
          f"(Δ {comparison['retrieval']['precision@5']['delta']:+.3f})")

    print(f"  Recall@10: {comparison['retrieval']['recall@10']['baseline']:.3f} → "
          f"{comparison['retrieval']['recall@10']['current']:.3f} "
          f"(Δ {comparison['retrieval']['recall@10']['delta']:+.3f})")

    print(f"  MRR: {comparison['retrieval']['mrr']['baseline']:.3f} → "
          f"{comparison['retrieval']['mrr']['current']:.3f} "
          f"(Δ {comparison['retrieval']['mrr']['delta']:+.3f})")

    print(
        f"  Latency P95: {comparison['performance']['latency_p95']['baseline']:.1f}ms → "
        f"{comparison['performance']['latency_p95']['current']:.1f}ms "
        f"(Δ {comparison['performance']['latency_p95']['delta']:+.1f}ms)"
    )

    if comparison["status"]["regression"]:
        print("\n⚠️  REGRESSION DETECTED!")
    else:
        print("\n✅ No regression - quality maintained or improved")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("🚀 RAG EVALUATION HARNESS DEMO")
    print("=" * 60)

    # Run demos
    demo_metrics()
    dataset = demo_dataset()
    result = demo_evaluation(dataset)
    demo_comparison()

    print("\n" + "=" * 60)
    print("✅ Demo completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Create your evaluation dataset: python scripts/create_eval_datasets.py")
    print("2. Run evaluation: make eval")
    print("3. Generate dashboard: make dashboard")
    print("4. Set up CI: Check .github/workflows/eval_gate.yml")


if __name__ == "__main__":
    main()
