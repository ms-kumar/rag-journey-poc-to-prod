"""
Verification script for evaluation harness.

Quick checks that all components are importable and functional.
"""

print("🔍 Verifying Evaluation Harness Components...")
print("=" * 60)

# Test imports
try:
    from src.services.evaluation import (
        EvalDataset,
        EvalExample,
        EvaluationHarness,
        RAGMetrics,
        MetricType,
    )
    from src.services.evaluation.metrics import MetricsCalculator
    from src.services.evaluation.harness import ThresholdConfig

    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    exit(1)

# Test MetricsCalculator
print("\n📊 Testing MetricsCalculator...")
calc = MetricsCalculator()

# Test precision@k
retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
relevant = {"doc1", "doc3", "doc5"}
precision = calc.precision_at_k(retrieved, relevant, k=5)
assert precision == 0.6, f"Expected 0.6, got {precision}"
print(f"  ✓ Precision@5: {precision:.3f}")

# Test recall@k
recall = calc.recall_at_k(retrieved, relevant, k=5)
assert recall == 1.0, f"Expected 1.0, got {recall}"
print(f"  ✓ Recall@5: {recall:.3f}")

# Test MRR
mrr = calc.mean_reciprocal_rank(retrieved, relevant)
assert mrr == 1.0, f"Expected 1.0, got {mrr}"
print(f"  ✓ MRR: {mrr:.3f}")

# Test NDCG
ndcg = calc.ndcg_at_k(retrieved, relevant, k=5)
assert 0.8 < ndcg <= 1.0, f"Expected ~0.88, got {ndcg}"
print(f"  ✓ NDCG@5: {ndcg:.3f}")

print("✅ MetricsCalculator works correctly")

# Test EvalDataset
print("\n📝 Testing EvalDataset...")
dataset = EvalDataset(name="test", description="Test dataset")

dataset.add_example(
    query="Test query 1", relevant_doc_ids=["doc1", "doc2"], metadata={"test": True}
)

dataset.add_example(
    query="Test query 2",
    relevant_doc_ids=["doc3"],
    expected_answer="Test answer",
)

assert len(dataset) == 2, f"Expected 2 examples, got {len(dataset)}"
print(f"  ✓ Created dataset with {len(dataset)} examples")

stats = dataset.get_statistics()
assert stats["num_examples"] == 2
print(f"  ✓ Statistics: {stats}")

print("✅ EvalDataset works correctly")

# Test RAGMetrics
print("\n📈 Testing RAGMetrics...")
metrics = RAGMetrics(
    precision_at_k={5: 0.8, 10: 0.7},
    recall_at_k={10: 0.75},
    mrr=0.85,
    ndcg_at_k={10: 0.82},
    mean_average_precision=0.78,
    num_queries=10,
)

metrics_dict = metrics.to_dict()
assert "retrieval" in metrics_dict
assert "performance" in metrics_dict
print(f"  ✓ Metrics to dict: {list(metrics_dict.keys())}")

summary = metrics.get_summary()
assert "RAG Evaluation Metrics" in summary
print(f"  ✓ Summary generated ({len(summary)} chars)")

print("✅ RAGMetrics works correctly")

# Test ThresholdConfig
print("\n🎯 Testing ThresholdConfig...")
thresholds = ThresholdConfig(
    min_precision_at_5=0.7,
    min_recall_at_10=0.8,
    min_mrr=0.6,
    max_latency_p95=2000.0,
)

threshold_dict = thresholds.to_dict()
assert "retrieval" in threshold_dict
assert "performance" in threshold_dict
print(f"  ✓ Thresholds configured: {list(threshold_dict.keys())}")

print("✅ ThresholdConfig works correctly")

# Summary
print("\n" + "=" * 60)
print("✅ All verification checks passed!")
print("=" * 60)
print("\nEvaluation harness is ready to use:")
print("  1. Create datasets: make eval-datasets")
print("  2. Run evaluation: make eval")
print("  3. Generate dashboard: make dashboard")
print("  4. Run demo: python examples/evaluation_demo.py")
