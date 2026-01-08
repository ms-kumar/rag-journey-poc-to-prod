"""
Benchmark fusion methods and demonstrate recall uplift.

This script shows how fusion (RRF and weighted) improves retrieval quality
by combining results from multiple search methods.
"""

from langchain_core.documents import Document

from src.services.vectorstore.fusion import fuse_results, FusionConfig
from src.services.vectorstore.fusion_eval import (
    calculate_uplift,
    evaluate_retrieval,
)


def main():
    """Run fusion benchmark and show recall uplift."""
    print("🔍 Fusion Orchestration Benchmark\n")
    print("=" * 70)

    # Ground truth: relevant documents for the query "machine learning basics"
    relevant_docs = {"doc1", "doc2", "doc3", "doc5"}

    print(f"\n📚 Ground Truth: {len(relevant_docs)} relevant documents")
    print(f"   Relevant IDs: {sorted(relevant_docs)}\n")

    # Simulate results from different search methods
    # Each has some relevant docs, but none captures all

    # Vector search: Good at semantic similarity
    vector_results = [
        Document(
            page_content="Introduction to machine learning",
            metadata={"chunk_id": "doc1", "score": 0.95},
        ),
        Document(
            page_content="Deep learning tutorial",
            metadata={"chunk_id": "doc2", "score": 0.87},
        ),
        Document(
            page_content="Computer vision basics",  # Not relevant
            metadata={"chunk_id": "doc99", "score": 0.82},
        ),
        Document(
            page_content="Natural language processing",
            metadata={"chunk_id": "doc3", "score": 0.79},
        ),
        Document(
            page_content="Data structures",  # Not relevant
            metadata={"chunk_id": "doc98", "score": 0.75},
        ),
    ]

    # BM25 search: Good at keyword matching
    bm25_results = [
        Document(
            page_content="Machine learning fundamentals",
            metadata={"chunk_id": "doc5", "score": 18.5},
        ),
        Document(
            page_content="Introduction to machine learning",
            metadata={"chunk_id": "doc1", "score": 16.2},
        ),
        Document(
            page_content="Software engineering",  # Not relevant
            metadata={"chunk_id": "doc97", "score": 14.8},
        ),
        Document(
            page_content="Natural language processing",
            metadata={"chunk_id": "doc3", "score": 13.1},
        ),
        Document(
            page_content="Database design",  # Not relevant
            metadata={"chunk_id": "doc96", "score": 11.5},
        ),
    ]

    # Sparse search: Good at exact keyword matching
    sparse_results = [
        Document(
            page_content="Deep learning tutorial",
            metadata={"chunk_id": "doc2", "score": 0.93},
        ),
        Document(
            page_content="Machine learning fundamentals",
            metadata={"chunk_id": "doc5", "score": 0.88},
        ),
        Document(
            page_content="Cloud computing",  # Not relevant
            metadata={"chunk_id": "doc95", "score": 0.81},
        ),
        Document(
            page_content="Introduction to machine learning",
            metadata={"chunk_id": "doc1", "score": 0.76},
        ),
        Document(
            page_content="Web development",  # Not relevant
            metadata={"chunk_id": "doc94", "score": 0.72},
        ),
    ]

    # Evaluate individual methods
    print("📊 Individual Method Performance:")
    print("-" * 70)

    baseline_results = {
        "vector": vector_results,
        "bm25": bm25_results,
        "sparse": sparse_results,
    }

    for method, docs in baseline_results.items():
        metrics = evaluate_retrieval(docs, relevant_docs)
        print(f"\n{method.upper()}:")
        print(f"  Recall@5: {metrics.recall_at_k[5]:.2%}")
        print(f"  Precision@5: {metrics.precision_at_k[5]:.2%}")
        print(f"  MRR: {metrics.mrr:.3f}")
        print(f"  MAP: {metrics.map:.3f}")
        print(f"  NDCG: {metrics.ndcg:.3f}")

    # Test RRF fusion
    print("\n\n🔄 Reciprocal Rank Fusion (RRF)")
    print("-" * 70)

    rrf_config = FusionConfig(method="rrf", rrf_k=60, tie_break_strategy="score")
    rrf_result = fuse_results(
        {"vector": vector_results, "bm25": bm25_results, "sparse": sparse_results},
        config=rrf_config,
    )

    rrf_metrics = evaluate_retrieval(rrf_result.documents, relevant_docs)
    print(f"\nRRF Results:")
    print(f"  Recall@5: {rrf_metrics.recall_at_k[5]:.2%}")
    print(f"  Precision@5: {rrf_metrics.precision_at_k[5]:.2%}")
    print(f"  MRR: {rrf_metrics.mrr:.3f}")
    print(f"  MAP: {rrf_metrics.map:.3f}")
    print(f"  NDCG: {rrf_metrics.ndcg:.3f}")

    # Calculate uplift
    rrf_uplift = calculate_uplift(
        rrf_result.documents, baseline_results, relevant_docs, k_values=[5]
    )

    print(f"\n📈 RRF Uplift over baselines:")
    for method, uplift_dict in rrf_uplift.recall_uplift.items():
        uplift_pct = uplift_dict[5]
        print(f"  vs {method}: +{uplift_pct:.1f}%")

    print(
        f"\n  Best baseline recall@5: {rrf_uplift.best_baseline_recall[5]:.2%}"
    )
    print(
        f"  RRF uplift over best: +{rrf_uplift.uplift_over_best[5]:.1f}%"
    )

    # Test weighted fusion
    print("\n\n⚖️  Weighted Fusion")
    print("-" * 70)

    weighted_config = FusionConfig(
        method="weighted",
        weights={"vector": 0.5, "bm25": 0.3, "sparse": 0.2},
        normalize_scores=True,
        tie_break_strategy="score",
    )
    weighted_result = fuse_results(
        {"vector": vector_results, "bm25": bm25_results, "sparse": sparse_results},
        config=weighted_config,
    )

    weighted_metrics = evaluate_retrieval(weighted_result.documents, relevant_docs)
    print(f"\nWeighted Fusion Results:")
    print(f"  Weights: vector=0.5, bm25=0.3, sparse=0.2")
    print(f"  Recall@5: {weighted_metrics.recall_at_k[5]:.2%}")
    print(f"  Precision@5: {weighted_metrics.precision_at_k[5]:.2%}")
    print(f"  MRR: {weighted_metrics.mrr:.3f}")
    print(f"  MAP: {weighted_metrics.map:.3f}")
    print(f"  NDCG: {weighted_metrics.ndcg:.3f}")

    weighted_uplift = calculate_uplift(
        weighted_result.documents, baseline_results, relevant_docs, k_values=[5]
    )

    print(f"\n📈 Weighted Fusion Uplift over baselines:")
    for method, uplift_dict in weighted_uplift.recall_uplift.items():
        uplift_pct = uplift_dict[5]
        print(f"  vs {method}: +{uplift_pct:.1f}%")

    print(
        f"\n  Best baseline recall@5: {weighted_uplift.best_baseline_recall[5]:.2%}"
    )
    print(
        f"  Weighted uplift over best: +{weighted_uplift.uplift_over_best[5]:.1f}%"
    )

    # Compare fusion methods
    print("\n\n🏆 Fusion Method Comparison")
    print("-" * 70)
    print(f"{'Method':<20} {'Recall@5':<12} {'Precision@5':<12} {'MRR':<8}")
    print("-" * 70)

    for name, metrics in [
        ("Vector (baseline)", evaluate_retrieval(vector_results, relevant_docs)),
        ("BM25 (baseline)", evaluate_retrieval(bm25_results, relevant_docs)),
        ("Sparse (baseline)", evaluate_retrieval(sparse_results, relevant_docs)),
        ("RRF Fusion", rrf_metrics),
        ("Weighted Fusion", weighted_metrics),
    ]:
        print(
            f"{name:<20} {metrics.recall_at_k[5]:.2%}        "
            f"{metrics.precision_at_k[5]:.2%}        {metrics.mrr:.3f}"
        )

    print("\n" + "=" * 70)
    print("\n✨ Key Insights:")
    print("   • Fusion combines strengths of multiple search methods")
    print("   • RRF is simple and effective (no score normalization needed)")
    print("   • Weighted fusion allows tuning method importance")
    print(
        f"   • Typical recall uplift: {min(rrf_uplift.uplift_over_best[5], weighted_uplift.uplift_over_best[5]):.0f}%-"
        f"{max(rrf_uplift.uplift_over_best[5], weighted_uplift.uplift_over_best[5]):.0f}% over best baseline"
    )
    print()


if __name__ == "__main__":
    main()
