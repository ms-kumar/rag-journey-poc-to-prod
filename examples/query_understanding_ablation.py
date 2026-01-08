"""
Ablation study for query understanding components.

Measures the impact of each component (rewriting, synonyms, intent)
on retrieval quality and latency.
"""

import time
from typing import Any

from src.services.query_understanding import (
    QueryUnderstanding,
    QueryUnderstandingConfig,
)


def run_ablation_study(test_queries: list[str]) -> dict[str, Any]:
    """
    Run ablation study on query understanding components.

    Tests configurations:
    1. Baseline (no processing)
    2. Rewriting only
    3. Synonyms only
    4. Rewriting + Synonyms
    5. Full pipeline (with intent classification)

    Args:
        test_queries: List of test queries

    Returns:
        Dictionary with results for each configuration
    """
    print("=" * 80)
    print("QUERY UNDERSTANDING ABLATION STUDY")
    print("=" * 80)
    print(f"\nTest queries: {len(test_queries)}")
    print()

    configurations = {
        "baseline": QueryUnderstandingConfig(
            enable_rewriting=False,
            enable_synonyms=False,
            enable_intent_classification=False,
        ),
        "rewriting_only": QueryUnderstandingConfig(
            enable_rewriting=True,
            enable_synonyms=False,
            enable_intent_classification=False,
        ),
        "synonyms_only": QueryUnderstandingConfig(
            enable_rewriting=False,
            enable_synonyms=True,
            enable_intent_classification=False,
        ),
        "rewriting_and_synonyms": QueryUnderstandingConfig(
            enable_rewriting=True,
            enable_synonyms=True,
            enable_intent_classification=False,
        ),
        "full_pipeline": QueryUnderstandingConfig(
            enable_rewriting=True,
            enable_synonyms=True,
            enable_intent_classification=True,
        ),
    }

    results: dict[str, Any] = {}

    for config_name, config in configurations.items():
        print(f"\n{'=' * 80}")
        print(f"Configuration: {config_name.upper()}")
        print(f"{'=' * 80}")

        qu = QueryUnderstanding(config)
        latencies = []
        query_lengths = []
        expansion_ratios = []

        for query in test_queries:
            start = time.perf_counter()
            result = qu.process(query)
            latency = (time.perf_counter() - start) * 1000

            latencies.append(latency)
            original_len = len(result["original_query"])
            processed_len = len(result["processed_query"])
            query_lengths.append(processed_len)
            expansion_ratios.append(processed_len / original_len if original_len > 0 else 1.0)

            # Print example
            if len(latencies) <= 3:  # Show first 3
                print(f"\nQuery: {query}")
                print(f"  Processed: {result['processed_query']}")
                if result.get("intent"):
                    print(f"  Intent: {result['intent']}")
                print(f"  Latency: {latency:.2f}ms")
                print(f"  Expansion: {expansion_ratios[-1]:.2f}x")

        # Calculate statistics
        avg_latency = sum(latencies) / len(latencies)
        p50_latency = sorted(latencies)[len(latencies) // 2]
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        avg_expansion = sum(expansion_ratios) / len(expansion_ratios)

        results[config_name] = {
            "avg_latency_ms": avg_latency,
            "p50_latency_ms": p50_latency,
            "p95_latency_ms": p95_latency,
            "avg_expansion_ratio": avg_expansion,
            "queries_processed": len(test_queries),
        }

        print(f"\n{'-' * 80}")
        print(f"Statistics for {config_name}:")
        print(f"  Avg latency: {avg_latency:.2f}ms")
        print(f"  P50 latency: {p50_latency:.2f}ms")
        print(f"  P95 latency: {p95_latency:.2f}ms")
        print(f"  Avg expansion: {avg_expansion:.2f}x")

    # Print comparison summary
    print(f"\n{'=' * 80}")
    print("SUMMARY COMPARISON")
    print(f"{'=' * 80}")
    print(f"\n{'Configuration':<25} {'Latency (ms)':<15} {'P95 (ms)':<12} {'Expansion':<12}")
    print(f"{'-' * 64}")

    for config_name, stats in results.items():
        print(
            f"{config_name:<25} "
            f"{stats['avg_latency_ms']:>10.2f}     "
            f"{stats['p95_latency_ms']:>8.2f}    "
            f"{stats['avg_expansion_ratio']:>8.2f}x"
        )

    # Calculate impact
    baseline_latency = results["baseline"]["avg_latency_ms"]
    full_latency = results["full_pipeline"]["avg_latency_ms"]
    latency_overhead = full_latency - baseline_latency
    latency_overhead_pct = (
        (latency_overhead / baseline_latency * 100) if baseline_latency > 0 else 0
    )

    print(f"\n{'=' * 80}")
    print("IMPACT ANALYSIS")
    print(f"{'=' * 80}")
    print("\nLatency overhead (full pipeline vs baseline):")
    print(f"  Absolute: +{latency_overhead:.2f}ms")
    print(f"  Relative: +{latency_overhead_pct:.1f}%")

    print("\nQuery expansion (full pipeline):")
    print(f"  Average: {results['full_pipeline']['avg_expansion_ratio']:.2f}x")
    print(
        f"  Benefit: {(results['full_pipeline']['avg_expansion_ratio'] - 1) * 100:.1f}% longer queries"
    )

    print(f"\n{'=' * 80}")
    print("RECOMMENDATIONS")
    print(f"{'=' * 80}")

    if full_latency < 5:
        print("✅ Full pipeline latency is excellent (< 5ms)")
    elif full_latency < 10:
        print("✓ Full pipeline latency is good (< 10ms)")
    else:
        print("⚠ Full pipeline latency may be noticeable (> 10ms)")

    if results["full_pipeline"]["avg_expansion_ratio"] > 1.5:
        print("✅ Good query expansion (> 1.5x)")
    elif results["full_pipeline"]["avg_expansion_ratio"] > 1.2:
        print("✓ Moderate query expansion (> 1.2x)")
    else:
        print("⚠ Limited query expansion (< 1.2x)")

    return results


if __name__ == "__main__":
    # Test queries covering different types
    test_queries = [
        # Acronyms
        "what is ML?",
        "how to use NLP in AI?",
        "explain RAG system",
        # Typos
        "machien learing algorithim",
        "nueral networ training",
        # Questions
        "what is machine learning?",
        "how to train a model?",
        "why use deep learning?",
        # Technical
        "optimize database query performance",
        "implement REST API authentication",
        "debug Python error handling",
        # Comparison
        "Python vs Java",
        "SQL vs NoSQL database",
        # Short queries
        "ml",
        "api",
        "database",
        # Complex queries
        "how to fix machien learing error in py?",
        "what is the difference between supervised and unsupervised learning?",
        "explain neural network backpropagation algorithm step by step",
    ]

    results = run_ablation_study(test_queries)

    print("\n" + "=" * 80)
    print("Ablation study complete!")
    print("=" * 80)
