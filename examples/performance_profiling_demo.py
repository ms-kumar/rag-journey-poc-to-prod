"""
Performance profiling example.

Demonstrates:
1. Using PerformanceProfiler with timers
2. Tracking multiple operations
3. Generating performance reports
4. Checking SLA compliance
"""

import time
from pathlib import Path

from src.services.performance import (
    PerformanceProfiler,
    PerformanceReporter,
    SLAConfig,
)


def simulate_embedding(text: str, profiler: PerformanceProfiler) -> list[float]:
    """Simulate embedding generation."""
    with profiler.timer("embedding", metadata={"text_length": len(text)}):
        # Simulate variable latency based on text length
        time.sleep(0.01 + len(text) * 0.0001)
        return [0.1] * 384  # Mock embedding


def simulate_retrieval(query_embedding: list[float], profiler: PerformanceProfiler) -> list[dict]:
    """Simulate vector search."""
    with profiler.timer("retrieval", metadata={"k": 5}):
        time.sleep(0.02 + (hash(str(query_embedding)) % 30) / 1000.0)
        return [
            {"id": f"doc_{i}", "score": 0.9 - i * 0.1}
            for i in range(5)
        ]


def simulate_reranking(docs: list[dict], profiler: PerformanceProfiler) -> list[dict]:
    """Simulate reranking."""
    with profiler.timer("reranking", metadata={"num_docs": len(docs)}):
        time.sleep(0.01 + len(docs) * 0.002)
        return sorted(docs, key=lambda x: x["score"], reverse=True)


def simulate_generation(context: str, profiler: PerformanceProfiler) -> str:
    """Simulate LLM generation."""
    with profiler.timer("generation", metadata={"context_length": len(context)}):
        time.sleep(0.1 + len(context) * 0.0001)
        return f"Generated response based on {len(context)} chars"


def process_query(query: str, profiler: PerformanceProfiler) -> dict:
    """Process a complete RAG query."""
    with profiler.timer("end_to_end"):
        # Step 1: Embedding
        embedding = simulate_embedding(query, profiler)

        # Step 2: Retrieval
        docs = simulate_retrieval(embedding, profiler)

        # Step 3: Reranking
        reranked_docs = simulate_reranking(docs, profiler)

        # Step 4: Generation
        context = " ".join([d["id"] for d in reranked_docs])
        answer = simulate_generation(context, profiler)

        return {"query": query, "answer": answer, "docs": reranked_docs}


def main():
    """Run performance profiling demo."""
    print("\n" + "=" * 80)
    print("PERFORMANCE PROFILING DEMO")
    print("=" * 80)

    # Configure SLA thresholds
    sla_config = SLAConfig(
        max_p50_latency_ms=150.0,  # 150ms median
        max_p95_latency_ms=300.0,  # 300ms p95
        max_p99_latency_ms=500.0,  # 500ms p99
        min_throughput_rps=10.0,  # 10 queries/second
        min_success_rate=0.95,  # 95% success rate
        operation_slas={
            "embedding": {"max_p95": 50.0},
            "retrieval": {"max_p95": 100.0},
            "reranking": {"max_p95": 30.0},
            "generation": {"max_p95": 200.0},
        },
    )

    # Initialize profiler
    profiler = PerformanceProfiler(sla_config=sla_config)
    reporter = PerformanceReporter()

    # Test queries
    queries = [
        "What is machine learning?",
        "Explain neural networks",
        "How does gradient descent work?",
        "What are transformers in NLP?",
        "Describe reinforcement learning",
        "What is the attention mechanism?",
        "Explain backpropagation algorithm",
        "How do CNNs work for images?",
        "What is transfer learning?",
        "Describe GANs and their applications",
    ] * 5  # 50 queries total

    print(f"\nProcessing {len(queries)} queries...")

    # Process queries with performance tracking
    results = []
    for i, query in enumerate(queries, 1):
        try:
            result = process_query(query, profiler)
            results.append(result)

            if i % 10 == 0:
                print(f"  Processed {i}/{len(queries)} queries...")

        except Exception as e:
            print(f"  Error on query {i}: {e}")
            # Mark as failure in metrics
            with profiler.timer("end_to_end") as timer:
                timer.mark_failure()

    print(f"\n✓ Completed {len(results)}/{len(queries)} queries")

    # Display results
    print("\n" + "-" * 80)
    print("PERFORMANCE SUMMARY")
    print("-" * 80)

    stats = profiler.get_all_stats()
    reporter.print_summary(stats)

    # Check SLA compliance
    print("\n" + "-" * 80)
    print("SLA COMPLIANCE")
    print("-" * 80)

    sla_result = profiler.check_sla()
    reporter.print_sla_result(sla_result)

    # Export reports
    output_dir = Path("reports/performance")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_name = f"profiling_demo_{timestamp}"

    reporter.export_json(stats, output_dir / f"{base_name}.json")
    reporter.export_markdown(stats, output_dir / f"{base_name}.md")
    reporter.export_html(stats, output_dir / f"{base_name}.html")

    print(f"\n✓ Reports exported to {output_dir}/")
    print(f"  - {base_name}.json")
    print(f"  - {base_name}.md")
    print(f"  - {base_name}.html")

    # Show profiler repr
    print(f"\nProfiler status: {profiler}")

    print("\n" + "=" * 80)
    print("Demo completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
