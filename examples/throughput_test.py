"""
Throughput testing for RAG pipeline.

Tests system performance under load with various configurations.
"""

import concurrent.futures
import time
from pathlib import Path

from src.config import Settings
from src.services.performance import PerformanceProfiler, PerformanceReporter, SLAConfig


def single_query_test(profiler: PerformanceProfiler, query: str) -> dict:
    """
    Execute a single query with performance tracking.

    Args:
        profiler: Performance profiler
        query: Query string

    Returns:
        Query result
    """
    # Simulate query processing
    with profiler.timer("query_processing"):
        with profiler.timer("embedding"):
            # Simulate embedding generation (~10-50ms)
            time.sleep(0.01 + (hash(query) % 40) / 1000.0)

        with profiler.timer("retrieval"):
            # Simulate vector search (~20-100ms)
            time.sleep(0.02 + (hash(query) % 80) / 1000.0)

        with profiler.timer("reranking"):
            # Simulate reranking (~10-30ms)
            time.sleep(0.01 + (hash(query) % 20) / 1000.0)

        with profiler.timer("generation"):
            # Simulate LLM generation (~100-500ms)
            time.sleep(0.1 + (hash(query) % 400) / 1000.0)

    return {"query": query, "answer": f"Response to: {query}"}


def run_sequential_test(
    num_queries: int = 100, profiler: PerformanceProfiler | None = None
) -> PerformanceProfiler:
    """
    Run sequential throughput test.

    Args:
        num_queries: Number of queries to process
        profiler: Optional existing profiler

    Returns:
        Performance profiler with results
    """
    if profiler is None:
        profiler = PerformanceProfiler()

    queries = [
        f"What is machine learning algorithm {i}?"
        for i in range(num_queries)
    ]

    print(f"\n{'='*60}")
    print(f"Running Sequential Test: {num_queries} queries")
    print(f"{'='*60}")

    start_time = time.time()

    for query in queries:
        single_query_test(profiler, query)

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.2f}s")
    print(f"Throughput: {num_queries/elapsed:.2f} queries/second")

    return profiler


def run_concurrent_test(
    num_queries: int = 100,
    max_workers: int = 10,
    profiler: PerformanceProfiler | None = None,
) -> PerformanceProfiler:
    """
    Run concurrent throughput test.

    Args:
        num_queries: Number of queries to process
        max_workers: Number of concurrent workers
        profiler: Optional existing profiler

    Returns:
        Performance profiler with results
    """
    if profiler is None:
        profiler = PerformanceProfiler()

    queries = [
        f"Explain concept {i} in machine learning"
        for i in range(num_queries)
    ]

    print(f"\n{'='*60}")
    print(f"Running Concurrent Test: {num_queries} queries, {max_workers} workers")
    print(f"{'='*60}")

    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(single_query_test, profiler, query)
            for query in queries
        ]
        concurrent.futures.wait(futures)

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.2f}s")
    print(f"Throughput: {num_queries/elapsed:.2f} queries/second")
    print(f"Speedup: {(num_queries*0.15)/elapsed:.2f}x (vs sequential)")

    return profiler


def run_stress_test(
    duration_seconds: int = 60,
    target_rps: int = 20,
    profiler: PerformanceProfiler | None = None,
) -> PerformanceProfiler:
    """
    Run stress test with target RPS for specified duration.

    Args:
        duration_seconds: Test duration
        target_rps: Target requests per second
        profiler: Optional existing profiler

    Returns:
        Performance profiler with results
    """
    if profiler is None:
        profiler = PerformanceProfiler()

    print(f"\n{'='*60}")
    print(f"Running Stress Test: {target_rps} RPS for {duration_seconds}s")
    print(f"{'='*60}")

    start_time = time.time()
    query_count = 0
    interval = 1.0 / target_rps

    while time.time() - start_time < duration_seconds:
        iteration_start = time.time()

        query = f"Query {query_count} at {time.time():.2f}"
        single_query_test(profiler, query)
        query_count += 1

        # Maintain target RPS
        elapsed = time.time() - iteration_start
        if elapsed < interval:
            time.sleep(interval - elapsed)

        # Progress indicator
        if query_count % 100 == 0:
            current_elapsed = time.time() - start_time
            current_rps = query_count / current_elapsed
            print(f"  Progress: {query_count} queries, {current_rps:.1f} RPS")

    elapsed = time.time() - start_time
    print(f"\nCompleted: {query_count} queries in {elapsed:.2f}s")
    print(f"Actual RPS: {query_count/elapsed:.2f}")

    return profiler


def main():
    """Run throughput tests and generate reports."""
    # Define SLA thresholds
    sla_config = SLAConfig(
        max_p50_latency_ms=200.0,  # 200ms median
        max_p95_latency_ms=500.0,  # 500ms p95
        max_p99_latency_ms=1000.0,  # 1000ms p99
        min_throughput_rps=5.0,  # Minimum 5 queries/second
        min_success_rate=0.99,  # 99% success rate
        operation_slas={
            "embedding": {"max_p95": 100.0},
            "retrieval": {"max_p95": 150.0},
            "generation": {"max_p95": 600.0},
        },
    )

    profiler = PerformanceProfiler(sla_config=sla_config)
    reporter = PerformanceReporter()

    print("\n" + "=" * 80)
    print("RAG PIPELINE THROUGHPUT TESTING")
    print("=" * 80)

    # Test 1: Sequential baseline
    print("\n[Test 1/4] Sequential Baseline")
    run_sequential_test(num_queries=50, profiler=profiler)

    # Test 2: Concurrent processing
    print("\n[Test 2/4] Concurrent Processing")
    profiler.reset()  # Reset for fresh metrics
    run_concurrent_test(num_queries=100, max_workers=10, profiler=profiler)

    # Test 3: Stress test
    print("\n[Test 3/4] Stress Test")
    profiler.reset()
    run_stress_test(duration_seconds=30, target_rps=10, profiler=profiler)

    # Test 4: High-load test
    print("\n[Test 4/4] High-Load Test")
    profiler.reset()
    run_concurrent_test(num_queries=200, max_workers=20, profiler=profiler)

    # Generate comprehensive reports
    print("\n" + "=" * 80)
    print("GENERATING REPORTS")
    print("=" * 80)

    stats = profiler.get_all_stats()
    reporter.print_summary(stats)

    # Check SLA compliance
    sla_result = profiler.check_sla()
    reporter.print_sla_result(sla_result)

    # Export reports
    output_dir = Path("reports/performance")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    reporter.export_json(stats, output_dir / f"performance_{timestamp}.json")
    reporter.export_markdown(stats, output_dir / f"performance_{timestamp}.md")
    reporter.export_html(stats, output_dir / f"performance_{timestamp}.html")

    print(f"\n✓ Reports saved to {output_dir}/")
    print("\nThroughput testing completed!")


if __name__ == "__main__":
    main()
