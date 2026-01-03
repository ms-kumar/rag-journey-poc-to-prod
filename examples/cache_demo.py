#!/usr/bin/env python3
"""
Demo script showing embedding cache performance improvements.

Run with: python examples/cache_demo.py
"""

import time

from src.services.embeddings.factory import get_embed_client

# Sample texts for demonstration
SAMPLE_TEXTS = [
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks with multiple layers",
    "Natural language processing enables computers to understand text",
    "Computer vision allows machines to interpret images",
    "Reinforcement learning trains agents through trial and error",
    "Transfer learning leverages pre-trained models",
    "Supervised learning requires labeled training data",
    "Unsupervised learning finds patterns in unlabeled data",
    "Neural networks are inspired by biological neurons",
    "Gradient descent optimizes model parameters",
]


def benchmark_without_cache():
    """Benchmark embedding without caching."""
    print("\n" + "=" * 60)
    print("Benchmark 1: WITHOUT CACHE")
    print("=" * 60)

    client = get_embed_client(
        provider="hash", model_name="demo-hash", dim=64, cache_enabled=False
    )

    # First run
    start = time.time()
    embeddings1 = client.embed(SAMPLE_TEXTS)
    time1 = time.time() - start

    # Second run (same texts)
    start = time.time()
    embeddings2 = client.embed(SAMPLE_TEXTS)
    time2 = time.time() - start

    print(f"First run:  {time1*1000:.2f}ms")
    print(f"Second run: {time2*1000:.2f}ms")
    print(f"Speedup:    {time1/time2:.1f}x")
    print(f"Note: No caching benefit expected")


def benchmark_with_cache():
    """Benchmark embedding with caching."""
    print("\n" + "=" * 60)
    print("Benchmark 2: WITH CACHE")
    print("=" * 60)

    client = get_embed_client(
        provider="hash", model_name="demo-hash", dim=64, cache_enabled=True, cache_max_size=1000
    )

    # First run - populate cache
    start = time.time()
    embeddings1 = client.embed(SAMPLE_TEXTS)
    time1 = time.time() - start

    # Second run - use cache
    start = time.time()
    embeddings2 = client.embed(SAMPLE_TEXTS)
    time2 = time.time() - start

    # Third run - use cache again
    start = time.time()
    embeddings3 = client.embed(SAMPLE_TEXTS)
    time3 = time.time() - start

    print(f"First run:  {time1*1000:.2f}ms (cache miss)")
    print(f"Second run: {time2*1000:.2f}ms (cache hit)")
    print(f"Third run:  {time3*1000:.2f}ms (cache hit)")
    print(f"Speedup:    {time1/time2:.1f}x")

    # Show cache stats
    stats = client.cache_stats
    print(f"\nCache Statistics:")
    print(f"  Size: {stats['size']}/{stats['max_size']}")
    print(f"  Hits: {stats['hits']}")
    print(f"  Misses: {stats['misses']}")
    print(f"  Hit rate: {stats['hit_rate']:.1%}")


def benchmark_partial_cache():
    """Benchmark with partial cache hits."""
    print("\n" + "=" * 60)
    print("Benchmark 3: PARTIAL CACHE HITS")
    print("=" * 60)

    client = get_embed_client(
        provider="hash", model_name="demo-hash", dim=64, cache_enabled=True
    )

    # Populate cache with first 5 texts
    cached_texts = SAMPLE_TEXTS[:5]
    client.embed(cached_texts)

    # Mix of cached and new texts
    mixed_texts = SAMPLE_TEXTS[:3] + ["New text 1", "New text 2"] + SAMPLE_TEXTS[3:5]

    start = time.time()
    embeddings = client.embed(mixed_texts)
    elapsed = time.time() - start

    stats = client.cache_stats
    print(f"Total texts: {len(mixed_texts)}")
    print(f"Cached:      {stats['hits']} texts")
    print(f"Computed:    {stats['misses'] - 5} texts")  # Subtract initial misses
    print(f"Time:        {elapsed*1000:.2f}ms")
    print(f"Hit rate:    {stats['hit_rate']:.1%}")


def benchmark_batching():
    """Benchmark batch processing."""
    print("\n" + "=" * 60)
    print("Benchmark 4: BATCH PROCESSING")
    print("=" * 60)

    # Generate larger dataset
    large_dataset = []
    for i in range(100):
        large_dataset.append(f"Document {i}: This is sample text about topic {i%10}")

    # Small batches
    client_small = get_embed_client(
        provider="hash",
        model_name="demo-hash",
        dim=64,
        cache_enabled=False,
        batch_size=10,
    )

    start = time.time()
    client_small.embed(large_dataset)
    time_small = time.time() - start

    # Large batches
    client_large = get_embed_client(
        provider="hash",
        model_name="demo-hash",
        dim=64,
        cache_enabled=False,
        batch_size=50,
    )

    start = time.time()
    client_large.embed(large_dataset)
    time_large = time.time() - start

    print(f"100 texts with batch_size=10: {time_small*1000:.2f}ms")
    print(f"100 texts with batch_size=50: {time_large*1000:.2f}ms")
    print(f"Speedup: {time_small/time_large:.1f}x")


def main():
    """Run all benchmarks."""
    print("\n" + "=" * 60)
    print("EMBEDDING CACHE & BATCH PROCESSING DEMO")
    print("=" * 60)

    benchmark_without_cache()
    benchmark_with_cache()
    benchmark_partial_cache()
    benchmark_batching()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("✓ Caching provides significant speedup for repeated texts")
    print("✓ Partial cache hits still improve performance")
    print("✓ Batch processing optimizes throughput")
    print("✓ Use cache_enabled=True in production for best performance")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
