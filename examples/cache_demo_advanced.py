"""
Demo of Redis caching layer with semantic caching and monitoring.

This example demonstrates:
1. Redis cache with TTL and invalidation
2. Semantic cache for similar queries
3. Cache metrics and monitoring
4. Flush hooks
5. Staleness detection
6. Hit rate tracking (targeting ≥ 60%)
"""

import time

from src.services.cache import (
    CacheMetrics,
    CacheTimer,
    RedisCache,
    RedisCacheConfig,
    SemanticCache,
    SemanticCacheConfig,
    StalenessConfig,
)
from src.services.embeddings import EmbedClient


def demo_redis_cache():
    """Demonstrate basic Redis cache operations."""
    print("=" * 60)
    print("DEMO 1: Redis Cache with TTL and Invalidation")
    print("=" * 60)

    # Configure Redis cache
    config = RedisCacheConfig(
        host="localhost",
        port=6379,
        default_ttl=300,  # 5 minutes
        key_prefix="demo:",
    )

    try:
        cache = RedisCache(config)

        # Health check
        if not cache.health_check():
            print("⚠️  Redis not available, skipping demo")
            return

        print("\n✅ Connected to Redis")

        # Set values with different TTLs
        print("\n1. Setting cache values...")
        cache.set("user:123", {"name": "Alice", "role": "admin"}, ttl=60)
        cache.set("user:456", {"name": "Bob", "role": "user"}, ttl=120)
        cache.set("config:app", {"theme": "dark", "language": "en"}, ttl=300)

        # Get values
        print("\n2. Getting cached values...")
        user = cache.get("user:123")
        print(f"   User 123: {user}")

        # Check TTL
        ttl = cache.ttl("user:123")
        print(f"   TTL remaining: {ttl} seconds")

        # Pattern invalidation
        print("\n3. Invalidating user keys...")
        count = cache.invalidate_pattern("user:*")
        print(f"   Invalidated {count} keys")

        # Verify deletion
        user = cache.get("user:123")
        print(f"   User 123 after invalidation: {user}")

        # Cleanup
        cache.flush()
        cache.close()

        print("\n✅ Redis cache demo complete")

    except Exception as e:
        print(f"\n❌ Error: {e}")


def demo_semantic_cache():
    """Demonstrate semantic caching based on similarity."""
    print("\n" + "=" * 60)
    print("DEMO 2: Semantic Cache with Similarity Matching")
    print("=" * 60)

    try:
        # Create embedding client
        embed_client = EmbedClient(dim=384, normalize=True)

        # Configure semantic cache
        config = SemanticCacheConfig(
            similarity_threshold=0.90,  # 90% similarity required
            max_candidates=50,
        )

        # Create Redis cache
        redis_config = RedisCacheConfig(key_prefix="semantic:")
        redis_cache = RedisCache(redis_config)

        if not redis_cache.health_check():
            print("⚠️  Redis not available, skipping demo")
            return

        cache = SemanticCache(embed_client, config, redis_cache)

        print("\n1. Caching query results...")

        # Cache some Q&A pairs
        qa_pairs = [
            ("What is machine learning?", "ML is a subset of AI that enables systems to learn..."),
            ("How does Python work?", "Python is an interpreted, high-level programming language..."),
            ("What is deep learning?", "Deep learning uses neural networks with multiple layers..."),
        ]

        for query, answer in qa_pairs:
            cache.set(query, {"answer": answer}, ttl=600)
            print(f"   Cached: {query[:40]}...")

        print("\n2. Testing semantic similarity matches...")

        # Test similar queries (should find cached results)
        similar_queries = [
            "What's machine learning?",  # Similar to first
            "Explain Python to me",  # Similar to second
            "Tell me about deep learning",  # Similar to third
        ]

        for query in similar_queries:
            result = cache.get(query)
            if result:
                print(f"   ✅ CACHE HIT for: {query}")
                print(f"      Answer: {result['answer'][:50]}...")
            else:
                print(f"   ❌ CACHE MISS for: {query}")

        # Test dissimilar query (should miss)
        print("\n3. Testing dissimilar query...")
        result = cache.get("What's the weather today?")
        if result:
            print("   ✅ Cache hit (unexpected)")
        else:
            print("   ❌ Cache miss (expected)")

        # Get stats
        stats = cache.get_stats()
        print(f"\n4. Cache Statistics:")
        print(f"   Total entries: {stats['total_entries']}")
        print(f"   Similarity threshold: {stats['similarity_threshold']}")

        # Cleanup
        cache.flush()
        redis_cache.close()

        print("\n✅ Semantic cache demo complete")

    except Exception as e:
        print(f"\n❌ Error: {e}")


def demo_cache_metrics():
    """Demonstrate cache metrics and monitoring."""
    print("\n" + "=" * 60)
    print("DEMO 3: Cache Metrics and Monitoring")
    print("=" * 60)

    # Configure staleness monitoring
    staleness_config = StalenessConfig(
        check_interval=5,  # 5 seconds for demo
        staleness_threshold=3,  # 3 seconds threshold
        auto_invalidate_stale=False,
    )

    metrics = CacheMetrics(staleness_config)

    print("\n1. Simulating cache operations...")

    # Simulate cache hits and misses
    operations = [
        ("get", "user:1", True, 2.5),
        ("get", "user:2", True, 2.1),
        ("get", "user:3", False, 5.3),
        ("get", "user:1", True, 1.8),
        ("get", "user:4", False, 4.9),
        ("get", "user:2", True, 2.3),
        ("get", "user:1", True, 1.9),
        ("set", "user:5", None, None),
    ]

    for op_type, key, is_hit, latency in operations:
        if op_type == "get":
            if is_hit:
                metrics.record_hit(key, latency)
                print(f"   ✅ HIT:  {key} ({latency:.1f}ms)")
            else:
                metrics.record_miss(key, latency)
                print(f"   ❌ MISS: {key} ({latency:.1f}ms)")
        elif op_type == "set":
            metrics.record_set(key)
            print(f"   💾 SET:  {key}")

    # Get statistics
    print("\n2. Cache Statistics:")
    stats = metrics.get_global_stats()
    print(f"   Total requests: {stats.total_requests}")
    print(f"   Hits: {stats.hits}")
    print(f"   Misses: {stats.misses}")
    print(f"   Hit rate: {stats.hit_rate:.2%}")
    print(f"   Avg latency: {stats.avg_latency_ms:.2f}ms")

    # Check if meets target
    target = 0.6
    meets_target = metrics.meets_target_hit_rate(target)
    print(f"\n3. Target Hit Rate Check:")
    print(f"   Target: {target:.0%}")
    if meets_target:
        print(f"   ✅ Meets target ({stats.hit_rate:.2%} ≥ {target:.0%})")
    else:
        print(f"   ❌ Below target ({stats.hit_rate:.2%} < {target:.0%})")

    # Staleness check
    print("\n4. Staleness Monitoring:")
    print("   Waiting for entries to become stale...")
    time.sleep(4)  # Wait for staleness threshold

    staleness_info = metrics.check_staleness(force=True)
    print(f"   Stale entries detected: {staleness_info['stale_count']}")
    print(f"   Total entries tracked: {staleness_info['total_entries']}")

    # Summary
    print("\n5. Comprehensive Summary:")
    summary = metrics.get_summary()
    print(f"   Top keys by hits: {summary['top_keys_by_hits'][:3]}")
    print(f"   Top keys by misses: {summary['top_keys_by_misses'][:3]}")

    print("\n✅ Metrics demo complete")


def demo_flush_hooks():
    """Demonstrate flush hooks."""
    print("\n" + "=" * 60)
    print("DEMO 4: Flush Hooks")
    print("=" * 60)

    try:
        config = RedisCacheConfig(key_prefix="hooks:")
        cache = RedisCache(config)

        if not cache.health_check():
            print("⚠️  Redis not available, skipping demo")
            return

        # Define hooks
        def log_flush():
            print("   📝 Hook 1: Logging flush operation")

        def notify_flush():
            print("   📢 Hook 2: Sending notification")

        def cleanup_flush():
            print("   🧹 Hook 3: Running cleanup tasks")

        # Register hooks
        print("\n1. Registering flush hooks...")
        cache.register_flush_hook(log_flush)
        cache.register_flush_hook(notify_flush)
        cache.register_flush_hook(cleanup_flush)
        print("   Registered 3 hooks")

        # Add some data
        print("\n2. Adding cache data...")
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        # Flush (will trigger hooks)
        print("\n3. Flushing cache (hooks will be called)...")
        cache.flush()

        cache.close()

        print("\n✅ Flush hooks demo complete")

    except Exception as e:
        print(f"\n❌ Error: {e}")


def demo_cache_timer():
    """Demonstrate cache timer for automatic metrics."""
    print("\n" + "=" * 60)
    print("DEMO 5: Cache Timer Context Manager")
    print("=" * 60)

    metrics = CacheMetrics()

    print("\n1. Using CacheTimer for automatic tracking...")

    # Mock cache for demo
    mock_cache = {
        "cached_key": {"data": "value"},
    }

    # Example with cache hit
    with CacheTimer(metrics, "cached_key", "get") as timer:
        time.sleep(0.002)  # Simulate lookup
        result = mock_cache.get("cached_key")
        if result:
            timer.mark_hit()
            print(f"   ✅ Cache hit for 'cached_key'")

    # Example with cache miss
    with CacheTimer(metrics, "missing_key", "get") as timer:
        time.sleep(0.005)  # Simulate lookup
        result = mock_cache.get("missing_key")
        if not result:
            print(f"   ❌ Cache miss for 'missing_key'")

    # Show metrics
    stats = metrics.get_global_stats()
    print(f"\n2. Automatically tracked metrics:")
    print(f"   Hits: {stats.hits}")
    print(f"   Misses: {stats.misses}")
    print(f"   Avg latency: {stats.avg_latency_ms:.2f}ms")

    print("\n✅ Cache timer demo complete")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("Redis Caching Layer Demo")
    print("=" * 60)
    print("\nThis demo requires Redis running on localhost:6379")
    print("Start Redis with: docker run -d -p 6379:6379 redis:latest\n")

    input("Press Enter to start demos...")

    try:
        demo_redis_cache()
        demo_semantic_cache()
        demo_cache_metrics()
        demo_flush_hooks()
        demo_cache_timer()

        print("\n" + "=" * 60)
        print("✅ All demos completed successfully!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted")
    except Exception as e:
        print(f"\n\n❌ Error running demos: {e}")


if __name__ == "__main__":
    main()
