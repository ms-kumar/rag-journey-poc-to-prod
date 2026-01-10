"""
Example demonstrating cache usage with centralized configuration.

This shows how to use cache services with settings from config.py.
"""

from src.config import get_settings
from src.services.cache import (
    create_cache_metrics,
    create_redis_cache,
    create_semantic_cache,
)


def demo_redis_cache():
    """Demo Redis cache with settings."""
    print("=" * 60)
    print("Demo 1: Redis Cache with Centralized Settings")
    print("=" * 60)

    # Load application settings
    settings = get_settings()

    # Create cache from settings
    cache = create_redis_cache(settings)

    print(f"\nCache Configuration:")
    print(f"  Host: {settings.cache.redis_host}:{settings.cache.redis_port}")
    print(f"  Database: {settings.cache.redis_db}")
    print(f"  Key Prefix: {settings.cache.key_prefix}")
    print(f"  Default TTL: {settings.cache.default_ttl}s")

    # Test basic operations
    print("\nBasic Operations:")
    cache.set("user:123", {"name": "Alice", "role": "admin"})
    user = cache.get("user:123")
    print(f"  Set and retrieved: {user}")

    ttl = cache.ttl("user:123")
    print(f"  TTL: {ttl}s")

    cache.close()
    print("✓ Redis cache demo complete\n")


def demo_semantic_cache():
    """Demo semantic cache with settings."""
    print("=" * 60)
    print("Demo 2: Semantic Cache with Centralized Settings")
    print("=" * 60)

    # Load application settings
    settings = get_settings()

    # Create semantic cache from settings
    # (automatically creates embedding client with configured dimension)
    semantic_cache = create_semantic_cache(settings)

    print(f"\nSemantic Cache Configuration:")
    print(f"  Similarity Threshold: {settings.cache.semantic_similarity_threshold}")
    print(f"  Embedding Dimension: {settings.cache.semantic_embedding_dim}")
    print(f"  Max Candidates: {settings.cache.semantic_max_candidates}")

    # Cache semantic queries
    print("\nSemantic Matching:")
    semantic_cache.set(
        "What is Python?",
        {"answer": "Python is a high-level programming language."},
    )

    # Try similar query
    result = semantic_cache.get("what is python")
    print(f"  Query: 'what is python'")
    print(f"  Cache hit: {result is not None}")
    if result:
        print(f"  Answer: {result['answer']}")

    semantic_cache.flush()
    print("✓ Semantic cache demo complete\n")


def demo_cache_metrics():
    """Demo cache metrics with settings."""
    print("=" * 60)
    print("Demo 3: Cache Metrics with Centralized Settings")
    print("=" * 60)

    # Load application settings
    settings = get_settings()

    # Create cache and metrics from settings
    cache = create_redis_cache(settings)
    metrics = create_cache_metrics(settings, cache)

    print(f"\nMetrics Configuration:")
    print(f"  Staleness Check Interval: {settings.cache.staleness_check_interval}s")
    print(f"  Staleness Threshold: {settings.cache.staleness_threshold}s")
    print(f"  Target Hit Rate: {settings.cache.target_hit_rate:.0%}")

    # Track operations
    print("\nTracking Operations:")
    cache.set("key1", "value1")
    metrics.record_set("key1")

    cache.get("key1")
    metrics.record_hit("key1", latency_ms=2.5)

    cache.get("key2")
    metrics.record_miss("key2", latency_ms=1.2)

    stats = metrics.get_global_stats()
    print(f"  Hits: {stats.hits}")
    print(f"  Misses: {stats.misses}")
    print(f"  Hit Rate: {stats.hit_rate:.1%}")
    print(f"  Avg Latency: {stats.avg_latency_ms:.2f}ms")

    # Check target
    meets_target = metrics.meets_target_hit_rate(settings.cache.target_hit_rate)
    print(f"  Meets {settings.cache.target_hit_rate:.0%} target: {meets_target}")

    cache.close()
    print("✓ Cache metrics demo complete\n")


def demo_environment_override():
    """Demo environment variable configuration override."""
    print("=" * 60)
    print("Demo 4: Environment Variable Configuration")
    print("=" * 60)

    settings = get_settings()

    print("\nCache settings can be overridden via environment variables:")
    print("  CACHE__REDIS_HOST=redis.example.com")
    print("  CACHE__REDIS_PORT=6380")
    print("  CACHE__DEFAULT_TTL=7200")
    print("  CACHE__KEY_PREFIX=myapp:")
    print("  CACHE__SEMANTIC_SIMILARITY_THRESHOLD=0.90")
    print("  CACHE__TARGET_HIT_RATE=0.70")

    print("\nCurrent settings:")
    print(f"  Redis Host: {settings.cache.redis_host}")
    print(f"  Redis Port: {settings.cache.redis_port}")
    print(f"  Default TTL: {settings.cache.default_ttl}s")
    print(f"  Key Prefix: {settings.cache.key_prefix}")
    print(f"  Similarity Threshold: {settings.cache.semantic_similarity_threshold}")
    print(f"  Target Hit Rate: {settings.cache.target_hit_rate:.0%}")

    print("\n✓ Configuration can be customized per environment\n")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("Cache Configuration Demo")
    print("Demonstrating centralized cache configuration via config.py")
    print("=" * 60 + "\n")

    try:
        demo_redis_cache()
        demo_semantic_cache()
        demo_cache_metrics()
        demo_environment_override()

        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print("""
All cache components now use centralized configuration from config.py:

1. Redis Cache: Uses CACHE__REDIS_* settings
2. Semantic Cache: Uses CACHE__SEMANTIC_* settings  
3. Metrics: Uses CACHE__STALENESS_* and CACHE__TARGET_HIT_RATE

Factory functions make it easy to create configured instances:
- create_redis_cache(settings)
- create_semantic_cache(settings)
- create_cache_metrics(settings)

All settings can be overridden via environment variables or .env file.
        """)

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("Note: Redis server must be running for cache operations to work")


if __name__ == "__main__":
    main()
