# Redis Caching Layer

Comprehensive caching system with Redis, semantic caching, TTL, invalidation, flush hooks, and monitoring.

## Overview

The caching layer provides three main components:

1. **RedisCache** - Redis-based cache with TTL and invalidation
2. **SemanticCache** - Semantic similarity-based caching using embeddings
3. **CacheMetrics** - Monitoring, staleness detection, and hit rate tracking

## Features

### ✅ Redis TTL + Invalidation
- Automatic time-to-live (TTL) for all cached entries
- Pattern-based key invalidation with wildcards
- Individual key deletion
- Configurable default TTL (default: 1 hour)

### ✅ Semantic Cache
- Caches based on semantic similarity instead of exact matches
- Uses embeddings to find similar queries
- Configurable similarity threshold (default: 0.95)
- Efficient candidate selection

### ✅ Flush Hooks
- Register callbacks to execute before cache flush
- Multiple hooks supported
- Useful for logging, cleanup, or notifications

### ✅ Staleness Monitoring
- Automatic detection of stale cache entries
- Configurable staleness threshold
- Periodic staleness checks
- Optional auto-invalidation

### ✅ Cache Hit Rate ≥ 60%
- Real-time hit rate calculation
- Per-key and global statistics
- Target hit rate validation
- Time-series metrics tracking

## Installation

Add Redis dependency (already included in pyproject.toml):

```bash
uv sync
```

Start Redis server:

```bash
# Using Docker
docker run -d -p 6379:6379 redis:latest

# Or using Docker Compose
docker compose -f infra/docker/compose-redis.yml up -d
```

## Quick Start

### Basic Redis Cache

```python
from src.services.cache import RedisCache, RedisCacheConfig

# Configure cache
config = RedisCacheConfig(
    host="localhost",
    port=6379,
    default_ttl=3600,  # 1 hour
    key_prefix="myapp:"
)

# Create cache
cache = RedisCache(config)

# Set value with TTL
cache.set("user:123", {"name": "John", "email": "john@example.com"}, ttl=300)

# Get value
user = cache.get("user:123")

# Check existence
if cache.exists("user:123"):
    print("User found in cache")

# Delete key
cache.delete("user:123")

# Invalidate pattern
cache.invalidate_pattern("user:*")
```

### Semantic Cache

```python
from src.services.cache import SemanticCache, SemanticCacheConfig
from src.services.embeddings import EmbedClient

# Create embedding client
embed_client = EmbedClient(dimension=384, normalize=True)

# Configure semantic cache
config = SemanticCacheConfig(
    similarity_threshold=0.92,  # 92% similarity required
    max_candidates=100,
)

# Create semantic cache
cache = SemanticCache(embed_client, config)

# Cache a query result
cache.set(
    query="What is machine learning?",
    value={"answer": "ML is a subset of AI..."},
    ttl=3600
)

# Retrieve with similar query (semantic match)
result = cache.get("What's ML?")  # Will find cached result if similar enough

# Get statistics
stats = cache.get_stats()
print(f"Total entries: {stats['total_entries']}")
```

### Cache Metrics & Monitoring

```python
from src.services.cache import CacheMetrics, StalenessConfig, CacheTimer

# Configure staleness monitoring
staleness_config = StalenessConfig(
    check_interval=300,  # Check every 5 minutes
    staleness_threshold=3600,  # 1 hour threshold
    auto_invalidate_stale=False
)

# Create metrics
metrics = CacheMetrics(staleness_config)

# Record operations
metrics.record_hit("key1", latency_ms=5.2)
metrics.record_miss("key2", latency_ms=3.1)
metrics.record_set("key3")

# Check hit rate
if metrics.meets_target_hit_rate(0.6):  # 60% target
    print("✅ Hit rate meets target")

# Get comprehensive summary
summary = metrics.get_summary()
print(f"Hit rate: {summary['global_stats']['hit_rate']:.2%}")
print(f"Avg latency: {summary['global_stats']['avg_latency_ms']:.2f}ms")

# Check for stale entries
staleness_info = metrics.check_staleness(force=True)
print(f"Stale entries: {staleness_info['stale_count']}")

# Use timer context manager
with CacheTimer(metrics, "my_key", "get") as timer:
    result = cache.get("my_key")
    if result:
        timer.mark_hit()
```

### Flush Hooks

```python
# Define flush hook
def log_flush():
    print("Cache is being flushed!")
    # Log to monitoring system, send alert, etc.

# Register hook
cache.register_flush_hook(log_flush)

# Flush will trigger the hook
cache.flush()  # Calls log_flush() before clearing cache

# Unregister hook
cache.unregister_flush_hook(log_flush)
```

## Configuration

### Redis Cache Configuration

```python
config = RedisCacheConfig(
    host="localhost",           # Redis host
    port=6379,                  # Redis port
    db=0,                       # Redis database number
    password=None,              # Redis password
    default_ttl=3600,          # Default TTL (seconds)
    key_prefix="rag:",         # Key prefix
    max_connections=10,        # Connection pool size
    socket_timeout=5,          # Socket timeout (seconds)
    decode_responses=True      # Decode to strings
)
```

### Semantic Cache Configuration

```python
config = SemanticCacheConfig(
    similarity_threshold=0.95,  # Similarity threshold (0.0-1.0)
    embedding_dim=384,          # Embedding dimension
    max_candidates=100,         # Max candidates to check
    redis_config=redis_config   # Redis configuration
)
```

### Staleness Configuration

```python
config = StalenessConfig(
    check_interval=300,         # Check interval (seconds)
    staleness_threshold=3600,   # Staleness threshold (seconds)
    auto_invalidate_stale=False # Auto-invalidate stale entries
)
```

## API Reference

### RedisCache

#### Methods

- `get(key: str) -> Any | None` - Get value from cache
- `set(key: str, value: Any, ttl: int | None = None, nx: bool = False, xx: bool = False) -> bool` - Set value
- `delete(key: str) -> bool` - Delete key
- `exists(key: str) -> bool` - Check if key exists
- `ttl(key: str) -> int` - Get remaining TTL
- `expire(key: str, ttl: int) -> bool` - Set expiration
- `invalidate_pattern(pattern: str) -> int` - Invalidate matching keys
- `flush() -> bool` - Flush cache with prefix
- `flush_all() -> bool` - Flush entire database
- `register_flush_hook(hook: Callable) -> None` - Register flush hook
- `health_check() -> bool` - Check Redis health

### SemanticCache

#### Methods

- `get(query: str) -> Any | None` - Get cached result for similar query
- `set(query: str, value: Any, ttl: int | None = None) -> bool` - Cache with semantic embedding
- `invalidate(pattern: str = "*") -> int` - Invalidate entries
- `flush() -> bool` - Flush all entries
- `get_stats() -> dict` - Get cache statistics
- `health_check() -> bool` - Check health

### CacheMetrics

#### Methods

- `record_hit(key: str, latency_ms: float = 0.0) -> None` - Record cache hit
- `record_miss(key: str, latency_ms: float = 0.0) -> None` - Record cache miss
- `record_set(key: str) -> None` - Record set operation
- `record_delete(key: str) -> None` - Record delete operation
- `record_invalidation(count: int = 1) -> None` - Record invalidation
- `check_staleness(force: bool = False) -> dict` - Check for stale entries
- `get_global_stats() -> CacheStats` - Get global statistics
- `get_key_stats(key: str) -> CacheStats` - Get key statistics
- `get_summary() -> dict` - Get comprehensive summary
- `meets_target_hit_rate(target: float = 0.6) -> bool` - Check if target met
- `reset() -> None` - Reset all metrics

## Best Practices

1. **Choose Appropriate TTL**
   - Short-lived data: 300-600 seconds
   - Medium-lived data: 1-6 hours
   - Long-lived data: 24 hours
   - Static data: No TTL or very long TTL

2. **Use Key Prefixes**
   - Organize keys by namespace
   - Makes invalidation easier
   - Prevents key collisions

3. **Monitor Hit Rates**
   - Target ≥ 60% hit rate
   - Track per-key patterns
   - Identify cold keys

4. **Semantic Cache Threshold**
   - Higher threshold (0.95-0.98): More precise, fewer false positives
   - Lower threshold (0.85-0.90): More flexible, more cache hits
   - Balance based on use case

5. **Staleness Management**
   - Set appropriate thresholds
   - Use auto-invalidation cautiously
   - Monitor stale entry counts

6. **Error Handling**
   - Cache operations return None/False on error
   - Application should handle cache misses gracefully
   - Don't let cache failures break application

## Performance Tips

- Use connection pooling (default: 10 connections)
- Batch operations when possible
- Use pattern invalidation instead of individual deletes
- Monitor memory usage in Redis
- Use appropriate data structures (JSON for complex objects)
- Consider cache warming for critical queries

## Monitoring

### Key Metrics to Track

1. **Hit Rate**: Should be ≥ 60%
2. **Average Latency**: Should be < 10ms
3. **Stale Entries**: Keep low
4. **Memory Usage**: Monitor Redis memory
5. **Eviction Rate**: Track LRU evictions

### Example Monitoring Integration

```python
# Periodic monitoring
import time

metrics = CacheMetrics()

while True:
    time.sleep(300)  # Every 5 minutes
    
    summary = metrics.get_summary()
    hit_rate = summary['global_stats']['hit_rate']
    
    if hit_rate < 0.6:
        print(f"⚠️  Low hit rate: {hit_rate:.2%}")
        # Send alert
    
    # Check staleness
    staleness = metrics.check_staleness(force=True)
    if staleness['stale_count'] > 100:
        print(f"⚠️  High stale count: {staleness['stale_count']}")
        # Consider invalidation
```

## Troubleshooting

### Redis Connection Issues

```python
# Check Redis health
if not cache.health_check():
    print("❌ Redis connection failed")
    # Fall back to in-memory cache or handle gracefully
```

### Low Hit Rates

- Verify TTL isn't too short
- Check if queries have high variance
- For semantic cache, lower similarity threshold
- Increase max_candidates

### High Memory Usage

- Reduce TTL
- Implement LRU eviction policy in Redis
- Use pattern invalidation to clear old keys
- Monitor key count

### Stale Data

- Reduce staleness threshold
- Increase check frequency
- Enable auto-invalidation
- Implement cache warming strategy

## Testing

Run cache tests:

```bash
# Unit tests (mocked Redis)
pytest tests/test_redis_cache.py -v
pytest tests/test_semantic_cache.py -v
pytest tests/test_cache_metrics.py -v

# Integration tests (requires Redis)
pytest tests/test_redis_cache.py::TestRedisCacheIntegration -v -m integration
```

## See Also

- [Embedding Cache](embedding-cache.md) - In-memory embedding cache
- [Token Budgets](token-budgets.md) - Token management
- [Performance](../README.md#performance) - Overall performance guidelines
