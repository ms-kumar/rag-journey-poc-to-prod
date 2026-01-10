# Cache Configuration Quick Reference

## TL;DR

```python
# Old way (still works)
from src.services.cache import RedisCache, RedisCacheConfig
cache = RedisCache(RedisCacheConfig(host="localhost"))

# New way (recommended)
from src.config import get_settings
from src.services.cache import create_redis_cache
cache = create_redis_cache(get_settings())
```

## All 21 Cache Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `CACHE__REDIS_HOST` | `localhost` | Redis hostname |
| `CACHE__REDIS_PORT` | `6379` | Redis port |
| `CACHE__REDIS_DB` | `0` | Redis database |
| `CACHE__REDIS_PASSWORD` | `None` | Redis password |
| `CACHE__REDIS_MAX_CONNECTIONS` | `10` | Connection pool size |
| `CACHE__REDIS_SOCKET_TIMEOUT` | `5` | Socket timeout (sec) |
| `CACHE__REDIS_DECODE_RESPONSES` | `true` | Decode to strings |
| `CACHE__DEFAULT_TTL` | `3600` | Default TTL (sec) |
| `CACHE__KEY_PREFIX` | `rag:` | Key prefix |
| `CACHE__ENABLED` | `true` | Enable caching |
| `CACHE__SEMANTIC_SIMILARITY_THRESHOLD` | `0.95` | Similarity threshold |
| `CACHE__SEMANTIC_EMBEDDING_DIM` | `384` | Embedding dimension |
| `CACHE__SEMANTIC_MAX_CANDIDATES` | `100` | Max candidates |
| `CACHE__STALENESS_CHECK_INTERVAL` | `300` | Check interval (sec) |
| `CACHE__STALENESS_THRESHOLD` | `3600` | Staleness threshold |
| `CACHE__STALENESS_AUTO_INVALIDATE` | `false` | Auto-invalidate |
| `CACHE__TARGET_HIT_RATE` | `0.6` | Target hit rate |

## Quick Examples

### Redis Cache
```python
from src.config import get_settings
from src.services.cache import create_redis_cache

cache = create_redis_cache(get_settings())
cache.set("key", {"data": "value"})
result = cache.get("key")
```

### Semantic Cache
```python
from src.config import get_settings
from src.services.cache import create_semantic_cache

semantic = create_semantic_cache(get_settings())
semantic.set("What is AI?", {"answer": "..."})
result = semantic.get("what is ai")  # Cache hit!
```

### Metrics
```python
from src.config import get_settings
from src.services.cache import create_cache_metrics

metrics = create_cache_metrics(get_settings())
metrics.record_hit("key", latency_ms=2.5)
stats = metrics.get_global_stats()
print(f"Hit rate: {stats.hit_rate:.1%}")
```

## Environment Configuration

Create `.env` file:
```bash
# Production settings
CACHE__REDIS_HOST=prod-redis.example.com
CACHE__REDIS_PORT=6380
CACHE__REDIS_PASSWORD=secret
CACHE__DEFAULT_TTL=7200
CACHE__TARGET_HIT_RATE=0.70
```

## Files to Know

| File | Purpose |
|------|---------|
| [config.py](../src/config.py) | Main configuration with CacheSettings |
| [factory.py](../src/services/cache/factory.py) | Factory functions |
| [cache-configuration.md](cache-configuration.md) | Full documentation |
| [cache_settings_demo.py](../examples/cache_settings_demo.py) | Interactive demo |

## See Also

- [Main Cache Docs](redis-cache.md) - Detailed cache documentation
- [Implementation Summary](caching-implementation-summary.md) - Original implementation
- [Integration Summary](cache-config-integration-summary.md) - This integration
