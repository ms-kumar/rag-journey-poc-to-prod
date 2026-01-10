# Cache Architecture - Mother of AI Pattern

This document shows how the cache service follows the Mother of AI project's architecture pattern.

## Mother of AI Pattern

The Mother of AI project uses a clean factory pattern for cache creation:

```python
# src/services/cache/factory.py
def make_redis_client(settings: Settings) -> redis.Redis:
    """Create Redis client with connection pooling."""
    redis_settings = settings.redis
    client = redis.Redis(
        host=redis_settings.host,
        port=redis_settings.port,
        # ... other config
    )
    client.ping()  # Test connection
    return client

def make_cache_client(settings: Settings) -> CacheClient:
    """Create exact match cache client."""
    redis_client = make_redis_client(settings)
    cache_client = CacheClient(redis_client, settings.redis)
    return cache_client
```

## Our Implementation

We've adopted the same pattern in `src/services/cache/`:

### 1. Unified CacheClient (`client.py`)
- Single class for all cache operations
- Simple `get()`, `set()`, `delete()`, `clear()` methods
- Automatic key generation with parameter hashing
- TTL-based expiration
- Graceful error handling

### 2. Factory Functions (`factory.py`)
```python
def make_redis_client(settings: Settings) -> redis.Redis:
    """Create Redis client with connection pooling."""
    # Returns configured Redis client
    # Raises RedisConnectionError on failure

def make_cache_client(settings: Settings) -> CacheClient | None:
    """Create unified cache client."""
    # Returns CacheClient or None (graceful fallback)
    # No exceptions raised - application can continue without cache
```

### 3. Clean Exports (`__init__.py`)
```python
from src.services.cache import CacheClient, make_cache_client

# Usage in application
cache_client = make_cache_client(settings)
if cache_client:
    cache_client.set(query, response)
    cached = cache_client.get(query)
```

## Key Similarities

### 1. Factory Pattern
Both use factory functions instead of direct instantiation:
- `make_redis_client()` - Creates Redis connection
- `make_cache_client()` - Creates cache client

### 2. Graceful Fallback
Both handle Redis connection failures gracefully:
```python
# Returns None if Redis unavailable
cache_client = make_cache_client(settings)
if cache_client:
    # Use cache
else:
    # Continue without cache
```

### 3. Simple Interface
Both provide simple, intuitive methods:
- `get(query, **params)` - Retrieve cached response
- `set(query, response, **params)` - Store response
- `ping()` - Check connection
- `clear()` - Clear cache

### 4. Parameter-Based Caching
Both use request parameters for cache keys:
```python
# Different parameters = different cache entries
cache.get(query, model="llama3.2", top_k=5)
cache.get(query, model="llama3.2", top_k=10)  # Different cache
```

## Usage Example

```python
from src.config import get_settings
from src.services.cache import CacheClient, make_cache_client

# Initialize
settings = get_settings()
cache_client = make_cache_client(settings)

if cache_client is None:
    logger.warning("Cache unavailable - continuing without caching")
    # Application continues normally

# Use cache
query = "What are transformers?"
params = {"model": "llama3.2", "top_k": 5}

# Check cache
cached_response = cache_client.get(query, **params)
if cached_response:
    return cached_response  # Cache hit

# Generate response
response = generate_rag_response(query, **params)

# Store in cache
cache_client.set(query, response, **params)
```

## Architecture Benefits

1. **Separation of Concerns**: Factory handles creation, client handles operations
2. **Testability**: Easy to mock `make_cache_client()` in tests
3. **Graceful Degradation**: Application works with or without cache
4. **Type Safety**: Proper typing throughout
5. **Error Handling**: Exceptions caught at factory level

## Files Structure

```
src/services/cache/
├── __init__.py          # Exports: CacheClient, make_cache_client, make_redis_client
├── client.py            # CacheClient class (unified interface)
├── factory.py           # make_cache_client(), make_redis_client()
├── redis_client.py      # RedisCache (advanced/legacy)
├── semantic_cache.py    # SemanticCache (advanced)
└── metrics.py           # CacheMetrics
```

## Migration from Legacy

If you're using the old `RedisCache` or `SemanticCache` directly, migrate to:

```python
# Old way
from src.services.cache import RedisCache, RedisCacheConfig
config = RedisCacheConfig.from_settings(settings)
cache = RedisCache(config)
cache.set("key", value)
cache.get("key")

# New way (Mother of AI pattern)
from src.services.cache import make_cache_client
cache_client = make_cache_client(settings)
if cache_client:
    cache_client.set(query, response, model="llama3.2")
    cached = cache_client.get(query, model="llama3.2")
```

## Demo

Run the demo to see it in action:
```bash
uv run python examples/cache_client_demo.py
```

This demonstrates:
- Basic get/set operations
- Parameter-based caching
- Cache statistics
- Cache management (clearing)
- Graceful fallback
