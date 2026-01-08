# Client Retry and Backoff

Comprehensive retry logic with exponential backoff for resilient external service calls.

## Overview

The retry/backoff system provides automatic retrying of failed operations with configurable exponential backoff and jitter. This is essential for handling transient network failures, temporary service unavailability, and rate limiting.

## Features

- ✅ **Exponential Backoff**: Delays increase exponentially between retries
- ✅ **Jitter**: Random jitter prevents thundering herd problem
- ✅ **Configurable Exceptions**: Define which exceptions trigger retry
- ✅ **Sync and Async**: Support for both synchronous and asynchronous code
- ✅ **Decorator Pattern**: Easy to apply with decorators
- ✅ **Base Class**: RetryableClient for building resilient clients

## Installation

The retry module is included in the core project. No additional dependencies required.

## Quick Start

### Basic Synchronous Retry

```python
from src.services.retry import retry_with_backoff, RetryConfig

@retry_with_backoff(RetryConfig(max_retries=3, initial_delay=1.0))
def fetch_data():
    response = requests.get("https://api.example.com/data")
    response.raise_for_status()
    return response.json()

# Automatically retries on ConnectionError, TimeoutError, OSError
result = fetch_data()
```

### Basic Async Retry

```python
from src.services.retry import async_retry_with_backoff, RetryConfig

@async_retry_with_backoff(RetryConfig(max_retries=3, initial_delay=1.0))
async def fetch_data_async():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.example.com/data")
        response.raise_for_status()
        return response.json()

result = await fetch_data_async()
```

## Configuration

### RetryConfig

Complete configuration options for retry behavior:

```python
from src.services.retry import RetryConfig

config = RetryConfig(
    max_retries=3,                    # Maximum number of retry attempts
    initial_delay=1.0,                # Initial delay in seconds
    max_delay=60.0,                   # Maximum delay cap in seconds
    exponential_base=2.0,             # Exponential base (delay *= base^attempt)
    jitter=True,                      # Add random jitter to delays
    retryable_exceptions=(            # Exceptions that trigger retry
        ConnectionError,
        TimeoutError,
        OSError,
    ),
    fatal_exceptions=(                # Exceptions that never retry
        ValueError,
        TypeError,
        KeyError,
    ),
)
```

### Delay Calculation

Delays follow exponential backoff with optional jitter:

```
delay = min(initial_delay * (exponential_base ^ attempt), max_delay)
if jitter:
    delay = random.uniform(0, delay)
```

**Example progression** (initial_delay=1.0, base=2.0, no jitter):
- Attempt 1: 1 second
- Attempt 2: 2 seconds
- Attempt 3: 4 seconds
- Attempt 4: 8 seconds
- Attempt 5: 16 seconds

## Usage Examples

### Example 1: Custom Retry Configuration

```python
from src.services.retry import retry_with_backoff, RetryConfig

# Aggressive retry (more attempts, shorter delays)
aggressive_config = RetryConfig(
    max_retries=5,
    initial_delay=0.5,
    max_delay=10.0,
    exponential_base=1.5,
    jitter=True,
)

@retry_with_backoff(aggressive_config)
def quick_retry_fetch():
    return api_call()
```

### Example 2: Custom Exception Handling

```python
from src.services.retry import retry_with_backoff, RetryConfig

class RateLimitError(Exception):
    pass

class AuthenticationError(Exception):
    pass

# Retry on rate limit, but not on auth errors
config = RetryConfig(
    max_retries=3,
    retryable_exceptions=(RateLimitError, TimeoutError),
    fatal_exceptions=(AuthenticationError, ValueError),
)

@retry_with_backoff(config)
def api_with_rate_limit():
    response = requests.get("https://api.example.com/endpoint")
    if response.status_code == 429:
        raise RateLimitError("Rate limit exceeded")
    if response.status_code == 401:
        raise AuthenticationError("Invalid credentials")
    return response.json()
```

### Example 3: Async with Custom Timing

```python
from src.services.retry import async_retry_with_backoff, RetryConfig

# Longer delays for expensive operations
config = RetryConfig(
    max_retries=3,
    initial_delay=5.0,
    max_delay=120.0,
    exponential_base=3.0,  # Faster exponential growth
    jitter=False,          # Deterministic delays
)

@async_retry_with_backoff(config)
async def expensive_operation():
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post("https://api.example.com/process")
        return response.json()
```

### Example 4: RetryableClient Base Class

```python
from src.services.retry import RetryableClient, RetryConfig

class MyAPIClient(RetryableClient):
    def __init__(self):
        # Initialize with custom retry config
        config = RetryConfig(max_retries=5, initial_delay=2.0)
        super().__init__(config)
        self.base_url = "https://api.example.com"
    
    def get_data(self, endpoint: str):
        # Wrap method with retry
        @self.with_retry
        def _fetch():
            response = requests.get(f"{self.base_url}/{endpoint}")
            response.raise_for_status()
            return response.json()
        
        return _fetch()
    
    async def get_data_async(self, endpoint: str):
        # Wrap async method with retry
        @self.with_async_retry
        async def _fetch():
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/{endpoint}")
                response.raise_for_status()
                return response.json()
        
        return await _fetch()

# Usage
client = MyAPIClient()
data = client.get_data("users")
```

## Integration with Existing Clients

### VectorStore Client

The vectorstore client automatically includes retry logic:

```python
from src.services.vectorstore.client import VectorStoreConfig
from src.services.retry import RetryConfig

# Configure retry behavior
config = VectorStoreConfig(
    qdrant_url="http://localhost:6333",
    collection_name="documents",
    retry_config=RetryConfig(
        max_retries=3,
        initial_delay=1.0,
        max_delay=30.0,
    ),
)

# All query methods automatically retry on failure
vectorstore = QdrantVectorStoreClient(embeddings, config)
docs = vectorstore.similarity_search("query")  # Retries on connection errors
```

### Generation Client

The generation client also supports retry configuration:

```python
from src.services.generation.client import GenerationConfig
from src.services.retry import RetryConfig

config = GenerationConfig(
    model_name="gpt2",
    retry_config=RetryConfig(
        max_retries=2,
        initial_delay=0.5,
        max_delay=10.0,
    ),
)

generator = HFGenerator(config)
output = generator.generate("prompt")  # Retries on transient failures
```

## Best Practices

### ✅ DO

1. **Use appropriate max_retries**: 3-5 retries for most services
2. **Set reasonable delays**: Start with 1-2 seconds initial delay
3. **Enable jitter**: Prevents thundering herd on service recovery
4. **Define specific retryable exceptions**: Only retry transient failures
5. **Log retry attempts**: Monitor retry patterns for issues

### ❌ DON'T

1. **Don't retry indefinitely**: Set sensible max_retries
2. **Don't retry on user errors**: ValueError, TypeError shouldn't retry
3. **Don't use tiny delays**: Sub-100ms delays may overwhelm services
4. **Don't retry without backoff**: Fixed delays can create cascading failures
5. **Don't ignore retry failures**: Log and alert on exhausted retries

## Retry Decision Logic

The system decides whether to retry based on exception type:

```python
# Retryable (transient failures)
ConnectionError          # Network connection failed
TimeoutError            # Operation timed out
OSError                 # Low-level I/O error

# Fatal (don't retry)
ValueError              # Invalid input data
TypeError               # Type mismatch
KeyError                # Missing key in data

# Custom exceptions
class RateLimitError(Exception):  # Should retry
    pass

class AuthError(Exception):       # Should not retry
    pass
```

## Performance Considerations

### Memory Usage

- Decorators are lightweight (minimal overhead)
- No state is stored between calls
- Each retry attempt runs in the same context

### Latency Impact

Example with 3 retries (initial_delay=1.0, base=2.0):
- Success on 1st try: 0ms additional latency
- Success on 2nd try: ~1000ms additional latency
- Success on 3rd try: ~3000ms additional latency (1s + 2s)
- All retries fail: ~7000ms total latency (1s + 2s + 4s)

### Recommendations

| Service Type | Max Retries | Initial Delay | Max Delay |
|-------------|-------------|---------------|-----------|
| Fast API calls | 3 | 0.5s | 10s |
| Database queries | 3 | 1.0s | 30s |
| Vector search | 3 | 1.0s | 30s |
| LLM generation | 2 | 2.0s | 60s |
| Batch operations | 5 | 5.0s | 120s |

## Monitoring and Logging

The retry system logs all retry attempts:

```python
# Log output example
2026-01-08 12:34:56 - WARNING - similarity_search: Attempt 1/4 failed: ConnectionError('Connection refused'). Retrying in 1.23s...
2026-01-08 12:34:57 - WARNING - similarity_search: Attempt 2/4 failed: ConnectionError('Connection refused'). Retrying in 2.45s...
2026-01-08 12:35:00 - INFO - similarity_search: Succeeded on attempt 3
```

### Monitoring Metrics

Track these metrics for retry effectiveness:

- **Retry rate**: Percentage of calls requiring retry
- **Success after retry**: Percentage succeeding after 1+ retries
- **Exhausted retries**: Calls that failed all retry attempts
- **Average retry count**: Mean retries per call
- **Total retry latency**: Time spent in retries

## Testing

The retry module includes comprehensive tests:

```bash
# Run retry tests
pytest tests/test_retry.py -v

# Test coverage
pytest tests/test_retry.py --cov=src.services.retry --cov-report=term-missing
```

### Test Scenarios Covered

- ✅ Success on first try (no retries)
- ✅ Success after transient failures
- ✅ Max retries exceeded
- ✅ Fatal exceptions (no retry)
- ✅ Exponential backoff timing
- ✅ Jitter randomization
- ✅ Custom exception configuration
- ✅ Async retry behavior

## API Reference

### Decorators

#### `retry_with_backoff(config)`

Decorator for synchronous functions with exponential backoff retry.

**Parameters:**
- `config` (RetryConfig): Retry configuration

**Returns:** Decorated function that retries on failure

**Example:**
```python
@retry_with_backoff(RetryConfig(max_retries=3))
def my_function():
    return api_call()
```

#### `async_retry_with_backoff(config)`

Decorator for asynchronous functions with exponential backoff retry.

**Parameters:**
- `config` (RetryConfig): Retry configuration

**Returns:** Decorated async function that retries on failure

**Example:**
```python
@async_retry_with_backoff(RetryConfig(max_retries=3))
async def my_async_function():
    return await api_call()
```

### Functions

#### `calculate_delay(attempt, initial_delay, exponential_base, max_delay, jitter)`

Calculate exponential backoff delay with optional jitter.

**Parameters:**
- `attempt` (int): Current attempt number (0-indexed)
- `initial_delay` (float): Base delay in seconds
- `exponential_base` (float): Exponential base
- `max_delay` (float): Maximum delay cap
- `jitter` (bool): Whether to add random jitter

**Returns:** float - Delay in seconds

#### `should_retry(exception, retryable_exceptions, fatal_exceptions)`

Determine if an exception should trigger a retry.

**Parameters:**
- `exception` (Exception): The raised exception
- `retryable_exceptions` (tuple): Exceptions that trigger retry
- `fatal_exceptions` (tuple): Exceptions that never retry

**Returns:** bool - True if should retry

### Classes

#### `RetryConfig`

Configuration dataclass for retry behavior.

**Attributes:**
- `max_retries` (int): Maximum retry attempts (default: 3)
- `initial_delay` (float): Initial delay in seconds (default: 1.0)
- `max_delay` (float): Maximum delay cap (default: 60.0)
- `exponential_base` (float): Exponential base (default: 2.0)
- `jitter` (bool): Enable jitter (default: True)
- `retryable_exceptions` (tuple): Exceptions to retry
- `fatal_exceptions` (tuple): Exceptions to never retry

#### `RetryableClient`

Base class for clients requiring retry logic.

**Methods:**
- `with_retry(func)`: Wrap synchronous function with retry
- `with_async_retry(func)`: Wrap async function with retry

**Example:**
```python
class MyClient(RetryableClient):
    def __init__(self):
        super().__init__(RetryConfig(max_retries=5))
```

## Troubleshooting

### Issue: Too Many Retries

**Symptom:** Requests take very long to fail

**Solution:** Reduce `max_retries` or `max_delay`:
```python
config = RetryConfig(max_retries=2, max_delay=10.0)
```

### Issue: Thundering Herd

**Symptom:** All clients retry at same time, overwhelming service

**Solution:** Enable jitter:
```python
config = RetryConfig(jitter=True)  # Adds randomness to delays
```

### Issue: Wrong Exceptions Being Retried

**Symptom:** Retrying on errors that won't recover

**Solution:** Configure appropriate exceptions:
```python
config = RetryConfig(
    retryable_exceptions=(ConnectionError, TimeoutError),
    fatal_exceptions=(ValueError, AuthenticationError),
)
```

## Future Enhancements

Potential improvements:
- [ ] Adaptive retry (adjust delays based on error patterns)
- [ ] Circuit breaker integration
- [ ] Retry budget (limit total retry attempts across service)
- [ ] Retry metrics collection
- [ ] Distributed tracing integration

## See Also

- [Health Check Documentation](./health-check.md)
- [VectorStore Client](./vectorstore-client.md)
- [Generation Client](./generation-client.md)
