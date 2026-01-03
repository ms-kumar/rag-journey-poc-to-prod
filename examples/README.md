# RAG Examples

This directory contains example scripts demonstrating RAG system features and benchmarks.

## Scripts

### `cache_demo.py`
Demonstrates the embedding cache performance with side-by-side comparison.

**Features:**
- Cache hit vs miss comparison
- Performance speedup measurement
- Cache statistics display

**Usage:**
```bash
uv run python examples/cache_demo.py
```

**Expected Output:**
- First run (cold cache): ~5ms
- Second run (warm cache): ~0.06ms
- **Speedup: 83x**

### `benchmark_retrieval.py`
Comprehensive benchmark of RAG pipeline components including retrieval quality and latency.

**Measures:**
- Retrieval@k latency for k=[1, 3, 5, 10]
- End-to-end pipeline latency
- Cache performance impact
- Component-level breakdown

**Usage:**
```bash
uv run python examples/benchmark_retrieval.py
```

**Key Metrics:**

#### Indexing Pipeline (with warm cache)
- **Ingestion**: ~1ms (2.2%)
- **Chunking**: ~0.9ms (1.9%)
- **Embedding**: ~0.6ms (1.2%) - **83x faster with cache**
- **Indexing**: ~40ms (85.9%)
- **Total**: ~46ms

#### Retrieval Performance
- **k=1**: 4.06ms
- **k=3**: 2.38ms
- **k=5**: 2.23ms
- **k=10**: 2.61ms

**Observations:**
- Retrieval latency is relatively constant across different k values (2-4ms)
- Cache provides **1.86x speedup** on indexing pipeline
- Indexing (vector storage) dominates the pipeline at 86%
- Generation is the slowest step at ~8-11 seconds (CPU-based GPT-2)

#### Full End-to-End
- **Indexing + Retrieval**: ~50ms
- **+ Generation**: ~8.4 seconds
- **Generation dominates** at 99.4% of total time

## Benchmark Insights

### Performance Bottlenecks
1. **Generation (99.4%)** - CPU-based text generation is the primary bottleneck
2. **Indexing (85% of non-generation time)** - Vector storage operations
3. **Retrieval (9%)** - Fast and scales well with k

### Optimization Opportunities
1. **Use GPU for generation** - Can reduce generation time by 10-100x
2. **Batch indexing** - Could improve vector storage throughput
3. **Cache embeddings** - Already implemented, provides 1.86x speedup
4. **Async retrieval** - For multiple queries simultaneously

### Cache Impact
- **Embedding cache speedup**: 9-83x depending on query patterns
- **Overall pipeline speedup**: 1.86x with warm cache
- **Cache hit rate**: 50% (11/22 operations on repeated queries)

## Running Benchmarks

### Quick Benchmark
```bash
make benchmark  # If added to Makefile
```

### Custom Configuration
```bash
# Override settings via environment variables
CHUNK_SIZE=500 EMBED_CACHE_ENABLED=false uv run python examples/benchmark_retrieval.py
```

### With Different Providers
```bash
# Test with different embedding providers
EMBED_PROVIDER=e5 uv run python examples/benchmark_retrieval.py
EMBED_PROVIDER=bge uv run python examples/benchmark_retrieval.py
```

## Adding New Benchmarks

Create new benchmark scripts following this pattern:

```python
from src.config import Settings
from examples.benchmark_retrieval import timeit, RetrievalBenchmark

@timeit
def your_benchmark_function():
    # Your benchmark code
    pass

if __name__ == "__main__":
    config = Settings()
    # Run your benchmark
```

## Interpreting Results

### Latency Guidelines
- **Excellent**: < 100ms end-to-end (excluding generation)
- **Good**: 100-500ms
- **Acceptable**: 500ms-2s
- **Slow**: > 2s (needs optimization)

### Retrieval@k Quality
- **Top score > 0.5**: Strong relevance
- **Top score 0.3-0.5**: Moderate relevance
- **Top score < 0.3**: Weak relevance (consider reindexing or better chunking)

### Cache Effectiveness
- **Hit rate > 80%**: Excellent cache utilization
- **Hit rate 50-80%**: Good, typical for production
- **Hit rate < 50%**: Consider increasing cache size or reviewing query patterns

## Future Benchmarks

Planned additions:
- [ ] Recall@k measurement with ground truth
- [ ] Query latency under load (concurrent requests)
- [ ] Memory usage profiling
- [ ] Different embedding provider comparisons
- [ ] Chunking strategy impact on retrieval quality
