# Performance Profiling & SLA Monitoring

Comprehensive performance profiling system for monitoring latency, throughput, and SLA compliance in RAG pipelines.

## Quick Start

```python
from src.services.performance import PerformanceProfiler, SLAConfig

# Create profiler with SLA configuration
sla_config = SLAConfig(
    max_p50_latency_ms=150.0,
    max_p95_latency_ms=300.0,
    max_p99_latency_ms=500.0,
    min_throughput_rps=10.0,
    min_success_rate=0.95,
)
profiler = PerformanceProfiler(sla_config=sla_config)

# Time operations using context manager
with profiler.timer("embedding"):
    embeddings = model.embed(query)

with profiler.timer("retrieval"):
    results = index.search(embeddings)

# Get performance statistics
stats = profiler.get_all_stats()

# Check SLA compliance
result = profiler.check_sla()
if not result.passed:
    print(f"SLA violations: {result.violations}")
```

## Features

### 1. Performance Timers

Context-managed timers for automatic operation tracking:

```python
# Basic timing
with profiler.timer("my_operation"):
    do_something()

# With metadata
with profiler.timer("query", metadata={"query_id": "123"}):
    process_query()

# Manual failure marking
with profiler.timer("risky_op") as timer:
    try:
        risky_operation()
    except SpecificError:
        timer.mark_failure()
        handle_error()
```

### 2. Percentile Tracking

Tracks latency percentiles for all operations:

- **P50** (median) - typical operation latency
- **P90** - 90th percentile
- **P95** - 95th percentile (common SLA target)
- **P99** - 99th percentile (tail latency)
- **Mean, Min, Max** - additional statistics

```python
stats = profiler.get_operation_stats("embedding")
print(f"P50: {stats['latency']['p50']:.2f}ms")
print(f"P95: {stats['latency']['p95']:.2f}ms")
print(f"P99: {stats['latency']['p99']:.2f}ms")
```

### 3. Throughput Measurement

Measures operations per second (RPS):

```python
stats = profiler.get_all_stats()
print(f"Throughput: {stats['overall']['throughput_rps']:.1f} RPS")

# Per-operation throughput
op_stats = profiler.get_operation_stats("retrieval")
print(f"Retrieval: {op_stats['throughput_rps']:.1f} RPS")
```

### 4. SLA Configuration

Define performance targets and validate compliance:

```python
# Overall SLA targets
sla_config = SLAConfig(
    max_p50_latency_ms=100.0,    # P50 must be under 100ms
    max_p95_latency_ms=250.0,    # P95 must be under 250ms
    max_p99_latency_ms=500.0,    # P99 must be under 500ms
    min_throughput_rps=20.0,     # Must handle 20 requests/sec
    min_success_rate=0.99,       # Must succeed 99% of the time
)

# Per-operation SLA targets
sla_config = SLAConfig(
    operation_slas={
        "embedding": {"max_p95": 50.0},
        "retrieval": {"max_p95": 100.0},
        "reranking": {"max_p95": 30.0},
        "generation": {"max_p95": 200.0},
    }
)

# Check compliance
result = profiler.check_sla()
if result.passed:
    print("✓ All SLA targets met")
else:
    print("✗ SLA violations:")
    for violation in result.violations:
        print(f"  - {violation}")
```

### 5. Multi-Format Reports

Generate reports in multiple formats:

```python
from src.services.performance import PerformanceReporter

reporter = PerformanceReporter()
stats = profiler.get_all_stats()

# Console output (for logs)
reporter.print_summary(stats)

# JSON (for programmatic access)
reporter.export_json(stats, "reports/performance.json")

# Markdown (for documentation)
reporter.export_markdown(stats, "reports/performance.md")

# HTML (for dashboards)
reporter.export_html(stats, "reports/performance.html")
```

## Architecture

### Components

1. **PerformanceProfiler** - Core profiling engine
   - Records operation metrics
   - Calculates percentiles and throughput
   - Validates SLA compliance

2. **PerformanceTimer** - Context manager for timing
   - Automatic duration tracking
   - Success/failure recording
   - Metadata collection

3. **SLAConfig** - SLA threshold configuration
   - Overall latency targets
   - Throughput requirements
   - Per-operation limits

4. **PerformanceReporter** - Report generation
   - Console summaries
   - JSON/Markdown/HTML export
   - Color-coded status indicators

### Data Flow

```
Operation Execution
        ↓
PerformanceTimer (context manager)
        ↓
Record duration + metadata
        ↓
OperationStats (per-operation tracking)
        ↓
PerformanceProfiler (aggregation)
        ↓
Calculate percentiles & throughput
        ↓
Check SLA compliance
        ↓
Generate reports
```

## Usage Patterns

### RAG Pipeline Integration

```python
class RAGPipeline:
    def __init__(self):
        sla_config = SLAConfig(
            max_p95_latency_ms=500.0,
            min_throughput_rps=10.0,
        )
        self.profiler = PerformanceProfiler(sla_config)

    async def process_query(self, query: str) -> dict:
        # Time end-to-end
        with self.profiler.timer("end_to_end", metadata={"query": query}):
            # Time embedding
            with self.profiler.timer("embedding"):
                embedding = await self.embed(query)

            # Time retrieval
            with self.profiler.timer("retrieval"):
                docs = await self.retrieve(embedding)

            # Time reranking
            with self.profiler.timer("reranking"):
                ranked_docs = await self.rerank(query, docs)

            # Time generation
            with self.profiler.timer("generation"):
                response = await self.generate(query, ranked_docs)

        return response

    def get_performance_report(self):
        return self.profiler.get_all_stats()
```

### API Endpoint Monitoring

```python
from fastapi import FastAPI

app = FastAPI()
profiler = PerformanceProfiler()

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    with profiler.timer("api_query", metadata={"user_id": request.user_id}):
        try:
            result = await process_query(request.query)
            return {"success": True, "result": result}
        except Exception as e:
            profiler.timer("api_query").mark_failure()
            raise

@app.get("/metrics")
async def metrics_endpoint():
    stats = profiler.get_all_stats()
    sla_result = profiler.check_sla()
    return {
        "performance": stats,
        "sla_status": sla_result.passed,
        "violations": sla_result.violations,
    }
```

### Throughput Testing

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def run_concurrent_test(num_queries: int, max_workers: int):
    profiler = PerformanceProfiler()

    async def process_single_query(query_id: int):
        with profiler.timer("concurrent_query", metadata={"id": query_id}):
            await process_query(f"Query {query_id}")

    # Run queries concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        tasks = [process_single_query(i) for i in range(num_queries)]
        await asyncio.gather(*tasks)

    # Report results
    stats = profiler.get_all_stats()
    print(f"Throughput: {stats['overall']['throughput_rps']:.1f} RPS")
    print(f"P95 latency: {stats['overall']['latency']['p95']:.2f}ms")
```

## Best Practices

### 1. Define Realistic SLAs

Base SLA targets on actual measurements:

```python
# Run baseline tests first
baseline_profiler = PerformanceProfiler()
# ... run representative queries ...
baseline_stats = baseline_profiler.get_all_stats()

# Set SLAs with buffer (e.g., 20% above baseline P95)
sla_config = SLAConfig(
    max_p95_latency_ms=baseline_stats['overall']['latency']['p95'] * 1.2,
    min_throughput_rps=baseline_stats['overall']['throughput_rps'] * 0.8,
)
```

### 2. Monitor Per-Operation Performance

Track each pipeline stage separately:

```python
# Identify bottlenecks
for op_name, op_stats in stats['operations'].items():
    p95 = op_stats['latency']['p95']
    if p95 > TARGET_LATENCY:
        logger.warning(f"{op_name} exceeds target: {p95:.2f}ms")
```

### 3. Handle Failures Explicitly

Mark failures for accurate success rate tracking:

```python
with profiler.timer("api_call") as timer:
    response = await call_external_api()
    if response.status_code >= 500:
        timer.mark_failure()
```

### 4. Export Reports Regularly

Generate reports for historical analysis:

```python
# Daily performance report
timestamp = datetime.now().strftime("%Y%m%d")
reporter.export_html(stats, f"reports/performance_{timestamp}.html")
```

### 5. Reset Metrics Periodically

Clear metrics for fresh measurements:

```python
# Reset every hour for real-time monitoring
if time.time() - last_reset > 3600:
    profiler.reset()
    last_reset = time.time()
```

## Examples

See example scripts for complete demonstrations:

- **[performance_profiling_demo.py](../examples/performance_profiling_demo.py)** - Basic usage with simulated RAG pipeline
- **[throughput_test.py](../examples/throughput_test.py)** - Load testing scenarios (sequential, concurrent, stress)

## Metrics Reference

### Overall Metrics

- `total_requests` - Total operations recorded
- `total_successes` - Successfully completed operations
- `total_failures` - Failed operations
- `success_rate` - Percentage of successful operations
- `throughput_rps` - Operations per second
- `latency.p50` - Median latency (ms)
- `latency.p90` - 90th percentile latency (ms)
- `latency.p95` - 95th percentile latency (ms)
- `latency.p99` - 99th percentile latency (ms)
- `latency.mean` - Average latency (ms)
- `latency.min` - Minimum latency (ms)
- `latency.max` - Maximum latency (ms)

### Per-Operation Metrics

Same metrics as overall, but specific to each operation:

```json
{
  "operations": {
    "embedding": {
      "count": 100,
      "successes": 100,
      "failures": 0,
      "success_rate": 1.0,
      "throughput_rps": 20.5,
      "latency": {
        "p50": 12.3,
        "p95": 15.8,
        "p99": 18.2
      }
    }
  }
}
```

## SLA Violations

Common violation types:

1. **High Latency** - `"Overall P95 latency (325.4ms) exceeds target (300.0ms)"`
2. **Low Throughput** - `"Overall throughput (8.5 RPS) below target (10.0 RPS)"`
3. **Low Success Rate** - `"Overall success rate (92.3%) below target (95.0%)"`
4. **Per-Operation** - `"Operation 'retrieval' P95 (125.6ms) exceeds target (100.0ms)"`

## Troubleshooting

### High Latency

- Check per-operation stats to identify bottleneck
- Review P99 for tail latency issues
- Consider caching for repeated operations

### Low Throughput

- Increase concurrency/parallelism
- Optimize slow operations
- Scale infrastructure

### Low Success Rate

- Check failure metadata for error patterns
- Review retry/timeout configurations
- Improve error handling

## Related Documentation

- [Evaluation Harness](./evaluation-harness.md) - Quality metrics and testing
- [Health Check](./health-check.md) - Service availability monitoring
- [Retry & Backoff](./retry-backoff.md) - Resilience patterns
- [Token Budgets](./token-budgets.md) - Resource management
