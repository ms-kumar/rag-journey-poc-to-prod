# Observability Quick Start

This guide covers the observability module for production monitoring of the RAG pipeline.

## Features

- **Distributed Tracing**: Track requests across pipeline stages with correlation IDs
- **Structured Logging**: JSON-formatted logs with context propagation
- **Metrics Collection**: Latency, cost, and quality metrics for dashboards
- **SLO Monitoring**: Service Level Objective tracking with alerting
- **Golden Traces**: Reference traces for regression testing

## Quick Start

### 1. Enable Tracing

```python
from src.services.observability import Tracer, RAGSpanNames

# Create a tracer
tracer = Tracer(service_name="rag-service", enabled=True)

# Trace a pipeline operation
with tracer.start_span(RAGSpanNames.RETRIEVAL) as span:
    span.set_attribute("query", "What is RAG?")
    results = retrieve_documents(query)
    span.set_attribute("num_results", len(results))
```

### 2. Add Structured Logging

```python
from src.services.observability import StructuredLogger, CorrelationContext

# Create a logger
logger = StructuredLogger(name="retrieval", service_name="rag-service")

# Log with correlation ID
with CorrelationContext() as correlation_id:
    logger.info("Processing query", query="What is RAG?", user_id="user-123")
    # All subsequent logs will include the correlation_id
```

### 3. Collect Metrics

```python
from src.services.observability import MetricsCollector

collector = MetricsCollector()

# Time an operation
with collector.time_operation("retrieval"):
    results = retrieve_documents(query)

# Record quality score
collector.record_quality_score(score=0.92, query="What is RAG?")

# Get dashboard data
dashboard = collector.get_dashboard_data()
print(f"P99 latency: {dashboard.latency.percentiles['p99']}ms")
```

### 4. Monitor SLOs

```python
from src.services.observability import SLOMonitor, SLODefinition

# Create SLO monitor
monitor = SLOMonitor()

# Define an SLO
latency_slo = SLODefinition(
    name="latency_p99",
    slo_type="latency",
    target=0.99,  # 99% of requests under threshold
    threshold=500.0,  # 500ms threshold
)
monitor.register_slo(latency_slo)

# Record latency
monitor.record_latency("latency_p99", latency_ms=150.0)

# Get status
status = monitor.get_slo_status("latency_p99")
print(f"SLO compliance: {status.compliance:.2%}")
```

### 5. Golden Traces for Regression

```python
from src.services.observability import GoldenTraceManager

manager = GoldenTraceManager(storage_path="data/golden_traces")

# Capture a golden trace
manager.capture_golden_trace(
    trace_name="simple_query",
    spans=[span.to_dict() for span in current_trace],
    expected_quality=0.9,
)

# Compare against golden trace
comparison = manager.compare_trace(
    trace_name="simple_query",
    actual_spans=new_trace_spans,
    actual_latency_ms=150.0,
    actual_quality=0.88,
)
print(f"Trace matches: {comparison.passed}")
```

## Configuration

Add to your `.env` file:

```bash
# Observability settings
OBSERVABILITY__SERVICE_NAME=rag-service
OBSERVABILITY__ENVIRONMENT=production
OBSERVABILITY__TRACING_ENABLED=true
OBSERVABILITY__TRACE_FILE_PATH=logs/traces.jsonl
OBSERVABILITY__STRUCTURED_LOGGING=true
OBSERVABILITY__LOG_LEVEL=INFO
OBSERVABILITY__METRICS_ENABLED=true

# SLO targets
OBSERVABILITY__SLO_AVAILABILITY_TARGET=0.999
OBSERVABILITY__SLO_LATENCY_P99_MS=500.0
OBSERVABILITY__SLO_QUALITY_TARGET=0.8
OBSERVABILITY__SLO_ERROR_RATE_TARGET=0.001
```

## Dashboard Integration

Export metrics for Prometheus/Grafana:

```python
# Export metrics in Prometheus format
prometheus_output = collector.export_prometheus()
print(prometheus_output)

# Output:
# # HELP rag_latency_seconds RAG pipeline latency
# # TYPE rag_latency_seconds histogram
# rag_latency_seconds{operation="retrieval",quantile="0.5"} 0.045
# rag_latency_seconds{operation="retrieval",quantile="0.95"} 0.120
# rag_latency_seconds{operation="retrieval",quantile="0.99"} 0.250
```

## Running Tests

```bash
# Run all observability tests
make test-observability

# Run specific test file
uv run pytest tests/unit/services/observability/test_tracing.py -v
```

## API Reference

See [docs/week-plans/week-8.md](week-plans/week-8.md) for full implementation details.
