"""
Observability module for RAG pipeline.

Provides tracing, structured logging, metrics collection,
SLO monitoring, and golden trace management.

Components:
- Tracing: Distributed tracing with correlation IDs across pipeline stages
- Logging: Structured logging with JSON format and correlation ID propagation
- Metrics: Latency, cost, and quality metrics for dashboards
- SLO: Service Level Objective monitoring and alerting
- Golden Traces: Reference traces for regression testing and debugging
"""

from src.services.observability.golden_traces import (
    GoldenTrace,
    GoldenTraceManager,
    GoldenTraceStore,
    TraceComparison,
)
from src.services.observability.logging import (
    CorrelationContext,
    LogEntry,
    StructuredLogger,
    get_correlation_id,
    set_correlation_id,
)
from src.services.observability.metrics import (
    CostMetrics,
    DashboardData,
    LatencyMetrics,
    MetricsCollector,
    QualityMetrics,
)
from src.services.observability.slo import (
    SLOAlert,
    SLODefinition,
    SLOMonitor,
    SLOStatus,
)
from src.services.observability.tracing import (
    Span,
    SpanContext,
    TraceExporter,
    Tracer,
)

__all__ = [
    # Tracing
    "Span",
    "SpanContext",
    "Tracer",
    "TraceExporter",
    # Logging
    "CorrelationContext",
    "LogEntry",
    "StructuredLogger",
    "get_correlation_id",
    "set_correlation_id",
    # Metrics
    "CostMetrics",
    "DashboardData",
    "LatencyMetrics",
    "MetricsCollector",
    "QualityMetrics",
    # SLO
    "SLOAlert",
    "SLODefinition",
    "SLOMonitor",
    "SLOStatus",
    # Golden Traces
    "GoldenTrace",
    "GoldenTraceManager",
    "GoldenTraceStore",
    "TraceComparison",
]
