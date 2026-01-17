"""
Distributed tracing module for RAG pipeline.

Provides span-based tracing with correlation IDs that propagate
across all pipeline stages (ingestion, retrieval, reranking, generation).

Features:
- Automatic span creation and timing
- Parent-child span relationships
- Correlation ID propagation
- Trace export to various backends
- Span attributes and events
"""

import uuid
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from functools import wraps
from threading import local
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


class SpanStatus(Enum):
    """Status of a span."""

    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


@dataclass
class SpanContext:
    """
    Context for a span, containing trace and span identifiers.

    Attributes:
        trace_id: Unique identifier for the entire trace
        span_id: Unique identifier for this specific span
        parent_span_id: ID of the parent span (None if root)
        correlation_id: Request-level correlation ID for log linking
        baggage: Key-value pairs propagated across span boundaries
    """

    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    correlation_id: str | None = None
    baggage: dict[str, str] = field(default_factory=dict)

    @classmethod
    def new_root(cls, correlation_id: str | None = None) -> "SpanContext":
        """Create a new root span context."""
        return cls(
            trace_id=str(uuid.uuid4()),
            span_id=str(uuid.uuid4()),
            parent_span_id=None,
            correlation_id=correlation_id or str(uuid.uuid4()),
        )

    def create_child(self) -> "SpanContext":
        """Create a child span context."""
        return SpanContext(
            trace_id=self.trace_id,
            span_id=str(uuid.uuid4()),
            parent_span_id=self.span_id,
            correlation_id=self.correlation_id,
            baggage=self.baggage.copy(),
        )


@dataclass
class SpanEvent:
    """An event recorded within a span."""

    name: str
    timestamp: datetime
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    """
    Represents a single operation in the trace.

    Spans track the duration and metadata of operations like:
    - Document ingestion
    - Embedding generation
    - Vector search
    - Reranking
    - LLM generation
    """

    name: str
    context: SpanContext
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime | None = None
    status: SpanStatus = SpanStatus.UNSET
    status_message: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[SpanEvent] = field(default_factory=list)

    @property
    def duration_ms(self) -> float | None:
        """Calculate span duration in milliseconds."""
        if self.end_time is None:
            return None
        delta = self.end_time - self.start_time
        return delta.total_seconds() * 1000

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Add an event to the span."""
        self.events.append(
            SpanEvent(
                name=name,
                timestamp=datetime.now(UTC),
                attributes=attributes or {},
            )
        )

    def set_status(self, status: SpanStatus, message: str | None = None) -> None:
        """Set the span status."""
        self.status = status
        self.status_message = message

    def end(self) -> None:
        """End the span and record the end time."""
        if self.end_time is None:
            self.end_time = datetime.now(UTC)

    def to_dict(self) -> dict[str, Any]:
        """Convert span to dictionary for export."""
        return {
            "name": self.name,
            "trace_id": self.context.trace_id,
            "span_id": self.context.span_id,
            "parent_span_id": self.context.parent_span_id,
            "correlation_id": self.context.correlation_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "status_message": self.status_message,
            "attributes": self.attributes,
            "events": [
                {
                    "name": e.name,
                    "timestamp": e.timestamp.isoformat(),
                    "attributes": e.attributes,
                }
                for e in self.events
            ],
        }


class TraceExporter:
    """
    Base class for trace exporters.

    Exporters send completed traces to various backends:
    - Console (for debugging)
    - JSON file (for local analysis)
    - OpenTelemetry collector (for production)
    """

    def export(self, spans: list[Span]) -> None:
        """Export a batch of spans."""
        raise NotImplementedError

    def shutdown(self) -> None:
        """Cleanup resources."""
        pass


class ConsoleExporter(TraceExporter):
    """Export spans to console for debugging."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def export(self, spans: list[Span]) -> None:
        for span in spans:
            if self.verbose:
                print(f"[TRACE] {span.to_dict()}")  # noqa: T201
            else:
                duration = f"{span.duration_ms:.2f}ms" if span.duration_ms else "in-progress"
                print(  # noqa: T201
                    f"[TRACE] {span.name} ({duration}) "
                    f"trace_id={span.context.trace_id[:8]}... "
                    f"status={span.status.value}"
                )


class InMemoryExporter(TraceExporter):
    """Store spans in memory for testing and analysis."""

    def __init__(self, max_spans: int = 10000):
        self.spans: list[Span] = []
        self.max_spans = max_spans

    def export(self, spans: list[Span]) -> None:
        self.spans.extend(spans)
        # Trim if over limit
        if len(self.spans) > self.max_spans:
            self.spans = self.spans[-self.max_spans :]

    def get_spans(self, trace_id: str | None = None) -> list[Span]:
        """Get stored spans, optionally filtered by trace ID."""
        if trace_id:
            return [s for s in self.spans if s.context.trace_id == trace_id]
        return self.spans.copy()

    def clear(self) -> None:
        """Clear all stored spans."""
        self.spans.clear()


class JSONFileExporter(TraceExporter):
    """Export spans to a JSON Lines file."""

    def __init__(self, file_path: str):
        import json
        from pathlib import Path

        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._json = json

    def export(self, spans: list[Span]) -> None:
        with self.file_path.open("a") as f:
            for span in spans:
                f.write(self._json.dumps(span.to_dict()) + "\n")


# Thread-local storage for current span context
_context_storage = local()


class Tracer:
    """
    Main tracing interface for the RAG pipeline.

    Usage:
        tracer = Tracer()

        with tracer.start_span("retrieval") as span:
            span.set_attribute("query", query)
            results = vectorstore.search(query)
            span.set_attribute("num_results", len(results))

        # Or use the decorator
        @tracer.trace("generation")
        def generate_response(prompt):
            return llm.generate(prompt)
    """

    def __init__(
        self,
        service_name: str = "rag-pipeline",
        exporters: list[TraceExporter] | None = None,
        enabled: bool = True,
    ):
        self.service_name = service_name
        self.exporters = exporters or []
        self.enabled = enabled
        self._pending_spans: list[Span] = []
        self._batch_size = 100

    def get_current_context(self) -> SpanContext | None:
        """Get the current span context from thread-local storage."""
        return getattr(_context_storage, "context", None)

    def set_current_context(self, context: SpanContext | None) -> None:
        """Set the current span context in thread-local storage."""
        _context_storage.context = context

    @contextmanager
    def start_span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ):
        """
        Start a new span as a context manager.

        Args:
            name: Name of the operation being traced
            attributes: Initial span attributes
            correlation_id: Optional correlation ID for request linking
        """
        if not self.enabled:
            yield None
            return

        # Get or create context
        parent_context = self.get_current_context()
        if parent_context:
            context = parent_context.create_child()
        else:
            context = SpanContext.new_root(correlation_id)

        # Create span
        span = Span(
            name=name,
            context=context,
            attributes=attributes or {},
        )
        span.set_attribute("service.name", self.service_name)

        # Set as current context
        previous_context = self.get_current_context()
        self.set_current_context(context)

        try:
            yield span
            if span.status == SpanStatus.UNSET:
                span.set_status(SpanStatus.OK)
        except Exception as e:
            span.set_status(SpanStatus.ERROR, str(e))
            span.add_event("exception", {"type": type(e).__name__, "message": str(e)})
            raise
        finally:
            span.end()
            self._pending_spans.append(span)
            self.set_current_context(previous_context)

            # Export if batch is full
            if len(self._pending_spans) >= self._batch_size:
                self._flush()

    def trace(
        self,
        name: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Callable[[F], F]:
        """
        Decorator to trace a function.

        Args:
            name: Span name (defaults to function name)
            attributes: Initial span attributes
        """

        def decorator(func: F) -> F:
            span_name = name or func.__name__

            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.start_span(span_name, attributes) as span:
                    if span:
                        span.set_attribute("function.args_count", len(args))
                        span.set_attribute("function.kwargs_keys", list(kwargs.keys()))
                    return func(*args, **kwargs)

            return wrapper  # type: ignore[return-value]

        return decorator

    def _flush(self) -> None:
        """Flush pending spans to exporters."""
        import contextlib

        if not self._pending_spans:
            return

        spans_to_export = self._pending_spans.copy()
        self._pending_spans.clear()

        for exporter in self.exporters:
            with contextlib.suppress(Exception):
                exporter.export(spans_to_export)

    def force_flush(self) -> None:
        """Force flush all pending spans."""
        self._flush()

    def shutdown(self) -> None:
        """Shutdown the tracer and exporters."""
        self._flush()
        for exporter in self.exporters:
            exporter.shutdown()


# Predefined span names for RAG pipeline stages
class RAGSpanNames:
    """Standard span names for RAG pipeline operations."""

    # Ingestion
    INGEST_DOCUMENTS = "rag.ingest.documents"
    INGEST_PARSE = "rag.ingest.parse"
    INGEST_CHUNK = "rag.ingest.chunk"

    # Embedding
    EMBED_TEXTS = "rag.embed.texts"
    EMBED_QUERY = "rag.embed.query"

    # Vector Store
    VECTORSTORE_INDEX = "rag.vectorstore.index"
    VECTORSTORE_SEARCH = "rag.vectorstore.search"
    VECTORSTORE_HYBRID = "rag.vectorstore.hybrid_search"

    # Retrieval
    RETRIEVE_DOCUMENTS = "rag.retrieve.documents"
    RETRIEVE_FILTER = "rag.retrieve.filter"

    # Reranking
    RERANK_DOCUMENTS = "rag.rerank.documents"
    RERANK_SCORE = "rag.rerank.score"

    # Generation
    GENERATE_PROMPT = "rag.generate.prompt"
    GENERATE_CONTEXT = "rag.generate.context"
    GENERATE_RESPONSE = "rag.generate.response"

    # Query Understanding
    QUERY_REWRITE = "rag.query.rewrite"
    QUERY_EXPAND = "rag.query.expand"
    QUERY_CLASSIFY = "rag.query.classify"

    # Agent
    AGENT_PLAN = "rag.agent.plan"
    AGENT_EXECUTE = "rag.agent.execute"
    AGENT_REFLECT = "rag.agent.reflect"

    # Guardrails
    GUARDRAILS_INPUT = "rag.guardrails.input"
    GUARDRAILS_OUTPUT = "rag.guardrails.output"

    # Cache
    CACHE_LOOKUP = "rag.cache.lookup"
    CACHE_STORE = "rag.cache.store"


# Default global tracer instance
_default_tracer: Tracer | None = None


def get_tracer() -> Tracer:
    """Get the default global tracer."""
    global _default_tracer
    if _default_tracer is None:
        _default_tracer = Tracer()
    return _default_tracer


def set_tracer(tracer: Tracer) -> None:
    """Set the default global tracer."""
    global _default_tracer
    _default_tracer = tracer


def create_rag_tracer(
    service_name: str = "rag-pipeline",
    enable_console: bool = False,
    enable_file: bool = True,
    file_path: str = "./logs/traces.jsonl",
    enabled: bool = True,
) -> Tracer:
    """
    Create a pre-configured tracer for RAG pipeline.

    Args:
        service_name: Name of the service for trace metadata
        enable_console: Enable console output for debugging
        enable_file: Enable JSON file export
        file_path: Path for trace file
        enabled: Enable/disable tracing

    Returns:
        Configured Tracer instance
    """
    exporters: list[TraceExporter] = []

    if enable_console:
        exporters.append(ConsoleExporter(verbose=False))

    if enable_file:
        exporters.append(JSONFileExporter(file_path))

    return Tracer(
        service_name=service_name,
        exporters=exporters,
        enabled=enabled,
    )
