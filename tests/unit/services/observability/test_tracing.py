"""Unit tests for the tracing module."""

import time

import pytest

from src.services.observability.tracing import (
    ConsoleExporter,
    InMemoryExporter,
    JSONFileExporter,
    RAGSpanNames,
    Span,
    SpanContext,
    SpanStatus,
    Tracer,
    create_rag_tracer,
    get_tracer,
    set_tracer,
)


class TestSpanContext:
    """Tests for SpanContext."""

    def test_new_root_creates_unique_ids(self):
        """Test that new root contexts have unique IDs."""
        ctx1 = SpanContext.new_root()
        ctx2 = SpanContext.new_root()

        assert ctx1.trace_id != ctx2.trace_id
        assert ctx1.span_id != ctx2.span_id
        assert ctx1.parent_span_id is None
        assert ctx2.parent_span_id is None

    def test_new_root_with_correlation_id(self):
        """Test creating root context with correlation ID."""
        ctx = SpanContext.new_root(correlation_id="test-corr-123")
        assert ctx.correlation_id == "test-corr-123"

    def test_create_child_preserves_trace_id(self):
        """Test child context inherits trace ID."""
        parent = SpanContext.new_root()
        child = parent.create_child()

        assert child.trace_id == parent.trace_id
        assert child.span_id != parent.span_id
        assert child.parent_span_id == parent.span_id
        assert child.correlation_id == parent.correlation_id

    def test_create_child_copies_baggage(self):
        """Test child context copies baggage."""
        parent = SpanContext.new_root()
        parent.baggage["key"] = "value"

        child = parent.create_child()
        assert child.baggage["key"] == "value"

        # Modifications don't affect parent
        child.baggage["new_key"] = "new_value"
        assert "new_key" not in parent.baggage


class TestSpan:
    """Tests for Span."""

    def test_span_creation(self):
        """Test basic span creation."""
        ctx = SpanContext.new_root()
        span = Span(name="test-span", context=ctx)

        assert span.name == "test-span"
        assert span.context == ctx
        assert span.status == SpanStatus.UNSET
        assert span.end_time is None

    def test_span_duration(self):
        """Test span duration calculation."""
        ctx = SpanContext.new_root()
        span = Span(name="test-span", context=ctx)

        time.sleep(0.01)  # 10ms
        span.end()

        assert span.duration_ms is not None
        assert span.duration_ms >= 10

    def test_span_set_attribute(self):
        """Test setting span attributes."""
        ctx = SpanContext.new_root()
        span = Span(name="test-span", context=ctx)

        span.set_attribute("key", "value")
        span.set_attribute("count", 42)

        assert span.attributes["key"] == "value"
        assert span.attributes["count"] == 42

    def test_span_add_event(self):
        """Test adding events to span."""
        ctx = SpanContext.new_root()
        span = Span(name="test-span", context=ctx)

        span.add_event("cache_hit", {"key": "query-123"})

        assert len(span.events) == 1
        assert span.events[0].name == "cache_hit"
        assert span.events[0].attributes["key"] == "query-123"

    def test_span_set_status(self):
        """Test setting span status."""
        ctx = SpanContext.new_root()
        span = Span(name="test-span", context=ctx)

        span.set_status(SpanStatus.ERROR, "Something went wrong")

        assert span.status == SpanStatus.ERROR
        assert span.status_message == "Something went wrong"

    def test_span_to_dict(self):
        """Test span serialization."""
        ctx = SpanContext.new_root(correlation_id="corr-123")
        span = Span(name="test-span", context=ctx)
        span.set_attribute("query", "test query")
        span.end()

        data = span.to_dict()

        assert data["name"] == "test-span"
        assert data["trace_id"] == ctx.trace_id
        assert data["span_id"] == ctx.span_id
        assert data["correlation_id"] == "corr-123"
        assert data["attributes"]["query"] == "test query"
        assert data["duration_ms"] is not None


class TestInMemoryExporter:
    """Tests for InMemoryExporter."""

    def test_export_stores_spans(self):
        """Test that exported spans are stored."""
        exporter = InMemoryExporter()
        ctx = SpanContext.new_root()
        span = Span(name="test", context=ctx)
        span.end()

        exporter.export([span])

        assert len(exporter.get_spans()) == 1
        assert exporter.get_spans()[0].name == "test"

    def test_export_limits_max_spans(self):
        """Test that exporter respects max_spans limit."""
        exporter = InMemoryExporter(max_spans=10)

        for i in range(20):
            ctx = SpanContext.new_root()
            span = Span(name=f"span-{i}", context=ctx)
            span.end()
            exporter.export([span])

        assert len(exporter.get_spans()) == 10

    def test_get_spans_by_trace_id(self):
        """Test filtering spans by trace ID."""
        exporter = InMemoryExporter()

        ctx1 = SpanContext.new_root()
        ctx2 = SpanContext.new_root()

        span1 = Span(name="span1", context=ctx1)
        span2 = Span(name="span2", context=ctx1.create_child())
        span3 = Span(name="span3", context=ctx2)

        for span in [span1, span2, span3]:
            span.end()

        exporter.export([span1, span2, span3])

        trace1_spans = exporter.get_spans(trace_id=ctx1.trace_id)
        assert len(trace1_spans) == 2

    def test_clear(self):
        """Test clearing stored spans."""
        exporter = InMemoryExporter()
        ctx = SpanContext.new_root()
        span = Span(name="test", context=ctx)
        span.end()
        exporter.export([span])

        exporter.clear()
        assert len(exporter.get_spans()) == 0


class TestJSONFileExporter:
    """Tests for JSONFileExporter."""

    def test_export_writes_to_file(self, tmp_path):
        """Test that exporter writes spans to file."""
        file_path = tmp_path / "traces.jsonl"
        exporter = JSONFileExporter(str(file_path))

        ctx = SpanContext.new_root()
        span = Span(name="test", context=ctx)
        span.end()

        exporter.export([span])

        assert file_path.exists()
        content = file_path.read_text()
        assert "test" in content


class TestTracer:
    """Tests for Tracer."""

    def test_start_span_creates_span(self):
        """Test starting a span creates proper span."""
        tracer = Tracer(enabled=True)

        with tracer.start_span("test-operation") as span:
            assert span is not None
            assert span.name == "test-operation"

    def test_start_span_sets_status_ok(self):
        """Test successful span has OK status."""
        exporter = InMemoryExporter()
        tracer = Tracer(exporters=[exporter], enabled=True)

        with tracer.start_span("test"):
            pass

        tracer.force_flush()
        spans = exporter.get_spans()
        assert len(spans) == 1
        assert spans[0].status == SpanStatus.OK

    def test_start_span_sets_status_error_on_exception(self):
        """Test exception sets span status to ERROR."""
        exporter = InMemoryExporter()
        tracer = Tracer(exporters=[exporter], enabled=True)

        with pytest.raises(ValueError), tracer.start_span("test"):
            raise ValueError("test error")

        tracer.force_flush()
        spans = exporter.get_spans()
        assert len(spans) == 1
        assert spans[0].status == SpanStatus.ERROR
        assert "test error" in spans[0].status_message

    def test_nested_spans_have_parent_child_relationship(self):
        """Test nested spans have correct parent-child relationship."""
        exporter = InMemoryExporter()
        tracer = Tracer(exporters=[exporter], enabled=True)

        with tracer.start_span("parent") as parent_span:  # noqa: SIM117
            with tracer.start_span("child") as child_span:
                assert child_span.context.parent_span_id == parent_span.context.span_id
                assert child_span.context.trace_id == parent_span.context.trace_id

    def test_disabled_tracer_yields_none(self):
        """Test disabled tracer yields None span."""
        tracer = Tracer(enabled=False)

        with tracer.start_span("test") as span:
            assert span is None

    def test_trace_decorator(self):
        """Test trace decorator creates spans."""
        exporter = InMemoryExporter()
        tracer = Tracer(exporters=[exporter], enabled=True)

        @tracer.trace("decorated-function")
        def my_function(x, y):
            return x + y

        result = my_function(1, 2)

        assert result == 3
        tracer.force_flush()
        spans = exporter.get_spans()
        assert len(spans) == 1
        assert spans[0].name == "decorated-function"

    def test_trace_decorator_uses_function_name_by_default(self):
        """Test trace decorator uses function name if not specified."""
        exporter = InMemoryExporter()
        tracer = Tracer(exporters=[exporter], enabled=True)

        @tracer.trace()
        def my_named_function():
            return 42

        my_named_function()
        tracer.force_flush()

        spans = exporter.get_spans()
        assert spans[0].name == "my_named_function"

    def test_correlation_id_propagation(self):
        """Test correlation ID propagates through spans."""
        exporter = InMemoryExporter()
        tracer = Tracer(exporters=[exporter], enabled=True)

        with tracer.start_span("parent", correlation_id="corr-123"):  # noqa: SIM117
            with tracer.start_span("child") as child:
                assert child.context.correlation_id == "corr-123"

    def test_shutdown_flushes_pending_spans(self):
        """Test shutdown flushes all pending spans."""
        exporter = InMemoryExporter()
        tracer = Tracer(exporters=[exporter], enabled=True)

        with tracer.start_span("test"):
            pass

        # Before shutdown, might not be flushed
        tracer.shutdown()

        assert len(exporter.get_spans()) >= 1


class TestRagSpanNames:
    """Tests for RAGSpanNames constants."""

    def test_span_names_are_strings(self):
        """Test that all span names are strings."""
        assert isinstance(RAGSpanNames.INGEST_DOCUMENTS, str)
        assert isinstance(RAGSpanNames.EMBED_TEXTS, str)
        assert isinstance(RAGSpanNames.RETRIEVE_DOCUMENTS, str)
        assert isinstance(RAGSpanNames.GENERATE_RESPONSE, str)

    def test_span_names_have_rag_prefix(self):
        """Test that span names follow naming convention."""
        assert RAGSpanNames.INGEST_DOCUMENTS.startswith("rag.")
        assert RAGSpanNames.RERANK_DOCUMENTS.startswith("rag.")


class TestCreateRagTracer:
    """Tests for create_rag_tracer helper."""

    def test_creates_tracer_with_console_exporter(self):
        """Test creating tracer with console exporter."""
        tracer = create_rag_tracer(
            service_name="test-service",
            enable_console=True,
            enable_file=False,
        )

        assert tracer.service_name == "test-service"
        assert tracer.enabled is True
        assert any(isinstance(e, ConsoleExporter) for e in tracer.exporters)

    def test_creates_tracer_with_file_exporter(self, tmp_path):
        """Test creating tracer with file exporter."""
        file_path = tmp_path / "traces.jsonl"
        tracer = create_rag_tracer(
            enable_console=False,
            enable_file=True,
            file_path=str(file_path),
        )

        assert any(isinstance(e, JSONFileExporter) for e in tracer.exporters)

    def test_creates_disabled_tracer(self):
        """Test creating disabled tracer."""
        tracer = create_rag_tracer(enabled=False)
        assert tracer.enabled is False


class TestGlobalTracer:
    """Tests for global tracer functions."""

    def test_get_tracer_returns_default(self):
        """Test get_tracer returns a tracer."""
        tracer = get_tracer()
        assert isinstance(tracer, Tracer)

    def test_set_tracer_changes_global(self):
        """Test set_tracer changes the global tracer."""
        custom_tracer = Tracer(service_name="custom")
        set_tracer(custom_tracer)

        assert get_tracer().service_name == "custom"
