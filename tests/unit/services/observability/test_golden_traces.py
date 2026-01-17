"""Unit tests for the golden traces module."""

from datetime import UTC, datetime

from src.services.observability.golden_traces import (
    GoldenTrace,
    GoldenTraceManager,
    GoldenTraceStore,
    TraceComparison,
    create_rag_golden_templates,
)
from src.services.observability.tracing import Span, SpanContext


class TestGoldenTrace:
    """Tests for GoldenTrace."""

    def test_golden_trace_creation(self):
        """Test creating a golden trace."""
        trace = GoldenTrace(
            trace_id="trace-123",
            name="test_trace",
            description="A test golden trace",
            query="What is RAG?",
            expected_spans=["rag.embed.query", "rag.vectorstore.search"],
            expected_latency_ms=500.0,
            expected_quality_score=0.85,
        )

        assert trace.trace_id == "trace-123"
        assert trace.name == "test_trace"
        assert len(trace.expected_spans) == 2
        assert trace.expected_latency_ms == 500.0

    def test_to_dict(self):
        """Test golden trace serialization."""
        trace = GoldenTrace(
            trace_id="trace-123",
            name="test",
            description="Test",
            query="Test query",
            expected_spans=["span1"],
            expected_latency_ms=100.0,
        )

        data = trace.to_dict()

        assert data["trace_id"] == "trace-123"
        assert data["name"] == "test"
        assert data["expected_spans"] == ["span1"]

    def test_from_dict(self):
        """Test golden trace deserialization."""
        data = {
            "trace_id": "trace-456",
            "name": "restored_trace",
            "description": "Restored",
            "query": "Test query",
            "expected_spans": ["span1", "span2"],
            "expected_latency_ms": 250.0,
            "expected_quality_score": 0.9,
        }

        trace = GoldenTrace.from_dict(data)

        assert trace.trace_id == "trace-456"
        assert trace.name == "restored_trace"
        assert len(trace.expected_spans) == 2
        assert trace.expected_quality_score == 0.9


class TestTraceComparison:
    """Tests for TraceComparison."""

    def test_trace_comparison_to_dict(self):
        """Test comparison result serialization."""
        comparison = TraceComparison(
            golden_trace_id="golden-123",
            actual_trace_id="actual-456",
            match=True,
            span_match=True,
            latency_match=True,
            quality_match=True,
            latency_diff_ms=10.5,
            latency_diff_percent=2.1,
        )

        data = comparison.to_dict()

        assert data["golden_trace_id"] == "golden-123"
        assert data["match"] is True
        assert data["latency_diff_ms"] == 10.5

    def test_trace_comparison_with_failures(self):
        """Test comparison with failures."""
        comparison = TraceComparison(
            golden_trace_id="golden-123",
            actual_trace_id="actual-456",
            match=False,
            span_match=False,
            latency_match=True,
            quality_match=False,
            missing_spans=["rag.rerank.documents"],
            extra_spans=["rag.cache.lookup"],
            quality_diff=-0.15,
        )

        assert comparison.match is False
        assert "rag.rerank.documents" in comparison.missing_spans
        assert "rag.cache.lookup" in comparison.extra_spans


class TestGoldenTraceStore:
    """Tests for GoldenTraceStore."""

    def test_save_and_get(self, tmp_path):
        """Test saving and retrieving a golden trace."""
        store = GoldenTraceStore(storage_path=str(tmp_path))

        trace = GoldenTrace(
            trace_id="store-test-123",
            name="stored_trace",
            description="Test",
            query="Test query",
            expected_spans=["span1"],
            expected_latency_ms=100.0,
        )

        store.save(trace)
        retrieved = store.get("store-test-123")

        assert retrieved is not None
        assert retrieved.name == "stored_trace"

    def test_get_by_name(self, tmp_path):
        """Test retrieving by name."""
        store = GoldenTraceStore(storage_path=str(tmp_path))

        trace = GoldenTrace(
            trace_id="name-test-123",
            name="named_trace",
            description="Test",
            query="Test",
            expected_spans=[],
            expected_latency_ms=100.0,
        )

        store.save(trace)
        retrieved = store.get_by_name("named_trace")

        assert retrieved is not None
        assert retrieved.trace_id == "name-test-123"

    def test_list_all(self, tmp_path):
        """Test listing all traces."""
        store = GoldenTraceStore(storage_path=str(tmp_path))

        for i in range(3):
            trace = GoldenTrace(
                trace_id=f"list-test-{i}",
                name=f"trace_{i}",
                description="Test",
                query="Test",
                expected_spans=[],
                expected_latency_ms=100.0,
            )
            store.save(trace)

        traces = store.list_all()
        assert len(traces) == 3

    def test_delete(self, tmp_path):
        """Test deleting a trace."""
        store = GoldenTraceStore(storage_path=str(tmp_path))

        trace = GoldenTrace(
            trace_id="delete-test-123",
            name="to_delete",
            description="Test",
            query="Test",
            expected_spans=[],
            expected_latency_ms=100.0,
        )

        store.save(trace)
        assert store.delete("delete-test-123") is True
        assert store.get("delete-test-123") is None

    def test_delete_nonexistent(self, tmp_path):
        """Test deleting nonexistent trace."""
        store = GoldenTraceStore(storage_path=str(tmp_path))
        assert store.delete("nonexistent") is False

    def test_export_and_import(self, tmp_path):
        """Test export and import functionality."""
        store = GoldenTraceStore(storage_path=str(tmp_path / "store1"))

        # Create traces
        for i in range(2):
            trace = GoldenTrace(
                trace_id=f"export-test-{i}",
                name=f"export_trace_{i}",
                description="Test",
                query="Test",
                expected_spans=[],
                expected_latency_ms=100.0,
            )
            store.save(trace)

        # Export
        export_path = str(tmp_path / "export.json")
        store.export_all(export_path)

        # Import to new store
        new_store = GoldenTraceStore(storage_path=str(tmp_path / "store2"))
        imported = new_store.import_all(export_path)

        assert imported == 2
        assert len(new_store.list_all()) == 2

    def test_persistence_across_instances(self, tmp_path):
        """Test that traces persist across store instances."""
        store_path = str(tmp_path / "persistent")

        # First instance - save
        store1 = GoldenTraceStore(storage_path=store_path)
        trace = GoldenTrace(
            trace_id="persist-test",
            name="persistent_trace",
            description="Test",
            query="Test",
            expected_spans=[],
            expected_latency_ms=100.0,
        )
        store1.save(trace)

        # Second instance - should load from disk
        store2 = GoldenTraceStore(storage_path=store_path)
        retrieved = store2.get("persist-test")

        assert retrieved is not None
        assert retrieved.name == "persistent_trace"


class TestGoldenTraceManager:
    """Tests for GoldenTraceManager."""

    def test_capture_golden_trace_from_spans(self, tmp_path):
        """Test capturing a golden trace from Span objects."""
        store = GoldenTraceStore(storage_path=str(tmp_path))
        manager = GoldenTraceManager(store=store)

        # Create test spans
        ctx = SpanContext.new_root()
        span1 = Span(name="rag.embed.query", context=ctx)
        span1.end_time = datetime.now(UTC)
        span1.attributes["duration_ms"] = 50.0

        span2 = Span(name="rag.vectorstore.search", context=ctx.create_child())
        span2.end_time = datetime.now(UTC)
        span2.attributes["duration_ms"] = 150.0

        golden = manager.capture_golden_trace(
            name="test_capture",
            query="What is RAG?",
            spans=[span1, span2],
            quality_score=0.85,
        )

        assert golden.name == "test_capture"
        assert "rag.embed.query" in golden.expected_spans
        assert "rag.vectorstore.search" in golden.expected_spans

    def test_capture_golden_trace_from_dicts(self, tmp_path):
        """Test capturing a golden trace from dictionaries."""
        store = GoldenTraceStore(storage_path=str(tmp_path))
        manager = GoldenTraceManager(store=store)

        spans = [
            {"name": "span1", "duration_ms": 100},
            {"name": "span2", "duration_ms": 200},
        ]

        golden = manager.capture_golden_trace(
            name="dict_capture",
            query="Test query",
            spans=spans,
        )

        assert golden.name == "dict_capture"
        assert golden.expected_latency_ms == 300

    def test_compare_trace_matching(self, tmp_path):
        """Test comparing traces that match."""
        store = GoldenTraceStore(storage_path=str(tmp_path))
        manager = GoldenTraceManager(
            store=store,
            latency_tolerance_percent=20.0,
            quality_tolerance=0.1,
        )

        # Capture golden
        golden_spans = [
            {"name": "span1", "duration_ms": 100},
            {"name": "span2", "duration_ms": 100},
        ]
        manager.capture_golden_trace(
            name="compare_test",
            query="Test",
            spans=golden_spans,
            quality_score=0.8,
        )

        # Compare with similar trace
        actual_spans = [
            {"name": "span1", "duration_ms": 110},  # Slightly slower
            {"name": "span2", "duration_ms": 90},  # Slightly faster
        ]

        comparison = manager.compare_trace(
            golden_name="compare_test",
            actual_spans=actual_spans,
            actual_quality_score=0.82,  # Within tolerance
        )

        assert comparison is not None
        assert comparison.match is True
        assert comparison.span_match is True
        assert comparison.latency_match is True
        assert comparison.quality_match is True

    def test_compare_trace_span_mismatch(self, tmp_path):
        """Test comparing traces with span mismatches."""
        store = GoldenTraceStore(storage_path=str(tmp_path))
        manager = GoldenTraceManager(store=store)

        # Capture golden
        golden_spans = [
            {"name": "span1", "duration_ms": 100},
            {"name": "span2", "duration_ms": 100},
        ]
        manager.capture_golden_trace(
            name="span_mismatch_test",
            query="Test",
            spans=golden_spans,
        )

        # Compare with different spans
        actual_spans = [
            {"name": "span1", "duration_ms": 100},
            {"name": "span3", "duration_ms": 100},  # Different span
        ]

        comparison = manager.compare_trace(
            golden_name="span_mismatch_test",
            actual_spans=actual_spans,
        )

        assert comparison.match is False
        assert comparison.span_match is False
        assert "span2" in comparison.missing_spans
        assert "span3" in comparison.extra_spans

    def test_compare_trace_latency_exceeded(self, tmp_path):
        """Test comparing traces with latency exceeding tolerance."""
        store = GoldenTraceStore(storage_path=str(tmp_path))
        manager = GoldenTraceManager(
            store=store,
            latency_tolerance_percent=10.0,
        )

        # Capture golden
        golden_spans = [{"name": "span1", "duration_ms": 100}]
        manager.capture_golden_trace(
            name="latency_test",
            query="Test",
            spans=golden_spans,
        )

        # Compare with much slower trace
        actual_spans = [{"name": "span1", "duration_ms": 150}]  # 50% slower

        comparison = manager.compare_trace(
            golden_name="latency_test",
            actual_spans=actual_spans,
        )

        assert comparison.match is False
        assert comparison.latency_match is False
        assert comparison.latency_diff_percent > 10.0

    def test_compare_trace_quality_mismatch(self, tmp_path):
        """Test comparing traces with quality score mismatch."""
        store = GoldenTraceStore(storage_path=str(tmp_path))
        manager = GoldenTraceManager(
            store=store,
            quality_tolerance=0.05,
        )

        # Capture golden with quality score
        golden_spans = [{"name": "span1", "duration_ms": 100}]
        manager.capture_golden_trace(
            name="quality_test",
            query="Test",
            spans=golden_spans,
            quality_score=0.9,
        )

        # Compare with lower quality
        actual_spans = [{"name": "span1", "duration_ms": 100}]

        comparison = manager.compare_trace(
            golden_name="quality_test",
            actual_spans=actual_spans,
            actual_quality_score=0.7,  # Much lower
        )

        assert comparison.match is False
        assert comparison.quality_match is False
        assert comparison.quality_diff < -0.1

    def test_compare_trace_not_found(self, tmp_path):
        """Test comparing with nonexistent golden trace."""
        store = GoldenTraceStore(storage_path=str(tmp_path))
        manager = GoldenTraceManager(store=store)

        comparison = manager.compare_trace(
            golden_name="nonexistent",
            actual_spans=[],
        )

        assert comparison is None

    def test_run_regression_tests(self, tmp_path):
        """Test running regression tests."""
        store = GoldenTraceStore(storage_path=str(tmp_path))
        manager = GoldenTraceManager(store=store)

        # Create golden traces
        for i in range(3):
            manager.capture_golden_trace(
                name=f"regression_test_{i}",
                query=f"Query {i}",
                spans=[{"name": "span1", "duration_ms": 100}],
                quality_score=0.8,
            )

        # Test function that returns consistent results
        def test_fn(query):
            spans = [{"name": "span1", "duration_ms": 100}]
            return spans, 100.0, 0.8

        results = manager.run_regression_tests(test_fn)

        assert len(results) == 3
        assert all(c.match for c in results.values())

    def test_get_regression_summary(self, tmp_path):
        """Test getting regression test summary."""
        store = GoldenTraceStore(storage_path=str(tmp_path))
        manager = GoldenTraceManager(store=store)

        # Create comparisons
        comparisons = {
            "test1": TraceComparison(
                golden_trace_id="g1",
                actual_trace_id="a1",
                match=True,
                span_match=True,
                latency_match=True,
                quality_match=True,
            ),
            "test2": TraceComparison(
                golden_trace_id="g2",
                actual_trace_id="a2",
                match=False,
                span_match=False,
                latency_match=True,
                quality_match=True,
                missing_spans=["span1"],
            ),
        }

        summary = manager.get_regression_summary(comparisons)

        assert summary["total"] == 2
        assert summary["passed"] == 1
        assert summary["failed"] == 1
        assert summary["pass_rate"] == 50.0
        assert len(summary["failures"]) == 1


class TestCreateRagGoldenTemplates:
    """Tests for create_rag_golden_templates."""

    def test_creates_templates(self):
        """Test that templates are created."""
        templates = create_rag_golden_templates()

        assert len(templates) >= 3
        assert any(t["name"] == "simple_factual_query" for t in templates)
        assert any(t["name"] == "complex_reasoning_query" for t in templates)

    def test_templates_have_expected_structure(self):
        """Test that templates have expected structure."""
        templates = create_rag_golden_templates()

        for template in templates:
            assert "name" in template
            assert "description" in template
            assert "expected_spans" in template
            assert "expected_latency_ms" in template
