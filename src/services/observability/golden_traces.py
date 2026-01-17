"""
Golden traces module for RAG pipeline.

Provides functionality to capture, store, and compare "golden" traces
which serve as reference implementations for regression testing
and debugging.

Features:
- Capture complete traces as golden references
- Compare new traces against golden traces
- Identify regressions in latency, quality, or behavior
- Export/import golden traces for CI/CD
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.services.observability.tracing import Span


@dataclass
class GoldenTrace:
    """
    A reference trace used for comparison and regression testing.

    Attributes:
        trace_id: Unique identifier for this golden trace
        name: Human-readable name for the trace
        description: Description of what this trace represents
        query: The input query that produced this trace
        expected_spans: List of expected span names in order
        expected_latency_ms: Expected total latency (tolerance applied)
        expected_quality_score: Expected quality score
        spans: The actual span data
        metadata: Additional metadata
        created_at: When this golden trace was created
        version: Version of the golden trace
    """

    trace_id: str
    name: str
    description: str
    query: str
    expected_spans: list[str]
    expected_latency_ms: float
    expected_quality_score: float | None = None
    spans: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "description": self.description,
            "query": self.query,
            "expected_spans": self.expected_spans,
            "expected_latency_ms": self.expected_latency_ms,
            "expected_quality_score": self.expected_quality_score,
            "spans": self.spans,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GoldenTrace":
        """Create from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        else:
            created_at = datetime.now(UTC)

        return cls(
            trace_id=data["trace_id"],
            name=data["name"],
            description=data.get("description", ""),
            query=data["query"],
            expected_spans=data.get("expected_spans", []),
            expected_latency_ms=data.get("expected_latency_ms", 0.0),
            expected_quality_score=data.get("expected_quality_score"),
            spans=data.get("spans", []),
            metadata=data.get("metadata", {}),
            created_at=created_at,
            version=data.get("version", "1.0"),
        )


@dataclass
class TraceComparison:
    """
    Result of comparing a trace against a golden trace.

    Attributes:
        golden_trace_id: ID of the golden trace
        actual_trace_id: ID of the actual trace
        match: Whether the traces match within tolerances
        span_match: Whether span structure matches
        latency_match: Whether latency is within tolerance
        quality_match: Whether quality score is within tolerance
        missing_spans: Spans in golden but not in actual
        extra_spans: Spans in actual but not in golden
        latency_diff_ms: Difference in latency
        latency_diff_percent: Percentage difference in latency
        quality_diff: Difference in quality score
        details: Additional comparison details
    """

    golden_trace_id: str
    actual_trace_id: str
    match: bool
    span_match: bool
    latency_match: bool
    quality_match: bool
    missing_spans: list[str] = field(default_factory=list)
    extra_spans: list[str] = field(default_factory=list)
    latency_diff_ms: float = 0.0
    latency_diff_percent: float = 0.0
    quality_diff: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "golden_trace_id": self.golden_trace_id,
            "actual_trace_id": self.actual_trace_id,
            "match": self.match,
            "span_match": self.span_match,
            "latency_match": self.latency_match,
            "quality_match": self.quality_match,
            "missing_spans": self.missing_spans,
            "extra_spans": self.extra_spans,
            "latency_diff_ms": round(self.latency_diff_ms, 2),
            "latency_diff_percent": round(self.latency_diff_percent, 2),
            "quality_diff": round(self.quality_diff, 4) if self.quality_diff else None,
            "details": self.details,
        }


class GoldenTraceStore:
    """
    Storage for golden traces.

    Supports file-based persistence for use in CI/CD pipelines.
    """

    def __init__(self, storage_path: str = "./data/golden_traces"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._traces: dict[str, GoldenTrace] = {}
        self._load_all()

    def _load_all(self) -> None:
        """Load all golden traces from storage."""
        for file_path in self.storage_path.glob("*.json"):
            try:
                with file_path.open() as f:
                    data = json.load(f)
                    trace = GoldenTrace.from_dict(data)
                    self._traces[trace.trace_id] = trace
            except (json.JSONDecodeError, KeyError):
                # Skip invalid files
                pass

    def save(self, trace: GoldenTrace) -> None:
        """Save a golden trace to storage."""
        self._traces[trace.trace_id] = trace
        file_path = self.storage_path / f"{trace.trace_id}.json"
        with file_path.open("w") as f:
            json.dump(trace.to_dict(), f, indent=2)

    def get(self, trace_id: str) -> GoldenTrace | None:
        """Get a golden trace by ID."""
        return self._traces.get(trace_id)

    def get_by_name(self, name: str) -> GoldenTrace | None:
        """Get a golden trace by name."""
        for trace in self._traces.values():
            if trace.name == name:
                return trace
        return None

    def list_all(self) -> list[GoldenTrace]:
        """List all golden traces."""
        return list(self._traces.values())

    def delete(self, trace_id: str) -> bool:
        """Delete a golden trace."""
        if trace_id not in self._traces:
            return False

        del self._traces[trace_id]
        file_path = self.storage_path / f"{trace_id}.json"
        if file_path.exists():
            file_path.unlink()
        return True

    def export_all(self, output_path: str) -> None:
        """Export all golden traces to a single file."""
        data = [trace.to_dict() for trace in self._traces.values()]
        with Path(output_path).open("w") as f:
            json.dump(data, f, indent=2)

    def import_all(self, input_path: str, overwrite: bool = False) -> int:
        """Import golden traces from a file."""
        with Path(input_path).open() as f:
            data = json.load(f)

        imported = 0
        for item in data:
            trace = GoldenTrace.from_dict(item)
            if overwrite or trace.trace_id not in self._traces:
                self.save(trace)
                imported += 1

        return imported


class GoldenTraceManager:
    """
    Manager for golden trace operations.

    Provides high-level operations for creating, comparing,
    and managing golden traces.

    Usage:
        manager = GoldenTraceManager()

        # Capture a golden trace
        golden = manager.capture_golden_trace(
            name="basic_rag_query",
            query="What is machine learning?",
            spans=actual_spans,
            quality_score=0.85,
        )

        # Compare against golden
        comparison = manager.compare_trace(
            golden_name="basic_rag_query",
            actual_spans=new_spans,
            actual_latency_ms=150,
            actual_quality_score=0.82,
        )

        print(f"Match: {comparison.match}")
    """

    def __init__(
        self,
        store: GoldenTraceStore | None = None,
        latency_tolerance_percent: float = 20.0,
        quality_tolerance: float = 0.1,
    ):
        self.store = store or GoldenTraceStore()
        self.latency_tolerance_percent = latency_tolerance_percent
        self.quality_tolerance = quality_tolerance

    def capture_golden_trace(
        self,
        name: str,
        query: str,
        spans: list[Span] | list[dict[str, Any]],
        quality_score: float | None = None,
        description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> GoldenTrace:
        """
        Capture a new golden trace from actual spans.

        Args:
            name: Name for the golden trace
            query: The input query
            spans: List of Span objects or span dictionaries
            quality_score: Optional quality score
            description: Description of the trace
            metadata: Additional metadata

        Returns:
            The created GoldenTrace
        """
        # Convert spans to dictionaries
        span_dicts = []
        span_names = []
        total_latency = 0.0

        for span in spans:
            if isinstance(span, Span):
                span_dict = span.to_dict()
                span_dicts.append(span_dict)
                span_names.append(span.name)
                if span.duration_ms:
                    total_latency += span.duration_ms
            else:
                span_dicts.append(span)
                span_names.append(span.get("name", "unknown"))
                if span.get("duration_ms"):
                    total_latency += span["duration_ms"]

        # Generate trace ID
        trace_id = self._generate_trace_id(name, query)

        golden = GoldenTrace(
            trace_id=trace_id,
            name=name,
            description=description,
            query=query,
            expected_spans=span_names,
            expected_latency_ms=total_latency,
            expected_quality_score=quality_score,
            spans=span_dicts,
            metadata=metadata or {},
        )

        self.store.save(golden)
        return golden

    def _generate_trace_id(self, name: str, query: str) -> str:
        """Generate a unique trace ID."""
        content = f"{name}:{query}:{datetime.now(UTC).isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def compare_trace(
        self,
        golden_name: str,
        actual_spans: list[Span] | list[dict[str, Any]],
        actual_latency_ms: float | None = None,
        actual_quality_score: float | None = None,
        actual_trace_id: str | None = None,
    ) -> TraceComparison | None:
        """
        Compare actual trace against a golden trace.

        Args:
            golden_name: Name of the golden trace to compare against
            actual_spans: List of actual Span objects or dictionaries
            actual_latency_ms: Actual total latency (calculated from spans if None)
            actual_quality_score: Actual quality score
            actual_trace_id: ID of the actual trace

        Returns:
            TraceComparison result or None if golden trace not found
        """
        golden = self.store.get_by_name(golden_name)
        if not golden:
            return None

        # Extract span names and calculate latency
        actual_span_names = []
        calculated_latency = 0.0

        for span in actual_spans:
            if isinstance(span, Span):
                actual_span_names.append(span.name)
                if span.duration_ms:
                    calculated_latency += span.duration_ms
            else:
                actual_span_names.append(span.get("name", "unknown"))
                if span.get("duration_ms"):
                    calculated_latency += span["duration_ms"]

        if actual_latency_ms is None:
            actual_latency_ms = calculated_latency

        # Compare spans
        missing_spans = [s for s in golden.expected_spans if s not in actual_span_names]
        extra_spans = [s for s in actual_span_names if s not in golden.expected_spans]
        span_match = len(missing_spans) == 0 and len(extra_spans) == 0

        # Compare latency
        latency_diff_ms = actual_latency_ms - golden.expected_latency_ms
        latency_diff_percent = (
            (latency_diff_ms / golden.expected_latency_ms * 100)
            if golden.expected_latency_ms > 0
            else 0.0
        )
        latency_match = abs(latency_diff_percent) <= self.latency_tolerance_percent

        # Compare quality
        quality_diff = None
        quality_match = True
        if golden.expected_quality_score is not None and actual_quality_score is not None:
            quality_diff = actual_quality_score - golden.expected_quality_score
            quality_match = abs(quality_diff) <= self.quality_tolerance

        # Overall match
        match = span_match and latency_match and quality_match

        return TraceComparison(
            golden_trace_id=golden.trace_id,
            actual_trace_id=actual_trace_id or "unknown",
            match=match,
            span_match=span_match,
            latency_match=latency_match,
            quality_match=quality_match,
            missing_spans=missing_spans,
            extra_spans=extra_spans,
            latency_diff_ms=latency_diff_ms,
            latency_diff_percent=latency_diff_percent,
            quality_diff=quality_diff,
            details={
                "expected_spans": golden.expected_spans,
                "actual_spans": actual_span_names,
                "expected_latency_ms": golden.expected_latency_ms,
                "actual_latency_ms": actual_latency_ms,
                "expected_quality": golden.expected_quality_score,
                "actual_quality": actual_quality_score,
            },
        )

    def run_regression_tests(
        self,
        test_function: Any,
    ) -> dict[str, TraceComparison]:
        """
        Run regression tests against all golden traces.

        Args:
            test_function: Function that takes a query and returns
                          (spans, latency_ms, quality_score)

        Returns:
            Dictionary mapping golden trace names to comparison results
        """
        results = {}
        for golden in self.store.list_all():
            try:
                spans, latency_ms, quality_score = test_function(golden.query)
                comparison = self.compare_trace(
                    golden_name=golden.name,
                    actual_spans=spans,
                    actual_latency_ms=latency_ms,
                    actual_quality_score=quality_score,
                )
                if comparison:
                    results[golden.name] = comparison
            except Exception as e:
                # Create a failed comparison
                results[golden.name] = TraceComparison(
                    golden_trace_id=golden.trace_id,
                    actual_trace_id="error",
                    match=False,
                    span_match=False,
                    latency_match=False,
                    quality_match=False,
                    details={"error": str(e)},
                )
        return results

    def get_regression_summary(
        self,
        comparisons: dict[str, TraceComparison],
    ) -> dict[str, Any]:
        """Get summary of regression test results."""
        total = len(comparisons)
        passed = sum(1 for c in comparisons.values() if c.match)
        failed = total - passed

        failures = []
        for name, comparison in comparisons.items():
            if not comparison.match:
                failures.append(
                    {
                        "name": name,
                        "span_match": comparison.span_match,
                        "latency_match": comparison.latency_match,
                        "quality_match": comparison.quality_match,
                        "missing_spans": comparison.missing_spans,
                        "extra_spans": comparison.extra_spans,
                        "latency_diff_percent": comparison.latency_diff_percent,
                    }
                )

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": (passed / total * 100) if total > 0 else 100.0,
            "failures": failures,
        }


# Pre-defined golden trace templates for RAG pipeline
def create_rag_golden_templates() -> list[dict[str, Any]]:
    """Create template golden traces for common RAG scenarios."""
    return [
        {
            "name": "simple_factual_query",
            "description": "Simple factual question with single-hop retrieval",
            "expected_spans": [
                "rag.query.rewrite",
                "rag.embed.query",
                "rag.vectorstore.search",
                "rag.generate.context",
                "rag.generate.response",
            ],
            "expected_latency_ms": 500,
            "expected_quality_score": 0.85,
        },
        {
            "name": "complex_reasoning_query",
            "description": "Complex query requiring multi-hop reasoning",
            "expected_spans": [
                "rag.query.rewrite",
                "rag.agent.plan",
                "rag.embed.query",
                "rag.vectorstore.search",
                "rag.rerank.documents",
                "rag.agent.execute",
                "rag.agent.reflect",
                "rag.generate.context",
                "rag.generate.response",
            ],
            "expected_latency_ms": 2000,
            "expected_quality_score": 0.80,
        },
        {
            "name": "hybrid_search_query",
            "description": "Query using hybrid vector + BM25 search",
            "expected_spans": [
                "rag.query.rewrite",
                "rag.embed.query",
                "rag.vectorstore.hybrid_search",
                "rag.rerank.documents",
                "rag.generate.context",
                "rag.generate.response",
            ],
            "expected_latency_ms": 800,
            "expected_quality_score": 0.88,
        },
        {
            "name": "cached_query",
            "description": "Query that should hit cache",
            "expected_spans": [
                "rag.cache.lookup",
            ],
            "expected_latency_ms": 50,
            "expected_quality_score": None,  # Cached, quality not re-evaluated
        },
    ]
