"""
Tests for performance profiling.
"""

import time

import pytest

from src.services.performance import (
    PerformanceProfiler,
    PerformanceReporter,
    SLAConfig,
)


class TestPerformanceTimer:
    """Test PerformanceTimer context manager."""

    def test_timer_basic(self):
        """Test basic timer functionality."""
        profiler = PerformanceProfiler()

        with profiler.timer("test_op"):
            time.sleep(0.01)

        assert profiler.operations["test_op"].count == 1
        assert profiler.operations["test_op"].successes == 1
        assert profiler.operations["test_op"].failures == 0

    def test_timer_with_metadata(self):
        """Test timer with metadata."""
        profiler = PerformanceProfiler()

        with profiler.timer("test_op", metadata={"key": "value"}):
            time.sleep(0.01)

        assert len(profiler.operations["test_op"].metadata) == 1
        assert profiler.operations["test_op"].metadata[0]["key"] == "value"

    def test_timer_failure(self):
        """Test timer with exception."""
        profiler = PerformanceProfiler()

        with pytest.raises(ValueError), profiler.timer("test_op"):
            raise ValueError("Test error")

        assert profiler.operations["test_op"].failures == 1
        assert profiler.operations["test_op"].successes == 0

    def test_timer_mark_failure(self):
        """Test manually marking failure."""
        profiler = PerformanceProfiler()

        with profiler.timer("test_op") as timer:
            timer.mark_failure()

        assert profiler.operations["test_op"].failures == 1


class TestOperationStats:
    """Test OperationStats."""

    def test_percentiles(self):
        """Test percentile calculations."""
        profiler = PerformanceProfiler()

        # Record multiple operations with known durations
        for i in range(10):
            profiler.record_operation("test", duration=i * 0.01, success=True)

        stats = profiler.get_operation_stats("test")
        percentiles = stats["latency"]

        assert percentiles["p50"] > 0
        assert percentiles["p95"] > percentiles["p50"]
        assert percentiles["p99"] > percentiles["p95"]

    def test_success_rate(self):
        """Test success rate calculation."""
        profiler = PerformanceProfiler()

        # 8 successes, 2 failures
        for _ in range(8):
            profiler.record_operation("test", duration=0.01, success=True)
        for _ in range(2):
            profiler.record_operation("test", duration=0.01, success=False)

        stats = profiler.get_operation_stats("test")
        assert stats["success_rate"] == pytest.approx(0.8, abs=0.01)

    def test_throughput(self):
        """Test throughput calculation."""
        profiler = PerformanceProfiler()

        time.sleep(0.1)  # Wait a bit for elapsed time

        for _ in range(10):
            profiler.record_operation("test", duration=0.01, success=True)

        stats = profiler.get_operation_stats("test")
        assert stats["throughput_rps"] > 0


class TestPerformanceProfiler:
    """Test PerformanceProfiler."""

    def test_initialization(self):
        """Test profiler initialization."""
        profiler = PerformanceProfiler()
        assert profiler.total_requests == 0
        assert len(profiler.operations) == 0

    def test_record_operation(self):
        """Test recording operations."""
        profiler = PerformanceProfiler()

        profiler.record_operation("op1", duration=0.01, success=True)
        profiler.record_operation("op2", duration=0.02, success=False)

        assert profiler.total_requests == 2
        assert "op1" in profiler.operations
        assert "op2" in profiler.operations

    def test_get_all_stats(self):
        """Test getting all statistics."""
        profiler = PerformanceProfiler()

        profiler.record_operation("op1", duration=0.01, success=True)
        profiler.record_operation("op2", duration=0.02, success=True)

        stats = profiler.get_all_stats()

        assert "overall" in stats
        assert "operations" in stats
        assert "timestamp" in stats
        assert stats["overall"]["total_requests"] == 2

    def test_reset(self):
        """Test resetting metrics."""
        profiler = PerformanceProfiler()

        profiler.record_operation("op1", duration=0.01, success=True)
        profiler.reset()

        assert profiler.total_requests == 0
        assert len(profiler.operations) == 0

    def test_repr(self):
        """Test string representation."""
        profiler = PerformanceProfiler()
        profiler.record_operation("op1", duration=0.01, success=True)

        repr_str = repr(profiler)
        assert "PerformanceProfiler" in repr_str
        assert "requests=" in repr_str


class TestSLAConfig:
    """Test SLA configuration."""

    def test_default_config(self):
        """Test default SLA configuration."""
        config = SLAConfig()

        assert config.max_p50_latency_ms == 500.0
        assert config.max_p95_latency_ms == 1000.0
        assert config.max_p99_latency_ms == 2000.0
        assert config.min_throughput_rps == 10.0
        assert config.min_success_rate == 0.95

    def test_custom_config(self):
        """Test custom SLA configuration."""
        config = SLAConfig(
            max_p50_latency_ms=100.0,
            min_throughput_rps=50.0,
            operation_slas={"embedding": {"max_p95": 50.0}},
        )

        assert config.max_p50_latency_ms == 100.0
        assert config.min_throughput_rps == 50.0
        assert "embedding" in config.operation_slas


class TestSLACompliance:
    """Test SLA compliance checking."""

    def test_sla_pass(self):
        """Test SLA compliance when all targets met."""
        config = SLAConfig(
            max_p50_latency_ms=1000.0,
            max_p95_latency_ms=2000.0,
            max_p99_latency_ms=3000.0,
            min_throughput_rps=1.0,
            min_success_rate=0.9,
        )
        profiler = PerformanceProfiler(sla_config=config)

        # Record fast operations
        for _ in range(20):
            profiler.record_operation("test", duration=0.01, success=True)

        time.sleep(0.1)  # Ensure throughput is measurable

        result = profiler.check_sla()
        assert result.passed
        assert len(result.violations) == 0

    def test_sla_latency_violation(self):
        """Test SLA violation for latency."""
        config = SLAConfig(max_p95_latency_ms=10.0)  # Very strict
        profiler = PerformanceProfiler(sla_config=config)

        # Record slow operations
        for _ in range(20):
            profiler.record_operation("test", duration=0.1, success=True)

        time.sleep(0.1)

        result = profiler.check_sla()
        assert not result.passed
        assert len(result.violations) > 0
        assert any("latency" in v.lower() for v in result.violations)

    def test_sla_throughput_violation(self):
        """Test SLA violation for throughput."""
        config = SLAConfig(min_throughput_rps=1000.0)  # Very high
        profiler = PerformanceProfiler(sla_config=config)

        profiler.record_operation("test", duration=0.01, success=True)
        time.sleep(0.5)

        result = profiler.check_sla()
        assert not result.passed
        assert any("throughput" in v.lower() for v in result.violations)

    def test_sla_success_rate_violation(self):
        """Test SLA violation for success rate."""
        config = SLAConfig(min_success_rate=0.99)
        profiler = PerformanceProfiler(sla_config=config)

        # 90% success rate
        for _ in range(9):
            profiler.record_operation("test", duration=0.01, success=True)
        profiler.record_operation("test", duration=0.01, success=False)

        time.sleep(0.1)

        result = profiler.check_sla()
        assert not result.passed
        assert any("success rate" in v.lower() for v in result.violations)


class TestPerformanceReporter:
    """Test PerformanceReporter."""

    def test_print_summary(self, capsys):
        """Test printing summary."""
        profiler = PerformanceProfiler()
        profiler.record_operation("test", duration=0.01, success=True)

        reporter = PerformanceReporter()
        stats = profiler.get_all_stats()
        reporter.print_summary(stats)

        captured = capsys.readouterr()
        assert "PERFORMANCE REPORT" in captured.out
        assert "LATENCY PERCENTILES" in captured.out

    def test_print_sla_result(self, capsys):
        """Test printing SLA result."""
        profiler = PerformanceProfiler()
        profiler.record_operation("test", duration=0.01, success=True)
        time.sleep(0.1)

        result = profiler.check_sla()

        reporter = PerformanceReporter()
        reporter.print_sla_result(result)

        captured = capsys.readouterr()
        assert "SLA COMPLIANCE" in captured.out

    def test_export_json(self, tmp_path):
        """Test JSON export."""
        profiler = PerformanceProfiler()
        profiler.record_operation("test", duration=0.01, success=True)

        reporter = PerformanceReporter()
        stats = profiler.get_all_stats()
        output_path = tmp_path / "test_report.json"

        reporter.export_json(stats, output_path)

        assert output_path.exists()

        import json

        with output_path.open() as f:
            loaded_stats = json.load(f)

        assert "overall" in loaded_stats
        assert "operations" in loaded_stats

    def test_export_markdown(self, tmp_path):
        """Test Markdown export."""
        profiler = PerformanceProfiler()
        profiler.record_operation("test", duration=0.01, success=True)

        reporter = PerformanceReporter()
        stats = profiler.get_all_stats()
        output_path = tmp_path / "test_report.md"

        reporter.export_markdown(stats, output_path)

        assert output_path.exists()

        content = output_path.read_text()
        assert "# Performance Report" in content
        assert "## Latency Percentiles" in content

    def test_export_html(self, tmp_path):
        """Test HTML export."""
        profiler = PerformanceProfiler()
        profiler.record_operation("test", duration=0.01, success=True)

        reporter = PerformanceReporter()
        stats = profiler.get_all_stats()
        output_path = tmp_path / "test_report.html"

        reporter.export_html(stats, output_path)

        assert output_path.exists()

        content = output_path.read_text()
        assert "<!DOCTYPE html>" in content
        assert "Performance Report" in content


class TestIntegration:
    """Integration tests."""

    def test_full_workflow(self, tmp_path):
        """Test complete profiling workflow."""
        # Configure SLA
        sla_config = SLAConfig(
            max_p50_latency_ms=100.0,
            max_p95_latency_ms=200.0,
        )

        # Create profiler
        profiler = PerformanceProfiler(sla_config=sla_config)

        # Simulate operations
        with profiler.timer("embedding"):
            time.sleep(0.01)

        with profiler.timer("retrieval"):
            time.sleep(0.02)

        with profiler.timer("generation"):
            time.sleep(0.05)

        # Get stats
        stats = profiler.get_all_stats()
        assert stats["overall"]["total_requests"] == 3

        # Check SLA
        time.sleep(0.1)
        result = profiler.check_sla()
        assert isinstance(result.passed, bool)

        # Export reports
        reporter = PerformanceReporter()
        reporter.export_json(stats, tmp_path / "report.json")
        reporter.export_markdown(stats, tmp_path / "report.md")
        reporter.export_html(stats, tmp_path / "report.html")

        assert (tmp_path / "report.json").exists()
        assert (tmp_path / "report.md").exists()
        assert (tmp_path / "report.html").exists()
