"""
Performance Client - Unified interface for performance profiling and monitoring.

Provides a clean client interface for performance operations:
- Operation timing with context managers
- Percentile tracking (p50/p90/p95/p99)
- Throughput measurement
- SLA compliance checking
- Multi-format report generation
"""

import logging
from typing import Any

from src.services.performance.profiler import PerformanceProfiler, PerformanceTimer, SLAConfig
from src.services.performance.reporter import PerformanceReporter

logger = logging.getLogger(__name__)


class PerformanceClient:
    """
    Unified client for performance profiling and monitoring.

    Wraps PerformanceProfiler and PerformanceReporter to provide a clean
    interface following the cache client pattern.

    Features:
    - Context manager timers for operations
    - Automatic percentile calculation (p50/p90/p95/p99)
    - Throughput measurement (RPS)
    - SLA compliance validation
    - Multi-format reporting (console, JSON, Markdown, HTML)

    Example:
        ```python
        from src.services.performance import PerformanceClient, SLAConfig

        # Create client with SLA config
        sla_config = SLAConfig(max_p95_latency_ms=300.0)
        client = PerformanceClient(sla_config=sla_config)

        # Time operations
        with client.timer("embedding"):
            embeddings = model.embed(texts)

        # Check SLA compliance
        result = client.check_sla()
        if not result.passed:
            print(f"Violations: {result.violations}")

        # Generate reports
        client.print_summary()
        client.export_html("performance_report.html")
        ```
    """

    def __init__(
        self,
        sla_config: SLAConfig | None = None,
        profiler: PerformanceProfiler | None = None,
        reporter: PerformanceReporter | None = None,
    ):
        """
        Initialize performance client.

        Args:
            sla_config: Optional SLA configuration with thresholds
            profiler: Optional PerformanceProfiler instance (creates default if None)
            reporter: Optional PerformanceReporter instance (creates default if None)
        """
        self.profiler = profiler or PerformanceProfiler(sla_config=sla_config)
        self.reporter = reporter or PerformanceReporter()
        self._sla_config = sla_config

    def timer(self, operation: str, metadata: dict[str, Any] | None = None) -> PerformanceTimer:
        """
        Create a timer context manager for an operation.

        Args:
            operation: Name of the operation being timed
            metadata: Optional metadata about the operation

        Returns:
            PerformanceTimer context manager

        Example:
            ```python
            with client.timer("query_processing", metadata={"query_id": "123"}):
                process_query()
            ```
        """
        return self.profiler.timer(operation=operation, metadata=metadata)

    def record_operation(
        self,
        operation: str,
        duration: float,
        success: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Manually record an operation's performance.

        Args:
            operation: Name of the operation
            duration: Duration in seconds
            success: Whether the operation succeeded
            metadata: Optional metadata about the operation

        Example:
            ```python
            start = time.perf_counter()
            result = do_something()
            duration = time.perf_counter() - start
            client.record_operation("my_op", duration, success=True)
            ```
        """
        self.profiler.record_operation(
            operation=operation, duration=duration, success=success, metadata=metadata
        )

    def get_stats(self, operation: str | None = None) -> dict[str, Any]:
        """
        Get performance statistics.

        Args:
            operation: Optional operation name to get stats for.
                      If None, returns all stats.

        Returns:
            Dictionary containing performance statistics

        Example:
            ```python
            # Get overall stats
            stats = client.get_stats()
            print(f"P95: {stats['overall']['latency']['p95']:.2f}ms")

            # Get operation-specific stats
            embed_stats = client.get_stats("embedding")
            print(f"Embedding P95: {embed_stats['latency']['p95']:.2f}ms")
            ```
        """
        if operation:
            return self.profiler.get_operation_stats(operation)
        return self.profiler.get_all_stats()

    def check_sla(self):
        """
        Check SLA compliance.

        Returns:
            SLAResult with pass/fail status and violations

        Example:
            ```python
            result = client.check_sla()
            if result.passed:
                print("✓ All SLA targets met")
            else:
                print(f"✗ Violations: {result.violations}")
            ```
        """
        return self.profiler.check_sla()

    def reset(self) -> None:
        """
        Reset all performance metrics.

        Clears all recorded operations and statistics. Useful for
        starting fresh measurements or periodic resets.

        Example:
            ```python
            # Reset hourly for real-time monitoring
            client.reset()
            ```
        """
        self.profiler.reset()

    def print_summary(self) -> None:
        """
        Print performance summary to console.

        Displays overall metrics, latency percentiles, and per-operation stats.

        Example:
            ```python
            client.print_summary()
            # Output:
            # PERFORMANCE REPORT
            # Total Requests: 1000
            # Throughput: 25.5 RPS
            # ...
            ```
        """
        stats = self.profiler.get_all_stats()
        self.reporter.print_summary(stats)

    def print_sla_result(self) -> None:
        """
        Print SLA compliance status to console.

        Example:
            ```python
            client.print_sla_result()
            # Output:
            # SLA COMPLIANCE CHECK
            # Status: ✓ PASSED
            # ...
            ```
        """
        result = self.profiler.check_sla()
        self.reporter.print_sla_result(result)

    def export_json(self, output_path: str) -> None:
        """
        Export performance stats to JSON file.

        Args:
            output_path: Path to output JSON file

        Example:
            ```python
            client.export_json("reports/performance.json")
            ```
        """
        stats = self.profiler.get_all_stats()
        self.reporter.export_json(stats, output_path)

    def export_markdown(self, output_path: str) -> None:
        """
        Export performance stats to Markdown file.

        Args:
            output_path: Path to output Markdown file

        Example:
            ```python
            client.export_markdown("reports/performance.md")
            ```
        """
        stats = self.profiler.get_all_stats()
        self.reporter.export_markdown(stats, output_path)

    def export_html(self, output_path: str) -> None:
        """
        Export performance stats to HTML file.

        Args:
            output_path: Path to output HTML file

        Example:
            ```python
            client.export_html("reports/performance.html")
            ```
        """
        stats = self.profiler.get_all_stats()
        self.reporter.export_html(stats, output_path)

    def export_all_formats(self, base_path: str) -> None:
        """
        Export reports in all formats (JSON, Markdown, HTML).

        Args:
            base_path: Base path for output files (without extension)

        Example:
            ```python
            # Creates: report.json, report.md, report.html
            client.export_all_formats("reports/performance_report")
            ```
        """
        stats = self.profiler.get_all_stats()
        self.reporter.export_json(stats, f"{base_path}.json")
        self.reporter.export_markdown(stats, f"{base_path}.md")
        self.reporter.export_html(stats, f"{base_path}.html")
        logger.info(f"Exported reports to {base_path}.{{json,md,html}}")

    @property
    def sla_config(self) -> SLAConfig | None:
        """Get current SLA configuration."""
        return self._sla_config

    @property
    def total_requests(self) -> int:
        """Get total number of requests tracked."""
        return self.profiler.total_requests

    @property
    def total_successes(self) -> int:
        """Get total number of successful requests."""
        return sum(op.successes for op in self.profiler.operations.values())

    @property
    def total_failures(self) -> int:
        """Get total number of failed requests."""
        return sum(op.failures for op in self.profiler.operations.values())

    def __repr__(self) -> str:
        """String representation."""
        stats = self.profiler.get_all_stats()
        overall = stats.get("overall", {})
        throughput = overall.get("throughput_rps", 0)
        p95 = overall.get("latency", {}).get("p95", 0)

        return (
            f"PerformanceClient("
            f"requests={self.total_requests}, "
            f"throughput={throughput:.1f} RPS, "
            f"p95={p95:.1f}ms, "
            f"sla={'configured' if self._sla_config else 'none'})"
        )
