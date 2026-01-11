"""
Performance profiling with timers and SLA monitoring.

Provides comprehensive performance tracking including:
- Latency percentiles (p50/p95/p99)
- Throughput measurement (requests/second)
- Operation-level timing
- SLA compliance checking
"""

import logging
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SLAConfig:
    """SLA thresholds for performance monitoring."""

    # Latency thresholds (in milliseconds)
    max_p50_latency_ms: float = 500.0
    max_p95_latency_ms: float = 1000.0
    max_p99_latency_ms: float = 2000.0

    # Throughput thresholds
    min_throughput_rps: float = 10.0  # requests per second

    # Success rate threshold
    min_success_rate: float = 0.95  # 95%

    # Per-operation SLAs (optional)
    operation_slas: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class SLAResult:
    """Result of SLA compliance check."""

    passed: bool
    violations: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""


class PerformanceTimer:
    """
    Context manager for timing operations.

    Usage:
        profiler = PerformanceProfiler()
        with PerformanceTimer(profiler, "embedding"):
            embeddings = model.embed(texts)
    """

    def __init__(
        self,
        profiler: "PerformanceProfiler",
        operation: str,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Initialize timer.

        Args:
            profiler: Performance profiler to record metrics
            operation: Name of operation being timed
            metadata: Optional metadata about the operation
        """
        self.profiler = profiler
        self.operation = operation
        self.metadata = metadata or {}
        self.start_time: float | None = None
        self.success = True

    def __enter__(self) -> "PerformanceTimer":
        """Start timing."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and record metrics."""
        if self.start_time is None:
            return

        duration = time.perf_counter() - self.start_time
        # Only update success if not explicitly marked as failure
        if self.success:
            self.success = exc_type is None

        self.profiler.record_operation(
            operation=self.operation,
            duration=duration,
            success=self.success,
            metadata=self.metadata,
        )

    def mark_failure(self) -> None:
        """Mark operation as failed."""
        self.success = False


@dataclass
class OperationStats:
    """Statistics for a specific operation."""

    name: str
    count: int = 0
    successes: int = 0
    failures: int = 0
    durations: list[float] = field(default_factory=list)
    metadata: list[dict[str, Any]] = field(default_factory=list)

    def record(self, duration: float, success: bool, metadata: dict[str, Any]) -> None:
        """Record operation execution."""
        self.count += 1
        self.durations.append(duration)
        self.metadata.append(metadata)

        if success:
            self.successes += 1
        else:
            self.failures += 1

    def get_percentiles(self) -> dict[str, float]:
        """Calculate latency percentiles in milliseconds."""
        if not self.durations:
            return {
                "p50": 0.0,
                "p90": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
            }

        sorted_durations = sorted(self.durations)
        n = len(sorted_durations)

        def percentile(p: float) -> float:
            """Calculate percentile value."""
            k = (n - 1) * p
            f = int(k)
            c = f + 1
            if c >= n:
                return sorted_durations[-1] * 1000  # Convert to ms
            d0 = sorted_durations[f] * 1000
            d1 = sorted_durations[c] * 1000
            return d0 + (d1 - d0) * (k - f)

        return {
            "p50": percentile(0.50),
            "p90": percentile(0.90),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
            "mean": statistics.mean(self.durations) * 1000,
            "min": min(self.durations) * 1000,
            "max": max(self.durations) * 1000,
        }

    def get_success_rate(self) -> float:
        """Calculate success rate."""
        if self.count == 0:
            return 1.0
        return self.successes / self.count

    def get_throughput(self, elapsed_time: float) -> float:
        """
        Calculate throughput (operations per second).

        Args:
            elapsed_time: Total elapsed time in seconds

        Returns:
            Operations per second
        """
        if elapsed_time <= 0:
            return 0.0
        return self.count / elapsed_time


class PerformanceProfiler:
    """
    Track and analyze performance metrics.

    Features:
    - Per-operation timing
    - Latency percentiles (p50/p95/p99)
    - Throughput measurement
    - SLA compliance checking
    - Detailed performance reports
    """

    def __init__(self, sla_config: SLAConfig | None = None):
        """
        Initialize profiler.

        Args:
            sla_config: Optional SLA configuration
        """
        self.sla_config = sla_config or SLAConfig()
        self.operations: dict[str, OperationStats] = defaultdict(
            lambda: OperationStats(name="unknown")
        )
        self.start_time = time.time()
        self.total_requests = 0
        logger.info("PerformanceProfiler initialized")

    def record_operation(
        self,
        operation: str,
        duration: float,
        success: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Record an operation execution.

        Args:
            operation: Name of the operation
            duration: Duration in seconds
            success: Whether operation succeeded
            metadata: Optional metadata
        """
        if operation not in self.operations:
            self.operations[operation] = OperationStats(name=operation)

        self.operations[operation].record(duration, success, metadata or {})
        self.total_requests += 1

    def timer(self, operation: str, metadata: dict[str, Any] | None = None) -> PerformanceTimer:
        """
        Create a timer context manager.

        Args:
            operation: Name of operation
            metadata: Optional metadata

        Returns:
            PerformanceTimer context manager
        """
        return PerformanceTimer(self, operation, metadata)

    def get_operation_stats(self, operation: str) -> dict[str, Any]:
        """
        Get statistics for a specific operation.

        Args:
            operation: Operation name

        Returns:
            Dict with operation statistics
        """
        if operation not in self.operations:
            return {}

        stats = self.operations[operation]
        elapsed = time.time() - self.start_time

        return {
            "operation": operation,
            "count": stats.count,
            "successes": stats.successes,
            "failures": stats.failures,
            "success_rate": stats.get_success_rate(),
            "latency": stats.get_percentiles(),
            "throughput_rps": stats.get_throughput(elapsed),
        }

    def get_all_stats(self) -> dict[str, Any]:
        """
        Get comprehensive statistics for all operations.

        Returns:
            Dict with all statistics
        """
        elapsed = time.time() - self.start_time
        overall_throughput = self.total_requests / elapsed if elapsed > 0 else 0.0

        # Aggregate latencies across all operations
        all_durations = []
        all_successes = 0
        all_failures = 0

        for stats in self.operations.values():
            all_durations.extend(stats.durations)
            all_successes += stats.successes
            all_failures += stats.failures

        # Calculate overall percentiles
        if all_durations:
            sorted_durations = sorted(all_durations)
            n = len(sorted_durations)

            def percentile(p: float) -> float:
                k = (n - 1) * p
                f = int(k)
                c = f + 1
                if c >= n:
                    return sorted_durations[-1] * 1000
                d0 = sorted_durations[f] * 1000
                d1 = sorted_durations[c] * 1000
                return d0 + (d1 - d0) * (k - f)

            overall_latency = {
                "p50": percentile(0.50),
                "p90": percentile(0.90),
                "p95": percentile(0.95),
                "p99": percentile(0.99),
                "mean": statistics.mean(all_durations) * 1000,
                "min": min(all_durations) * 1000,
                "max": max(all_durations) * 1000,
            }
        else:
            overall_latency = {
                "p50": 0.0,
                "p90": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
            }

        # Per-operation stats
        operations = {name: self.get_operation_stats(name) for name in self.operations}

        return {
            "overall": {
                "total_requests": self.total_requests,
                "elapsed_time_s": elapsed,
                "throughput_rps": overall_throughput,
                "success_rate": all_successes / self.total_requests
                if self.total_requests > 0
                else 1.0,
                "latency": overall_latency,
            },
            "operations": operations,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def check_sla(self) -> SLAResult:
        """
        Check SLA compliance.

        Returns:
            SLAResult with pass/fail and violations
        """
        from datetime import datetime

        stats = self.get_all_stats()
        violations = []
        overall = stats["overall"]

        # Check overall latency SLAs
        latency = overall["latency"]
        if latency["p50"] > self.sla_config.max_p50_latency_ms:
            violations.append(
                f"P50 latency {latency['p50']:.1f}ms exceeds SLA "
                f"{self.sla_config.max_p50_latency_ms:.1f}ms"
            )

        if latency["p95"] > self.sla_config.max_p95_latency_ms:
            violations.append(
                f"P95 latency {latency['p95']:.1f}ms exceeds SLA "
                f"{self.sla_config.max_p95_latency_ms:.1f}ms"
            )

        if latency["p99"] > self.sla_config.max_p99_latency_ms:
            violations.append(
                f"P99 latency {latency['p99']:.1f}ms exceeds SLA "
                f"{self.sla_config.max_p99_latency_ms:.1f}ms"
            )

        # Check throughput SLA
        if overall["throughput_rps"] < self.sla_config.min_throughput_rps:
            violations.append(
                f"Throughput {overall['throughput_rps']:.1f} RPS below SLA "
                f"{self.sla_config.min_throughput_rps:.1f} RPS"
            )

        # Check success rate SLA
        if overall["success_rate"] < self.sla_config.min_success_rate:
            violations.append(
                f"Success rate {overall['success_rate']:.1%} below SLA "
                f"{self.sla_config.min_success_rate:.1%}"
            )

        # Check per-operation SLAs
        for op_name, op_sla in self.sla_config.operation_slas.items():
            if op_name in stats["operations"]:
                op_stats = stats["operations"][op_name]
                op_latency = op_stats["latency"]

                if "max_p95" in op_sla and op_latency["p95"] > op_sla["max_p95"]:
                    violations.append(
                        f"{op_name} P95 latency {op_latency['p95']:.1f}ms "
                        f"exceeds SLA {op_sla['max_p95']:.1f}ms"
                    )

        passed = len(violations) == 0

        if passed:
            logger.info("✓ SLA check passed")
        else:
            logger.warning(f"✗ SLA check failed with {len(violations)} violations")
            for violation in violations:
                logger.warning(f"  - {violation}")

        return SLAResult(
            passed=passed,
            violations=violations,
            metrics=stats,
            timestamp=datetime.now().isoformat(),
        )

    def reset(self) -> None:
        """Reset all metrics."""
        self.operations.clear()
        self.start_time = time.time()
        self.total_requests = 0
        logger.info("Performance metrics reset")

    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_all_stats()
        overall = stats["overall"]
        return (
            f"PerformanceProfiler(requests={overall['total_requests']}, "
            f"throughput={overall['throughput_rps']:.1f} RPS, "
            f"p50={overall['latency']['p50']:.1f}ms, "
            f"p95={overall['latency']['p95']:.1f}ms)"
        )
