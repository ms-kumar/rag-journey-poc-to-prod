"""
Factory for creating performance client instances.
"""

import logging
from typing import TYPE_CHECKING, Optional

from src.services.performance.client import PerformanceClient
from src.services.performance.profiler import PerformanceProfiler, SLAConfig
from src.services.performance.reporter import PerformanceReporter

if TYPE_CHECKING:
    from src.config import Settings

logger = logging.getLogger(__name__)


def make_performance_client(
    settings: Optional["Settings"] = None, sla_config: SLAConfig | None = None
) -> PerformanceClient:
    """
    Create performance client from application settings.

    Args:
        settings: Optional application settings
        sla_config: Optional SLA configuration (overrides settings)

    Returns:
        Configured PerformanceClient instance

    Example:
        ```python
        from src.config import get_settings
        from src.services.performance import SLAConfig

        settings = get_settings()
        sla_config = SLAConfig(max_p95_latency_ms=300.0)
        client = make_performance_client(settings, sla_config)
        ```
    """
    # Use provided SLA config or create from settings
    if sla_config is None and settings is not None:
        perf_settings = getattr(settings, "performance", None)
        if perf_settings:
            sla_config = SLAConfig(
                max_p50_latency_ms=getattr(perf_settings, "max_p50_latency_ms", 500.0),
                max_p95_latency_ms=getattr(perf_settings, "max_p95_latency_ms", 1000.0),
                max_p99_latency_ms=getattr(perf_settings, "max_p99_latency_ms", 2000.0),
                min_throughput_rps=getattr(perf_settings, "min_throughput_rps", 10.0),
                min_success_rate=getattr(perf_settings, "min_success_rate", 0.95),
            )

    # Create profiler and reporter
    profiler = PerformanceProfiler(sla_config=sla_config)
    reporter = PerformanceReporter()

    # Create client
    client = PerformanceClient(sla_config=sla_config, profiler=profiler, reporter=reporter)

    logger.info(f"Performance client created with SLA: {sla_config is not None}")
    return client


def make_profiler(sla_config: SLAConfig | None = None) -> PerformanceProfiler:
    """
    Create standalone performance profiler.

    Args:
        sla_config: Optional SLA configuration

    Returns:
        PerformanceProfiler instance

    Example:
        ```python
        from src.services.performance import SLAConfig

        sla_config = SLAConfig(max_p95_latency_ms=250.0)
        profiler = make_profiler(sla_config)
        ```
    """
    return PerformanceProfiler(sla_config=sla_config)


def make_reporter() -> PerformanceReporter:
    """
    Create standalone performance reporter.

    Returns:
        PerformanceReporter instance

    Example:
        ```python
        reporter = make_reporter()
        reporter.export_html(stats, "report.html")
        ```
    """
    return PerformanceReporter()


def make_default_sla_config() -> SLAConfig:
    """
    Create default SLA configuration.

    Returns:
        SLAConfig with default thresholds

    Example:
        ```python
        sla_config = make_default_sla_config()
        # Customize as needed
        sla_config.max_p95_latency_ms = 200.0
        ```
    """
    return SLAConfig(
        max_p50_latency_ms=500.0,
        max_p95_latency_ms=1000.0,
        max_p99_latency_ms=2000.0,
        min_throughput_rps=10.0,
        min_success_rate=0.95,
    )


def make_strict_sla_config() -> SLAConfig:
    """
    Create strict SLA configuration for high-performance requirements.

    Returns:
        SLAConfig with strict thresholds

    Example:
        ```python
        sla_config = make_strict_sla_config()
        client = make_performance_client(sla_config=sla_config)
        ```
    """
    return SLAConfig(
        max_p50_latency_ms=100.0,
        max_p95_latency_ms=250.0,
        max_p99_latency_ms=500.0,
        min_throughput_rps=50.0,
        min_success_rate=0.99,
    )


def make_relaxed_sla_config() -> SLAConfig:
    """
    Create relaxed SLA configuration for development/testing.

    Returns:
        SLAConfig with relaxed thresholds

    Example:
        ```python
        sla_config = make_relaxed_sla_config()
        client = make_performance_client(sla_config=sla_config)
        ```
    """
    return SLAConfig(
        max_p50_latency_ms=1000.0,
        max_p95_latency_ms=2000.0,
        max_p99_latency_ms=5000.0,
        min_throughput_rps=1.0,
        min_success_rate=0.90,
    )
