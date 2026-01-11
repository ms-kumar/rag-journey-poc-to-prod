"""
Performance profiling and SLA monitoring.

Provides:
- Performance timers with context managers
- Latency tracking (p50/p95/p99)
- Throughput measurement
- SLA compliance checking
- Performance reports
"""

from src.services.performance.client import PerformanceClient
from src.services.performance.factory import (
    make_default_sla_config,
    make_performance_client,
    make_profiler,
    make_relaxed_sla_config,
    make_reporter,
    make_strict_sla_config,
)
from src.services.performance.profiler import (
    PerformanceProfiler,
    PerformanceTimer,
    SLAConfig,
    SLAResult,
)
from src.services.performance.reporter import PerformanceReporter

__all__ = [
    # Client & Factory
    "PerformanceClient",
    "make_performance_client",
    "make_profiler",
    "make_reporter",
    "make_default_sla_config",
    "make_strict_sla_config",
    "make_relaxed_sla_config",
    # Core Components (for advanced usage)
    "PerformanceProfiler",
    "PerformanceTimer",
    "SLAConfig",
    "SLAResult",
    "PerformanceReporter",
]
