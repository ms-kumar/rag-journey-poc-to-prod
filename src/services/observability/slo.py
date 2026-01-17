"""
SLO (Service Level Objective) monitoring module for RAG pipeline.

Provides SLO definition, monitoring, and alerting capabilities
for tracking service quality against defined targets.

Features:
- SLO definition with targets and error budgets
- Real-time SLO status monitoring
- Alert generation for SLO violations
- Burn rate calculation
- Error budget tracking
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from threading import Lock
from typing import Any

from src.services.observability.metrics import MetricsCollector


class SLOType(Enum):
    """Types of SLOs."""

    AVAILABILITY = "availability"  # Percentage of successful requests
    LATENCY = "latency"  # Percentage of requests under threshold
    ERROR_RATE = "error_rate"  # Percentage of error-free requests
    QUALITY = "quality"  # Percentage meeting quality threshold


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status."""

    FIRING = "firing"
    RESOLVED = "resolved"


@dataclass
class SLODefinition:
    """
    Definition of a Service Level Objective.

    Attributes:
        name: Unique name for the SLO
        slo_type: Type of SLO (availability, latency, etc.)
        target: Target percentage (0-100)
        window_days: Rolling window for calculation
        threshold: Threshold value for latency/quality SLOs
        error_budget_percent: Allowed error budget as percentage
    """

    name: str
    slo_type: SLOType
    target: float  # Target percentage (e.g., 99.9)
    window_days: int = 30
    threshold: float | None = None  # For latency (ms) or quality (0-1)
    error_budget_percent: float | None = None  # Derived from target if not set
    description: str = ""
    labels: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Calculate error budget if not provided."""
        if self.error_budget_percent is None:
            self.error_budget_percent = 100.0 - self.target

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.slo_type.value,
            "target": self.target,
            "window_days": self.window_days,
            "threshold": self.threshold,
            "error_budget_percent": self.error_budget_percent,
            "description": self.description,
            "labels": self.labels,
        }


@dataclass
class SLOStatus:
    """
    Current status of an SLO.

    Attributes:
        slo: The SLO definition
        current_value: Current measured value
        is_met: Whether SLO is currently met
        error_budget_remaining: Remaining error budget percentage
        burn_rate: Current error budget burn rate
        samples: Number of samples in the window
    """

    slo: SLODefinition
    current_value: float
    is_met: bool
    error_budget_remaining: float
    burn_rate: float
    samples: int
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "slo": self.slo.to_dict(),
            "current_value": round(self.current_value, 4),
            "is_met": self.is_met,
            "error_budget_remaining": round(self.error_budget_remaining, 4),
            "burn_rate": round(self.burn_rate, 4),
            "samples": self.samples,
            "last_updated": self.last_updated.isoformat(),
        }


@dataclass
class SLOAlert:
    """
    Alert generated for SLO violations.

    Attributes:
        slo_name: Name of the SLO
        severity: Alert severity
        status: Alert status (firing/resolved)
        message: Human-readable alert message
        current_value: Value that triggered the alert
        threshold: Threshold that was violated
        started_at: When the alert started firing
        resolved_at: When the alert was resolved (if resolved)
    """

    slo_name: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    current_value: float
    threshold: float
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    resolved_at: datetime | None = None
    labels: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "slo_name": self.slo_name,
            "severity": self.severity.value,
            "status": self.status.value,
            "message": self.message,
            "current_value": round(self.current_value, 4),
            "threshold": round(self.threshold, 4),
            "started_at": self.started_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "labels": self.labels,
        }


@dataclass
class SLOSample:
    """A single measurement sample for SLO calculation."""

    timestamp: datetime
    success: bool
    value: float | None = None  # For latency/quality measurements


class SLOMonitor:
    """
    Monitor and track SLO compliance.

    Usage:
        monitor = SLOMonitor()

        # Define SLOs
        monitor.register_slo(SLODefinition(
            name="api_availability",
            slo_type=SLOType.AVAILABILITY,
            target=99.9,
        ))

        monitor.register_slo(SLODefinition(
            name="p95_latency",
            slo_type=SLOType.LATENCY,
            target=95.0,
            threshold=200,  # 200ms
        ))

        # Record measurements
        monitor.record_success("api_availability")
        monitor.record_latency("p95_latency", 150)

        # Check status
        status = monitor.get_slo_status("api_availability")
        alerts = monitor.get_active_alerts()
    """

    def __init__(
        self,
        metrics_collector: MetricsCollector | None = None,
        alert_callback: Any | None = None,
        max_samples_per_slo: int = 100000,
    ):
        self._slos: dict[str, SLODefinition] = {}
        self._samples: dict[str, deque[SLOSample]] = {}
        self._alerts: dict[str, SLOAlert] = {}
        self._alert_history: list[SLOAlert] = []
        self._metrics_collector = metrics_collector
        self._alert_callback = alert_callback
        self._max_samples = max_samples_per_slo
        self._lock = Lock()

    def register_slo(self, slo: SLODefinition) -> None:
        """Register a new SLO to monitor."""
        with self._lock:
            self._slos[slo.name] = slo
            self._samples[slo.name] = deque(maxlen=self._max_samples)

    def unregister_slo(self, name: str) -> None:
        """Unregister an SLO."""
        with self._lock:
            self._slos.pop(name, None)
            self._samples.pop(name, None)
            self._alerts.pop(name, None)

    def get_slo(self, name: str) -> SLODefinition | None:
        """Get an SLO definition by name."""
        return self._slos.get(name)

    def list_slos(self) -> list[SLODefinition]:
        """List all registered SLOs."""
        return list(self._slos.values())

    def record_success(self, slo_name: str) -> None:
        """Record a successful measurement for an availability SLO."""
        self._record_sample(slo_name, success=True)

    def record_failure(self, slo_name: str) -> None:
        """Record a failed measurement for an availability SLO."""
        self._record_sample(slo_name, success=False)

    def record_latency(self, slo_name: str, latency_ms: float) -> None:
        """Record a latency measurement for a latency SLO."""
        slo = self._slos.get(slo_name)
        if slo and slo.threshold:
            success = latency_ms <= slo.threshold
            self._record_sample(slo_name, success=success, value=latency_ms)

    def record_quality(self, slo_name: str, score: float) -> None:
        """Record a quality score for a quality SLO."""
        slo = self._slos.get(slo_name)
        if slo and slo.threshold:
            success = score >= slo.threshold
            self._record_sample(slo_name, success=success, value=score)

    def _record_sample(
        self,
        slo_name: str,
        success: bool,
        value: float | None = None,
    ) -> None:
        """Record a sample and check for alerts."""
        with self._lock:
            if slo_name not in self._samples:
                return

            sample = SLOSample(
                timestamp=datetime.now(UTC),
                success=success,
                value=value,
            )
            self._samples[slo_name].append(sample)

            # Check for SLO violation
            self._check_and_alert(slo_name)

    def _check_and_alert(self, slo_name: str) -> None:
        """Check SLO status and generate alerts if needed."""
        slo = self._slos.get(slo_name)
        if not slo:
            return

        status = self._calculate_status(slo_name)
        if not status:
            return

        existing_alert = self._alerts.get(slo_name)

        # Check for new violation
        if not status.is_met:
            if not existing_alert or existing_alert.status == AlertStatus.RESOLVED:
                # Generate new alert
                severity = self._determine_severity(status)
                alert = SLOAlert(
                    slo_name=slo_name,
                    severity=severity,
                    status=AlertStatus.FIRING,
                    message=self._format_alert_message(slo, status),
                    current_value=status.current_value,
                    threshold=slo.target,
                    labels=slo.labels,
                )
                self._alerts[slo_name] = alert
                self._alert_history.append(alert)

                if self._alert_callback:
                    self._alert_callback(alert)

        # Check for resolution
        elif existing_alert and existing_alert.status == AlertStatus.FIRING:
            existing_alert.status = AlertStatus.RESOLVED
            existing_alert.resolved_at = datetime.now(UTC)

            if self._alert_callback:
                self._alert_callback(existing_alert)

    def _determine_severity(self, status: SLOStatus) -> AlertSeverity:
        """Determine alert severity based on SLO status."""
        if status.error_budget_remaining <= 0:
            return AlertSeverity.CRITICAL
        if status.error_budget_remaining <= 25:
            return AlertSeverity.WARNING
        return AlertSeverity.INFO

    def _format_alert_message(self, slo: SLODefinition, status: SLOStatus) -> str:
        """Format a human-readable alert message."""
        return (
            f"SLO '{slo.name}' is not meeting target. "
            f"Current: {status.current_value:.2f}%, Target: {slo.target:.2f}%. "
            f"Error budget remaining: {status.error_budget_remaining:.2f}%"
        )

    def _calculate_status(self, slo_name: str) -> SLOStatus | None:
        """Calculate current SLO status."""
        slo = self._slos.get(slo_name)
        samples = self._samples.get(slo_name)

        if not slo or not samples:
            return None

        # Filter samples within window
        now = datetime.now(UTC)
        window_seconds = slo.window_days * 24 * 60 * 60
        valid_samples = [
            s for s in samples if (now - s.timestamp).total_seconds() <= window_seconds
        ]

        if not valid_samples:
            return SLOStatus(
                slo=slo,
                current_value=100.0,
                is_met=True,
                error_budget_remaining=slo.error_budget_percent or 0.0,
                burn_rate=0.0,
                samples=0,
            )

        # Calculate success rate
        success_count = sum(1 for s in valid_samples if s.success)
        total = len(valid_samples)
        current_value = (success_count / total) * 100 if total > 0 else 100.0

        # Calculate error budget
        error_budget_total = slo.error_budget_percent or (100.0 - slo.target)
        error_rate = 100.0 - current_value
        error_budget_consumed = error_rate
        error_budget_remaining = max(0.0, error_budget_total - error_budget_consumed)

        # Calculate burn rate (how fast we're consuming budget)
        # Burn rate > 1 means we're consuming budget faster than allowed
        if error_budget_total > 0:
            burn_rate = error_budget_consumed / error_budget_total
        else:
            burn_rate = 0.0

        return SLOStatus(
            slo=slo,
            current_value=current_value,
            is_met=current_value >= slo.target,
            error_budget_remaining=error_budget_remaining,
            burn_rate=burn_rate,
            samples=total,
        )

    def get_slo_status(self, slo_name: str) -> SLOStatus | None:
        """Get current status for an SLO."""
        with self._lock:
            return self._calculate_status(slo_name)

    def get_all_statuses(self) -> dict[str, SLOStatus]:
        """Get status for all SLOs."""
        with self._lock:
            return {name: status for name in self._slos if (status := self._calculate_status(name))}

    def get_active_alerts(self) -> list[SLOAlert]:
        """Get all currently firing alerts."""
        with self._lock:
            return [a for a in self._alerts.values() if a.status == AlertStatus.FIRING]

    def get_alert_history(
        self,
        limit: int = 100,
        slo_name: str | None = None,
    ) -> list[SLOAlert]:
        """Get alert history."""
        with self._lock:
            history = self._alert_history
            if slo_name:
                history = [a for a in history if a.slo_name == slo_name]
            return history[-limit:]

    def get_dashboard_summary(self) -> dict[str, Any]:
        """Get summary for dashboards."""
        statuses = self.get_all_statuses()
        active_alerts = self.get_active_alerts()

        # Calculate overall health
        total_slos = len(statuses)
        met_slos = sum(1 for s in statuses.values() if s.is_met)
        health_score = (met_slos / total_slos * 100) if total_slos > 0 else 100.0

        return {
            "overall_health": round(health_score, 2),
            "total_slos": total_slos,
            "slos_met": met_slos,
            "slos_violated": total_slos - met_slos,
            "active_alerts": len(active_alerts),
            "critical_alerts": sum(
                1 for a in active_alerts if a.severity == AlertSeverity.CRITICAL
            ),
            "warning_alerts": sum(1 for a in active_alerts if a.severity == AlertSeverity.WARNING),
            "statuses": {name: s.to_dict() for name, s in statuses.items()},
            "alerts": [a.to_dict() for a in active_alerts],
        }

    def reset(self) -> None:
        """Reset all SLO data."""
        with self._lock:
            for name in self._samples:
                self._samples[name].clear()
            self._alerts.clear()
            self._alert_history.clear()


# Default SLO definitions for RAG pipeline
def create_default_rag_slos() -> list[SLODefinition]:
    """Create default SLO definitions for a RAG pipeline."""
    return [
        # Availability SLO: 99.5% of requests succeed
        SLODefinition(
            name="rag_availability",
            slo_type=SLOType.AVAILABILITY,
            target=99.5,
            window_days=30,
            description="RAG pipeline availability - percentage of successful requests",
        ),
        # Latency SLO: 95% of requests complete under 2 seconds
        SLODefinition(
            name="rag_p95_latency",
            slo_type=SLOType.LATENCY,
            target=95.0,
            threshold=2000,  # 2000ms = 2 seconds
            window_days=7,
            description="RAG response latency - 95% under 2 seconds",
        ),
        # Latency SLO: 99% of requests complete under 5 seconds
        SLODefinition(
            name="rag_p99_latency",
            slo_type=SLOType.LATENCY,
            target=99.0,
            threshold=5000,  # 5000ms = 5 seconds
            window_days=7,
            description="RAG response latency - 99% under 5 seconds",
        ),
        # Quality SLO: 90% of responses have relevance >= 0.7
        SLODefinition(
            name="rag_quality",
            slo_type=SLOType.QUALITY,
            target=90.0,
            threshold=0.7,
            window_days=7,
            description="RAG response quality - 90% with relevance >= 0.7",
        ),
        # Error rate SLO: Less than 1% errors
        SLODefinition(
            name="rag_error_rate",
            slo_type=SLOType.ERROR_RATE,
            target=99.0,  # 99% error-free = <1% error rate
            window_days=7,
            description="RAG error rate - less than 1% errors",
        ),
    ]


# Global SLO monitor instance
_default_monitor: SLOMonitor | None = None


def get_slo_monitor() -> SLOMonitor:
    """Get the default global SLO monitor."""
    global _default_monitor
    if _default_monitor is None:
        _default_monitor = SLOMonitor()
        # Register default SLOs
        for slo in create_default_rag_slos():
            _default_monitor.register_slo(slo)
    return _default_monitor


def set_slo_monitor(monitor: SLOMonitor) -> None:
    """Set the default global SLO monitor."""
    global _default_monitor
    _default_monitor = monitor
