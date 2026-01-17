"""
Canary deployment support for safe rollouts.

Provides canary release management with:
- Gradual traffic shifting
- Automatic health monitoring
- Rollback on failure
- Metrics comparison
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


class CanaryStatus(str, Enum):
    """Status of a canary deployment."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    PROMOTING = "promoting"
    PROMOTED = "promoted"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


class RollbackReason(str, Enum):
    """Reason for canary rollback."""

    MANUAL = "manual"
    ERROR_RATE = "error_rate"
    LATENCY = "latency"
    QUALITY = "quality"
    HEALTH_CHECK = "health_check"
    TIMEOUT = "timeout"


@dataclass
class CanaryConfig:
    """Configuration for canary deployment."""

    # Traffic settings
    initial_percentage: float = 1.0
    max_percentage: float = 100.0
    increment_percentage: float = 10.0
    increment_interval_minutes: int = 5

    # Thresholds for automatic rollback
    max_error_rate: float = 0.05  # 5%
    max_latency_p99_ms: float = 500.0
    min_quality_score: float = 0.8

    # Monitoring settings
    min_samples_before_decision: int = 100
    evaluation_window_minutes: int = 5

    # Timing
    max_duration_hours: int = 24
    stabilization_minutes: int = 10

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "initial_percentage": self.initial_percentage,
            "max_percentage": self.max_percentage,
            "increment_percentage": self.increment_percentage,
            "increment_interval_minutes": self.increment_interval_minutes,
            "max_error_rate": self.max_error_rate,
            "max_latency_p99_ms": self.max_latency_p99_ms,
            "min_quality_score": self.min_quality_score,
            "min_samples_before_decision": self.min_samples_before_decision,
            "evaluation_window_minutes": self.evaluation_window_minutes,
            "max_duration_hours": self.max_duration_hours,
            "stabilization_minutes": self.stabilization_minutes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CanaryConfig:
        """Create config from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class CanaryMetrics:
    """Metrics for canary comparison."""

    request_count: int = 0
    error_count: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    quality_scores: list[float] = field(default_factory=list)
    recorded_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        if self.request_count == 0:
            return 0.0
        return self.error_count / self.request_count

    @property
    def latency_p99(self) -> float:
        """Calculate P99 latency."""
        if not self.latencies_ms:
            return 0.0
        sorted_latencies = sorted(self.latencies_ms)
        idx = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]

    @property
    def latency_p50(self) -> float:
        """Calculate P50 latency."""
        if not self.latencies_ms:
            return 0.0
        sorted_latencies = sorted(self.latencies_ms)
        idx = len(sorted_latencies) // 2
        return sorted_latencies[idx]

    @property
    def avg_quality(self) -> float:
        """Calculate average quality score."""
        if not self.quality_scores:
            return 0.0
        return sum(self.quality_scores) / len(self.quality_scores)

    def record_request(
        self,
        latency_ms: float,
        is_error: bool = False,
        quality_score: float | None = None,
    ) -> None:
        """Record a request."""
        self.request_count += 1
        self.latencies_ms.append(latency_ms)
        if is_error:
            self.error_count += 1
        if quality_score is not None:
            self.quality_scores.append(quality_score)

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_rate,
            "latency_p50": self.latency_p50,
            "latency_p99": self.latency_p99,
            "avg_quality": self.avg_quality,
            "recorded_at": self.recorded_at.isoformat(),
        }


@dataclass
class CanaryDeployment:
    """A canary deployment."""

    id: str
    name: str
    baseline_version: str
    canary_version: str
    config: CanaryConfig
    status: CanaryStatus = CanaryStatus.PENDING
    current_percentage: float = 0.0
    baseline_metrics: CanaryMetrics = field(default_factory=CanaryMetrics)
    canary_metrics: CanaryMetrics = field(default_factory=CanaryMetrics)
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    ended_at: datetime | None = None
    last_increment_at: datetime | None = None
    rollback_reason: RollbackReason | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def start(self) -> None:
        """Start the canary deployment."""
        if self.status != CanaryStatus.PENDING:
            raise ValueError(f"Cannot start canary in {self.status} status")
        self.status = CanaryStatus.RUNNING
        self.started_at = datetime.utcnow()
        self.current_percentage = self.config.initial_percentage
        self.last_increment_at = self.started_at

    def pause(self) -> None:
        """Pause the canary deployment."""
        if self.status != CanaryStatus.RUNNING:
            raise ValueError(f"Cannot pause canary in {self.status} status")
        self.status = CanaryStatus.PAUSED

    def resume(self) -> None:
        """Resume the canary deployment."""
        if self.status != CanaryStatus.PAUSED:
            raise ValueError(f"Cannot resume canary in {self.status} status")
        self.status = CanaryStatus.RUNNING
        self.last_increment_at = datetime.utcnow()

    def promote(self) -> None:
        """Promote canary to full production."""
        if self.status not in (CanaryStatus.RUNNING, CanaryStatus.PAUSED):
            raise ValueError(f"Cannot promote canary in {self.status} status")
        self.status = CanaryStatus.PROMOTING
        self.current_percentage = 100.0
        self.status = CanaryStatus.PROMOTED
        self.ended_at = datetime.utcnow()

    def rollback(self, reason: RollbackReason = RollbackReason.MANUAL) -> None:
        """Rollback canary deployment."""
        if self.status in (CanaryStatus.PROMOTED, CanaryStatus.ROLLED_BACK):
            raise ValueError(f"Cannot rollback canary in {self.status} status")
        self.status = CanaryStatus.ROLLING_BACK
        self.rollback_reason = reason
        self.current_percentage = 0.0
        self.status = CanaryStatus.ROLLED_BACK
        self.ended_at = datetime.utcnow()

    def increment_traffic(self) -> bool:
        """
        Increment canary traffic percentage.

        Returns True if increment was successful, False if max reached.
        """
        if self.status != CanaryStatus.RUNNING:
            return False

        new_percentage = min(
            self.current_percentage + self.config.increment_percentage,
            self.config.max_percentage,
        )

        if new_percentage == self.current_percentage:
            return False

        self.current_percentage = new_percentage
        self.last_increment_at = datetime.utcnow()
        return True

    def should_increment(self) -> bool:
        """Check if it's time to increment traffic."""
        if self.status != CanaryStatus.RUNNING:
            return False
        if self.current_percentage >= self.config.max_percentage:
            return False
        if not self.last_increment_at:
            return True

        time_since_increment = datetime.utcnow() - self.last_increment_at
        increment_interval = timedelta(minutes=self.config.increment_interval_minutes)
        return time_since_increment >= increment_interval

    def should_rollback(self) -> tuple[bool, RollbackReason | None]:
        """
        Check if canary should be rolled back based on metrics.

        Returns (should_rollback, reason).
        """
        if self.status != CanaryStatus.RUNNING:
            return False, None

        # Check if we have enough samples
        if self.canary_metrics.request_count < self.config.min_samples_before_decision:
            return False, None

        # Check error rate
        if self.canary_metrics.error_rate > self.config.max_error_rate:
            return True, RollbackReason.ERROR_RATE

        # Check latency
        if self.canary_metrics.latency_p99 > self.config.max_latency_p99_ms:
            return True, RollbackReason.LATENCY

        # Check quality (if we have quality scores)
        if (
            self.canary_metrics.quality_scores
            and self.canary_metrics.avg_quality < self.config.min_quality_score
        ):
            return True, RollbackReason.QUALITY

        # Check timeout
        if self.started_at:
            max_duration = timedelta(hours=self.config.max_duration_hours)
            if datetime.utcnow() - self.started_at > max_duration:
                return True, RollbackReason.TIMEOUT

        return False, None

    def should_promote(self) -> bool:
        """Check if canary is ready for promotion."""
        if self.status != CanaryStatus.RUNNING:
            return False
        if self.current_percentage < self.config.max_percentage:
            return False

        # Check stabilization period
        if self.last_increment_at:
            stabilization = timedelta(minutes=self.config.stabilization_minutes)
            if datetime.utcnow() - self.last_increment_at < stabilization:
                return False

        # Check metrics are healthy
        should_roll, _ = self.should_rollback()
        return not should_roll

    def get_comparison(self) -> dict[str, Any]:
        """Get comparison of baseline vs canary metrics."""
        return {
            "baseline": self.baseline_metrics.to_dict(),
            "canary": self.canary_metrics.to_dict(),
            "comparison": {
                "error_rate_diff": (
                    self.canary_metrics.error_rate - self.baseline_metrics.error_rate
                ),
                "latency_p99_diff": (
                    self.canary_metrics.latency_p99 - self.baseline_metrics.latency_p99
                ),
                "quality_diff": (
                    self.canary_metrics.avg_quality - self.baseline_metrics.avg_quality
                ),
            },
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert deployment to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "baseline_version": self.baseline_version,
            "canary_version": self.canary_version,
            "config": self.config.to_dict(),
            "status": self.status.value,
            "current_percentage": self.current_percentage,
            "baseline_metrics": self.baseline_metrics.to_dict(),
            "canary_metrics": self.canary_metrics.to_dict(),
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "rollback_reason": self.rollback_reason.value if self.rollback_reason else None,
            "metadata": self.metadata,
        }


class CanaryManager:
    """Manages canary deployments."""

    def __init__(
        self,
        on_rollback: Callable[[CanaryDeployment, RollbackReason], None] | None = None,
        on_promote: Callable[[CanaryDeployment], None] | None = None,
    ):
        """Initialize canary manager."""
        self._deployments: dict[str, CanaryDeployment] = {}
        self._on_rollback = on_rollback
        self._on_promote = on_promote

    def create_deployment(
        self,
        deployment_id: str,
        name: str,
        baseline_version: str,
        canary_version: str,
        config: CanaryConfig | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CanaryDeployment:
        """Create a new canary deployment."""
        deployment = CanaryDeployment(
            id=deployment_id,
            name=name,
            baseline_version=baseline_version,
            canary_version=canary_version,
            config=config or CanaryConfig(),
            metadata=metadata or {},
        )
        self._deployments[deployment_id] = deployment
        return deployment

    def get_deployment(self, deployment_id: str) -> CanaryDeployment | None:
        """Get deployment by ID."""
        return self._deployments.get(deployment_id)

    def list_deployments(
        self,
        status: CanaryStatus | None = None,
    ) -> list[CanaryDeployment]:
        """List deployments with optional status filter."""
        deployments = list(self._deployments.values())
        if status:
            deployments = [d for d in deployments if d.status == status]
        return deployments

    def start_deployment(self, deployment_id: str) -> CanaryDeployment:
        """Start a canary deployment."""
        deployment = self.get_deployment(deployment_id)
        if not deployment:
            raise ValueError(f"Deployment {deployment_id} not found")
        deployment.start()
        return deployment

    def pause_deployment(self, deployment_id: str) -> CanaryDeployment:
        """Pause a canary deployment."""
        deployment = self.get_deployment(deployment_id)
        if not deployment:
            raise ValueError(f"Deployment {deployment_id} not found")
        deployment.pause()
        return deployment

    def resume_deployment(self, deployment_id: str) -> CanaryDeployment:
        """Resume a canary deployment."""
        deployment = self.get_deployment(deployment_id)
        if not deployment:
            raise ValueError(f"Deployment {deployment_id} not found")
        deployment.resume()
        return deployment

    def promote_deployment(self, deployment_id: str) -> CanaryDeployment:
        """Promote a canary to production."""
        deployment = self.get_deployment(deployment_id)
        if not deployment:
            raise ValueError(f"Deployment {deployment_id} not found")
        deployment.promote()
        if self._on_promote:
            self._on_promote(deployment)
        return deployment

    def rollback_deployment(
        self,
        deployment_id: str,
        reason: RollbackReason = RollbackReason.MANUAL,
    ) -> CanaryDeployment:
        """Rollback a canary deployment."""
        deployment = self.get_deployment(deployment_id)
        if not deployment:
            raise ValueError(f"Deployment {deployment_id} not found")
        deployment.rollback(reason)
        if self._on_rollback:
            self._on_rollback(deployment, reason)
        return deployment

    def record_request(
        self,
        deployment_id: str,
        is_canary: bool,
        latency_ms: float,
        is_error: bool = False,
        quality_score: float | None = None,
    ) -> None:
        """Record a request for metrics tracking."""
        deployment = self.get_deployment(deployment_id)
        if not deployment:
            return

        metrics = deployment.canary_metrics if is_canary else deployment.baseline_metrics
        metrics.record_request(
            latency_ms=latency_ms,
            is_error=is_error,
            quality_score=quality_score,
        )

    def route_request(self, deployment_id: str, user_id: str | None = None) -> bool:
        """
        Route a request to baseline or canary.

        Returns True if request should go to canary, False for baseline.
        Uses consistent hashing for sticky routing if user_id provided.
        """
        deployment = self.get_deployment(deployment_id)
        if not deployment or deployment.status != CanaryStatus.RUNNING:
            return False

        if user_id:
            # Consistent hashing for sticky routing (not security-sensitive)
            import hashlib

            hash_input = f"canary:{deployment_id}:{user_id}"
            hash_value = int(
                hashlib.md5(hash_input.encode(), usedforsecurity=False).hexdigest(), 16
            )  # noqa: S324
            bucket = (hash_value % 10000) / 100.0
            return bucket < deployment.current_percentage
        # Random routing for traffic splitting (not security-sensitive)
        import random

        return random.random() * 100 < deployment.current_percentage  # nosec B311

    def evaluate_deployments(self) -> list[tuple[str, str]]:
        """
        Evaluate all running deployments.

        Returns list of (deployment_id, action) tuples.
        Actions: 'increment', 'rollback', 'promote', 'none'
        """
        actions = []

        for deployment_id, deployment in self._deployments.items():
            if deployment.status != CanaryStatus.RUNNING:
                continue

            # Check for rollback
            should_roll, reason = deployment.should_rollback()
            if should_roll and reason:
                deployment.rollback(reason)
                if self._on_rollback:
                    self._on_rollback(deployment, reason)
                actions.append((deployment_id, f"rollback:{reason.value}"))
                continue

            # Check for promotion
            if deployment.should_promote():
                deployment.promote()
                if self._on_promote:
                    self._on_promote(deployment)
                actions.append((deployment_id, "promote"))
                continue

            # Check for increment
            if deployment.should_increment():
                deployment.increment_traffic()
                actions.append((deployment_id, "increment"))
            else:
                actions.append((deployment_id, "none"))

        return actions

    def get_active_deployment(self) -> CanaryDeployment | None:
        """Get the currently active (running) deployment."""
        running = self.list_deployments(status=CanaryStatus.RUNNING)
        return running[0] if running else None

    def get_deployment_summary(self, deployment_id: str) -> dict[str, Any]:
        """Get summary of a deployment."""
        deployment = self.get_deployment(deployment_id)
        if not deployment:
            raise ValueError(f"Deployment {deployment_id} not found")

        return {
            "id": deployment.id,
            "name": deployment.name,
            "status": deployment.status.value,
            "current_percentage": deployment.current_percentage,
            "baseline_version": deployment.baseline_version,
            "canary_version": deployment.canary_version,
            "comparison": deployment.get_comparison(),
            "should_rollback": deployment.should_rollback(),
            "should_promote": deployment.should_promote(),
        }


# Global canary manager
_canary_manager: CanaryManager | None = None


def get_canary_manager() -> CanaryManager:
    """Get the global canary manager."""
    global _canary_manager
    if _canary_manager is None:
        _canary_manager = CanaryManager()
    return _canary_manager


def set_canary_manager(manager: CanaryManager) -> None:
    """Set the global canary manager."""
    global _canary_manager
    _canary_manager = manager
