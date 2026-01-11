"""
Autoscaling and concurrency management.

Implements autoscaling policies to handle varying loads
while managing costs and maintaining performance SLAs.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ScalingDecision(Enum):
    """Autoscaling decision."""

    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_CHANGE = "no_change"


@dataclass
class AutoscalingPolicy:
    """Autoscaling policy configuration."""

    # Thresholds
    scale_up_cpu_threshold: float = 70.0  # CPU % to scale up
    scale_down_cpu_threshold: float = 30.0  # CPU % to scale down
    scale_up_queue_threshold: int = 50  # Queue size to scale up
    scale_down_queue_threshold: int = 5  # Queue size to scale down
    scale_up_latency_threshold: float = 1000.0  # ms

    # Limits
    min_instances: int = 1
    max_instances: int = 10
    min_concurrency: int = 1
    max_concurrency: int = 100

    # Scaling behavior
    scale_up_factor: float = 2.0  # Multiply instances by this
    scale_down_factor: float = 0.5  # Multiply instances by this
    cooldown_period: int = 60  # Seconds between scaling actions

    # Cost controls
    max_cost_per_hour: float | None = None
    budget_limit: float | None = None

    # Quality requirements
    min_quality_score: float = 0.7
    target_p95_latency_ms: float = 500.0


@dataclass
class LoadMetrics:
    """Current load metrics."""

    cpu_usage: float  # 0-100
    queue_size: int
    active_requests: int
    avg_latency_ms: float
    p95_latency_ms: float
    error_rate: float  # 0-100
    requests_per_second: float


class Autoscaler:
    """
    Autoscaling and concurrency manager.

    Features:
    - CPU and queue-based scaling
    - Latency-aware scaling
    - Cost-aware scaling decisions
    - Cooldown periods
    - Quality SLA enforcement
    """

    def __init__(self, policy: AutoscalingPolicy | None = None):
        """
        Initialize autoscaler.

        Args:
            policy: Autoscaling policy configuration
        """
        self.policy = policy or AutoscalingPolicy()
        self.current_instances = self.policy.min_instances
        self.current_concurrency = self.policy.min_concurrency
        self.last_scale_time = 0.0
        self.total_cost = 0.0
        self.history: list[dict] = []

    def should_scale(self, metrics: LoadMetrics) -> ScalingDecision:
        """
        Determine if scaling action is needed.

        Args:
            metrics: Current load metrics

        Returns:
            Scaling decision
        """
        # Check cooldown period
        time_since_last_scale = time.time() - self.last_scale_time
        if time_since_last_scale < self.policy.cooldown_period:
            return ScalingDecision.NO_CHANGE

        # Check quality SLAs
        if metrics.p95_latency_ms > self.policy.scale_up_latency_threshold:
            logger.warning(
                f"Latency SLA breach: {metrics.p95_latency_ms:.0f}ms > "
                f"{self.policy.scale_up_latency_threshold:.0f}ms"
            )
            if self._can_scale_up():
                return ScalingDecision.SCALE_UP

        # Check CPU usage
        if metrics.cpu_usage > self.policy.scale_up_cpu_threshold and self._can_scale_up():
            return ScalingDecision.SCALE_UP
        if metrics.cpu_usage < self.policy.scale_down_cpu_threshold and self._can_scale_down():
            return ScalingDecision.SCALE_DOWN

        # Check queue size
        if metrics.queue_size > self.policy.scale_up_queue_threshold and self._can_scale_up():
            return ScalingDecision.SCALE_UP
        if metrics.queue_size < self.policy.scale_down_queue_threshold and self._can_scale_down():
            return ScalingDecision.SCALE_DOWN

        return ScalingDecision.NO_CHANGE

    def _can_scale_up(self) -> bool:
        """Check if can scale up."""
        if self.current_instances >= self.policy.max_instances:
            logger.warning(f"Already at max instances: {self.policy.max_instances}")
            return False

        # Check budget constraints
        if self.policy.budget_limit and self.total_cost >= self.policy.budget_limit:
            logger.warning(f"Budget limit reached: ${self.total_cost:.2f}")
            return False

        return True

    def _can_scale_down(self) -> bool:
        """Check if can scale down."""
        return self.current_instances > self.policy.min_instances

    def scale(self, decision: ScalingDecision) -> int:
        """
        Execute scaling decision.

        Args:
            decision: Scaling decision to execute

        Returns:
            New number of instances
        """
        if decision == ScalingDecision.NO_CHANGE:
            return self.current_instances

        old_instances = self.current_instances

        if decision == ScalingDecision.SCALE_UP:
            new_instances = int(self.current_instances * self.policy.scale_up_factor)
            new_instances = min(new_instances, self.policy.max_instances)
            self.current_instances = new_instances

            # Increase concurrency
            self.current_concurrency = min(
                int(self.current_concurrency * 1.5),
                self.policy.max_concurrency,
            )

            logger.info(
                f"Scaled UP: {old_instances} -> {new_instances} instances, "
                f"concurrency: {self.current_concurrency}"
            )

        elif decision == ScalingDecision.SCALE_DOWN:
            new_instances = int(self.current_instances * self.policy.scale_down_factor)
            new_instances = max(new_instances, self.policy.min_instances)
            self.current_instances = new_instances

            # Decrease concurrency
            self.current_concurrency = max(
                int(self.current_concurrency * 0.75),
                self.policy.min_concurrency,
            )

            logger.info(
                f"Scaled DOWN: {old_instances} -> {new_instances} instances, "
                f"concurrency: {self.current_concurrency}"
            )

        self.last_scale_time = time.time()

        # Record in history
        self.history.append(
            {
                "timestamp": time.time(),
                "decision": decision.value,
                "old_instances": old_instances,
                "new_instances": self.current_instances,
                "concurrency": self.current_concurrency,
            }
        )

        return self.current_instances

    def auto_scale(self, metrics: LoadMetrics) -> tuple[ScalingDecision, int]:
        """
        Automatically scale based on metrics.

        Args:
            metrics: Current load metrics

        Returns:
            Tuple of (decision, new_instance_count)
        """
        decision = self.should_scale(metrics)
        new_instances = self.scale(decision)
        return decision, new_instances

    def record_cost(self, cost: float) -> None:
        """Record operational cost."""
        self.total_cost += cost

    def get_current_capacity(self) -> dict:
        """Get current capacity configuration."""
        return {
            "instances": self.current_instances,
            "concurrency": self.current_concurrency,
            "total_capacity": self.current_instances * self.current_concurrency,
            "utilization": "n/a",
        }

    def get_scaling_history(self, limit: int = 10) -> list[dict]:
        """Get recent scaling history."""
        return self.history[-limit:]

    def reset(self) -> None:
        """Reset autoscaler state."""
        self.current_instances = self.policy.min_instances
        self.current_concurrency = self.policy.min_concurrency
        self.last_scale_time = 0.0
        self.total_cost = 0.0
        self.history.clear()
        logger.info("Autoscaler reset")

    def print_status(self) -> None:
        """Print current autoscaling status."""
        print("\n" + "=" * 60)
        print("AUTOSCALING STATUS")
        print("=" * 60)
        print(f"Instances: {self.current_instances}/{self.policy.max_instances}")
        print(f"Concurrency: {self.current_concurrency}/{self.policy.max_concurrency}")
        print(f"Total Capacity: {self.current_instances * self.current_concurrency}")
        print(f"Total Cost: ${self.total_cost:.2f}")

        if self.policy.budget_limit:
            print(f"Budget: ${self.total_cost:.2f} / ${self.policy.budget_limit:.2f}")

        if self.history:
            print(f"\nRecent Scaling Actions: {len(self.history)}")
            for event in self.history[-5:]:
                ts = time.strftime("%H:%M:%S", time.localtime(event["timestamp"]))
                print(
                    f"  {ts}: {event['decision']} "
                    f"({event['old_instances']} -> {event['new_instances']} instances)"
                )
