"""
Cost tracking for model inference operations.

Tracks costs per model, calculates cost per 1k queries,
and provides cost reporting and analytics.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Metrics for a specific model."""

    model_name: str
    total_requests: int = 0
    total_cost: float = 0.0
    total_tokens: int = 0
    total_latency: float = 0.0  # Total latency in seconds
    errors: int = 0
    quality_scores: list[float] = field(default_factory=list)

    @property
    def avg_latency(self) -> float:
        """Average latency per request in ms."""
        if self.total_requests == 0:
            return 0.0
        return (self.total_latency / self.total_requests) * 1000

    @property
    def cost_per_1k(self) -> float:
        """Cost per 1000 queries."""
        if self.total_requests == 0:
            return 0.0
        return (self.total_cost / self.total_requests) * 1000

    @property
    def avg_quality(self) -> float:
        """Average quality score."""
        if not self.quality_scores:
            return 0.0
        return sum(self.quality_scores) / len(self.quality_scores)

    @property
    def error_rate(self) -> float:
        """Error rate as percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.errors / self.total_requests) * 100

    @property
    def cost_per_token(self) -> float:
        """Cost per token."""
        if self.total_tokens == 0:
            return 0.0
        return self.total_cost / self.total_tokens


@dataclass
class CostReport:
    """Cost report for all models."""

    total_cost: float
    total_requests: int
    models: dict[str, ModelMetrics]
    period_start: float
    period_end: float

    @property
    def period_duration_hours(self) -> float:
        """Duration of reporting period in hours."""
        return (self.period_end - self.period_start) / 3600

    @property
    def cost_per_1k(self) -> float:
        """Overall cost per 1000 queries."""
        if self.total_requests == 0:
            return 0.0
        return (self.total_cost / self.total_requests) * 1000

    @property
    def requests_per_hour(self) -> float:
        """Average requests per hour."""
        if self.period_duration_hours == 0:
            return 0.0
        return self.total_requests / self.period_duration_hours

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "total_cost": round(self.total_cost, 4),
            "total_requests": self.total_requests,
            "cost_per_1k": round(self.cost_per_1k, 4),
            "period_hours": round(self.period_duration_hours, 2),
            "requests_per_hour": round(self.requests_per_hour, 2),
            "models": {
                name: {
                    "requests": m.total_requests,
                    "cost": round(m.total_cost, 4),
                    "cost_per_1k": round(m.cost_per_1k, 4),
                    "avg_latency_ms": round(m.avg_latency, 2),
                    "avg_quality": round(m.avg_quality, 3),
                    "error_rate": round(m.error_rate, 2),
                    "cost_per_token": round(m.cost_per_token, 6),
                }
                for name, m in self.models.items()
            },
        }


class CostTracker:
    """
    Track costs and metrics for model inference operations.

    Features:
    - Per-model cost tracking
    - Latency and quality metrics
    - Cost per 1k queries calculation
    - Time-series data for trend analysis
    """

    def __init__(self):
        """Initialize cost tracker."""
        self.models: dict[str, ModelMetrics] = {}
        self.start_time = time.time()

    def record_request(
        self,
        model_name: str,
        cost: float,
        latency: float,
        tokens: int = 0,
        quality_score: float | None = None,
        error: bool = False,
    ) -> None:
        """
        Record a model inference request.

        Args:
            model_name: Name of the model used
            cost: Cost of the request in dollars
            latency: Latency in seconds
            tokens: Number of tokens processed
            quality_score: Quality score (0-1)
            error: Whether the request resulted in an error
        """
        if model_name not in self.models:
            self.models[model_name] = ModelMetrics(model_name=model_name)

        metrics = self.models[model_name]
        metrics.total_requests += 1
        metrics.total_cost += cost
        metrics.total_tokens += tokens
        metrics.total_latency += latency

        if error:
            metrics.errors += 1

        if quality_score is not None:
            metrics.quality_scores.append(quality_score)

    def get_model_metrics(self, model_name: str) -> ModelMetrics | None:
        """Get metrics for a specific model."""
        return self.models.get(model_name)

    def get_report(self, reset: bool = False) -> CostReport:
        """
        Generate a cost report.

        Args:
            reset: Whether to reset metrics after generating report

        Returns:
            CostReport with aggregated metrics
        """
        total_cost = sum(m.total_cost for m in self.models.values())
        total_requests = sum(m.total_requests for m in self.models.values())

        report = CostReport(
            total_cost=total_cost,
            total_requests=total_requests,
            models=self.models.copy(),
            period_start=self.start_time,
            period_end=time.time(),
        )

        if reset:
            self.reset()

        return report

    def reset(self) -> None:
        """Reset all metrics."""
        self.models.clear()
        self.start_time = time.time()

    def get_top_models_by_cost(self, top_n: int = 5) -> list[tuple[str, float]]:
        """Get top N models by total cost."""
        sorted_models = sorted(
            self.models.items(),
            key=lambda x: x[1].total_cost,
            reverse=True,
        )
        return [(name, m.total_cost) for name, m in sorted_models[:top_n]]

    def get_models_by_efficiency(self) -> list[tuple[str, float]]:
        """
        Get models ranked by efficiency (quality/cost ratio).

        Returns:
            List of (model_name, efficiency_score) tuples
        """
        efficiency_scores = []

        for name, metrics in self.models.items():
            if metrics.total_cost > 0 and metrics.avg_quality > 0:
                # Efficiency = quality / cost per 1k
                efficiency = metrics.avg_quality / metrics.cost_per_1k
                efficiency_scores.append((name, efficiency))

        return sorted(efficiency_scores, key=lambda x: x[1], reverse=True)

    def print_summary(self) -> None:
        """Print a summary of costs and metrics."""
        report = self.get_report()

        print("\n" + "=" * 60)
        print("COST TRACKING SUMMARY")
        print("=" * 60)
        print(f"Total Cost: ${report.total_cost:.4f}")
        print(f"Total Requests: {report.total_requests}")
        print(f"Cost per 1k queries: ${report.cost_per_1k:.4f}")
        print(f"Period: {report.period_duration_hours:.2f} hours")
        print(f"Requests/hour: {report.requests_per_hour:.2f}")
        print("\nPer-Model Breakdown:")
        print("-" * 60)

        for name, metrics in sorted(
            report.models.items(),
            key=lambda x: x[1].total_cost,
            reverse=True,
        ):
            print(f"\n{name}:")
            print(f"  Requests: {metrics.total_requests}")
            print(f"  Cost: ${metrics.total_cost:.4f}")
            print(f"  Cost/1k: ${metrics.cost_per_1k:.4f}")
            print(f"  Avg Latency: {metrics.avg_latency:.2f}ms")
            print(f"  Avg Quality: {metrics.avg_quality:.3f}")
            print(f"  Error Rate: {metrics.error_rate:.2f}%")
