"""Metrics tracker for agent tools."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ToolMetrics:
    """Metrics for a tool.

    Attributes:
        tool_name: Tool name
        total_invocations: Total number of invocations
        successful_invocations: Number of successful invocations
        failed_invocations: Number of failed invocations
        total_latency_ms: Total latency in milliseconds
        total_cost: Total cost
        confidence_scores: List of confidence scores
        error_types: Dictionary of error types and counts
    """

    tool_name: str
    total_invocations: int = 0
    successful_invocations: int = 0
    failed_invocations: int = 0
    total_latency_ms: float = 0.0
    total_cost: float = 0.0
    confidence_scores: list[float] = field(default_factory=list)
    error_types: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize mutable defaults."""
        if self.confidence_scores is None:
            self.confidence_scores = []
        if self.error_types is None:
            self.error_types = {}

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_invocations == 0:
            return 0.0
        return self.successful_invocations / self.total_invocations

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency."""
        if self.total_invocations == 0:
            return 0.0
        return self.total_latency_ms / self.total_invocations

    @property
    def avg_confidence(self) -> float:
        """Calculate average confidence."""
        if not self.confidence_scores:
            return 0.0
        return sum(self.confidence_scores) / len(self.confidence_scores)


class MetricsTracker:
    """Track metrics for agent tools."""

    def __init__(self, storage_path: str | None = None):
        """Initialize metrics tracker.

        Args:
            storage_path: Optional path to store metrics
        """
        self.logger = logging.getLogger(__name__)
        self.storage_path = Path(storage_path) if storage_path else None
        self._metrics: dict[str, ToolMetrics] = {}
        self._load_metrics()

    def _load_metrics(self) -> None:
        """Load metrics from storage."""
        if not self.storage_path or not self.storage_path.exists():
            return

        try:
            with self.storage_path.open() as f:
                data = json.load(f)

            for tool_name, metrics_data in data.items():
                self._metrics[tool_name] = ToolMetrics(
                    tool_name=tool_name,
                    total_invocations=metrics_data.get("total_invocations", 0),
                    successful_invocations=metrics_data.get("successful_invocations", 0),
                    failed_invocations=metrics_data.get("failed_invocations", 0),
                    total_latency_ms=metrics_data.get("total_latency_ms", 0.0),
                    total_cost=metrics_data.get("total_cost", 0.0),
                    confidence_scores=metrics_data.get("confidence_scores", []),
                    error_types=metrics_data.get("error_types", {}),
                )

            self.logger.info(f"Loaded metrics for {len(self._metrics)} tools")
        except Exception as e:
            self.logger.error(f"Failed to load metrics: {e}")

    def _save_metrics(self) -> None:
        """Save metrics to storage."""
        if not self.storage_path:
            return

        try:
            # Create directory if needed
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert to dict
            data = {}
            for tool_name, metrics in self._metrics.items():
                data[tool_name] = {
                    "tool_name": metrics.tool_name,
                    "total_invocations": metrics.total_invocations,
                    "successful_invocations": metrics.successful_invocations,
                    "failed_invocations": metrics.failed_invocations,
                    "total_latency_ms": metrics.total_latency_ms,
                    "total_cost": metrics.total_cost,
                    "confidence_scores": metrics.confidence_scores[-100:],  # Keep last 100
                    "error_types": metrics.error_types,
                    "success_rate": metrics.success_rate,
                    "avg_latency_ms": metrics.avg_latency_ms,
                    "avg_confidence": metrics.avg_confidence,
                }

            with self.storage_path.open("w") as f:
                json.dump(data, f, indent=2)

            self.logger.debug("Metrics saved successfully")
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}")

    def track_invocation(
        self,
        tool_name: str,
        success: bool,
        latency_ms: float,
        cost: float = 0.0,
        confidence: float | None = None,
        error: str | None = None,
    ) -> None:
        """Track a tool invocation.

        Args:
            tool_name: Tool name
            success: Whether invocation was successful
            latency_ms: Latency in milliseconds
            cost: Cost of invocation
            confidence: Confidence score (if available)
            error: Error message (if failed)
        """
        # Get or create metrics
        if tool_name not in self._metrics:
            self._metrics[tool_name] = ToolMetrics(tool_name=tool_name)

        metrics = self._metrics[tool_name]

        # Update metrics
        metrics.total_invocations += 1
        if success:
            metrics.successful_invocations += 1
        else:
            metrics.failed_invocations += 1

            # Track error type
            if error:
                error_type = error.split(":")[0] if ":" in error else "unknown"
                metrics.error_types[error_type] = metrics.error_types.get(error_type, 0) + 1  # type: ignore[index]
        metrics.total_cost += cost

        if confidence is not None:
            metrics.confidence_scores.append(confidence)
            # Keep only last 100 scores
            if len(metrics.confidence_scores) > 100:
                metrics.confidence_scores = metrics.confidence_scores[-100:]

        # Save periodically (every 10 invocations)
        if metrics.total_invocations % 10 == 0:
            self._save_metrics()

    def get_tool_metrics(self, tool_name: str) -> ToolMetrics | None:
        """Get metrics for a tool.

        Args:
            tool_name: Tool name

        Returns:
            Tool metrics or None if not found
        """
        return self._metrics.get(tool_name)

    def get_all_metrics(self) -> dict[str, ToolMetrics]:
        """Get all tool metrics.

        Returns:
            Dictionary of tool metrics
        """
        return self._metrics.copy()

    def get_summary(self) -> dict:
        """Get summary of all metrics.

        Returns:
            Summary dictionary
        """
        summary = {
            "total_tools": len(self._metrics),
            "tools": {},
        }

        for tool_name, metrics in self._metrics.items():
            summary["tools"][tool_name] = {
                "invocations": metrics.total_invocations,
                "success_rate": round(metrics.success_rate, 3),
                "avg_latency_ms": round(metrics.avg_latency_ms, 2),
                "avg_confidence": round(metrics.avg_confidence, 3),
                "total_cost": round(metrics.total_cost, 4),
            }

        return summary

    def reset_metrics(self, tool_name: str | None = None) -> None:
        """Reset metrics for a tool or all tools.

        Args:
            tool_name: Optional tool name to reset (resets all if not provided)
        """
        if tool_name:
            if tool_name in self._metrics:
                self._metrics[tool_name] = ToolMetrics(tool_name=tool_name)
                self.logger.info(f"Reset metrics for tool: {tool_name}")
        else:
            self._metrics.clear()
            self.logger.info("Reset all metrics")

        self._save_metrics()


# Global tracker instance
_global_tracker: MetricsTracker | None = None


def get_metrics_tracker(storage_path: str | None = None) -> MetricsTracker:
    """Get the global metrics tracker instance.

    Args:
        storage_path: Optional path to store metrics

    Returns:
        Global metrics tracker
    """
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = MetricsTracker(storage_path)
    return _global_tracker
