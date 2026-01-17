"""
Metrics collection module for RAG pipeline dashboards.

Provides latency, cost, and quality metrics collection
for observability dashboards.

Features:
- Latency metrics (p50, p95, p99 percentiles)
- Cost tracking (tokens, API calls)
- Quality metrics (relevance, faithfulness)
- Time-series data aggregation
- Dashboard data export
"""

import statistics
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from threading import Lock
from typing import Any


class MetricType(Enum):
    """Types of metrics."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricValue:
    """A single metric value with timestamp."""

    value: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class LatencyMetrics:
    """
    Latency metrics for pipeline operations.

    Tracks timing for various RAG pipeline stages.
    """

    # Per-operation latencies (in milliseconds)
    ingestion_ms: list[float] = field(default_factory=list)
    embedding_ms: list[float] = field(default_factory=list)
    retrieval_ms: list[float] = field(default_factory=list)
    reranking_ms: list[float] = field(default_factory=list)
    generation_ms: list[float] = field(default_factory=list)
    total_request_ms: list[float] = field(default_factory=list)

    # Timestamps for time-series
    timestamps: list[datetime] = field(default_factory=list)

    def add_sample(
        self,
        operation: str,
        latency_ms: float,
        timestamp: datetime | None = None,
    ) -> None:
        """Add a latency sample for an operation."""
        ts = timestamp or datetime.now(UTC)
        self.timestamps.append(ts)

        latency_map = {
            "ingestion": self.ingestion_ms,
            "embedding": self.embedding_ms,
            "retrieval": self.retrieval_ms,
            "reranking": self.reranking_ms,
            "generation": self.generation_ms,
            "total": self.total_request_ms,
        }

        if operation in latency_map:
            latency_map[operation].append(latency_ms)

    def get_percentiles(
        self,
        operation: str,
        percentiles: list[int] | None = None,
    ) -> dict[str, float]:
        """Get percentile values for an operation."""
        percentiles = percentiles or [50, 95, 99]

        latency_map = {
            "ingestion": self.ingestion_ms,
            "embedding": self.embedding_ms,
            "retrieval": self.retrieval_ms,
            "reranking": self.reranking_ms,
            "generation": self.generation_ms,
            "total": self.total_request_ms,
        }

        values = latency_map.get(operation, [])
        if not values:
            return {f"p{p}": 0.0 for p in percentiles}

        sorted_values = sorted(values)
        result = {}
        for p in percentiles:
            idx = int(len(sorted_values) * p / 100)
            idx = min(idx, len(sorted_values) - 1)
            result[f"p{p}"] = sorted_values[idx]

        return result

    def get_summary(self) -> dict[str, dict[str, float]]:
        """Get summary statistics for all operations."""
        operations = [
            "ingestion",
            "embedding",
            "retrieval",
            "reranking",
            "generation",
            "total",
        ]

        summary = {}
        for op in operations:
            percentiles = self.get_percentiles(op)
            latency_map = {
                "ingestion": self.ingestion_ms,
                "embedding": self.embedding_ms,
                "retrieval": self.retrieval_ms,
                "reranking": self.reranking_ms,
                "generation": self.generation_ms,
                "total": self.total_request_ms,
            }
            values = latency_map.get(op, [])
            summary[op] = {
                "count": len(values),
                "mean": statistics.mean(values) if values else 0.0,
                "min": min(values) if values else 0.0,
                "max": max(values) if values else 0.0,
                **percentiles,
            }

        return summary


@dataclass
class CostMetrics:
    """
    Cost metrics for API usage and token consumption.

    Tracks costs across different LLM providers and operations.
    """

    # Token counts
    input_tokens: int = 0
    output_tokens: int = 0
    embedding_tokens: int = 0
    total_tokens: int = 0

    # API call counts
    embedding_calls: int = 0
    llm_calls: int = 0
    reranker_calls: int = 0
    vectorstore_calls: int = 0

    # Estimated costs (in USD)
    embedding_cost_usd: float = 0.0
    llm_cost_usd: float = 0.0
    total_cost_usd: float = 0.0

    # Per-request tracking
    request_costs: list[dict[str, Any]] = field(default_factory=list)

    def add_llm_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        cost_per_1k_input: float = 0.0015,
        cost_per_1k_output: float = 0.002,
        model: str = "unknown",
    ) -> float:
        """Add LLM usage and calculate cost."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens += input_tokens + output_tokens
        self.llm_calls += 1

        cost = (input_tokens * cost_per_1k_input / 1000) + (
            output_tokens * cost_per_1k_output / 1000
        )
        self.llm_cost_usd += cost
        self.total_cost_usd += cost

        self.request_costs.append(
            {
                "type": "llm",
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

        return cost

    def add_embedding_usage(
        self,
        tokens: int,
        cost_per_1k: float = 0.0001,
        model: str = "unknown",
    ) -> float:
        """Add embedding usage and calculate cost."""
        self.embedding_tokens += tokens
        self.total_tokens += tokens
        self.embedding_calls += 1

        cost = tokens * cost_per_1k / 1000
        self.embedding_cost_usd += cost
        self.total_cost_usd += cost

        self.request_costs.append(
            {
                "type": "embedding",
                "model": model,
                "tokens": tokens,
                "cost_usd": cost,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

        return cost

    def add_api_call(self, call_type: str) -> None:
        """Track an API call."""
        call_map = {
            "embedding": "embedding_calls",
            "llm": "llm_calls",
            "reranker": "reranker_calls",
            "vectorstore": "vectorstore_calls",
        }
        if call_type in call_map:
            setattr(self, call_map[call_type], getattr(self, call_map[call_type]) + 1)

    def get_summary(self) -> dict[str, Any]:
        """Get cost summary."""
        return {
            "tokens": {
                "input": self.input_tokens,
                "output": self.output_tokens,
                "embedding": self.embedding_tokens,
                "total": self.total_tokens,
            },
            "api_calls": {
                "embedding": self.embedding_calls,
                "llm": self.llm_calls,
                "reranker": self.reranker_calls,
                "vectorstore": self.vectorstore_calls,
            },
            "costs_usd": {
                "embedding": round(self.embedding_cost_usd, 6),
                "llm": round(self.llm_cost_usd, 6),
                "total": round(self.total_cost_usd, 6),
            },
        }


@dataclass
class QualityMetrics:
    """
    Quality metrics for RAG responses.

    Tracks relevance, faithfulness, and other quality indicators.
    """

    # Relevance scores (0-1)
    relevance_scores: list[float] = field(default_factory=list)

    # Faithfulness scores (0-1)
    faithfulness_scores: list[float] = field(default_factory=list)

    # Answer completeness (0-1)
    completeness_scores: list[float] = field(default_factory=list)

    # User ratings (1-5)
    user_ratings: list[int] = field(default_factory=list)

    # Retrieval quality
    retrieval_precision: list[float] = field(default_factory=list)
    retrieval_recall: list[float] = field(default_factory=list)

    # Error tracking
    error_count: int = 0
    guardrail_triggers: int = 0
    hallucination_detections: int = 0

    def add_relevance_score(self, score: float) -> None:
        """Add a relevance score."""
        self.relevance_scores.append(max(0.0, min(1.0, score)))

    def add_faithfulness_score(self, score: float) -> None:
        """Add a faithfulness score."""
        self.faithfulness_scores.append(max(0.0, min(1.0, score)))

    def add_completeness_score(self, score: float) -> None:
        """Add a completeness score."""
        self.completeness_scores.append(max(0.0, min(1.0, score)))

    def add_user_rating(self, rating: int) -> None:
        """Add a user rating (1-5)."""
        self.user_ratings.append(max(1, min(5, rating)))

    def add_retrieval_metrics(self, precision: float, recall: float) -> None:
        """Add retrieval precision and recall."""
        self.retrieval_precision.append(max(0.0, min(1.0, precision)))
        self.retrieval_recall.append(max(0.0, min(1.0, recall)))

    def record_error(self) -> None:
        """Record an error occurrence."""
        self.error_count += 1

    def record_guardrail_trigger(self) -> None:
        """Record a guardrail trigger."""
        self.guardrail_triggers += 1

    def record_hallucination(self) -> None:
        """Record a hallucination detection."""
        self.hallucination_detections += 1

    def get_summary(self) -> dict[str, Any]:
        """Get quality summary."""

        def safe_mean(values: list[float]) -> float:
            return statistics.mean(values) if values else 0.0

        return {
            "relevance": {
                "mean": safe_mean(self.relevance_scores),
                "count": len(self.relevance_scores),
            },
            "faithfulness": {
                "mean": safe_mean(self.faithfulness_scores),
                "count": len(self.faithfulness_scores),
            },
            "completeness": {
                "mean": safe_mean(self.completeness_scores),
                "count": len(self.completeness_scores),
            },
            "user_ratings": {
                "mean": safe_mean([float(r) for r in self.user_ratings]),
                "count": len(self.user_ratings),
            },
            "retrieval": {
                "precision": safe_mean(self.retrieval_precision),
                "recall": safe_mean(self.retrieval_recall),
            },
            "errors": {
                "total": self.error_count,
                "guardrail_triggers": self.guardrail_triggers,
                "hallucinations": self.hallucination_detections,
            },
        }


@dataclass
class DashboardData:
    """
    Aggregated data for dashboards.

    Combines latency, cost, and quality metrics
    with time-series data for visualization.
    """

    latency: LatencyMetrics = field(default_factory=LatencyMetrics)
    cost: CostMetrics = field(default_factory=CostMetrics)
    quality: QualityMetrics = field(default_factory=QualityMetrics)

    # Request counts
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0

    # Time window
    window_start: datetime = field(default_factory=lambda: datetime.now(UTC))
    window_end: datetime | None = None

    def get_success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests

    def to_dict(self) -> dict[str, Any]:
        """Export dashboard data as dictionary."""
        return {
            "time_window": {
                "start": self.window_start.isoformat(),
                "end": (
                    self.window_end.isoformat()
                    if self.window_end
                    else datetime.now(UTC).isoformat()
                ),
            },
            "requests": {
                "total": self.total_requests,
                "successful": self.successful_requests,
                "failed": self.failed_requests,
                "success_rate": self.get_success_rate(),
            },
            "latency": self.latency.get_summary(),
            "cost": self.cost.get_summary(),
            "quality": self.quality.get_summary(),
        }


class MetricsCollector:
    """
    Central metrics collection point.

    Thread-safe collector for all pipeline metrics.

    Usage:
        collector = MetricsCollector()

        # Record latency
        with collector.time_operation("retrieval"):
            results = vectorstore.search(query)

        # Record cost
        collector.record_llm_usage(input_tokens=100, output_tokens=50)

        # Get dashboard data
        data = collector.get_dashboard_data()
    """

    def __init__(
        self,
        max_samples: int = 10000,
        enable_detailed_tracking: bool = True,
    ):
        self.max_samples = max_samples
        self.enable_detailed_tracking = enable_detailed_tracking

        self._latency = LatencyMetrics()
        self._cost = CostMetrics()
        self._quality = QualityMetrics()

        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0

        self._start_time = datetime.now(UTC)
        self._lock = Lock()

        # Per-correlation-id tracking
        self._request_metrics: dict[str, dict[str, Any]] = {}

    class _TimingContext:
        """Context manager for timing operations."""

        def __init__(self, collector: "MetricsCollector", operation: str):
            self.collector = collector
            self.operation = operation
            self.start_time = 0.0

        def __enter__(self) -> "MetricsCollector._TimingContext":
            self.start_time = time.perf_counter()
            return self

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            elapsed_ms = (time.perf_counter() - self.start_time) * 1000
            self.collector.record_latency(self.operation, elapsed_ms)

    def time_operation(self, operation: str) -> _TimingContext:
        """Time an operation using a context manager."""
        return self._TimingContext(self, operation)

    def record_latency(
        self,
        operation: str,
        latency_ms: float,
        correlation_id: str | None = None,
    ) -> None:
        """Record latency for an operation."""
        with self._lock:
            self._latency.add_sample(operation, latency_ms)

            # Trim if over limit
            if len(self._latency.timestamps) > self.max_samples:
                self._trim_latency_samples()

            # Track per-request if correlation_id provided
            if correlation_id and self.enable_detailed_tracking:
                if correlation_id not in self._request_metrics:
                    self._request_metrics[correlation_id] = {}
                self._request_metrics[correlation_id][f"{operation}_ms"] = latency_ms

    def _trim_latency_samples(self) -> None:
        """Trim latency samples to max_samples."""
        keep = self.max_samples // 2
        self._latency.timestamps = self._latency.timestamps[-keep:]
        self._latency.ingestion_ms = self._latency.ingestion_ms[-keep:]
        self._latency.embedding_ms = self._latency.embedding_ms[-keep:]
        self._latency.retrieval_ms = self._latency.retrieval_ms[-keep:]
        self._latency.reranking_ms = self._latency.reranking_ms[-keep:]
        self._latency.generation_ms = self._latency.generation_ms[-keep:]
        self._latency.total_request_ms = self._latency.total_request_ms[-keep:]

    def record_llm_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str = "unknown",
        cost_per_1k_input: float = 0.0015,
        cost_per_1k_output: float = 0.002,
    ) -> float:
        """Record LLM token usage."""
        with self._lock:
            return self._cost.add_llm_usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=model,
                cost_per_1k_input=cost_per_1k_input,
                cost_per_1k_output=cost_per_1k_output,
            )

    def record_embedding_usage(
        self,
        tokens: int,
        model: str = "unknown",
        cost_per_1k: float = 0.0001,
    ) -> float:
        """Record embedding token usage."""
        with self._lock:
            return self._cost.add_embedding_usage(
                tokens=tokens,
                model=model,
                cost_per_1k=cost_per_1k,
            )

    def record_api_call(self, call_type: str) -> None:
        """Record an API call."""
        with self._lock:
            self._cost.add_api_call(call_type)

    def record_quality_score(
        self,
        metric: str,
        score: float,
    ) -> None:
        """Record a quality metric score."""
        with self._lock:
            if metric == "relevance":
                self._quality.add_relevance_score(score)
            elif metric == "faithfulness":
                self._quality.add_faithfulness_score(score)
            elif metric == "completeness":
                self._quality.add_completeness_score(score)

    def record_user_rating(self, rating: int) -> None:
        """Record a user rating."""
        with self._lock:
            self._quality.add_user_rating(rating)

    def record_retrieval_quality(self, precision: float, recall: float) -> None:
        """Record retrieval quality metrics."""
        with self._lock:
            self._quality.add_retrieval_metrics(precision, recall)

    def record_error(self, error_type: str = "general") -> None:
        """Record an error."""
        with self._lock:
            self._failed_requests += 1
            self._quality.record_error()
            if error_type == "guardrail":
                self._quality.record_guardrail_trigger()
            elif error_type == "hallucination":
                self._quality.record_hallucination()

    def record_request_start(self, correlation_id: str | None = None) -> None:
        """Record the start of a request."""
        with self._lock:
            self._total_requests += 1
            if correlation_id and self.enable_detailed_tracking:
                self._request_metrics[correlation_id] = {
                    "start_time": datetime.now(UTC).isoformat(),
                }

    def record_request_end(
        self,
        success: bool,
        correlation_id: str | None = None,
    ) -> None:
        """Record the end of a request."""
        with self._lock:
            if success:
                self._successful_requests += 1
            else:
                self._failed_requests += 1

            if correlation_id and correlation_id in self._request_metrics:
                self._request_metrics[correlation_id]["end_time"] = datetime.now(UTC).isoformat()
                self._request_metrics[correlation_id]["success"] = success

    def get_dashboard_data(self) -> DashboardData:
        """Get aggregated dashboard data."""
        with self._lock:
            data = DashboardData(
                latency=self._latency,
                cost=self._cost,
                quality=self._quality,
                total_requests=self._total_requests,
                successful_requests=self._successful_requests,
                failed_requests=self._failed_requests,
                window_start=self._start_time,
                window_end=datetime.now(UTC),
            )
            return data

    def get_request_metrics(self, correlation_id: str) -> dict[str, Any] | None:
        """Get metrics for a specific request."""
        with self._lock:
            return self._request_metrics.get(correlation_id)

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._latency = LatencyMetrics()
            self._cost = CostMetrics()
            self._quality = QualityMetrics()
            self._total_requests = 0
            self._successful_requests = 0
            self._failed_requests = 0
            self._start_time = datetime.now(UTC)
            self._request_metrics.clear()

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        data = self.get_dashboard_data()

        # Request metrics
        lines.append("# HELP rag_requests_total Total number of requests")
        lines.append("# TYPE rag_requests_total counter")
        lines.append(f"rag_requests_total {data.total_requests}")

        lines.append("# HELP rag_requests_success_total Successful requests")
        lines.append("# TYPE rag_requests_success_total counter")
        lines.append(f"rag_requests_success_total {data.successful_requests}")

        # Latency metrics
        for op, stats in data.latency.get_summary().items():
            if stats["count"] > 0:
                lines.append(f"# HELP rag_{op}_latency_ms {op} latency in milliseconds")
                lines.append(f"# TYPE rag_{op}_latency_ms summary")
                lines.append(f'rag_{op}_latency_ms{{quantile="0.5"}} {stats["p50"]:.2f}')
                lines.append(f'rag_{op}_latency_ms{{quantile="0.95"}} {stats["p95"]:.2f}')
                lines.append(f'rag_{op}_latency_ms{{quantile="0.99"}} {stats["p99"]:.2f}')

        # Cost metrics
        cost_summary = data.cost.get_summary()
        lines.append("# HELP rag_tokens_total Total tokens used")
        lines.append("# TYPE rag_tokens_total counter")
        lines.append(f"rag_tokens_total {cost_summary['tokens']['total']}")

        lines.append("# HELP rag_cost_usd_total Total cost in USD")
        lines.append("# TYPE rag_cost_usd_total counter")
        lines.append(f"rag_cost_usd_total {cost_summary['costs_usd']['total']}")

        return "\n".join(lines)


# Global metrics collector instance
_default_collector: MetricsCollector | None = None


def get_metrics_collector() -> MetricsCollector:
    """Get the default global metrics collector."""
    global _default_collector
    if _default_collector is None:
        _default_collector = MetricsCollector()
    return _default_collector


def set_metrics_collector(collector: MetricsCollector) -> None:
    """Set the default global metrics collector."""
    global _default_collector
    _default_collector = collector
