"""Unit tests for the metrics module."""

import time

import pytest

from src.services.observability.metrics import (
    CostMetrics,
    DashboardData,
    LatencyMetrics,
    MetricsCollector,
    QualityMetrics,
    get_metrics_collector,
    set_metrics_collector,
)


class TestLatencyMetrics:
    """Tests for LatencyMetrics."""

    def test_add_sample(self):
        """Test adding latency samples."""
        metrics = LatencyMetrics()
        metrics.add_sample("retrieval", 150.0)
        metrics.add_sample("retrieval", 200.0)
        metrics.add_sample("generation", 500.0)

        assert len(metrics.retrieval_ms) == 2
        assert len(metrics.generation_ms) == 1

    def test_get_percentiles(self):
        """Test percentile calculation."""
        metrics = LatencyMetrics()

        # Add 100 samples: 1, 2, 3, ..., 100
        for i in range(1, 101):
            metrics.add_sample("retrieval", float(i))

        percentiles = metrics.get_percentiles("retrieval", [50, 95, 99])

        # Use approximate comparisons due to percentile interpolation
        assert 49.0 <= percentiles["p50"] <= 51.0
        assert 94.0 <= percentiles["p95"] <= 96.0
        assert 98.0 <= percentiles["p99"] <= 100.0

    def test_get_percentiles_empty(self):
        """Test percentiles for empty data."""
        metrics = LatencyMetrics()
        percentiles = metrics.get_percentiles("retrieval")

        assert percentiles["p50"] == 0.0
        assert percentiles["p95"] == 0.0
        assert percentiles["p99"] == 0.0

    def test_get_summary(self):
        """Test summary statistics."""
        metrics = LatencyMetrics()
        metrics.add_sample("embedding", 100.0)
        metrics.add_sample("embedding", 200.0)
        metrics.add_sample("embedding", 300.0)

        summary = metrics.get_summary()

        assert summary["embedding"]["count"] == 3
        assert summary["embedding"]["mean"] == 200.0
        assert summary["embedding"]["min"] == 100.0
        assert summary["embedding"]["max"] == 300.0


class TestCostMetrics:
    """Tests for CostMetrics."""

    def test_add_llm_usage(self):
        """Test adding LLM usage."""
        metrics = CostMetrics()

        cost = metrics.add_llm_usage(
            input_tokens=1000,
            output_tokens=500,
            cost_per_1k_input=0.01,
            cost_per_1k_output=0.03,
            model="gpt-4",
        )

        assert metrics.input_tokens == 1000
        assert metrics.output_tokens == 500
        assert metrics.total_tokens == 1500
        assert metrics.llm_calls == 1
        assert cost == pytest.approx(0.01 * 1 + 0.03 * 0.5)

    def test_add_embedding_usage(self):
        """Test adding embedding usage."""
        metrics = CostMetrics()

        cost = metrics.add_embedding_usage(
            tokens=5000,
            cost_per_1k=0.0001,
            model="ada-002",
        )

        assert metrics.embedding_tokens == 5000
        assert metrics.embedding_calls == 1
        assert cost == pytest.approx(0.0005)

    def test_add_api_call(self):
        """Test tracking API calls."""
        metrics = CostMetrics()

        metrics.add_api_call("embedding")
        metrics.add_api_call("llm")
        metrics.add_api_call("vectorstore")
        metrics.add_api_call("vectorstore")

        assert metrics.embedding_calls == 1
        assert metrics.llm_calls == 1
        assert metrics.vectorstore_calls == 2

    def test_get_summary(self):
        """Test cost summary."""
        metrics = CostMetrics()
        metrics.add_llm_usage(100, 50, model="gpt-4")
        metrics.add_embedding_usage(1000, model="ada")

        summary = metrics.get_summary()

        assert "tokens" in summary
        assert "api_calls" in summary
        assert "costs_usd" in summary
        assert summary["tokens"]["total"] == 1150


class TestQualityMetrics:
    """Tests for QualityMetrics."""

    def test_add_relevance_score(self):
        """Test adding relevance scores."""
        metrics = QualityMetrics()
        metrics.add_relevance_score(0.85)
        metrics.add_relevance_score(0.92)

        assert len(metrics.relevance_scores) == 2
        assert metrics.relevance_scores[0] == 0.85

    def test_score_clamping(self):
        """Test that scores are clamped to [0, 1]."""
        metrics = QualityMetrics()
        metrics.add_relevance_score(1.5)  # Should be clamped to 1.0
        metrics.add_relevance_score(-0.5)  # Should be clamped to 0.0

        assert metrics.relevance_scores[0] == 1.0
        assert metrics.relevance_scores[1] == 0.0

    def test_add_user_rating(self):
        """Test adding user ratings."""
        metrics = QualityMetrics()
        metrics.add_user_rating(5)
        metrics.add_user_rating(3)

        assert metrics.user_ratings == [5, 3]

    def test_user_rating_clamping(self):
        """Test that ratings are clamped to [1, 5]."""
        metrics = QualityMetrics()
        metrics.add_user_rating(0)  # Should be clamped to 1
        metrics.add_user_rating(10)  # Should be clamped to 5

        assert metrics.user_ratings == [1, 5]

    def test_record_error(self):
        """Test recording errors."""
        metrics = QualityMetrics()
        metrics.record_error()
        metrics.record_error()

        assert metrics.error_count == 2

    def test_record_guardrail_trigger(self):
        """Test recording guardrail triggers."""
        metrics = QualityMetrics()
        metrics.record_guardrail_trigger()

        assert metrics.guardrail_triggers == 1

    def test_get_summary(self):
        """Test quality summary."""
        metrics = QualityMetrics()
        metrics.add_relevance_score(0.8)
        metrics.add_relevance_score(0.9)
        metrics.add_faithfulness_score(0.85)
        metrics.add_user_rating(4)
        metrics.record_error()

        summary = metrics.get_summary()

        # Use approximate comparison for floating point
        assert abs(summary["relevance"]["mean"] - 0.85) < 0.001
        assert summary["relevance"]["count"] == 2
        assert abs(summary["faithfulness"]["mean"] - 0.85) < 0.001
        assert summary["user_ratings"]["mean"] == 4.0
        assert summary["errors"]["total"] == 1


class TestDashboardData:
    """Tests for DashboardData."""

    def test_get_success_rate(self):
        """Test success rate calculation."""
        data = DashboardData(
            total_requests=100,
            successful_requests=95,
            failed_requests=5,
        )

        assert data.get_success_rate() == 0.95

    def test_get_success_rate_no_requests(self):
        """Test success rate with no requests."""
        data = DashboardData()
        assert data.get_success_rate() == 1.0

    def test_to_dict(self):
        """Test dashboard data serialization."""
        data = DashboardData(
            total_requests=10,
            successful_requests=9,
            failed_requests=1,
        )

        result = data.to_dict()

        assert "time_window" in result
        assert "requests" in result
        assert "latency" in result
        assert "cost" in result
        assert "quality" in result
        assert result["requests"]["success_rate"] == 0.9


class TestMetricsCollector:
    """Tests for MetricsCollector."""

    def test_time_operation(self):
        """Test timing an operation."""
        collector = MetricsCollector()

        with collector.time_operation("retrieval"):
            time.sleep(0.01)  # 10ms

        data = collector.get_dashboard_data()
        assert len(data.latency.retrieval_ms) == 1
        assert data.latency.retrieval_ms[0] >= 10

    def test_record_latency(self):
        """Test recording latency directly."""
        collector = MetricsCollector()
        collector.record_latency("generation", 250.0)
        collector.record_latency("generation", 300.0)

        data = collector.get_dashboard_data()
        assert len(data.latency.generation_ms) == 2

    def test_record_latency_with_correlation_id(self):
        """Test recording latency with correlation ID."""
        collector = MetricsCollector(enable_detailed_tracking=True)
        collector.record_latency("retrieval", 150.0, correlation_id="req-123")

        metrics = collector.get_request_metrics("req-123")
        assert metrics is not None
        assert metrics["retrieval_ms"] == 150.0

    def test_record_llm_usage(self):
        """Test recording LLM usage."""
        collector = MetricsCollector()
        cost = collector.record_llm_usage(
            input_tokens=100,
            output_tokens=50,
            model="gpt-4",
        )

        data = collector.get_dashboard_data()
        assert data.cost.input_tokens == 100
        assert data.cost.output_tokens == 50
        assert cost > 0

    def test_record_quality_score(self):
        """Test recording quality scores."""
        collector = MetricsCollector()
        collector.record_quality_score("relevance", 0.85)
        collector.record_quality_score("faithfulness", 0.9)

        data = collector.get_dashboard_data()
        assert len(data.quality.relevance_scores) == 1
        assert len(data.quality.faithfulness_scores) == 1

    def test_record_request_lifecycle(self):
        """Test recording request start and end."""
        collector = MetricsCollector()
        collector.record_request_start(correlation_id="req-456")
        collector.record_request_end(success=True, correlation_id="req-456")

        data = collector.get_dashboard_data()
        assert data.total_requests == 1
        assert data.successful_requests == 1
        assert data.failed_requests == 0

    def test_record_request_failure(self):
        """Test recording failed request."""
        collector = MetricsCollector()
        collector.record_request_start()
        collector.record_request_end(success=False)

        data = collector.get_dashboard_data()
        assert data.failed_requests == 1

    def test_record_error(self):
        """Test recording errors."""
        collector = MetricsCollector()
        collector.record_error("general")
        collector.record_error("guardrail")
        collector.record_error("hallucination")

        data = collector.get_dashboard_data()
        assert data.quality.error_count == 3
        assert data.quality.guardrail_triggers == 1
        assert data.quality.hallucination_detections == 1

    def test_reset(self):
        """Test resetting collector."""
        collector = MetricsCollector()
        collector.record_latency("retrieval", 100.0)
        collector.record_request_start()

        collector.reset()

        data = collector.get_dashboard_data()
        assert data.total_requests == 0
        assert len(data.latency.retrieval_ms) == 0

    def test_export_prometheus(self):
        """Test Prometheus export format."""
        collector = MetricsCollector()
        collector.record_request_start()
        collector.record_request_end(success=True)
        collector.record_latency("retrieval", 150.0)
        collector.record_llm_usage(100, 50)

        prometheus_output = collector.export_prometheus()

        assert "rag_requests_total" in prometheus_output
        assert "rag_tokens_total" in prometheus_output

    def test_max_samples_limit(self):
        """Test that samples are trimmed when exceeding limit."""
        collector = MetricsCollector(max_samples=10)

        for i in range(20):
            collector.record_latency("retrieval", float(i))

        data = collector.get_dashboard_data()
        # Should have trimmed to approximately half
        assert len(data.latency.retrieval_ms) <= 10


class TestGlobalMetricsCollector:
    """Tests for global metrics collector functions."""

    def test_get_metrics_collector_returns_default(self):
        """Test get_metrics_collector returns a collector."""
        collector = get_metrics_collector()
        assert isinstance(collector, MetricsCollector)

    def test_set_metrics_collector_changes_global(self):
        """Test set_metrics_collector changes the global collector."""
        custom_collector = MetricsCollector(max_samples=500)
        set_metrics_collector(custom_collector)

        assert get_metrics_collector().max_samples == 500
