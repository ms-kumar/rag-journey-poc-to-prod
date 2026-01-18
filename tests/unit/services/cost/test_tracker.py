"""Tests for the CostTracker class."""

import time

import pytest

from src.services.cost.tracker import CostReport, CostTracker, ModelMetrics


class TestModelMetrics:
    """Tests for ModelMetrics dataclass."""

    def test_model_metrics_defaults(self):
        """Test default values."""
        metrics = ModelMetrics(model_name="test-model")

        assert metrics.model_name == "test-model"
        assert metrics.total_requests == 0
        assert metrics.total_cost == 0.0
        assert metrics.total_tokens == 0
        assert metrics.errors == 0
        assert metrics.quality_scores == []

    def test_avg_latency_calculation(self):
        """Test average latency calculation."""
        metrics = ModelMetrics(
            model_name="test-model",
            total_requests=10,
            total_latency=5.0,  # 5 seconds total
        )

        # 5s / 10 requests = 0.5s per request = 500ms
        assert metrics.avg_latency == 500.0

    def test_avg_latency_zero_requests(self):
        """Test avg latency with no requests."""
        metrics = ModelMetrics(model_name="test-model")
        assert metrics.avg_latency == 0.0

    def test_cost_per_1k_calculation(self):
        """Test cost per 1000 queries."""
        metrics = ModelMetrics(
            model_name="test-model",
            total_requests=100,
            total_cost=1.50,
        )

        # $1.50 / 100 requests * 1000 = $15 per 1k
        assert metrics.cost_per_1k == 15.0

    def test_cost_per_1k_zero_requests(self):
        """Test cost per 1k with no requests."""
        metrics = ModelMetrics(model_name="test-model")
        assert metrics.cost_per_1k == 0.0

    def test_avg_quality_calculation(self):
        """Test average quality score calculation."""
        metrics = ModelMetrics(
            model_name="test-model",
            quality_scores=[0.8, 0.9, 0.85],
        )

        assert metrics.avg_quality == pytest.approx(0.85, rel=0.01)

    def test_avg_quality_empty_scores(self):
        """Test avg quality with no scores."""
        metrics = ModelMetrics(model_name="test-model")
        assert metrics.avg_quality == 0.0

    def test_error_rate_calculation(self):
        """Test error rate calculation."""
        metrics = ModelMetrics(
            model_name="test-model",
            total_requests=100,
            errors=5,
        )

        assert metrics.error_rate == 5.0

    def test_error_rate_zero_requests(self):
        """Test error rate with no requests."""
        metrics = ModelMetrics(model_name="test-model")
        assert metrics.error_rate == 0.0

    def test_cost_per_token_calculation(self):
        """Test cost per token calculation."""
        metrics = ModelMetrics(
            model_name="test-model",
            total_cost=10.0,
            total_tokens=10000,
        )

        assert metrics.cost_per_token == 0.001

    def test_cost_per_token_zero_tokens(self):
        """Test cost per token with no tokens."""
        metrics = ModelMetrics(model_name="test-model")
        assert metrics.cost_per_token == 0.0


class TestCostReport:
    """Tests for CostReport dataclass."""

    def test_period_duration_hours(self):
        """Test period duration calculation."""
        now = time.time()
        report = CostReport(
            total_cost=0,
            total_requests=0,
            models={},
            period_start=now - 3600,  # 1 hour ago
            period_end=now,
        )

        assert report.period_duration_hours == pytest.approx(1.0, rel=0.01)

    def test_cost_per_1k(self):
        """Test cost per 1k calculation."""
        report = CostReport(
            total_cost=5.0,
            total_requests=500,
            models={},
            period_start=0,
            period_end=1,
        )

        # $5 / 500 * 1000 = $10 per 1k
        assert report.cost_per_1k == 10.0

    def test_requests_per_hour(self):
        """Test requests per hour calculation."""
        now = time.time()
        report = CostReport(
            total_cost=0,
            total_requests=1000,
            models={},
            period_start=now - 3600,  # 1 hour ago
            period_end=now,
        )

        assert report.requests_per_hour == pytest.approx(1000.0, rel=0.01)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = ModelMetrics(
            model_name="gpt-4",
            total_requests=100,
            total_cost=5.0,
        )

        report = CostReport(
            total_cost=5.0,
            total_requests=100,
            models={"gpt-4": metrics},
            period_start=0,
            period_end=3600,
        )

        result = report.to_dict()

        assert result["total_cost"] == 5.0
        assert result["total_requests"] == 100
        assert "gpt-4" in result["models"]
        assert result["models"]["gpt-4"]["requests"] == 100


class TestCostTracker:
    """Tests for CostTracker class."""

    def test_initialization(self, cost_tracker: CostTracker):
        """Test tracker initialization."""
        assert cost_tracker.models == {}
        assert cost_tracker.start_time > 0

    def test_record_request_new_model(self, cost_tracker: CostTracker):
        """Test recording request for new model."""
        cost_tracker.record_request(
            model_name="gpt-4",
            cost=0.01,
            latency=0.5,
            tokens=100,
        )

        assert "gpt-4" in cost_tracker.models
        metrics = cost_tracker.models["gpt-4"]
        assert metrics.total_requests == 1
        assert metrics.total_cost == 0.01
        assert metrics.total_tokens == 100
        assert metrics.total_latency == 0.5

    def test_record_request_existing_model(self, cost_tracker: CostTracker):
        """Test recording multiple requests for same model."""
        cost_tracker.record_request(model_name="gpt-4", cost=0.01, latency=0.5)
        cost_tracker.record_request(model_name="gpt-4", cost=0.02, latency=0.3)

        metrics = cost_tracker.models["gpt-4"]
        assert metrics.total_requests == 2
        assert metrics.total_cost == pytest.approx(0.03, rel=0.001)
        assert metrics.total_latency == 0.8

    def test_record_request_with_quality_score(self, cost_tracker: CostTracker):
        """Test recording request with quality score."""
        cost_tracker.record_request(
            model_name="gpt-4",
            cost=0.01,
            latency=0.5,
            quality_score=0.85,
        )

        metrics = cost_tracker.models["gpt-4"]
        assert metrics.quality_scores == [0.85]
        assert metrics.avg_quality == 0.85

    def test_record_request_with_error(self, cost_tracker: CostTracker):
        """Test recording request with error."""
        cost_tracker.record_request(
            model_name="gpt-4",
            cost=0.01,
            latency=0.5,
            error=True,
        )

        metrics = cost_tracker.models["gpt-4"]
        assert metrics.errors == 1
        assert metrics.error_rate == 100.0

    def test_record_multiple_models(self, cost_tracker: CostTracker):
        """Test recording requests for multiple models."""
        cost_tracker.record_request(model_name="gpt-4", cost=0.01, latency=0.5)
        cost_tracker.record_request(model_name="gpt-3.5", cost=0.001, latency=0.2)

        assert len(cost_tracker.models) == 2
        assert "gpt-4" in cost_tracker.models
        assert "gpt-3.5" in cost_tracker.models

    def test_get_model_metrics(self, cost_tracker: CostTracker):
        """Test getting metrics for specific model."""
        cost_tracker.record_request(model_name="gpt-4", cost=0.01, latency=0.5)

        metrics = cost_tracker.get_model_metrics("gpt-4")
        assert metrics is not None
        assert metrics.model_name == "gpt-4"

    def test_get_model_metrics_nonexistent(self, cost_tracker: CostTracker):
        """Test getting metrics for nonexistent model."""
        metrics = cost_tracker.get_model_metrics("nonexistent")
        assert metrics is None

    def test_get_report(self, cost_tracker: CostTracker):
        """Test generating cost report."""
        cost_tracker.record_request(model_name="gpt-4", cost=0.01, latency=0.5)
        cost_tracker.record_request(model_name="gpt-3.5", cost=0.001, latency=0.2)

        report = cost_tracker.get_report()

        assert report.total_cost == pytest.approx(0.011, rel=0.001)
        assert report.total_requests == 2
        assert len(report.models) == 2

    def test_get_report_with_reset(self, cost_tracker: CostTracker):
        """Test generating report with reset."""
        cost_tracker.record_request(model_name="gpt-4", cost=0.01, latency=0.5)

        report = cost_tracker.get_report(reset=True)

        assert report.total_requests == 1
        assert cost_tracker.models == {}

    def test_reset(self, cost_tracker: CostTracker):
        """Test resetting tracker."""
        cost_tracker.record_request(model_name="gpt-4", cost=0.01, latency=0.5)

        cost_tracker.reset()

        assert cost_tracker.models == {}
        assert cost_tracker.start_time > 0

    def test_multiple_quality_scores(self, cost_tracker: CostTracker):
        """Test tracking multiple quality scores."""
        for score in [0.8, 0.85, 0.9, 0.75]:
            cost_tracker.record_request(
                model_name="gpt-4",
                cost=0.01,
                latency=0.5,
                quality_score=score,
            )

        metrics = cost_tracker.models["gpt-4"]
        assert len(metrics.quality_scores) == 4
        assert metrics.avg_quality == pytest.approx(0.825, rel=0.01)
