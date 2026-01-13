"""Tests for cost tracking functionality."""

import pytest

from src.services.cost.tracker import CostTracker, ModelMetrics


class TestModelMetrics:
    """Test ModelMetrics class."""

    def test_default_metrics(self):
        """Test default metric values."""
        metrics = ModelMetrics(model_name="test-model")

        assert metrics.model_name == "test-model"
        assert metrics.total_requests == 0
        assert metrics.total_cost == 0.0
        assert metrics.avg_latency == 0.0
        assert metrics.cost_per_1k == 0.0
        assert metrics.avg_quality == 0.0
        assert metrics.error_rate == 0.0

    def test_avg_latency_calculation(self):
        """Test average latency calculation."""
        metrics = ModelMetrics(model_name="test")
        metrics.total_requests = 10
        metrics.total_latency = 1.0  # 1 second total

        # Should be in milliseconds
        assert metrics.avg_latency == 100.0

    def test_cost_per_1k_calculation(self):
        """Test cost per 1k queries calculation."""
        metrics = ModelMetrics(model_name="test")
        metrics.total_requests = 100
        metrics.total_cost = 1.0  # $1 for 100 requests

        assert metrics.cost_per_1k == 10.0  # $10 per 1k

    def test_avg_quality_calculation(self):
        """Test average quality calculation."""
        metrics = ModelMetrics(model_name="test")
        metrics.quality_scores = [0.8, 0.9, 0.85]

        assert metrics.avg_quality == pytest.approx(0.85)

    def test_error_rate_calculation(self):
        """Test error rate calculation."""
        metrics = ModelMetrics(model_name="test")
        metrics.total_requests = 100
        metrics.errors = 5

        assert metrics.error_rate == 5.0


class TestCostTracker:
    """Test CostTracker class."""

    def test_initialization(self):
        """Test tracker initialization."""
        tracker = CostTracker()

        assert len(tracker.models) == 0
        assert tracker.start_time > 0

    def test_record_request(self):
        """Test recording a request."""
        tracker = CostTracker()

        tracker.record_request(
            model_name="gpt2",
            cost=0.001,
            latency=0.1,
            tokens=100,
            quality_score=0.8,
        )

        assert "gpt2" in tracker.models
        metrics = tracker.models["gpt2"]
        assert metrics.total_requests == 1
        assert metrics.total_cost == 0.001
        assert metrics.total_tokens == 100
        assert metrics.quality_scores == [0.8]

    def test_record_multiple_requests(self):
        """Test recording multiple requests."""
        tracker = CostTracker()

        for _ in range(5):
            tracker.record_request(
                model_name="gpt2",
                cost=0.001,
                latency=0.1,
                tokens=100,
            )

        metrics = tracker.models["gpt2"]
        assert metrics.total_requests == 5
        assert metrics.total_cost == 0.005
        assert metrics.total_tokens == 500

    def test_record_error(self):
        """Test recording errors."""
        tracker = CostTracker()

        tracker.record_request(
            model_name="gpt2",
            cost=0.001,
            latency=0.1,
            error=True,
        )

        metrics = tracker.models["gpt2"]
        assert metrics.errors == 1
        assert metrics.error_rate == 100.0

    def test_multiple_models(self):
        """Test tracking multiple models."""
        tracker = CostTracker()

        tracker.record_request("gpt2", 0.001, 0.1)
        tracker.record_request("gpt-3.5", 0.01, 0.5)
        tracker.record_request("gpt2", 0.001, 0.1)

        assert len(tracker.models) == 2
        assert tracker.models["gpt2"].total_requests == 2
        assert tracker.models["gpt-3.5"].total_requests == 1

    def test_get_report(self):
        """Test generating cost report."""
        tracker = CostTracker()

        tracker.record_request("gpt2", 0.001, 0.1)
        tracker.record_request("gpt2", 0.001, 0.1)

        report = tracker.get_report()

        assert report.total_cost == 0.002
        assert report.total_requests == 2
        assert "gpt2" in report.models
        assert report.cost_per_1k == 1.0

    def test_get_report_with_reset(self):
        """Test report generation with reset."""
        tracker = CostTracker()

        tracker.record_request("gpt2", 0.001, 0.1)
        report = tracker.get_report(reset=True)

        assert report.total_requests == 1
        assert len(tracker.models) == 0  # Should be reset

    def test_get_top_models_by_cost(self):
        """Test getting top models by cost."""
        tracker = CostTracker()

        tracker.record_request("cheap", 0.001, 0.1)
        tracker.record_request("expensive", 1.0, 0.5)
        tracker.record_request("medium", 0.1, 0.2)

        top_models = tracker.get_top_models_by_cost(top_n=2)

        assert len(top_models) == 2
        assert top_models[0][0] == "expensive"
        assert top_models[1][0] == "medium"

    def test_get_models_by_efficiency(self):
        """Test getting models by efficiency."""
        tracker = CostTracker()

        # High quality, low cost = high efficiency
        tracker.record_request("efficient", 0.001, 0.1, quality_score=0.9)

        # Low quality, high cost = low efficiency
        tracker.record_request("inefficient", 1.0, 0.5, quality_score=0.5)

        efficiency_rankings = tracker.get_models_by_efficiency()

        assert len(efficiency_rankings) == 2
        assert efficiency_rankings[0][0] == "efficient"
        assert efficiency_rankings[1][0] == "inefficient"

    def test_reset(self):
        """Test resetting tracker."""
        tracker = CostTracker()

        tracker.record_request("gpt2", 0.001, 0.1)
        tracker.reset()

        assert len(tracker.models) == 0
        assert tracker.start_time > 0

    def test_report_to_dict(self):
        """Test converting report to dictionary."""
        tracker = CostTracker()

        tracker.record_request("gpt2", 0.001, 0.1, tokens=100, quality_score=0.8)
        report = tracker.get_report()

        data = report.to_dict()

        assert "total_cost" in data
        assert "total_requests" in data
        assert "cost_per_1k" in data
        assert "models" in data
        assert "gpt2" in data["models"]

        model_data = data["models"]["gpt2"]
        assert "requests" in model_data
        assert "cost" in model_data
        assert "avg_quality" in model_data
