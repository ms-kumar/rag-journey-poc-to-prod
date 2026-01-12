"""Unit tests for agent metrics tracking."""

import tempfile
from pathlib import Path

import pytest

from src.services.agent.metrics.confidence import ConfidenceScorer
from src.services.agent.metrics.tracker import MetricsTracker, ToolMetrics


class TestToolMetrics:
    """Test ToolMetrics dataclass."""

    def test_metrics_creation(self):
        """Test creating tool metrics."""
        metrics = ToolMetrics(
            tool_name="test_tool",
            total_invocations=10,
            successful_invocations=8,
            failed_invocations=2,
        )

        assert metrics.tool_name == "test_tool"
        assert metrics.total_invocations == 10
        assert metrics.successful_invocations == 8

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        metrics = ToolMetrics(
            tool_name="test_tool",
            total_invocations=100,
            successful_invocations=85,
            failed_invocations=15,
        )

        assert metrics.success_rate == 0.85

    def test_success_rate_no_invocations(self):
        """Test success rate with no invocations."""
        metrics = ToolMetrics(tool_name="test_tool")
        assert metrics.success_rate == 0.0

    def test_avg_latency_calculation(self):
        """Test average latency calculation."""
        metrics = ToolMetrics(
            tool_name="test_tool",
            total_invocations=10,
            total_latency_ms=1500.0,
        )

        assert metrics.avg_latency_ms == 150.0

    def test_avg_confidence_calculation(self):
        """Test average confidence calculation."""
        metrics = ToolMetrics(
            tool_name="test_tool",
            confidence_scores=[0.8, 0.9, 0.7, 0.85],
        )

        assert metrics.avg_confidence == pytest.approx(0.8125)

    def test_avg_confidence_no_scores(self):
        """Test average confidence with no scores."""
        metrics = ToolMetrics(tool_name="test_tool")
        assert metrics.avg_confidence == 0.0


class TestMetricsTracker:
    """Test MetricsTracker class."""

    def test_tracker_initialization(self):
        """Test tracker initialization."""
        tracker = MetricsTracker()
        assert len(tracker.get_all_metrics()) == 0

    def test_tracker_with_storage_path(self):
        """Test tracker with storage path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "metrics.json"
            tracker = MetricsTracker(storage_path=str(storage_path))

            assert tracker.storage_path == storage_path

    def test_track_invocation_success(self):
        """Test tracking a successful invocation."""
        tracker = MetricsTracker()

        tracker.track_invocation(
            tool_name="test_tool",
            success=True,
            latency_ms=100.0,
            cost=0.001,
            confidence=0.85,
        )

        metrics = tracker.get_tool_metrics("test_tool")
        assert metrics is not None
        assert metrics.total_invocations == 1
        assert metrics.successful_invocations == 1
        assert metrics.failed_invocations == 0

    def test_track_invocation_failure(self):
        """Test tracking a failed invocation."""
        tracker = MetricsTracker()

        tracker.track_invocation(
            tool_name="test_tool",
            success=False,
            latency_ms=100.0,
            error="Test error",
        )

        metrics = tracker.get_tool_metrics("test_tool")
        assert metrics.failed_invocations == 1
        assert "Test error" in metrics.error_types or "unknown" in metrics.error_types

    def test_track_multiple_invocations(self):
        """Test tracking multiple invocations."""
        tracker = MetricsTracker()

        for i in range(10):
            tracker.track_invocation(
                tool_name="test_tool",
                success=i % 2 == 0,  # 50% success rate
                latency_ms=100.0 + i,
                confidence=0.8,
            )

        metrics = tracker.get_tool_metrics("test_tool")
        assert metrics.total_invocations == 10
        assert metrics.successful_invocations == 5
        assert metrics.success_rate == 0.5

    def test_get_tool_metrics_nonexistent(self):
        """Test getting metrics for nonexistent tool."""
        tracker = MetricsTracker()
        metrics = tracker.get_tool_metrics("nonexistent")
        assert metrics is None

    def test_get_all_metrics(self):
        """Test getting all metrics."""
        tracker = MetricsTracker()

        # Track invocations for multiple tools
        for tool_name in ["tool1", "tool2", "tool3"]:
            tracker.track_invocation(
                tool_name=tool_name,
                success=True,
                latency_ms=100.0,
            )

        all_metrics = tracker.get_all_metrics()
        assert len(all_metrics) == 3
        assert "tool1" in all_metrics
        assert "tool2" in all_metrics
        assert "tool3" in all_metrics

    def test_get_summary(self):
        """Test getting metrics summary."""
        tracker = MetricsTracker()

        tracker.track_invocation(
            tool_name="test_tool",
            success=True,
            latency_ms=150.0,
            confidence=0.85,
        )

        summary = tracker.get_summary()
        assert "total_tools" in summary
        assert "tools" in summary
        assert "test_tool" in summary["tools"]

        tool_summary = summary["tools"]["test_tool"]
        assert tool_summary["invocations"] == 1
        assert tool_summary["success_rate"] == 1.0

    def test_reset_metrics_single_tool(self):
        """Test resetting metrics for a single tool."""
        tracker = MetricsTracker()

        tracker.track_invocation("tool1", True, 100.0)
        tracker.track_invocation("tool2", True, 100.0)

        tracker.reset_metrics("tool1")

        assert tracker.get_tool_metrics("tool1").total_invocations == 0
        assert tracker.get_tool_metrics("tool2").total_invocations == 1

    def test_reset_metrics_all(self):
        """Test resetting all metrics."""
        tracker = MetricsTracker()

        tracker.track_invocation("tool1", True, 100.0)
        tracker.track_invocation("tool2", True, 100.0)

        tracker.reset_metrics()

        assert len(tracker.get_all_metrics()) == 0

    def test_save_and_load_metrics(self):
        """Test saving and loading metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "metrics.json"

            # Create tracker and add metrics
            tracker1 = MetricsTracker(storage_path=str(storage_path))
            tracker1.track_invocation("test_tool", True, 100.0, confidence=0.85)
            tracker1._save_metrics()

            # Load metrics in new tracker
            tracker2 = MetricsTracker(storage_path=str(storage_path))
            metrics = tracker2.get_tool_metrics("test_tool")

            assert metrics is not None
            assert metrics.total_invocations == 1

    def test_confidence_scores_limit(self):
        """Test that confidence scores are limited to last 100."""
        tracker = MetricsTracker()

        # Track 150 invocations
        for _i in range(150):
            tracker.track_invocation(
                tool_name="test_tool",
                success=True,
                latency_ms=100.0,
                confidence=0.8,
            )

        metrics = tracker.get_tool_metrics("test_tool")
        assert len(metrics.confidence_scores) == 100


class TestConfidenceScorer:
    """Test ConfidenceScorer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.scorer = ConfidenceScorer()

    def test_score_tool_selection_basic(self):
        """Test basic tool selection scoring."""
        score = self.scorer.score_tool_selection(
            query="search for documents",
            tool_name="vectordb",
            tool_description="Retrieve documents from database",
            capabilities=["search", "retrieval"],
            success_rate=0.95,
        )

        assert 0.0 <= score <= 1.0

    def test_score_tool_selection_semantic_match(self):
        """Test scoring with semantic match."""
        score = self.scorer.score_tool_selection(
            query="search documents database",
            tool_name="vectordb",
            tool_description="search documents in database",
            capabilities=[],
            success_rate=0.5,
        )

        # Should have high score due to word overlap
        assert score > 0.3

    def test_score_tool_selection_success_rate(self):
        """Test scoring with high success rate."""
        score_high = self.scorer.score_tool_selection(
            query="test query",
            tool_name="tool",
            tool_description="test tool",
            capabilities=[],
            success_rate=0.95,
        )

        score_low = self.scorer.score_tool_selection(
            query="test query",
            tool_name="tool",
            tool_description="test tool",
            capabilities=[],
            success_rate=0.50,
        )

        assert score_high > score_low

    def test_score_tool_selection_capabilities(self):
        """Test scoring with capability match."""
        score = self.scorer.score_tool_selection(
            query="search for documents",
            tool_name="tool",
            tool_description="tool",
            capabilities=["search", "documents", "retrieval"],
            success_rate=0.5,
        )

        # Should have contribution from capability match
        assert score > 0.2

    def test_score_tool_selection_cost_latency(self):
        """Test scoring considers cost and latency."""
        score_cheap = self.scorer.score_tool_selection(
            query="test",
            tool_name="tool",
            tool_description="test",
            capabilities=[],
            success_rate=0.5,
            cost=0.0,
            latency_ms=100.0,
        )

        score_expensive = self.scorer.score_tool_selection(
            query="test",
            tool_name="tool",
            tool_description="test",
            capabilities=[],
            success_rate=0.5,
            cost=1.0,
            latency_ms=5000.0,
        )

        assert score_cheap > score_expensive

    def test_score_result_quality_success(self):
        """Test result quality scoring for successful result."""
        result = {
            "success": True,
            "result": {"data": "test"},
            "error": None,
        }

        score = self.scorer.score_result_quality(result)
        assert score >= 0.5

    def test_score_result_quality_failure(self):
        """Test result quality scoring for failed result."""
        result = {
            "success": False,
            "result": None,
            "error": "Test error",
        }

        score = self.scorer.score_result_quality(result)
        assert score == 0.0

    def test_score_result_quality_with_expected_fields(self):
        """Test result quality with expected fields."""
        result = {
            "success": True,
            "result": {
                "field1": "value1",
                "field2": "value2",
            },
        }

        score = self.scorer.score_result_quality(
            result,
            expected_fields=["field1", "field2"],
        )

        assert score > 0.8

    def test_score_result_quality_missing_fields(self):
        """Test result quality with missing expected fields."""
        result = {
            "success": True,
            "result": {
                "field1": "value1",
            },
        }

        score = self.scorer.score_result_quality(
            result,
            expected_fields=["field1", "field2", "field3"],
        )

        # Should have lower score due to missing fields
        assert score < 1.0
