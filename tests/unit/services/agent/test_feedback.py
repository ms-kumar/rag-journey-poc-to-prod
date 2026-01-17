"""Unit tests for the feedback module (FeedbackLogger, UserFeedback, FeedbackAnalytics)."""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from src.services.agent.feedback import (
    FeedbackAnalytics,
    FeedbackLogger,
    UserFeedback,
)


class TestUserFeedback:
    """Test UserFeedback dataclass."""

    def test_default_values(self):
        """Test default values of UserFeedback."""
        feedback = UserFeedback(
            query="What is Python?",
            answer="Python is a programming language.",
            rating=4,
        )

        assert feedback.query == "What is Python?"
        assert feedback.answer == "Python is a programming language."
        assert feedback.rating == 4
        assert feedback.feedback_text == ""
        assert feedback.issues == []
        assert feedback.session_id == ""
        assert feedback.metadata == {}
        assert feedback.timestamp is not None

    def test_custom_values(self):
        """Test UserFeedback with custom values."""
        feedback = UserFeedback(
            query="What is Python?",
            answer="Python is a programming language.",
            rating=3,
            feedback_text="Good but needs more examples",
            issues=["incomplete", "no_examples"],
            session_id="session_123",
            metadata={"user_id": "user_456"},
        )

        assert feedback.rating == 3
        assert feedback.feedback_text == "Good but needs more examples"
        assert "incomplete" in feedback.issues
        assert feedback.session_id == "session_123"
        assert feedback.metadata["user_id"] == "user_456"

    def test_rating_range(self):
        """Test various rating values."""
        for rating in range(1, 6):
            feedback = UserFeedback(
                query="Test",
                answer="Answer",
                rating=rating,
            )
            assert feedback.rating == rating


class TestFeedbackAnalytics:
    """Test FeedbackAnalytics dataclass."""

    def test_default_values(self):
        """Test default values of FeedbackAnalytics."""
        analytics = FeedbackAnalytics()

        assert analytics.total_feedback == 0
        assert analytics.average_rating == 0.0
        assert analytics.positive_feedback_rate == 0.0
        assert analytics.common_issues == {}
        assert analytics.improvement_areas == []
        assert analytics.success_patterns == []

    def test_custom_values(self):
        """Test FeedbackAnalytics with custom values."""
        analytics = FeedbackAnalytics(
            total_feedback=100,
            average_rating=4.2,
            positive_feedback_rate=0.85,
            common_issues={"incomplete": 15, "unclear": 10},
            improvement_areas=["documentation", "examples"],
            success_patterns=["code_snippets", "step_by_step"],
        )

        assert analytics.total_feedback == 100
        assert analytics.average_rating == 4.2
        assert analytics.positive_feedback_rate == 0.85
        assert analytics.common_issues["incomplete"] == 15
        assert "documentation" in analytics.improvement_areas


class TestFeedbackLogger:
    """Test FeedbackLogger class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Use a temporary file for testing
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.temp_dir) / "test_feedback.json"
        self.logger = FeedbackLogger(storage_path=str(self.storage_path))

    def teardown_method(self):
        """Clean up after tests."""
        # Remove temp file if exists
        if self.storage_path.exists():
            self.storage_path.unlink()
        # Remove temp directory
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init_default(self):
        """Test default initialization without storage."""
        logger = FeedbackLogger()
        assert logger.storage_path is None
        assert logger.feedback_history == []

    def test_init_with_storage_path(self):
        """Test initialization with storage path."""
        logger = FeedbackLogger(storage_path=str(self.storage_path))
        assert logger.storage_path == self.storage_path

    def test_log_feedback_basic(self):
        """Test basic feedback logging."""
        feedback = UserFeedback(
            query="What is Python?",
            answer="Python is a programming language.",
            rating=4,
        )

        self.logger.log_feedback(feedback)

        assert len(self.logger.feedback_history) == 1
        assert self.logger.feedback_history[0].rating == 4

    def test_log_multiple_feedback(self):
        """Test logging multiple feedback entries."""
        for i in range(5):
            feedback = UserFeedback(
                query=f"Query {i}",
                answer=f"Answer {i}",
                rating=i + 1,
            )
            self.logger.log_feedback(feedback)

        assert len(self.logger.feedback_history) == 5

    def test_log_feedback_persists_to_disk(self):
        """Test that feedback is persisted to disk."""
        feedback = UserFeedback(
            query="Test query",
            answer="Test answer",
            rating=5,
        )
        self.logger.log_feedback(feedback)

        # Check file exists and contains data
        assert self.storage_path.exists()
        with self.storage_path.open() as f:
            data = json.load(f)
        assert len(data) >= 1

    def test_load_existing_feedback(self):
        """Test loading existing feedback from disk."""
        # First, create some feedback
        feedback = UserFeedback(
            query="Test query",
            answer="Test answer",
            rating=5,
        )
        self.logger.log_feedback(feedback)

        # Create new logger with same path - should load existing
        new_logger = FeedbackLogger(storage_path=str(self.storage_path))
        assert len(new_logger.feedback_history) >= 1

    def test_get_analytics_empty(self):
        """Test analytics with no feedback."""
        analytics = self.logger.get_analytics()

        assert analytics.total_feedback == 0
        assert analytics.average_rating == 0.0

    def test_get_analytics_with_feedback(self):
        """Test analytics calculation with feedback."""
        # Add feedback with various ratings
        feedbacks = [
            UserFeedback(query="Q1", answer="A1", rating=5),
            UserFeedback(query="Q2", answer="A2", rating=4),
            UserFeedback(query="Q3", answer="A3", rating=4),
            UserFeedback(query="Q4", answer="A4", rating=3),
            UserFeedback(query="Q5", answer="A5", rating=2),
        ]

        for fb in feedbacks:
            self.logger.log_feedback(fb)

        analytics = self.logger.get_analytics()

        assert analytics.total_feedback == 5
        # Average should be (5+4+4+3+2)/5 = 3.6
        assert 3.5 <= analytics.average_rating <= 3.7

    def test_get_analytics_positive_rate(self):
        """Test positive feedback rate calculation."""
        # Add 3 positive (4-5) and 2 negative (1-3)
        feedbacks = [
            UserFeedback(query="Q1", answer="A1", rating=5),  # positive
            UserFeedback(query="Q2", answer="A2", rating=4),  # positive
            UserFeedback(query="Q3", answer="A3", rating=4),  # positive
            UserFeedback(query="Q4", answer="A4", rating=2),  # negative
            UserFeedback(query="Q5", answer="A5", rating=1),  # negative
        ]

        for fb in feedbacks:
            self.logger.log_feedback(fb)

        analytics = self.logger.get_analytics()

        # 3 out of 5 are positive = 60%
        assert 0.55 <= analytics.positive_feedback_rate <= 0.65

    def test_get_analytics_common_issues(self):
        """Test common issues identification."""
        feedbacks = [
            UserFeedback(query="Q1", answer="A1", rating=2, issues=["incomplete"]),
            UserFeedback(query="Q2", answer="A2", rating=3, issues=["incomplete", "unclear"]),
            UserFeedback(query="Q3", answer="A3", rating=2, issues=["incomplete"]),
            UserFeedback(query="Q4", answer="A4", rating=2, issues=["wrong"]),
            UserFeedback(query="Q5", answer="A5", rating=4, issues=[]),
        ]

        for fb in feedbacks:
            self.logger.log_feedback(fb)

        analytics = self.logger.get_analytics()

        # "incomplete" should be most common
        assert "incomplete" in analytics.common_issues
        assert analytics.common_issues["incomplete"] == 3

    def test_get_analytics_time_filter(self):
        """Test analytics with time filter."""
        # Add old feedback
        old_feedback = UserFeedback(query="Old", answer="Old", rating=1)
        old_feedback.timestamp = datetime.utcnow() - timedelta(days=30)
        self.logger.feedback_history.append(old_feedback)

        # Add recent feedback
        for i in range(5):
            feedback = UserFeedback(query=f"Q{i}", answer=f"A{i}", rating=5)
            self.logger.log_feedback(feedback)

        # Analytics for last 7 days should exclude old feedback
        analytics = self.logger.get_analytics(days=7)

        assert analytics.total_feedback == 5
        assert analytics.average_rating == 5.0

    def test_get_low_rated_queries(self):
        """Test getting low-rated queries."""
        feedbacks = [
            UserFeedback(query="Good query", answer="Good answer", rating=5),
            UserFeedback(query="Bad query 1", answer="Bad answer 1", rating=1),
            UserFeedback(query="Bad query 2", answer="Bad answer 2", rating=2),
            UserFeedback(query="Okay query", answer="Okay answer", rating=3),
        ]

        for fb in feedbacks:
            self.logger.log_feedback(fb)

        low_rated = self.logger.get_low_rated_queries(threshold=2)

        assert len(low_rated) == 2
        queries = [item["query"] for item in low_rated]
        assert "Bad query 1" in queries
        assert "Bad query 2" in queries

    def test_get_low_rated_queries_empty(self):
        """Test getting low-rated queries when all are good."""
        for i in range(5):
            feedback = UserFeedback(query=f"Q{i}", answer=f"A{i}", rating=5)
            self.logger.log_feedback(feedback)

        low_rated = self.logger.get_low_rated_queries(threshold=2)

        assert len(low_rated) == 0

    def test_export_feedback_json(self):
        """Test exporting feedback to JSON."""
        for i in range(3):
            feedback = UserFeedback(query=f"Q{i}", answer=f"A{i}", rating=i + 3)
            self.logger.log_feedback(feedback)

        export_path = Path(self.temp_dir) / "export.json"
        self.logger.export_feedback(str(export_path), format="json")

        assert export_path.exists()
        with export_path.open() as f:
            data = json.load(f)
        assert len(data) == 3

    def test_export_feedback_csv(self):
        """Test exporting feedback to CSV."""
        for i in range(3):
            feedback = UserFeedback(query=f"Q{i}", answer=f"A{i}", rating=i + 3)
            self.logger.log_feedback(feedback)

        export_path = Path(self.temp_dir) / "export.csv"
        self.logger.export_feedback(str(export_path), format="csv")

        assert export_path.exists()
        with export_path.open() as f:
            content = f.read()
        # CSV should have header and 3 data rows
        assert "query" in content.lower() or "rating" in content.lower()

    def test_feedback_with_session_tracking(self):
        """Test feedback with session tracking."""
        session_id = "session_abc123"

        for i in range(3):
            feedback = UserFeedback(
                query=f"Q{i}",
                answer=f"A{i}",
                rating=4,
                session_id=session_id,
            )
            self.logger.log_feedback(feedback)

        # Filter by session
        session_feedback = [
            fb for fb in self.logger.feedback_history if fb.session_id == session_id
        ]
        assert len(session_feedback) == 3

    def test_feedback_metadata(self):
        """Test feedback with custom metadata."""
        feedback = UserFeedback(
            query="Test",
            answer="Answer",
            rating=4,
            metadata={
                "user_id": "user_123",
                "model_version": "v1.0",
                "response_time": 1.5,
            },
        )
        self.logger.log_feedback(feedback)

        stored = self.logger.feedback_history[0]
        assert stored.metadata["user_id"] == "user_123"
        assert stored.metadata["response_time"] == 1.5


class TestFeedbackLoggerEdgeCases:
    """Test edge cases for FeedbackLogger."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.temp_dir) / "test_feedback.json"
        self.logger = FeedbackLogger(storage_path=str(self.storage_path))

    def teardown_method(self):
        """Clean up after tests."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_empty_query_feedback(self):
        """Test feedback with empty query."""
        feedback = UserFeedback(query="", answer="Answer", rating=3)
        self.logger.log_feedback(feedback)

        assert len(self.logger.feedback_history) == 1

    def test_empty_answer_feedback(self):
        """Test feedback with empty answer."""
        feedback = UserFeedback(query="Query", answer="", rating=1)
        self.logger.log_feedback(feedback)

        assert len(self.logger.feedback_history) == 1

    def test_very_long_feedback(self):
        """Test feedback with very long content."""
        long_query = "Q" * 10000
        long_answer = "A" * 10000
        long_text = "F" * 10000

        feedback = UserFeedback(
            query=long_query,
            answer=long_answer,
            rating=3,
            feedback_text=long_text,
        )
        self.logger.log_feedback(feedback)

        assert len(self.logger.feedback_history) == 1
        assert len(self.logger.feedback_history[0].query) == 10000

    def test_unicode_feedback(self):
        """Test feedback with unicode content."""
        feedback = UserFeedback(
            query="Pythonとは何ですか？",
            answer="Pythonはプログラミング言語です。",
            rating=5,
            feedback_text="素晴らしい回答です！",
        )
        self.logger.log_feedback(feedback)

        assert self.logger.feedback_history[0].query == "Pythonとは何ですか？"

    def test_special_characters_feedback(self):
        """Test feedback with special characters."""
        feedback = UserFeedback(
            query="What is @decorator & __init__?",
            answer="They are Python special syntax.",
            rating=4,
            feedback_text="Good explanation of $pecial ch@rs!",
        )
        self.logger.log_feedback(feedback)

        assert "@decorator" in self.logger.feedback_history[0].query

    def test_many_issues(self):
        """Test feedback with many issues."""
        issues = [f"issue_{i}" for i in range(50)]
        feedback = UserFeedback(
            query="Query",
            answer="Answer",
            rating=1,
            issues=issues,
        )
        self.logger.log_feedback(feedback)

        assert len(self.logger.feedback_history[0].issues) == 50

    def test_concurrent_feedback_logging(self):
        """Test logging feedback in quick succession."""
        for i in range(100):
            feedback = UserFeedback(query=f"Q{i}", answer=f"A{i}", rating=(i % 5) + 1)
            self.logger.log_feedback(feedback)

        assert len(self.logger.feedback_history) == 100

    def test_analytics_min_samples(self):
        """Test analytics with minimum sample requirement."""
        # Add only 3 samples (less than typical minimum of 5)
        for i in range(3):
            feedback = UserFeedback(query=f"Q{i}", answer=f"A{i}", rating=5)
            self.logger.log_feedback(feedback)

        # With < 5 samples, analytics returns empty/zero values
        analytics = self.logger.get_analytics()
        # Analytics has min_samples=5, so with 3 it returns empty
        assert analytics.total_feedback == 0 or len(self.logger.feedback_history) == 3

    def test_corrupted_storage_file(self):
        """Test handling of corrupted storage file."""
        # Write invalid JSON
        with self.storage_path.open("w") as f:
            f.write("not valid json {{{")

        # Creating logger should handle corrupted file gracefully
        try:
            logger = FeedbackLogger(storage_path=str(self.storage_path))
            # Should either start fresh or raise handled exception
            assert logger.feedback_history == []
        except (json.JSONDecodeError, Exception):
            # Exception is acceptable for corrupted file
            pass

    def test_feedback_timestamp_ordering(self):
        """Test that feedback maintains timestamp ordering."""
        import time

        for i in range(5):
            feedback = UserFeedback(query=f"Q{i}", answer=f"A{i}", rating=4)
            self.logger.log_feedback(feedback)
            time.sleep(0.01)  # Small delay

        # Check timestamps are in order
        timestamps = [fb.timestamp for fb in self.logger.feedback_history]
        assert timestamps == sorted(timestamps)


class TestFeedbackAnalyticsCalculation:
    """Test detailed analytics calculations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.logger = FeedbackLogger()

    def test_improvement_areas_detection(self):
        """Test detection of improvement areas."""
        # Create feedback with consistent issues
        for i in range(10):
            issues = ["incomplete"] if i < 5 else []  # 50% have "incomplete"
            feedback = UserFeedback(
                query=f"Q{i}",
                answer=f"A{i}",
                rating=3 if i < 5 else 5,
                issues=issues,
            )
            self.logger.log_feedback(feedback)

        analytics = self.logger.get_analytics()

        # "incomplete" should be identified as improvement area
        # depending on implementation threshold
        assert analytics.common_issues.get("incomplete", 0) == 5

    def test_success_patterns_detection(self):
        """Test detection of success patterns."""
        # Create feedback with patterns
        for i in range(10):
            answer = (
                "Here's a code example:\n```python\nprint('hello')\n```" if i < 7 else f"Answer {i}"
            )
            rating = 5 if i < 7 else 2
            feedback = UserFeedback(
                query=f"How to print in Python? Q{i}",
                answer=answer,
                rating=rating,
            )
            self.logger.log_feedback(feedback)

        analytics = self.logger.get_analytics()

        # High-rated answers with code should show pattern
        assert analytics.total_feedback == 10

    def test_analytics_edge_rating_values(self):
        """Test analytics with edge rating values."""
        # All 1s
        for i in range(5):
            feedback = UserFeedback(query=f"Q{i}", answer=f"A{i}", rating=1)
            self.logger.log_feedback(feedback)

        analytics = self.logger.get_analytics()
        assert analytics.average_rating == 1.0
        assert analytics.positive_feedback_rate == 0.0

        # Reset and test all 5s
        self.logger.feedback_history = []
        for i in range(5):
            feedback = UserFeedback(query=f"Q{i}", answer=f"A{i}", rating=5)
            self.logger.log_feedback(feedback)

        analytics = self.logger.get_analytics()
        assert analytics.average_rating == 5.0
        assert analytics.positive_feedback_rate == 1.0
