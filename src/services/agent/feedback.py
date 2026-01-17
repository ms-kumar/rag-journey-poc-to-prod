"""User feedback collection and learning system."""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class UserFeedback:
    """User feedback on an answer.

    Attributes:
        query: Original query
        answer: Answer provided
        rating: User rating (1-5 stars)
        feedback_text: Optional textual feedback
        issues: List of specific issues identified
        timestamp: When feedback was provided
        session_id: Session identifier
        metadata: Additional metadata
    """

    query: str
    answer: str
    rating: int
    feedback_text: str = ""
    issues: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    session_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FeedbackAnalytics:
    """Analytics derived from user feedback.

    Attributes:
        total_feedback: Total feedback count
        average_rating: Average rating
        positive_feedback_rate: Percentage of 4-5 star ratings
        common_issues: Most common issues reported
        improvement_areas: Areas needing improvement
        success_patterns: Patterns in successful answers
    """

    total_feedback: int = 0
    average_rating: float = 0.0
    positive_feedback_rate: float = 0.0
    common_issues: dict[str, int] = field(default_factory=dict)
    improvement_areas: list[str] = field(default_factory=list)
    success_patterns: list[str] = field(default_factory=list)


class FeedbackLogger:
    """Logs and analyzes user feedback."""

    def __init__(self, storage_path: str | None = None):
        """Initialize feedback logger.

        Args:
            storage_path: Path to store feedback (JSON file)
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self.feedback_history: list[UserFeedback] = []
        self.logger = logging.getLogger(__name__)

        # Load existing feedback if available
        if self.storage_path and self.storage_path.exists():
            self._load_feedback()

    def log_feedback(self, feedback: UserFeedback) -> None:
        """Log user feedback.

        Args:
            feedback: UserFeedback object
        """
        self.logger.info(
            f"Logging feedback: rating={feedback.rating}, query='{feedback.query[:50]}...'"
        )

        self.feedback_history.append(feedback)

        # Save to disk
        if self.storage_path:
            self._save_feedback()

        # Log to system logger
        level = logging.INFO if feedback.rating >= 3 else logging.WARNING
        self.logger.log(
            level,
            f"User feedback: {feedback.rating}/5 - {feedback.feedback_text[:100]}",
        )

    def get_analytics(self, days: int | None = None, min_samples: int = 5) -> FeedbackAnalytics:
        """Get feedback analytics.

        Args:
            days: Optional filter for recent N days
            min_samples: Minimum samples needed for analytics

        Returns:
            FeedbackAnalytics object
        """
        self.logger.info("Generating feedback analytics...")

        # Filter by date if specified
        feedback = self.feedback_history
        if days:
            cutoff = datetime.utcnow().timestamp() - (days * 86400)
            feedback = [f for f in feedback if f.timestamp.timestamp() > cutoff]

        if len(feedback) < min_samples:
            self.logger.warning(f"Insufficient feedback samples: {len(feedback)} < {min_samples}")
            return FeedbackAnalytics()

        analytics = FeedbackAnalytics()
        analytics.total_feedback = len(feedback)

        # Calculate average rating
        ratings = [f.rating for f in feedback]
        analytics.average_rating = sum(ratings) / len(ratings)

        # Calculate positive feedback rate
        positive = sum(1 for r in ratings if r >= 4)
        analytics.positive_feedback_rate = positive / len(ratings)

        # Identify common issues
        issue_counts: dict[str, int] = {}
        for f in feedback:
            for issue in f.issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
        analytics.common_issues = dict(
            sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        )

        # Identify improvement areas
        analytics.improvement_areas = self._identify_improvement_areas(feedback)

        # Identify success patterns
        analytics.success_patterns = self._identify_success_patterns(feedback)

        self.logger.info(
            f"Analytics: avg_rating={analytics.average_rating:.2f}, "
            f"positive_rate={analytics.positive_feedback_rate:.1%}"
        )

        return analytics

    def get_recent_feedback(self, limit: int = 10) -> list[UserFeedback]:
        """Get recent feedback entries.

        Args:
            limit: Maximum number of entries

        Returns:
            List of recent UserFeedback
        """
        return self.feedback_history[-limit:]

    def get_low_rated_queries(self, threshold: int = 2, limit: int = 20) -> list[dict[str, Any]]:
        """Get queries with low ratings for improvement.

        Args:
            threshold: Rating threshold (inclusive)
            limit: Maximum number to return

        Returns:
            List of query-feedback pairs
        """
        low_rated = [
            {"query": f.query, "answer": f.answer, "rating": f.rating, "feedback": f.feedback_text}
            for f in self.feedback_history
            if f.rating <= threshold
        ]

        def get_rating(item: dict[str, Any]) -> int:
            """Extract rating for sorting."""
            rating = item.get("rating", 0)
            return rating if isinstance(rating, int) else 0

        return sorted(low_rated, key=get_rating)[:limit]

    def export_feedback(self, export_path: str, format: str = "json") -> None:
        """Export feedback to file.

        Args:
            export_path: Path to export file
            format: Export format (json or csv)
        """
        self.logger.info(f"Exporting {len(self.feedback_history)} feedback entries...")

        export_file = Path(export_path)
        export_file.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            data = [asdict(f) for f in self.feedback_history]
            # Convert datetime to string
            for entry in data:
                entry["timestamp"] = entry["timestamp"].isoformat()

            with export_file.open("w") as file:
                json.dump(data, file, indent=2)
        elif format == "csv":
            import csv

            with export_file.open("w", newline="") as file:
                if not self.feedback_history:
                    return

                fieldnames = [
                    "timestamp",
                    "query",
                    "rating",
                    "feedback_text",
                    "session_id",
                ]
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()

                for f in self.feedback_history:
                    writer.writerow(
                        {
                            "timestamp": f.timestamp.isoformat(),
                            "query": f.query,
                            "rating": f.rating,
                            "feedback_text": f.feedback_text,
                            "session_id": f.session_id,
                        }
                    )

        self.logger.info(f"Feedback exported to {export_path}")

    def _save_feedback(self) -> None:
        """Save feedback to disk."""
        if not self.storage_path:
            return

        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)

            data = [asdict(f) for f in self.feedback_history]
            # Convert datetime to string
            for entry in data:
                entry["timestamp"] = entry["timestamp"].isoformat()

            with self.storage_path.open("w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to save feedback: {e}")

    def _load_feedback(self) -> None:
        """Load feedback from disk."""
        if not self.storage_path or not self.storage_path.exists():
            return

        try:
            with self.storage_path.open() as f:
                data = json.load(f)

            for entry in data:
                # Convert timestamp string back to datetime
                entry["timestamp"] = datetime.fromisoformat(entry["timestamp"])
                self.feedback_history.append(UserFeedback(**entry))

            self.logger.info(f"Loaded {len(self.feedback_history)} feedback entries")

        except Exception as e:
            self.logger.error(f"Failed to load feedback: {e}")

    def _identify_improvement_areas(self, feedback: list[UserFeedback]) -> list[str]:
        """Identify areas needing improvement.

        Args:
            feedback: List of feedback

        Returns:
            List of improvement areas
        """
        areas = []

        # Analyze low-rated feedback
        low_rated = [f for f in feedback if f.rating <= 2]

        if len(low_rated) > len(feedback) * 0.3:
            areas.append("Overall answer quality needs improvement")

        # Check common issue patterns
        issue_counts: dict[str, int] = {}
        for f in low_rated:
            for issue in f.issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1

        for issue, count in issue_counts.items():
            if count > len(low_rated) * 0.3:
                areas.append(f"Frequent issue: {issue}")

        # Check feedback text for patterns
        feedback_texts = [f.feedback_text.lower() for f in low_rated if f.feedback_text]

        problem_keywords = {
            "incomplete": "Completeness of answers",
            "wrong": "Answer accuracy",
            "unclear": "Answer clarity",
            "slow": "Response time",
            "irrelevant": "Answer relevance",
        }

        for keyword, area in problem_keywords.items():
            if sum(1 for text in feedback_texts if keyword in text) > len(feedback_texts) * 0.2:
                areas.append(area)

        return areas[:5]  # Top 5 areas

    def _identify_success_patterns(self, feedback: list[UserFeedback]) -> list[str]:
        """Identify patterns in successful answers.

        Args:
            feedback: List of feedback

        Returns:
            List of success patterns
        """
        patterns: list[str] = []

        if not feedback:
            return patterns

        # Check answer characteristics
        avg_length = sum(len(f.answer) for f in feedback) / len(feedback)
        if avg_length > 500:
            patterns.append("Detailed answers (500+ chars) are well-received")

        # Check for structured content
        structured_count = sum(
            1
            for f in feedback
            if "\n\n" in f.answer or any(marker in f.answer for marker in ["- ", "1.", "2."])
        )
        if structured_count > len(feedback) * 0.6:
            patterns.append("Well-structured answers with lists/paragraphs perform better")

        # Check feedback text for positive patterns
        positive_feedback = [f.feedback_text.lower() for f in feedback if f.feedback_text]

        success_keywords = {
            "comprehensive": "Comprehensive coverage",
            "clear": "Clear explanations",
            "helpful": "Practical and helpful",
            "detailed": "Detailed information",
            "accurate": "High accuracy",
        }

        for keyword, pattern in success_keywords.items():
            if (
                sum(1 for text in positive_feedback if keyword in text)
                > len(positive_feedback) * 0.2
            ):
                patterns.append(pattern)

        return patterns[:5]  # Top 5 patterns
