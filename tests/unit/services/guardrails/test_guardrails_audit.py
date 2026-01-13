"""
Unit tests for audit logging.
"""

import json
from datetime import datetime

import pytest

from src.services.guardrails.audit_log import (
    AuditEvent,
    AuditEventType,
    AuditLogger,
    AuditSeverity,
)


class TestAuditEvent:
    """Tests for AuditEvent dataclass."""

    def test_create_audit_event(self):
        """Test creating an audit event."""
        event = AuditEvent(
            event_type=AuditEventType.PII_DETECTED,
            severity=AuditSeverity.WARNING,
            timestamp=datetime.now(),
            user_id="user123",
            session_id="session456",
            message="PII detected in query",
        )

        assert event.event_type == AuditEventType.PII_DETECTED
        assert event.severity == AuditSeverity.WARNING
        assert event.user_id == "user123"
        assert event.session_id == "session456"

    def test_to_dict(self):
        """Test converting event to dictionary."""
        event = AuditEvent(
            event_type=AuditEventType.QUERY_PROCESSED,
            severity=AuditSeverity.INFO,
            timestamp=datetime.now(),
            details={"query_length": 50},
        )

        event_dict = event.to_dict()

        assert isinstance(event_dict, dict)
        assert event_dict["event_type"] == AuditEventType.QUERY_PROCESSED.value
        assert event_dict["severity"] == AuditSeverity.INFO.value
        assert "timestamp" in event_dict
        assert event_dict["details"]["query_length"] == 50

    def test_to_json(self):
        """Test converting event to JSON."""
        event = AuditEvent(
            event_type=AuditEventType.ERROR,
            severity=AuditSeverity.ERROR,
            timestamp=datetime.now(),
            message="Test error",
        )

        json_str = event.to_json()
        parsed = json.loads(json_str)

        assert parsed["event_type"] == AuditEventType.ERROR.value
        assert parsed["message"] == "Test error"


class TestAuditLogger:
    """Tests for AuditLogger class."""

    @pytest.fixture
    def temp_log_file(self, tmp_path):
        """Create temporary log file."""
        return tmp_path / "test_audit.log"

    @pytest.fixture
    def logger(self, temp_log_file):
        """Create audit logger with temporary file."""
        return AuditLogger(log_file=temp_log_file, log_to_console=False, structured_logs=True)

    def test_log_event(self, logger, temp_log_file):
        """Test logging an event."""
        event = AuditEvent(
            event_type=AuditEventType.QUERY_PROCESSED,
            severity=AuditSeverity.INFO,
            timestamp=datetime.now(),
            message="Test query processed",
        )

        logger.log_event(event)

        # Check log file was created and contains the event
        assert temp_log_file.exists()
        content = temp_log_file.read_text()
        assert "QUERY_PROCESSED" in content.upper() or "query_processed" in content

    def test_log_pii_detection(self, logger, temp_log_file):
        """Test logging PII detection."""
        logger.log_pii_detection(
            pii_types=["email", "phone"], user_id="user123", session_id="session456", redacted=True
        )

        content = temp_log_file.read_text()
        assert "email" in content
        assert "phone" in content

    def test_log_toxicity_detection(self, logger, temp_log_file):
        """Test logging toxicity detection."""
        logger.log_toxicity_detection(
            toxicity_level="high",
            categories=["threat", "harassment"],
            score=0.85,
            user_id="user123",
            filtered=True,
        )

        content = temp_log_file.read_text()
        assert "threat" in content or "THREAT" in content
        assert "harassment" in content or "HARASSMENT" in content

    def test_log_query(self, logger, temp_log_file):
        """Test logging query processing."""
        logger.log_query(query="Test query", user_id="user123", metadata={"source": "api"})

        content = temp_log_file.read_text()
        assert "query" in content.lower()

    def test_log_response(self, logger, temp_log_file):
        """Test logging response generation."""
        logger.log_response(response_length=150, user_id="user123", metadata={"tokens": 50})

        content = temp_log_file.read_text()
        assert "response" in content.lower()

    def test_log_access_denied(self, logger, temp_log_file):
        """Test logging access denied."""
        logger.log_access_denied(reason="Unauthorized", user_id="user123", ip_address="192.168.1.1")

        content = temp_log_file.read_text()
        assert "access" in content.lower() or "denied" in content.lower()

    def test_log_error(self, logger, temp_log_file):
        """Test logging error."""
        logger.log_error(
            error_message="Test error occurred", error_type="ValueError", user_id="user123"
        )

        content = temp_log_file.read_text()
        assert "error" in content.lower()
        assert "ValueError" in content

    def test_severity_levels(self, logger, temp_log_file):
        """Test different severity levels."""
        severities = [
            AuditSeverity.INFO,
            AuditSeverity.WARNING,
            AuditSeverity.ERROR,
            AuditSeverity.CRITICAL,
        ]

        for severity in severities:
            event = AuditEvent(
                event_type=AuditEventType.SYSTEM_EVENT,
                severity=severity,
                timestamp=datetime.now(),
                message=f"Test {severity.value} event",
            )
            logger.log_event(event)

        content = temp_log_file.read_text()
        # At least some severity indicators should be present
        assert len(content) > 0

    def test_structured_logging(self, logger, temp_log_file):
        """Test structured JSON logging."""
        event = AuditEvent(
            event_type=AuditEventType.QUERY_PROCESSED,
            severity=AuditSeverity.INFO,
            timestamp=datetime.now(),
            details={"key": "value"},
        )

        logger.log_event(event)

        content = temp_log_file.read_text()
        # Should be valid JSON
        parsed = json.loads(content.strip().split("\n")[0])
        assert parsed["details"]["key"] == "value"

    def test_get_recent_events(self, logger, temp_log_file):
        """Test retrieving recent events."""
        # Log multiple events
        for i in range(5):
            event = AuditEvent(
                event_type=AuditEventType.QUERY_PROCESSED,
                severity=AuditSeverity.INFO,
                timestamp=datetime.now(),
                message=f"Query {i}",
            )
            logger.log_event(event)

        # Retrieve recent events
        events = logger.get_recent_events(count=3)

        assert len(events) <= 3

    def test_filter_events_by_type(self, logger, temp_log_file):
        """Test filtering events by type."""
        # Log different event types
        logger.log_pii_detection(["email"], user_id="user1")
        logger.log_query("test query", user_id="user2")

        # Get only PII events
        pii_events = logger.get_recent_events(count=10, event_type=AuditEventType.PII_DETECTED)

        # Should only contain PII events
        for event in pii_events:
            assert event["event_type"] in [
                AuditEventType.PII_DETECTED.value,
                AuditEventType.PII_REDACTED.value,
            ]


class TestAuditLoggerIntegration:
    """Integration tests for audit logger."""

    @pytest.fixture
    def temp_log_dir(self, tmp_path):
        """Create temporary log directory."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        return log_dir

    def test_complete_audit_trail(self, temp_log_dir):
        """Test complete audit trail for a user session."""
        log_file = temp_log_dir / "audit.log"
        logger = AuditLogger(log_file=log_file, log_to_console=False)

        user_id = "user123"
        session_id = "session456"

        # User makes a query
        logger.log_query(query="Test query", user_id=user_id, session_id=session_id)

        # PII detected
        logger.log_pii_detection(
            pii_types=["email"], user_id=user_id, session_id=session_id, redacted=True
        )

        # Response generated
        logger.log_response(response_length=200, user_id=user_id, session_id=session_id)

        # Verify log file contains all events
        content = log_file.read_text()
        lines = content.strip().split("\n")

        assert len(lines) >= 3
        assert any("query" in line.lower() for line in lines)
        assert any("pii" in line.lower() or "email" in line.lower() for line in lines)
        assert any("response" in line.lower() for line in lines)

    def test_concurrent_logging(self, temp_log_dir):
        """Test logging from multiple sources."""
        log_file = temp_log_dir / "concurrent.log"
        logger = AuditLogger(log_file=log_file, log_to_console=False)

        # Simulate multiple users
        for i in range(10):
            logger.log_query(query=f"Query {i}", user_id=f"user{i}", session_id=f"session{i}")

        # All events should be logged
        events = logger.get_recent_events(count=10)
        assert len(events) == 10

    def test_log_file_creation(self, tmp_path):
        """Test automatic log file and directory creation."""
        log_file = tmp_path / "nested" / "logs" / "audit.log"
        logger = AuditLogger(log_file=log_file, log_to_console=False)

        event = AuditEvent(
            event_type=AuditEventType.SYSTEM_EVENT,
            severity=AuditSeverity.INFO,
            timestamp=datetime.now(),
            message="Test",
        )
        logger.log_event(event)

        assert log_file.exists()
        assert log_file.parent.exists()
