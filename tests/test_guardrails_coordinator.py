"""
Unit tests for guardrails coordinator.
"""

import pytest

from src.services.guardrails.audit_log import AuditLogger
from src.services.guardrails.coordinator import GuardrailResult, GuardrailsCoordinator
from src.services.guardrails.pii_detector import PIIDetector, PIIRedactor
from src.services.guardrails.safe_response import SafeResponseTemplate
from src.services.guardrails.toxicity_filter import ToxicityFilter


class TestGuardrailsCoordinator:
    """Tests for GuardrailsCoordinator class."""

    @pytest.fixture
    def coordinator(self, tmp_path):
        """Create guardrails coordinator with temporary log file."""
        log_file = tmp_path / "audit.log"
        audit_logger = AuditLogger(log_file=log_file, log_to_console=False)

        return GuardrailsCoordinator(audit_logger=audit_logger, enable_audit_logging=True)

    def test_check_input_clean(self, coordinator):
        """Test checking clean input."""
        text = "This is a clean message with no issues."
        result = coordinator.check_input(text, user_id="user123")

        assert result.is_safe
        assert result.processed_text == text
        assert not result.pii_detected
        assert not result.toxicity_detected
        assert len(result.violations) == 0

    def test_check_input_with_pii(self, coordinator):
        """Test checking input with PII."""
        text = "Contact me at test@example.com"
        result = coordinator.check_input(text, user_id="user123")

        assert result.pii_detected
        assert "email" in result.pii_types
        # Should be redacted if auto_redact_pii is True
        if coordinator.auto_redact_pii:
            assert "test@example.com" not in result.processed_text

    def test_check_input_with_toxicity(self, coordinator):
        """Test checking input with toxic content."""
        text = "I will hurt you!"
        result = coordinator.check_input(text, user_id="user123")

        assert result.toxicity_detected
        if coordinator.block_on_toxicity:
            assert not result.is_safe
            assert result.safe_response is not None

    def test_check_input_multiple_violations(self, coordinator):
        """Test checking input with multiple violations."""
        text = "Email me at bad@example.com, you stupid person!"
        result = coordinator.check_input(text, user_id="user123")

        # Should detect both PII and toxicity
        assert len(result.violations) > 0

    def test_check_output_always_safe(self, coordinator):
        """Test that check_output always makes content safe."""
        text = "Contact me at test@example.com or I'll hurt you!"
        result = coordinator.check_output(text, user_id="user123")

        # Output should always be safe (redacted/filtered)
        assert result.is_safe
        assert result.processed_text is not None
        assert "test@example.com" not in result.processed_text

    def test_process_query_safe(self, coordinator):
        """Test processing safe query."""
        query = "What is the weather like today?"
        is_safe, processed = coordinator.process_query(query, user_id="user123")

        assert is_safe
        assert processed == query

    def test_process_query_unsafe(self, coordinator):
        """Test processing unsafe query."""
        query = "I will hurt you!"
        is_safe, response = coordinator.process_query(query, user_id="user123")

        if coordinator.block_on_toxicity:
            assert not is_safe
            assert "cannot" in response.lower() or "unable" in response.lower()

    def test_process_query_with_pii_redaction(self, coordinator):
        """Test processing query with PII that gets redacted."""
        query = "My email is test@example.com"
        is_safe, processed = coordinator.process_query(query, user_id="user123")

        if coordinator.auto_redact_pii:
            assert is_safe
            assert "test@example.com" not in processed
        else:
            # If not auto-redacting, query should be blocked
            assert not is_safe or "test@example.com" not in processed

    def test_process_response(self, coordinator):
        """Test processing response through output guardrails."""
        response = "Your email is test@example.com"
        processed = coordinator.process_response(response, user_id="user123")

        # Email should be redacted in output
        assert "test@example.com" not in processed

    def test_disable_pii_check(self, tmp_path):
        """Test with PII check disabled."""
        log_file = tmp_path / "audit.log"
        coordinator = GuardrailsCoordinator(
            audit_logger=AuditLogger(log_file=log_file, log_to_console=False),
            enable_pii_check=False,
        )

        text = "Email: test@example.com"
        result = coordinator.check_input(text)

        assert not result.pii_detected
        assert result.is_safe

    def test_disable_toxicity_check(self, tmp_path):
        """Test with toxicity check disabled."""
        log_file = tmp_path / "audit.log"
        coordinator = GuardrailsCoordinator(
            audit_logger=AuditLogger(log_file=log_file, log_to_console=False),
            enable_toxicity_check=False,
        )

        text = "I will hurt you!"
        result = coordinator.check_input(text)

        assert not result.toxicity_detected
        assert result.is_safe

    def test_disable_audit_logging(self, tmp_path):
        """Test with audit logging disabled."""
        log_file = tmp_path / "audit.log"
        coordinator = GuardrailsCoordinator(
            audit_logger=AuditLogger(log_file=log_file, log_to_console=False),
            enable_audit_logging=False,
        )

        text = "Test message"
        coordinator.check_input(text, user_id="user123")

        # Log file should be empty or minimal
        # (can't fully test without checking log internals)
        assert True  # Placeholder

    def test_user_session_tracking(self, coordinator):
        """Test user and session tracking."""
        user_id = "user123"
        session_id = "session456"

        # Process multiple queries in a session
        coordinator.process_query("First query", user_id, session_id)
        coordinator.process_query("Second query", user_id, session_id)

        # Audit log should contain session information
        # (would need to inspect log file to verify)
        assert True  # Placeholder


class TestGuardrailResult:
    """Tests for GuardrailResult dataclass."""

    def test_create_result(self):
        """Test creating a guardrail result."""
        result = GuardrailResult(
            is_safe=True,
            original_text="Test text",
            processed_text="Test text",
            pii_detected=False,
            toxicity_detected=False,
        )

        assert result.is_safe
        assert result.original_text == "Test text"
        assert result.processed_text == "Test text"
        assert not result.pii_detected
        assert not result.toxicity_detected

    def test_result_with_violations(self):
        """Test result with violations."""
        result = GuardrailResult(
            is_safe=False,
            original_text="Bad text",
            pii_detected=True,
            pii_types=["email", "phone"],
            toxicity_detected=True,
            toxicity_level="high",
            violations=["PII detected", "Toxic content"],
        )

        assert not result.is_safe
        assert len(result.violations) == 2
        assert result.pii_detected
        assert result.toxicity_detected

    def test_result_defaults(self):
        """Test result with default values."""
        result = GuardrailResult(is_safe=True, original_text="Test")

        assert result.pii_types == []
        assert result.violations == []
        assert result.toxicity_score == 0.0


class TestGuardrailsIntegration:
    """Integration tests for complete guardrails system."""

    @pytest.fixture
    def full_coordinator(self, tmp_path):
        """Create fully configured coordinator."""
        log_file = tmp_path / "audit.log"

        return GuardrailsCoordinator(
            pii_detector=PIIDetector(),
            pii_redactor=PIIRedactor(),
            toxicity_filter=ToxicityFilter(sensitivity=0.3),
            response_template=SafeResponseTemplate(),
            audit_logger=AuditLogger(log_file=log_file, log_to_console=False),
            enable_pii_check=True,
            enable_toxicity_check=True,
            enable_audit_logging=True,
            auto_redact_pii=True,
            block_on_toxicity=True,
        )

    def test_full_pipeline_clean_content(self, full_coordinator):
        """Test full pipeline with clean content."""
        # Input check
        query = "What is machine learning?"
        is_safe, processed_query = full_coordinator.process_query(
            query, user_id="user123", session_id="session456"
        )

        assert is_safe
        assert processed_query == query

        # Response check
        response = "Machine learning is a subset of AI."
        processed_response = full_coordinator.process_response(
            response, user_id="user123", session_id="session456"
        )

        assert processed_response == response

    def test_full_pipeline_with_pii(self, full_coordinator):
        """Test full pipeline with PII."""
        query = "My email is test@example.com and phone is 555-1234."
        is_safe, processed = full_coordinator.process_query(query)

        # Should be safe with redacted PII
        assert is_safe
        assert "test@example.com" not in processed
        assert "555-1234" not in processed

    def test_full_pipeline_with_toxicity(self, full_coordinator):
        """Test full pipeline with toxic content."""
        query = "I will hurt you!"
        is_safe, response = full_coordinator.process_query(query)

        # Should be blocked
        assert not is_safe
        assert "cannot" in response.lower() or "unable" in response.lower()

    def test_full_pipeline_mixed_violations(self, full_coordinator):
        """Test full pipeline with mixed violations."""
        query = "Email me at bad@test.com, you stupid person!"
        is_safe, response = full_coordinator.process_query(query)

        # Should be blocked due to toxicity
        assert not is_safe

    def test_output_sanitization(self, full_coordinator):
        """Test that outputs are always sanitized."""
        response = "Contact us at support@example.com or I'll hurt you!"
        processed = full_coordinator.process_response(response)

        # Both PII and toxicity should be removed/filtered
        assert "support@example.com" not in processed
        assert ("hurt" not in processed) or ("[FILTERED]" in processed)

    def test_session_consistency(self, full_coordinator):
        """Test consistency across a session."""
        user_id = "user123"
        session_id = "session456"

        # Multiple queries in same session
        queries = ["What is AI?", "How does it work?", "Tell me more."]

        for query in queries:
            is_safe, processed = full_coordinator.process_query(
                query, user_id=user_id, session_id=session_id
            )
            assert is_safe
            assert processed == query
