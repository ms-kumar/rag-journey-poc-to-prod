"""
Unit tests for PII detection and redaction.
"""

from src.services.guardrails.pii_detector import PIIDetector, PIIRedactor, PIIType


class TestPIIDetector:
    """Tests for PIIDetector class."""

    def test_detect_email(self):
        """Test email detection."""
        detector = PIIDetector()
        text = "Contact me at john.doe@example.com for more info."
        matches = detector.detect(text)

        assert len(matches) == 1
        assert matches[0].pii_type == PIIType.EMAIL
        assert matches[0].value == "john.doe@example.com"

    def test_detect_phone_number(self):
        """Test phone number detection."""
        detector = PIIDetector()
        text = "Call me at 555-123-4567 or (555) 987-6543."
        matches = detector.detect(text)

        assert len(matches) == 2
        assert all(m.pii_type == PIIType.PHONE for m in matches)

    def test_detect_ssn(self):
        """Test SSN detection."""
        detector = PIIDetector()
        text = "My SSN is 123-45-6789."
        matches = detector.detect(text)

        assert len(matches) == 1
        assert matches[0].pii_type == PIIType.SSN
        assert matches[0].value == "123-45-6789"

    def test_detect_credit_card(self):
        """Test credit card detection."""
        detector = PIIDetector()
        text = "Card number: 4532-1488-0343-6467"
        matches = detector.detect(text)

        assert len(matches) == 1
        assert matches[0].pii_type == PIIType.CREDIT_CARD

    def test_detect_ip_address(self):
        """Test IP address detection."""
        detector = PIIDetector()
        text = "Server IP is 192.168.1.1 and 10.0.0.5."
        matches = detector.detect(text)

        assert len(matches) == 2
        assert all(m.pii_type == PIIType.IP_ADDRESS for m in matches)

    def test_detect_multiple_pii_types(self):
        """Test detection of multiple PII types in one text."""
        detector = PIIDetector()
        text = (
            "Email me at test@example.com or call 555-1234. "
            "My SSN is 123-45-6789 and IP is 192.168.1.1."
        )
        matches = detector.detect(text)

        assert len(matches) == 4
        pii_types = {m.pii_type for m in matches}
        assert PIIType.EMAIL in pii_types
        assert PIIType.PHONE in pii_types
        assert PIIType.SSN in pii_types
        assert PIIType.IP_ADDRESS in pii_types

    def test_no_pii_detected(self):
        """Test text with no PII."""
        detector = PIIDetector()
        text = "This is a clean message with no personal information."
        matches = detector.detect(text)

        assert len(matches) == 0

    def test_has_pii(self):
        """Test has_pii method."""
        detector = PIIDetector()

        assert detector.has_pii("Email: test@example.com")
        assert not detector.has_pii("No PII here")

    def test_enabled_types_filter(self):
        """Test filtering by enabled PII types."""
        detector = PIIDetector(enabled_types=[PIIType.EMAIL])
        text = "Email: test@example.com, Phone: 555-1234"
        matches = detector.detect(text)

        assert len(matches) == 1
        assert matches[0].pii_type == PIIType.EMAIL

    def test_luhn_algorithm_validation(self):
        """Test credit card validation with Luhn algorithm."""
        detector = PIIDetector()

        # Valid credit card number (passes Luhn check)
        valid_card = "4532015112830366"
        matches = detector.detect(valid_card)
        assert len(matches) == 1
        assert matches[0].confidence > 0.9

        # Invalid credit card number (fails Luhn check)
        invalid_card = "1234567890123456"
        matches = detector.detect(invalid_card)
        if matches:  # May or may not match pattern
            assert matches[0].confidence < 0.9


class TestPIIRedactor:
    """Tests for PIIRedactor class."""

    def test_redact_email(self):
        """Test email redaction."""
        redactor = PIIRedactor()
        text = "Contact me at john.doe@example.com for more info."
        redacted = redactor.redact(text)

        assert "john.doe@example.com" not in redacted
        assert "[EMAIL]" in redacted or "*" in redacted

    def test_redact_preserve_length(self):
        """Test redaction preserving text length."""
        redactor = PIIRedactor(preserve_length=True, redaction_char="*")
        text = "Email: test@example.com"
        redacted = redactor.redact(text)

        # Length should be preserved
        assert len(redacted) == len(text)
        assert "test@example.com" not in redacted

    def test_redact_custom_placeholder(self):
        """Test redaction with custom placeholders."""
        redactor = PIIRedactor(preserve_length=False)
        text = "Email: test@example.com"
        custom_placeholder = {PIIType.EMAIL: "[REDACTED_EMAIL]"}
        redacted = redactor.redact(text, custom_placeholder=custom_placeholder)

        assert "[REDACTED_EMAIL]" in redacted
        assert "test@example.com" not in redacted

    def test_redact_multiple_pii(self):
        """Test redacting multiple PII instances."""
        redactor = PIIRedactor(preserve_length=False)
        text = "Email: test@example.com, Phone: 555-1234, SSN: 123-45-6789"
        redacted = redactor.redact(text)

        assert "test@example.com" not in redacted
        assert "555-1234" not in redacted
        assert "123-45-6789" not in redacted

    def test_redact_with_confidence_threshold(self):
        """Test redaction with confidence threshold."""
        redactor = PIIRedactor()
        text = "Contact: test@example.com"
        redacted = redactor.redact(text, min_confidence=0.95)

        # Email should still be redacted (high confidence)
        assert "test@example.com" not in redacted

    def test_redact_with_metadata(self):
        """Test redaction with metadata return."""
        redactor = PIIRedactor()
        text = "Email: test@example.com, Phone: 555-1234"
        redacted, matches = redactor.redact_with_metadata(text)

        assert len(matches) == 2
        assert "test@example.com" not in redacted
        assert "555-1234" not in redacted

    def test_no_redaction_needed(self):
        """Test text with no PII to redact."""
        redactor = PIIRedactor()
        text = "This is a clean message."
        redacted = redactor.redact(text)

        assert redacted == text


class TestPIIIntegration:
    """Integration tests for PII detection and redaction."""

    def test_full_pipeline(self):
        """Test complete PII detection and redaction pipeline."""
        detector = PIIDetector()
        redactor = PIIRedactor(detector=detector, preserve_length=False)

        text = (
            "Hi, I'm John. Contact me at john@example.com or call 555-1234. "
            "My SSN is 123-45-6789 and credit card is 4532-1488-0343-6467."
        )

        # Detect
        matches = detector.detect(text)
        assert len(matches) >= 3

        # Redact
        redacted = redactor.redact(text)
        assert "john@example.com" not in redacted
        assert "555-1234" not in redacted
        assert "123-45-6789" not in redacted
        assert "4532-1488-0343-6467" not in redacted

    def test_selective_redaction(self):
        """Test redacting only specific PII types."""
        detector = PIIDetector(enabled_types=[PIIType.EMAIL, PIIType.SSN])
        redactor = PIIRedactor(detector=detector)

        text = "Email: test@example.com, Phone: 555-1234, SSN: 123-45-6789"
        redacted = redactor.redact(text)

        # Email and SSN should be redacted
        assert "test@example.com" not in redacted
        assert "123-45-6789" not in redacted

        # Phone should remain (not in enabled types)
        assert "555-1234" in redacted
