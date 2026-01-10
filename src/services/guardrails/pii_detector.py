"""
PII (Personally Identifiable Information) Detection and Redaction.

Detects and redacts various types of PII including:
- Email addresses
- Phone numbers
- Social Security Numbers
- Credit card numbers
- IP addresses
- Names (basic pattern-based detection)
"""

import re
from typing import List, Optional

from src.models.guardrails import PIIMatch, PIIType


class PIIDetector:
    """Detects PII in text using regex patterns and heuristics."""

    # Regex patterns for different PII types
    PATTERNS = {
        PIIType.EMAIL: r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        # Match 10-digit (with optional area code formatting) or 7-digit phone numbers
        PIIType.PHONE: r"\b(?:(?:\+?1[-.]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\d{3}[-.\s]\d{4})\b",
        PIIType.SSN: r"\b\d{3}-\d{2}-\d{4}\b",
        PIIType.CREDIT_CARD: r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        PIIType.IP_ADDRESS: r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        PIIType.DATE_OF_BIRTH: r"\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})\b",
    }

    def __init__(
        self,
        enabled_types: Optional[List[PIIType]] = None,
        case_sensitive: bool = False,
    ):
        """
        Initialize PII detector.

        Args:
            enabled_types: List of PII types to detect. If None, all types are enabled.
            case_sensitive: Whether pattern matching should be case sensitive.
        """
        self.enabled_types = enabled_types or list(PIIType)
        self.case_sensitive = case_sensitive
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile regex patterns for enabled PII types."""
        flags = 0 if self.case_sensitive else re.IGNORECASE
        self.compiled_patterns = {
            pii_type: re.compile(pattern, flags)
            for pii_type, pattern in self.PATTERNS.items()
            if pii_type in self.enabled_types
        }

    def detect(self, text: str) -> List[PIIMatch]:
        """
        Detect PII in the given text.

        Args:
            text: Text to scan for PII.

        Returns:
            List of PIIMatch objects representing detected PII.
        """
        matches: List[PIIMatch] = []

        for pii_type, pattern in self.compiled_patterns.items():
            for match in pattern.finditer(text):
                pii_match = PIIMatch(
                    pii_type=pii_type,
                    value=match.group(),
                    start=match.start(),
                    end=match.end(),
                    confidence=self._calculate_confidence(pii_type, match.group()),
                )
                matches.append(pii_match)

        # Sort by start position
        matches.sort(key=lambda x: x.start)
        return matches

    def _calculate_confidence(self, pii_type: PIIType, value: str) -> float:
        """
        Calculate confidence score for detected PII.

        Args:
            pii_type: Type of PII detected.
            value: The detected value.

        Returns:
            Confidence score between 0 and 1.
        """
        # Basic Luhn algorithm check for credit cards
        if pii_type == PIIType.CREDIT_CARD:
            clean_value = re.sub(r"[-\s]", "", value)
            if self._luhn_check(clean_value):
                return 0.95
            return 0.6

        # IP address validation
        if pii_type == PIIType.IP_ADDRESS:
            parts = value.split(".")
            if all(0 <= int(part) <= 255 for part in parts):
                return 0.9
            return 0.5

        return 1.0

    @staticmethod
    def _luhn_check(card_number: str) -> bool:
        """
        Validate credit card number using Luhn algorithm.

        Args:
            card_number: Credit card number string (digits only).

        Returns:
            True if valid according to Luhn algorithm.
        """
        try:
            digits = [int(d) for d in card_number]
            checksum = 0
            for i, digit in enumerate(reversed(digits)):
                if i % 2 == 1:
                    digit *= 2
                    if digit > 9:
                        digit -= 9
                checksum += digit
            return checksum % 10 == 0
        except (ValueError, AttributeError):
            return False

    def has_pii(self, text: str) -> bool:
        """
        Check if text contains any PII.

        Args:
            text: Text to check.

        Returns:
            True if PII is detected, False otherwise.
        """
        return len(self.detect(text)) > 0


class PIIRedactor:
    """Redacts PII from text."""

    def __init__(
        self,
        detector: Optional[PIIDetector] = None,
        redaction_char: str = "*",
        preserve_length: bool = True,
    ):
        """
        Initialize PII redactor.

        Args:
            detector: PIIDetector instance. If None, a default detector is created.
            redaction_char: Character to use for redaction.
            preserve_length: Whether to preserve original text length when redacting.
        """
        self.detector = detector or PIIDetector()
        self.redaction_char = redaction_char
        self.preserve_length = preserve_length

    def redact(
        self,
        text: str,
        min_confidence: float = 0.5,
        custom_placeholder: Optional[dict[PIIType, str]] = None,
    ) -> str:
        """
        Redact PII from text.

        Args:
            text: Text to redact PII from.
            min_confidence: Minimum confidence threshold for redaction.
            custom_placeholder: Custom placeholders for specific PII types.

        Returns:
            Text with PII redacted.
        """
        matches = self.detector.detect(text)
        
        # Filter by confidence
        matches = [m for m in matches if m.confidence >= min_confidence]
        
        if not matches:
            return text

        # Sort matches by start position in reverse to maintain indices
        matches.sort(key=lambda x: x.start, reverse=True)

        result = text
        for match in matches:
            replacement = self._get_replacement(match, custom_placeholder)
            result = result[: match.start] + replacement + result[match.end :]

        return result

    def _get_replacement(
        self,
        match: PIIMatch,
        custom_placeholder: Optional[dict[PIIType, str]] = None,
    ) -> str:
        """
        Get replacement string for a PII match.

        Args:
            match: PIIMatch to replace.
            custom_placeholder: Custom placeholders for specific PII types.

        Returns:
            Replacement string.
        """
        # Check for custom placeholder
        if custom_placeholder and match.pii_type in custom_placeholder:
            return custom_placeholder[match.pii_type]

        # Use default placeholders
        default_placeholders = {
            PIIType.EMAIL: "[EMAIL]",
            PIIType.PHONE: "[PHONE]",
            PIIType.SSN: "[SSN]",
            PIIType.CREDIT_CARD: "[CREDIT_CARD]",
            PIIType.IP_ADDRESS: "[IP_ADDRESS]",
            PIIType.NAME: "[NAME]",
            PIIType.ADDRESS: "[ADDRESS]",
            PIIType.DATE_OF_BIRTH: "[DOB]",
        }

        if not self.preserve_length:
            return default_placeholders.get(match.pii_type, "[REDACTED]")

        # Preserve length with redaction characters
        length = match.end - match.start
        return self.redaction_char * length

    def redact_with_metadata(
        self,
        text: str,
        min_confidence: float = 0.5,
    ) -> tuple[str, List[PIIMatch]]:
        """
        Redact PII and return both redacted text and detected PII metadata.

        Args:
            text: Text to redact PII from.
            min_confidence: Minimum confidence threshold for redaction.

        Returns:
            Tuple of (redacted_text, list of PIIMatch objects).
        """
        matches = self.detector.detect(text)
        matches = [m for m in matches if m.confidence >= min_confidence]
        redacted_text = self.redact(text, min_confidence)
        return redacted_text, matches
