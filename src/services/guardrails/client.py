"""
Guardrails Client - Unified interface for content safety checks.

Provides a clean client interface for all guardrail operations:
- PII detection and redaction
- Toxicity filtering
- Jailbreak and prompt injection detection
- Safe response generation
- Audit logging
"""

from src.models.guardrails import AuditEvent, AuditEventType, AuditSeverity, GuardrailResult

from .audit_log import AuditLogger
from .coordinator import GuardrailsCoordinator
from .jailbreak_detector import JailbreakDetector
from .pii_detector import PIIDetector, PIIRedactor
from .safe_response import SafeResponseTemplate
from .toxicity_filter import ToxicityFilter


class GuardrailsClient:
    """
    Unified client for guardrails operations.

    Wraps GuardrailsCoordinator to provide a clean interface following
    the cache client pattern.

    Features:
    - PII detection and redaction
    - Toxicity filtering
    - Jailbreak detection
    - Safe response generation
    - Comprehensive audit logging
    - Configurable enforcement policies

    Example:
        ```python
        client = GuardrailsClient()
        result = client.check_input("user query")
        if result.passed:
            # Process query
            response = generate_response(result.sanitized_content)
            result = client.check_output(response)
        ```
    """

    def __init__(
        self,
        pii_detector: PIIDetector | None = None,
        pii_redactor: PIIRedactor | None = None,
        toxicity_filter: ToxicityFilter | None = None,
        jailbreak_detector: JailbreakDetector | None = None,
        response_template: SafeResponseTemplate | None = None,
        audit_logger: AuditLogger | None = None,
        enable_pii_check: bool = True,
        enable_toxicity_check: bool = True,
        enable_jailbreak_check: bool = True,
        enable_audit_logging: bool = True,
        auto_redact_pii: bool = True,
        block_on_toxicity: bool = True,
        block_sensitive_pii: bool = True,
    ):
        """
        Initialize guardrails client.

        Args:
            pii_detector: PIIDetector instance
            pii_redactor: PIIRedactor instance
            toxicity_filter: ToxicityFilter instance
            jailbreak_detector: JailbreakDetector instance
            response_template: SafeResponseTemplate instance
            audit_logger: AuditLogger instance
            enable_pii_check: Enable PII detection
            enable_toxicity_check: Enable toxicity filtering
            enable_jailbreak_check: Enable jailbreak detection
            enable_audit_logging: Enable audit logging
            auto_redact_pii: Automatically redact PII
            block_on_toxicity: Block requests with toxic content
            block_sensitive_pii: Block requests with sensitive PII
        """
        self.coordinator = GuardrailsCoordinator(
            pii_detector=pii_detector,
            pii_redactor=pii_redactor,
            toxicity_filter=toxicity_filter,
            jailbreak_detector=jailbreak_detector,
            response_template=response_template,
            audit_logger=audit_logger,
            enable_pii_check=enable_pii_check,
            enable_toxicity_check=enable_toxicity_check,
            enable_jailbreak_check=enable_jailbreak_check,
            enable_audit_logging=enable_audit_logging,
            auto_redact_pii=auto_redact_pii,
            block_on_toxicity=block_on_toxicity,
            block_sensitive_pii=block_sensitive_pii,
        )

    def check_input(
        self,
        text: str,
        user_id: str | None = None,
        session_id: str | None = None,
        metadata: dict | None = None,
    ) -> GuardrailResult:
        """
        Check user input for safety violations.

        Runs all enabled guardrails on the input text and returns a result
        indicating whether the input is safe to process.

        Args:
            text: Input text to check
            user_id: Optional user identifier for audit logging
            session_id: Optional session identifier for audit logging
            metadata: Optional metadata for audit logging

        Returns:
            GuardrailResult with pass/fail status and details

        Example:
            ```python
            result = client.check_input("Hello, my SSN is 123-45-6789")
            if not result.passed:
                print(f"Blocked: {result.reason}")
            print(result.sanitized_content)  # "Hello, my SSN is [SSN_REDACTED]"
            ```
        """
        return self.coordinator.check_input(  # type: ignore[call-arg]
            text=text, user_id=user_id, session_id=session_id, metadata=metadata
        )

    def check_output(
        self,
        text: str,
        user_id: str | None = None,
        session_id: str | None = None,
        metadata: dict | None = None,
    ) -> GuardrailResult:
        """
        Check generated output for safety violations.

        Validates that generated content is safe to return to users.

        Args:
            text: Output text to check
            user_id: Optional user identifier for audit logging
            session_id: Optional session identifier for audit logging
            metadata: Optional metadata for audit logging

        Returns:
            GuardrailResult with pass/fail status and details

        Example:
            ```python
            result = client.check_output(generated_response)
            if result.passed:
                return result.sanitized_content
            else:
                return result.safe_response
            ```
        """
        return self.coordinator.check_output(
            text=text, user_id=user_id, session_id=session_id, metadata=metadata
        )

    def check_query_response_pair(
        self,
        query: str,
        response: str,
        user_id: str | None = None,
        session_id: str | None = None,
        metadata: dict | None = None,
    ) -> tuple[GuardrailResult, GuardrailResult]:
        """
        Check both query and response for safety.

        Convenience method to check input and output in one call.

        Args:
            query: User query
            response: Generated response
            user_id: Optional user identifier
            session_id: Optional session identifier
            metadata: Optional metadata

        Returns:
            Tuple of (query_result, response_result)

        Example:
            ```python
            query_result, response_result = client.check_query_response_pair(
                query="user input",
                response="generated output"
            )
            if query_result.passed and response_result.passed:
                return response_result.sanitized_content
            ```
        """
        return self.coordinator.check_query_response_pair(
            query=query,
            response=response,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata,
        )

    def redact_pii(self, text: str) -> tuple[str, list]:
        """
        Redact PII from text.

        Args:
            text: Text to redact

        Returns:
            Tuple of (redacted_text, pii_matches)
        """
        return self.coordinator.pii_redactor.redact(text)

    def detect_pii(self, text: str) -> list:
        """
        Detect PII in text without redaction.

        Args:
            text: Text to analyze

        Returns:
            List of PIIMatch objects
        """
        return self.coordinator.pii_detector.detect(text)

    def check_toxicity(self, text: str) -> dict:
        """
        Check text for toxic content.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with toxicity analysis
        """
        return self.coordinator.toxicity_filter.analyze(text)

    def check_jailbreak(self, text: str) -> dict:
        """
        Check for jailbreak/prompt injection attempts.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with jailbreak detection results
        """
        return self.coordinator.jailbreak_detector.detect(text)

    def get_audit_logs(
        self,
        user_id: str | None = None,
        session_id: str | None = None,
        event_type: AuditEventType | None = None,
        severity: AuditSeverity | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """
        Retrieve audit logs.

        Args:
            user_id: Filter by user ID
            session_id: Filter by session ID
            event_type: Filter by event type
            severity: Filter by severity
            limit: Maximum number of events to return

        Returns:
            List of AuditEvent objects
        """
        return self.coordinator.audit_logger.get_events(
            user_id=user_id,
            session_id=session_id,
            event_type=event_type,
            severity=severity,
            limit=limit,
        )

    def clear_audit_logs(self) -> None:
        """Clear all audit logs."""
        self.coordinator.audit_logger.clear()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"GuardrailsClient("
            f"pii={'enabled' if self.coordinator.enable_pii_check else 'disabled'}, "
            f"toxicity={'enabled' if self.coordinator.enable_toxicity_check else 'disabled'}, "
            f"jailbreak={'enabled' if self.coordinator.enable_jailbreak_check else 'disabled'})"
        )
