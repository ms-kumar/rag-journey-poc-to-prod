"""
Guardrails Coordinator - Main interface for content safety checks.

Coordinates all guardrail components:
- PII detection and redaction
- Toxicity filtering
- Jailbreak and prompt injection detection
- Safe response generation
- Audit logging
"""

from datetime import UTC, datetime

from src.models.guardrails import AuditEvent, AuditEventType, AuditSeverity, GuardrailResult

from .audit_log import AuditLogger
from .jailbreak_detector import JailbreakDetector
from .pii_detector import PIIDetector, PIIRedactor
from .safe_response import SafeResponseTemplate
from .toxicity_filter import ToxicityFilter


class GuardrailsCoordinator:
    """Coordinates all guardrail checks and enforcements."""

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
        Initialize guardrails coordinator.

        Args:
            pii_detector: PIIDetector instance.
            pii_redactor: PIIRedactor instance.
            toxicity_filter: ToxicityFilter instance.
            jailbreak_detector: JailbreakDetector instance.
            response_template: SafeResponseTemplate instance.
            audit_logger: AuditLogger instance.
            enable_pii_check: Enable PII detection.
            enable_toxicity_check: Enable toxicity filtering.
            enable_jailbreak_check: Enable jailbreak detection.
            enable_audit_logging: Enable audit logging.
            auto_redact_pii: Automatically redact PII.
            block_on_toxicity: Block requests with toxic content.
            block_sensitive_pii: Block requests with sensitive PII (SSN, credit cards).
        """
        self.pii_detector = pii_detector or PIIDetector()
        self.pii_redactor = pii_redactor or PIIRedactor(detector=self.pii_detector)
        self.toxicity_filter = toxicity_filter or ToxicityFilter()
        self.jailbreak_detector = jailbreak_detector or JailbreakDetector()
        self.response_template = response_template or SafeResponseTemplate()
        self.audit_logger = audit_logger or AuditLogger()

        self.enable_pii_check = enable_pii_check
        self.enable_toxicity_check = enable_toxicity_check
        self.enable_jailbreak_check = enable_jailbreak_check
        self.enable_audit_logging = enable_audit_logging
        self.auto_redact_pii = auto_redact_pii
        self.block_on_toxicity = block_on_toxicity
        self.block_sensitive_pii = block_sensitive_pii

        # Sensitive PII types that should always be blocked
        self.sensitive_pii_types = {"ssn", "credit_card", "passport", "drivers_license"}

    def check_input(
        self,
        text: str,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> GuardrailResult:
        """
        Check input text against all guardrails.

        Args:
            text: Text to check.
            user_id: User identifier for audit logging.
            session_id: Session identifier for audit logging.

        Returns:
            GuardrailResult with check results.
        """
        violations = []
        processed_text = text
        pii_detected = False
        pii_types_list = []
        toxicity_detected = False
        toxicity_level = None
        toxicity_score = 0.0
        safe_response = None
        jailbreak_detected = False
        jailbreak_severity = None

        # Jailbreak/Prompt Injection Check (highest priority)
        if self.enable_jailbreak_check:
            jailbreak_matches = self.jailbreak_detector.detect(text)
            if jailbreak_matches:
                jailbreak_detected = True
                jailbreak_severity = self.jailbreak_detector.get_severity(text)
                violations.append(f"Jailbreak attempt detected ({jailbreak_severity})")

                if self.enable_audit_logging:
                    event = AuditEvent(
                        event_type=AuditEventType.JAILBREAK_DETECTED,
                        severity=AuditSeverity.CRITICAL,
                        timestamp=datetime.now(UTC),
                        user_id=user_id,
                        session_id=session_id,
                        details={
                            "severity": jailbreak_severity,
                            "patterns": [m.pattern for m in jailbreak_matches],
                            "matched_text": [m.matched_text for m in jailbreak_matches],
                        },
                        message=f"Jailbreak attempt detected: {jailbreak_severity}",
                    )
                    self.audit_logger.log_event(event)

        # PII Check
        if self.enable_pii_check:
            pii_matches = self.pii_detector.detect(text)
            if pii_matches:
                pii_detected = True
                pii_types_list = list({m.pii_type.value for m in pii_matches})
                violations.append("PII detected")

                if self.enable_audit_logging:
                    self.audit_logger.log_pii_detection(
                        pii_types=pii_types_list,
                        user_id=user_id,
                        session_id=session_id,
                        redacted=self.auto_redact_pii,
                    )

                if self.auto_redact_pii:
                    processed_text = self.pii_redactor.redact(processed_text)

        # Toxicity Check
        if self.enable_toxicity_check:
            toxicity_result = self.toxicity_filter.check(text)
            if toxicity_result.is_toxic:
                toxicity_detected = True
                toxicity_level = toxicity_result.max_level.value
                toxicity_score = toxicity_result.overall_score
                violations.append(f"Toxic content ({toxicity_level})")

                if self.enable_audit_logging:
                    self.audit_logger.log_toxicity_detection(
                        toxicity_level=toxicity_level,
                        categories=[c.value for c in toxicity_result.categories],
                        score=toxicity_score,
                        user_id=user_id,
                        session_id=session_id,
                        filtered=self.block_on_toxicity,
                    )

        # Determine if content is safe
        is_safe = True

        # Block jailbreak attempts (highest priority)
        if jailbreak_detected:
            is_safe = False
            safe_response = self.response_template.get_fallback_response()
        # Block sensitive PII (high priority)
        elif self.block_sensitive_pii and pii_detected:
            # Check if any detected PII is sensitive
            has_sensitive_pii = any(
                pii_type in self.sensitive_pii_types for pii_type in pii_types_list
            )
            if has_sensitive_pii:
                is_safe = False
                safe_response = self.response_template.get_pii_response(pii_types_list)
        # Block if PII detected and auto-redact disabled
        elif pii_detected and not self.auto_redact_pii:
            is_safe = False
            safe_response = self.response_template.get_pii_response(pii_types_list)

        # Block toxic content (check separately, not as elif, so it runs even if non-sensitive PII was detected)
        if is_safe and toxicity_detected and self.block_on_toxicity:
            is_safe = False
            safe_response = self.response_template.get_toxicity_response(
                severity=toxicity_level,
                categories=[c.value for c in toxicity_result.categories]
                if toxicity_detected
                else None,
            )

        return GuardrailResult(
            is_safe=is_safe,
            original_text=text,
            processed_text=processed_text if is_safe else None,
            pii_detected=pii_detected,
            pii_types=pii_types_list,
            toxicity_detected=toxicity_detected,
            toxicity_level=toxicity_level,
            toxicity_score=toxicity_score,
            safe_response=safe_response,
            violations=violations,
        )

    def check_output(
        self,
        text: str,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> GuardrailResult:
        """
        Check output text against guardrails (always auto-redact PII).

        Args:
            text: Text to check.
            user_id: User identifier for audit logging.
            session_id: Session identifier for audit logging.

        Returns:
            GuardrailResult with check results.
        """
        violations = []
        processed_text = text
        pii_detected = False
        pii_types_list = []
        toxicity_detected = False
        toxicity_level = None
        toxicity_score = 0.0

        # Always check and redact PII in outputs
        if self.enable_pii_check:
            pii_matches = self.pii_detector.detect(text)
            if pii_matches:
                pii_detected = True
                pii_types_list = list({m.pii_type.value for m in pii_matches})
                violations.append("PII detected in output")

                if self.enable_audit_logging:
                    self.audit_logger.log_pii_detection(
                        pii_types=pii_types_list,
                        user_id=user_id,
                        session_id=session_id,
                        redacted=True,
                    )

                # Always redact PII in outputs
                processed_text = self.pii_redactor.redact(processed_text)

        # Check for toxicity in outputs
        if self.enable_toxicity_check:
            toxicity_result = self.toxicity_filter.check(text)
            if toxicity_result.is_toxic:
                toxicity_detected = True
                toxicity_level = toxicity_result.max_level.value
                toxicity_score = toxicity_result.overall_score
                violations.append(f"Toxic content in output ({toxicity_level})")

                if self.enable_audit_logging:
                    self.audit_logger.log_toxicity_detection(
                        toxicity_level=toxicity_level,
                        categories=[c.value for c in toxicity_result.categories],
                        score=toxicity_score,
                        user_id=user_id,
                        session_id=session_id,
                        filtered=True,
                    )

                # Filter toxic content
                processed_text = self.toxicity_filter.filter_text(processed_text)

        # Log response generation
        if self.enable_audit_logging:
            self.audit_logger.log_response(
                response_length=len(processed_text),
                user_id=user_id,
                session_id=session_id,
                metadata={
                    "pii_detected": pii_detected,
                    "toxicity_detected": toxicity_detected,
                },
            )

        return GuardrailResult(
            is_safe=True,  # Outputs are always made safe
            original_text=text,
            processed_text=processed_text,
            pii_detected=pii_detected,
            pii_types=pii_types_list,
            toxicity_detected=toxicity_detected,
            toxicity_level=toxicity_level,
            toxicity_score=toxicity_score,
            violations=violations,
        )

    def process_query(
        self,
        query: str,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> tuple[bool, str]:
        """
        Process a query through guardrails.

        Args:
            query: Query text to process.
            user_id: User identifier.
            session_id: Session identifier.

        Returns:
            Tuple of (is_safe, processed_text_or_safe_response).
        """
        result = self.check_input(query, user_id, session_id)

        if self.enable_audit_logging:
            self.audit_logger.log_query(
                query=result.processed_text or "[BLOCKED]",
                user_id=user_id,
                session_id=session_id,
                metadata={
                    "is_safe": result.is_safe,
                    "violations": result.violations,
                },
            )

        if not result.is_safe:
            return False, result.safe_response or self.response_template.get_fallback_response()

        return True, result.processed_text or query

    def process_response(
        self,
        response: str,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> str:
        """
        Process a response through output guardrails.

        Args:
            response: Response text to process.
            user_id: User identifier.
            session_id: Session identifier.

        Returns:
            Processed (safe) response.
        """
        result = self.check_output(response, user_id, session_id)
        return result.processed_text or response
