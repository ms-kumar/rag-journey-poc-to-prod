"""
Factory for creating guardrails client instances.
"""

import logging
from typing import TYPE_CHECKING

from src.services.guardrails.audit_log import AuditLogger
from src.services.guardrails.client import GuardrailsClient
from src.services.guardrails.jailbreak_detector import JailbreakDetector
from src.services.guardrails.pii_detector import PIIDetector, PIIRedactor
from src.services.guardrails.safe_response import SafeResponseTemplate
from src.services.guardrails.toxicity_filter import ToxicityFilter

if TYPE_CHECKING:
    from src.config import Settings

logger = logging.getLogger(__name__)


def make_guardrails_client(settings: "Settings") -> GuardrailsClient:
    """
    Create guardrails client from application settings.

    Args:
        settings: Application settings

    Returns:
        Configured GuardrailsClient instance

    Example:
        ```python
        from src.config import get_settings
        settings = get_settings()
        client = make_guardrails_client(settings)
        result = client.check_input("user query")
        ```
    """
    # Get guardrails settings
    guardrails_settings = settings.guardrails if hasattr(settings, "guardrails") else None

    # Initialize components
    pii_detector = PIIDetector()
    pii_redactor = PIIRedactor(detector=pii_detector)
    toxicity_filter = ToxicityFilter()
    jailbreak_detector = JailbreakDetector()
    response_template = SafeResponseTemplate()
    audit_logger = AuditLogger()

    # Configure from settings if available
    if guardrails_settings:
        enable_pii_check = getattr(guardrails_settings, "enable_pii_check", True)
        enable_toxicity_check = getattr(guardrails_settings, "enable_toxicity_check", True)
        enable_jailbreak_check = getattr(guardrails_settings, "enable_jailbreak_check", True)
        enable_audit_logging = getattr(guardrails_settings, "enable_audit_logging", True)
        auto_redact_pii = getattr(guardrails_settings, "auto_redact_pii", True)
        block_on_toxicity = getattr(guardrails_settings, "block_on_toxicity", True)
        block_sensitive_pii = getattr(guardrails_settings, "block_sensitive_pii", True)
    else:
        # Default settings
        enable_pii_check = True
        enable_toxicity_check = True
        enable_jailbreak_check = True
        enable_audit_logging = True
        auto_redact_pii = True
        block_on_toxicity = True
        block_sensitive_pii = True

    client = GuardrailsClient(
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

    logger.info("Guardrails client created successfully")
    return client


def make_pii_detector() -> PIIDetector:
    """
    Create standalone PII detector.

    Returns:
        PIIDetector instance
    """
    return PIIDetector()


def make_pii_redactor(detector: PIIDetector | None = None) -> PIIRedactor:
    """
    Create standalone PII redactor.

    Args:
        detector: Optional PIIDetector instance

    Returns:
        PIIRedactor instance
    """
    return PIIRedactor(detector=detector)


def make_toxicity_filter() -> ToxicityFilter:
    """
    Create standalone toxicity filter.

    Returns:
        ToxicityFilter instance
    """
    return ToxicityFilter()


def make_jailbreak_detector() -> JailbreakDetector:
    """
    Create standalone jailbreak detector.

    Returns:
        JailbreakDetector instance
    """
    return JailbreakDetector()


def make_audit_logger() -> AuditLogger:
    """
    Create standalone audit logger.

    Returns:
        AuditLogger instance
    """
    return AuditLogger()
