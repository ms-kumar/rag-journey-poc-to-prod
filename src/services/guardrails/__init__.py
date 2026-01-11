"""
Guardrails module for safety and compliance in RAG system.

This module provides:
- PII detection and redaction
- Toxicity filtering
- Safe response templates
- Audit logging
- Coordinated guardrail checks
"""

from src.schemas.services.guardrails import (
    AuditEvent,
    AuditEventType,
    AuditSeverity,
    GuardrailResult,
    PIIMatch,
    PIIType,
    ResponseType,
    ToxicityCategory,
    ToxicityLevel,
    ToxicityMatch,
    ToxicityScore,
)

from .audit_log import AuditLogger
from .client import GuardrailsClient
from .coordinator import GuardrailsCoordinator
from .factory import (
    make_audit_logger,
    make_guardrails_client,
    make_jailbreak_detector,
    make_pii_detector,
    make_pii_redactor,
    make_toxicity_filter,
)
from .pii_detector import PIIDetector, PIIRedactor
from .safe_response import ResponseBuilder, SafeResponseTemplate
from .toxicity_filter import ToxicityFilter

__all__ = [
    # Client & Factory
    "GuardrailsClient",
    "make_guardrails_client",
    "make_pii_detector",
    "make_pii_redactor",
    "make_toxicity_filter",
    "make_jailbreak_detector",
    "make_audit_logger",
    # Legacy (for backward compatibility)
    "GuardrailsCoordinator",
    # PII
    "PIIDetector",
    "PIIRedactor",
    "PIIType",
    "PIIMatch",
    # Toxicity
    "ToxicityFilter",
    "ToxicityLevel",
    "ToxicityCategory",
    "ToxicityMatch",
    "ToxicityScore",
    # Safe Response
    "SafeResponseTemplate",
    "ResponseType",
    "ResponseBuilder",
    # Audit
    "AuditLogger",
    "AuditEvent",
    "AuditEventType",
    "AuditSeverity",
    # Coordinator
    "GuardrailsCoordinator",
    "GuardrailResult",
]
