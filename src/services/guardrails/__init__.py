"""
Guardrails module for safety and compliance in RAG system.

This module provides:
- PII detection and redaction
- Toxicity filtering
- Safe response templates
- Audit logging
- Coordinated guardrail checks
"""

from src.models.guardrails import (
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
from .coordinator import GuardrailsCoordinator
from .pii_detector import PIIDetector, PIIRedactor
from .safe_response import ResponseBuilder, SafeResponseTemplate
from .toxicity_filter import ToxicityFilter

__all__ = [
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
