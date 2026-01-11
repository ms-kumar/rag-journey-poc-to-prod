"""Pydantic schemas for request/response validation.

This package organizes schemas by domain following best practices:
- api/: API endpoint request and response models
- services/: Service-specific schemas (guardrails, token budgets, etc.)
"""

from src.schemas.api import (
    AskRequest,
    AskResponse,
    DetailedHealthResponse,
    GenerateRequest,
    GenerateResponse,
    HealthCheckResponse,
    IngestRequest,
    IngestResponse,
    ServiceStatus,
)
from src.schemas.services import (
    AuditEvent,
    AuditEventType,
    AuditSeverity,
    GuardrailResult,
    PIIMatch,
    PIIType,
    ResponseType,
    TokenBudget,
    ToxicityCategory,
    ToxicityLevel,
    ToxicityMatch,
    ToxicityScore,
)

__all__ = [
    # API schemas
    "AskRequest",
    "AskResponse",
    "HealthCheckResponse",
    "DetailedHealthResponse",
    "ServiceStatus",
    "IngestRequest",
    "IngestResponse",
    "GenerateRequest",
    "GenerateResponse",
    # Service schemas
    "AuditEvent",
    "AuditEventType",
    "AuditSeverity",
    "GuardrailResult",
    "PIIMatch",
    "PIIType",
    "ResponseType",
    "TokenBudget",
    "ToxicityCategory",
    "ToxicityLevel",
    "ToxicityMatch",
    "ToxicityScore",
]
