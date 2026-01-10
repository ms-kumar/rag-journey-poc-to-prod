"""
Pydantic models for guardrails system.

Centralized models for PII detection, toxicity filtering, audit logging, and safe responses.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# ============================================================================
# PII Models
# ============================================================================


class PIIType(str, Enum):
    """Types of PII that can be detected."""

    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    NAME = "name"
    ADDRESS = "address"
    DATE_OF_BIRTH = "dob"


class PIIMatch(BaseModel):
    """Represents a detected PII match."""

    pii_type: PIIType
    value: str
    start: int
    end: int
    confidence: float = 1.0


# ============================================================================
# Toxicity Models
# ============================================================================


class ToxicityLevel(str, Enum):
    """Levels of toxicity."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    SEVERE = "severe"


class ToxicityCategory(str, Enum):
    """Categories of toxic content."""

    PROFANITY = "profanity"
    HATE_SPEECH = "hate_speech"
    THREAT = "threat"
    HARASSMENT = "harassment"
    SEXUAL = "sexual"
    VIOLENCE = "violence"
    SELF_HARM = "self_harm"


class ToxicityMatch(BaseModel):
    """Represents a detected toxic content match."""

    category: ToxicityCategory
    level: ToxicityLevel
    matched_text: str
    start: int
    end: int
    confidence: float


class ToxicityScore(BaseModel):
    """Overall toxicity assessment of text."""

    is_toxic: bool
    max_level: ToxicityLevel
    overall_score: float
    matches: list[ToxicityMatch]
    categories: set[ToxicityCategory]


# ============================================================================
# Audit Models
# ============================================================================


class AuditEventType(str, Enum):
    """Types of audit events."""

    PII_DETECTED = "pii_detected"
    PII_REDACTED = "pii_redacted"
    TOXICITY_DETECTED = "toxicity_detected"
    TOXICITY_FILTERED = "toxicity_filtered"
    JAILBREAK_DETECTED = "jailbreak_detected"
    PROMPT_INJECTION_DETECTED = "prompt_injection_detected"
    UNSAFE_QUERY = "unsafe_query"
    QUERY_PROCESSED = "query_processed"
    RESPONSE_GENERATED = "response_generated"
    ACCESS_DENIED = "access_denied"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    ERROR = "error"
    SYSTEM_EVENT = "system_event"


class AuditSeverity(str, Enum):
    """Severity levels for audit events."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditEvent(BaseModel):
    """Represents an audit event."""

    event_type: AuditEventType
    severity: AuditSeverity
    timestamp: datetime
    user_id: str | None = None
    session_id: str | None = None
    ip_address: str | None = None
    details: dict[str, Any] | None = None
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for compatibility."""
        data = self.model_dump()
        data["timestamp"] = self.timestamp.isoformat()
        return data

    def to_json(self) -> str:
        """Convert to JSON string for compatibility."""
        return self.model_dump_json()


# ============================================================================
# Response Models
# ============================================================================


class ResponseType(str, Enum):
    """Types of safe responses."""

    PII_DETECTED = "pii_detected"
    TOXIC_CONTENT = "toxic_content"
    UNSAFE_QUERY = "unsafe_query"
    RATE_LIMIT = "rate_limit"
    ERROR = "error"
    UNAUTHORIZED = "unauthorized"
    CONTENT_POLICY_VIOLATION = "content_policy_violation"


# ============================================================================
# Coordinator Models
# ============================================================================


class GuardrailResult(BaseModel):
    """Result of guardrail checks."""

    is_safe: bool
    original_text: str
    processed_text: str | None = None
    pii_detected: bool = False
    pii_types: list[str] = Field(default_factory=list)
    toxicity_detected: bool = False
    toxicity_level: str | None = None
    toxicity_score: float = 0.0
    safe_response: str | None = None
    violations: list[str] = Field(default_factory=list)
