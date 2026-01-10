"""
Audit Logging for guardrail events and security monitoring.

Logs all guardrail events including:
- PII detections
- Toxicity violations
- Access attempts
- System events
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.models.guardrails import AuditEvent, AuditEventType, AuditSeverity


class AuditLogger:
    """Manages audit logging for guardrail events."""

    def __init__(
        self,
        log_file: Optional[Path] = None,
        log_to_console: bool = True,
        log_level: str = "INFO",
        structured_logs: bool = True,
    ):
        """
        Initialize audit logger.

        Args:
            log_file: Path to log file. If None, logs to default location.
            log_to_console: Whether to also log to console.
            log_level: Logging level.
            structured_logs: Whether to use structured JSON logging.
        """
        self.log_file = log_file or Path("logs/audit.log")
        self.log_to_console = log_to_console
        self.structured_logs = structured_logs

        # Create log directory if it doesn't exist
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Configure logger
        self.logger = logging.getLogger("guardrails.audit")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        self.logger.handlers.clear()

        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)

        if structured_logs:
            file_handler.setFormatter(
                logging.Formatter("%(message)s")  # JSON will be the message
            )
        else:
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )

        self.logger.addHandler(file_handler)

        # Console handler
        if log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            self.logger.addHandler(console_handler)

    def log_event(self, event: AuditEvent) -> None:
        """
        Log an audit event.

        Args:
            event: AuditEvent to log.
        """
        severity_mapping = {
            AuditSeverity.INFO: logging.INFO,
            AuditSeverity.WARNING: logging.WARNING,
            AuditSeverity.ERROR: logging.ERROR,
            AuditSeverity.CRITICAL: logging.CRITICAL,
        }

        log_level = severity_mapping.get(event.severity, logging.INFO)

        if self.structured_logs:
            message = event.model_dump_json()
        else:
            message = f"[{event.event_type.value}] {event.message or 'No message'}"
            if event.details:
                message += f" | Details: {json.dumps(event.details)}"

        self.logger.log(log_level, message)

    def log_pii_detection(
        self,
        pii_types: list[str],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        redacted: bool = False,
    ) -> None:
        """
        Log PII detection event.

        Args:
            pii_types: Types of PII detected.
            user_id: User identifier.
            session_id: Session identifier.
            redacted: Whether PII was redacted.
        """
        event_type = (
            AuditEventType.PII_REDACTED if redacted else AuditEventType.PII_DETECTED
        )

        event = AuditEvent(
            event_type=event_type,
            severity=AuditSeverity.WARNING,
            timestamp=datetime.now(),
            user_id=user_id,
            session_id=session_id,
            details={"pii_types": pii_types, "count": len(pii_types)},
            message=f"PII detected: {', '.join(pii_types)}",
        )

        self.log_event(event)

    def log_toxicity_detection(
        self,
        toxicity_level: str,
        categories: list[str],
        score: float,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        filtered: bool = False,
    ) -> None:
        """
        Log toxicity detection event.

        Args:
            toxicity_level: Level of toxicity.
            categories: Categories of toxic content.
            score: Toxicity score.
            user_id: User identifier.
            session_id: Session identifier.
            filtered: Whether content was filtered.
        """
        event_type = (
            AuditEventType.TOXICITY_FILTERED
            if filtered
            else AuditEventType.TOXICITY_DETECTED
        )

        # Determine severity based on toxicity level
        severity_map = {
            "low": AuditSeverity.INFO,
            "medium": AuditSeverity.WARNING,
            "high": AuditSeverity.ERROR,
            "severe": AuditSeverity.CRITICAL,
        }
        severity = severity_map.get(toxicity_level.lower(), AuditSeverity.WARNING)

        event = AuditEvent(
            event_type=event_type,
            severity=severity,
            timestamp=datetime.now(),
            user_id=user_id,
            session_id=session_id,
            details={
                "toxicity_level": toxicity_level,
                "categories": categories,
                "score": score,
            },
            message=f"Toxic content detected: {toxicity_level} ({', '.join(categories)})",
        )

        self.log_event(event)

    def log_query(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Log query processing.

        Args:
            query: Query text (should be sanitized/redacted).
            user_id: User identifier.
            session_id: Session identifier.
            metadata: Additional metadata.
        """
        event = AuditEvent(
            event_type=AuditEventType.QUERY_PROCESSED,
            severity=AuditSeverity.INFO,
            timestamp=datetime.now(),
            user_id=user_id,
            session_id=session_id,
            details={
                "query_length": len(query),
                **(metadata or {}),
            },
            message="Query processed",
        )

        self.log_event(event)

    def log_response(
        self,
        response_length: int,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Log response generation.

        Args:
            response_length: Length of generated response.
            user_id: User identifier.
            session_id: Session identifier.
            metadata: Additional metadata.
        """
        event = AuditEvent(
            event_type=AuditEventType.RESPONSE_GENERATED,
            severity=AuditSeverity.INFO,
            timestamp=datetime.now(),
            user_id=user_id,
            session_id=session_id,
            details={
                "response_length": response_length,
                **(metadata or {}),
            },
            message="Response generated",
        )

        self.log_event(event)

    def log_access_denied(
        self,
        reason: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> None:
        """
        Log access denied event.

        Args:
            reason: Reason for denial.
            user_id: User identifier.
            ip_address: IP address.
        """
        event = AuditEvent(
            event_type=AuditEventType.ACCESS_DENIED,
            severity=AuditSeverity.WARNING,
            timestamp=datetime.now(),
            user_id=user_id,
            ip_address=ip_address,
            details={"reason": reason},
            message=f"Access denied: {reason}",
        )

        self.log_event(event)

    def log_error(
        self,
        error_message: str,
        error_type: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        stack_trace: Optional[str] = None,
    ) -> None:
        """
        Log error event.

        Args:
            error_message: Error message.
            error_type: Type of error.
            user_id: User identifier.
            session_id: Session identifier.
            stack_trace: Stack trace if available.
        """
        event = AuditEvent(
            event_type=AuditEventType.ERROR,
            severity=AuditSeverity.ERROR,
            timestamp=datetime.now(),
            user_id=user_id,
            session_id=session_id,
            details={
                "error_type": error_type,
                "stack_trace": stack_trace,
            },
            message=error_message,
        )

        self.log_event(event)

    def get_recent_events(
        self,
        count: int = 100,
        event_type: Optional[AuditEventType] = None,
    ) -> list[dict[str, Any]]:
        """
        Get recent audit events from log file.

        Args:
            count: Number of events to retrieve.
            event_type: Filter by event type.

        Returns:
            List of event dictionaries.
        """
        events: list[dict[str, Any]] = []

        try:
            with open(self.log_file, "r") as f:
                lines = f.readlines()

            # Get last N lines
            recent_lines = lines[-count:]

            for line in recent_lines:
                try:
                    event_data = json.loads(line.strip())
                    if event_type is None or event_data.get("event_type") == event_type.value:
                        events.append(event_data)
                except json.JSONDecodeError:
                    continue

        except FileNotFoundError:
            pass

        return events
