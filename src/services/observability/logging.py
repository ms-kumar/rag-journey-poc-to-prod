"""
Structured logging module for RAG pipeline.

Provides JSON-structured logging with correlation ID propagation
for request tracing across all pipeline components.

Features:
- Correlation ID propagation via context
- Structured JSON log format
- Log level configuration
- Request context enrichment
- Log filtering and sampling
"""

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, TextIO

# Context variable for correlation ID propagation
_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)


def get_correlation_id() -> str | None:
    """Get the current correlation ID from context."""
    return _correlation_id.get()


def set_correlation_id(correlation_id: str | None) -> str | None:
    """
    Set the correlation ID in context.

    Returns:
        The previous correlation ID
    """
    old = _correlation_id.get()
    _correlation_id.set(correlation_id)
    return old


def generate_correlation_id() -> str:
    """Generate a new correlation ID."""
    return str(uuid.uuid4())


class CorrelationContext:
    """
    Context manager for correlation ID scope.

    Usage:
        with CorrelationContext() as ctx:
            logger.info("Processing request")  # Includes correlation_id
            process_request()
    """

    def __init__(self, correlation_id: str | None = None):
        self.correlation_id = correlation_id or generate_correlation_id()
        self._previous_id: str | None = None

    def __enter__(self) -> "CorrelationContext":
        self._previous_id = set_correlation_id(self.correlation_id)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        set_correlation_id(self._previous_id)


class LogLevel(Enum):
    """Log levels matching Python logging."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogEntry:
    """
    Structured log entry.

    Contains all fields for a structured log message,
    including context, timing, and request metadata.
    """

    timestamp: str
    level: str
    message: str
    logger_name: str
    correlation_id: str | None = None
    trace_id: str | None = None
    span_id: str | None = None
    service_name: str = "rag-pipeline"
    environment: str = "development"
    # Request context
    request_path: str | None = None
    request_method: str | None = None
    user_id: str | None = None
    # Performance
    duration_ms: float | None = None
    # Error details
    error_type: str | None = None
    error_message: str | None = None
    stack_trace: str | None = None
    # Custom fields
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = asdict(self)
        return {k: v for k, v in result.items() if v is not None}

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class JSONFormatter(logging.Formatter):
    """
    JSON log formatter for structured logging.

    Formats log records as JSON with correlation ID and context.
    """

    def __init__(
        self,
        service_name: str = "rag-pipeline",
        environment: str = "development",
        include_extra: bool = True,
    ):
        super().__init__()
        self.service_name = service_name
        self.environment = environment
        self.include_extra = include_extra

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Get correlation ID from context
        correlation_id = get_correlation_id()

        # Build log entry
        entry = LogEntry(
            timestamp=datetime.now(UTC).isoformat(),
            level=record.levelname,
            message=record.getMessage(),
            logger_name=record.name,
            correlation_id=correlation_id,
            service_name=self.service_name,
            environment=self.environment,
        )

        # Add exception info if present
        if record.exc_info:
            import traceback

            entry.error_type = record.exc_info[0].__name__ if record.exc_info[0] else None
            entry.error_message = str(record.exc_info[1]) if record.exc_info[1] else None
            entry.stack_trace = "".join(traceback.format_exception(*record.exc_info))

        # Add extra fields from record
        if self.include_extra and hasattr(record, "__dict__"):
            standard_attrs = {
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "exc_info",
                "exc_text",
                "thread",
                "threadName",
                "taskName",
                "message",
            }
            extra = {k: v for k, v in record.__dict__.items() if k not in standard_attrs}
            if extra:
                entry.extra = extra

        # Extract trace context if present
        if hasattr(record, "trace_id"):
            entry.trace_id = record.trace_id
        if hasattr(record, "span_id"):
            entry.span_id = record.span_id
        if hasattr(record, "duration_ms"):
            entry.duration_ms = record.duration_ms

        return entry.to_json()


class StructuredLogger:
    """
    Structured logger wrapper with correlation ID support.

    Provides a convenient interface for structured logging
    with automatic correlation ID injection.

    Usage:
        logger = StructuredLogger("my-service")

        with CorrelationContext() as ctx:
            logger.info("Processing request", request_id="123")
            logger.error("Failed", error_code=500)
    """

    def __init__(
        self,
        name: str,
        level: str | LogLevel = LogLevel.INFO,
        service_name: str = "rag-pipeline",
        environment: str = "development",
        output: TextIO | None = None,
    ):
        self.name = name
        self.service_name = service_name
        self.environment = environment

        # Create underlying logger
        self._logger = logging.getLogger(name)
        level_value = level.value if isinstance(level, LogLevel) else level
        self._logger.setLevel(getattr(logging, level_value.upper()))

        # Add JSON handler if not already present
        if not any(isinstance(h, logging.StreamHandler) for h in self._logger.handlers):
            handler = logging.StreamHandler(output or sys.stdout)
            handler.setFormatter(
                JSONFormatter(
                    service_name=service_name,
                    environment=environment,
                )
            )
            self._logger.addHandler(handler)

    def _log(self, level: int, message: str, **kwargs: Any) -> None:
        """Internal log method with extra context."""
        extra = {
            "correlation_id": get_correlation_id(),
            **kwargs,
        }
        self._logger.log(level, message, extra=extra)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, exc_info: bool = False, **kwargs: Any) -> None:
        """Log error message."""
        self._logger.error(message, exc_info=exc_info, extra=kwargs)

    def critical(self, message: str, exc_info: bool = False, **kwargs: Any) -> None:
        """Log critical message."""
        self._logger.critical(message, exc_info=exc_info, extra=kwargs)

    def with_context(self, **context: Any) -> "ContextualLogger":
        """Create a logger with additional context."""
        return ContextualLogger(self, context)


class ContextualLogger:
    """Logger with pre-bound context."""

    def __init__(self, logger: StructuredLogger, context: dict[str, Any]):
        self._logger = logger
        self._context = context

    def debug(self, message: str, **kwargs: Any) -> None:
        self._logger.debug(message, **{**self._context, **kwargs})

    def info(self, message: str, **kwargs: Any) -> None:
        self._logger.info(message, **{**self._context, **kwargs})

    def warning(self, message: str, **kwargs: Any) -> None:
        self._logger.warning(message, **{**self._context, **kwargs})

    def error(self, message: str, exc_info: bool = False, **kwargs: Any) -> None:
        self._logger.error(message, exc_info=exc_info, **{**self._context, **kwargs})

    def critical(self, message: str, exc_info: bool = False, **kwargs: Any) -> None:
        self._logger.critical(message, exc_info=exc_info, **{**self._context, **kwargs})


# Pre-configured loggers for RAG pipeline components
class RAGLoggers:
    """Factory for component-specific loggers."""

    _loggers: dict[str, StructuredLogger] = {}

    @classmethod
    def get_logger(
        cls,
        component: str,
        service_name: str = "rag-pipeline",
        environment: str = "development",
    ) -> StructuredLogger:
        """Get or create a logger for a component."""
        key = f"{service_name}.{component}"
        if key not in cls._loggers:
            cls._loggers[key] = StructuredLogger(
                name=f"rag.{component}",
                service_name=service_name,
                environment=environment,
            )
        return cls._loggers[key]

    @classmethod
    def ingestion(cls) -> StructuredLogger:
        return cls.get_logger("ingestion")

    @classmethod
    def embedding(cls) -> StructuredLogger:
        return cls.get_logger("embedding")

    @classmethod
    def retrieval(cls) -> StructuredLogger:
        return cls.get_logger("retrieval")

    @classmethod
    def reranking(cls) -> StructuredLogger:
        return cls.get_logger("reranking")

    @classmethod
    def generation(cls) -> StructuredLogger:
        return cls.get_logger("generation")

    @classmethod
    def agent(cls) -> StructuredLogger:
        return cls.get_logger("agent")

    @classmethod
    def guardrails(cls) -> StructuredLogger:
        return cls.get_logger("guardrails")

    @classmethod
    def cache(cls) -> StructuredLogger:
        return cls.get_logger("cache")


def setup_structured_logging(
    service_name: str = "rag-pipeline",
    environment: str = "development",
    level: str = "INFO",
    output: TextIO | None = None,
) -> None:
    """
    Configure the root logger for structured JSON output.

    This sets up the Python root logger to output JSON-formatted
    logs with correlation ID support.

    Args:
        service_name: Name of the service
        environment: Environment name (development, staging, production)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        output: Output stream (defaults to stdout)
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add JSON handler
    handler = logging.StreamHandler(output or sys.stdout)
    handler.setFormatter(
        JSONFormatter(
            service_name=service_name,
            environment=environment,
        )
    )
    root_logger.addHandler(handler)


def log_request_start(
    logger: StructuredLogger,
    method: str,
    path: str,
    correlation_id: str | None = None,
    **kwargs: Any,
) -> None:
    """Log the start of a request."""
    logger.info(
        f"Request started: {method} {path}",
        request_method=method,
        request_path=path,
        correlation_id=correlation_id or get_correlation_id(),
        **kwargs,
    )


def log_request_end(
    logger: StructuredLogger,
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    **kwargs: Any,
) -> None:
    """Log the end of a request."""
    level = "info" if status_code < 400 else "error"
    getattr(logger, level)(
        f"Request completed: {method} {path} -> {status_code}",
        request_method=method,
        request_path=path,
        status_code=status_code,
        duration_ms=duration_ms,
        **kwargs,
    )
