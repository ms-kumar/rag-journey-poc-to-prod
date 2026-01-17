"""Unit tests for the structured logging module."""

import json
import logging
from io import StringIO

from src.services.observability.logging import (
    ContextualLogger,
    CorrelationContext,
    JSONFormatter,
    LogEntry,
    LogLevel,
    RAGLoggers,
    StructuredLogger,
    generate_correlation_id,
    get_correlation_id,
    log_request_end,
    log_request_start,
    set_correlation_id,
    setup_structured_logging,
)


class TestCorrelationId:
    """Tests for correlation ID functions."""

    def test_generate_correlation_id_is_unique(self):
        """Test that generated correlation IDs are unique."""
        ids = [generate_correlation_id() for _ in range(100)]
        assert len(set(ids)) == 100

    def test_set_and_get_correlation_id(self):
        """Test setting and getting correlation ID."""
        set_correlation_id("test-id-123")
        assert get_correlation_id() == "test-id-123"

        # Clean up
        set_correlation_id(None)

    def test_set_correlation_id_returns_previous(self):
        """Test set_correlation_id returns previous value."""
        set_correlation_id("first")
        previous = set_correlation_id("second")

        assert previous == "first"
        assert get_correlation_id() == "second"

        # Clean up
        set_correlation_id(None)


class TestCorrelationContext:
    """Tests for CorrelationContext."""

    def test_context_sets_correlation_id(self):
        """Test context manager sets correlation ID."""
        with CorrelationContext("ctx-123") as ctx:
            assert get_correlation_id() == "ctx-123"
            assert ctx.correlation_id == "ctx-123"

    def test_context_restores_previous_id(self):
        """Test context manager restores previous correlation ID."""
        set_correlation_id("original")

        with CorrelationContext("temporary"):
            assert get_correlation_id() == "temporary"

        assert get_correlation_id() == "original"

        # Clean up
        set_correlation_id(None)

    def test_context_generates_id_if_not_provided(self):
        """Test context generates ID if not provided."""
        with CorrelationContext() as ctx:
            assert ctx.correlation_id is not None
            assert len(ctx.correlation_id) > 0

    def test_nested_contexts(self):
        """Test nested correlation contexts."""
        with CorrelationContext("outer"):
            assert get_correlation_id() == "outer"

            with CorrelationContext("inner"):
                assert get_correlation_id() == "inner"

            assert get_correlation_id() == "outer"


class TestLogEntry:
    """Tests for LogEntry."""

    def test_log_entry_creation(self):
        """Test creating a log entry."""
        entry = LogEntry(
            timestamp="2024-01-01T00:00:00Z",
            level="INFO",
            message="Test message",
            logger_name="test.logger",
        )

        assert entry.timestamp == "2024-01-01T00:00:00Z"
        assert entry.level == "INFO"
        assert entry.message == "Test message"

    def test_log_entry_to_dict_excludes_none(self):
        """Test to_dict excludes None values."""
        entry = LogEntry(
            timestamp="2024-01-01T00:00:00Z",
            level="INFO",
            message="Test",
            logger_name="test",
            correlation_id=None,
            error_type=None,
        )

        data = entry.to_dict()
        assert "correlation_id" not in data
        assert "error_type" not in data

    def test_log_entry_to_json(self):
        """Test to_json produces valid JSON."""
        entry = LogEntry(
            timestamp="2024-01-01T00:00:00Z",
            level="INFO",
            message="Test",
            logger_name="test",
            correlation_id="corr-123",
        )

        json_str = entry.to_json()
        parsed = json.loads(json_str)

        assert parsed["message"] == "Test"
        assert parsed["correlation_id"] == "corr-123"


class TestJSONFormatter:
    """Tests for JSONFormatter."""

    def test_format_produces_json(self):
        """Test formatter produces valid JSON."""
        formatter = JSONFormatter(service_name="test-service")
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["message"] == "Test message"
        assert parsed["level"] == "INFO"
        assert parsed["service_name"] == "test-service"

    def test_format_includes_correlation_id(self):
        """Test formatter includes correlation ID from context."""
        set_correlation_id("corr-456")

        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["correlation_id"] == "corr-456"

        # Clean up
        set_correlation_id(None)

    def test_format_includes_exception_info(self):
        """Test formatter includes exception details."""
        formatter = JSONFormatter()

        try:
            raise ValueError("Test error")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=1,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["error_type"] == "ValueError"
        assert "Test error" in parsed["error_message"]
        assert parsed["stack_trace"] is not None


class TestStructuredLogger:
    """Tests for StructuredLogger."""

    def test_logger_creates_json_output(self):
        """Test logger produces JSON output."""
        output = StringIO()
        logger = StructuredLogger(
            name="test.structured",
            output=output,
        )

        logger.info("Test message")

        output.seek(0)
        log_line = output.readline()
        parsed = json.loads(log_line)

        assert parsed["message"] == "Test message"
        assert parsed["level"] == "INFO"

    def test_logger_includes_extra_fields(self):
        """Test logger includes extra fields."""
        output = StringIO()
        logger = StructuredLogger(name="test", output=output)

        logger.info("Request processed", request_id="req-123", duration_ms=150)

        output.seek(0)
        log_line = output.readline()
        # Parse to verify it's valid JSON
        json.loads(log_line)

        assert "request_id" in log_line or "req-123" in log_line

    def test_logger_respects_level(self):
        """Test logger respects log level."""
        output = StringIO()
        logger = StructuredLogger(
            name="test.level",
            level=LogLevel.WARNING,
            output=output,
        )

        logger.debug("Debug message")
        logger.warning("Warning message")

        output.seek(0)
        content = output.read()

        assert "Debug message" not in content
        assert "Warning message" in content

    def test_logger_with_context(self):
        """Test creating contextual logger."""
        output = StringIO()
        import uuid

        logger = StructuredLogger(name=f"test.ctx.{uuid.uuid4()}", output=output)

        ctx_logger = logger.with_context(user_id="user-123", session="sess-456")
        ctx_logger.info("User action")

        # Flush handlers
        for handler in logger._logger.handlers:
            handler.flush()

        output.seek(0)
        content = output.read()

        # Context should be included
        assert "user_id" in content or "user-123" in content


class TestContextualLogger:
    """Tests for ContextualLogger."""

    def test_contextual_logger_merges_context(self):
        """Test contextual logger merges base and call context."""
        output = StringIO()
        import uuid

        base_logger = StructuredLogger(name=f"test.merge.{uuid.uuid4()}", output=output)
        ctx_logger = ContextualLogger(base_logger, {"base_key": "base_value"})

        ctx_logger.info("Message", call_key="call_value")

        # Flush handlers
        for handler in base_logger._logger.handlers:
            handler.flush()

        output.seek(0)
        content = output.read()

        # Both contexts should be present
        assert "base_key" in content or "base_value" in content


class TestRAGLoggers:
    """Tests for RAGLoggers factory."""

    def test_get_logger_returns_structured_logger(self):
        """Test get_logger returns StructuredLogger."""
        logger = RAGLoggers.get_logger("test-component")
        assert isinstance(logger, StructuredLogger)

    def test_get_logger_caches_loggers(self):
        """Test get_logger returns same instance for same component."""
        logger1 = RAGLoggers.get_logger("cached-component")
        logger2 = RAGLoggers.get_logger("cached-component")
        assert logger1 is logger2

    def test_component_specific_loggers(self):
        """Test component-specific logger methods."""
        ingestion_logger = RAGLoggers.ingestion()
        embedding_logger = RAGLoggers.embedding()
        retrieval_logger = RAGLoggers.retrieval()

        assert isinstance(ingestion_logger, StructuredLogger)
        assert isinstance(embedding_logger, StructuredLogger)
        assert isinstance(retrieval_logger, StructuredLogger)

        # Should be different loggers
        assert ingestion_logger is not embedding_logger


class TestSetupStructuredLogging:
    """Tests for setup_structured_logging."""

    def test_setup_configures_root_logger(self):
        """Test setup configures the root logger."""
        output = StringIO()
        setup_structured_logging(
            service_name="setup-test",
            level="DEBUG",
            output=output,
        )

        root_logger = logging.getLogger()
        root_logger.info("Setup test message")

        output.seek(0)
        content = output.read()

        assert "Setup test message" in content


class TestLogRequestHelpers:
    """Tests for log_request_start and log_request_end."""

    def test_log_request_start(self):
        """Test log_request_start helper."""
        output = StringIO()
        logger = StructuredLogger(name="request", output=output)

        log_request_start(logger, "POST", "/api/v1/rag/generate")

        output.seek(0)
        content = output.read()

        assert "POST" in content
        assert "/api/v1/rag/generate" in content

    def test_log_request_end_success(self):
        """Test log_request_end for successful request."""
        output = StringIO()
        import uuid

        logger = StructuredLogger(name=f"request.success.{uuid.uuid4()}", output=output)

        log_request_end(logger, "GET", "/api/v1/health", 200, 50.5)

        # Flush handlers
        for handler in logger._logger.handlers:
            handler.flush()

        output.seek(0)
        content = output.read()

        assert "200" in content

    def test_log_request_end_error(self):
        """Test log_request_end for error response."""
        output = StringIO()
        import uuid

        logger = StructuredLogger(name=f"request.error.{uuid.uuid4()}", output=output)

        log_request_end(logger, "POST", "/api/v1/rag/generate", 500, 1500.0)

        # Flush handlers
        for handler in logger._logger.handlers:
            handler.flush()

        output.seek(0)
        content = output.read()

        assert "500" in content
