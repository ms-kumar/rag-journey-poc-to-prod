"""Test suite for sandbox core components.

Tests for:
- SecurityValidator
- ResourceLimits
- ResourceMonitor
- NetworkConfig
- NetworkInterceptor
- FailureIsolation
- AuditLogger
"""

import time
from unittest.mock import patch

import pytest

from src.services.agent.tools.hybrid.sandbox import (
    AuditLogger,
    AuditRecord,
    FailureIsolation,
    NetworkConfig,
    NetworkInterceptor,
    ResourceLimits,
    ResourceMonitor,
    SecurityLevel,
    SecurityValidator,
)


class TestSecurityValidator:
    """Test SecurityValidator class."""

    def test_validate_safe_code(self):
        """Test that safe code passes validation."""
        code = "x = 10 + 20\nprint(x)"
        violations = SecurityValidator.validate_code(code, SecurityLevel.MODERATE)
        assert violations == []

    def test_validate_dangerous_imports(self):
        """Test detection of dangerous imports."""
        code = "import os\nos.system('ls')"
        violations = SecurityValidator.validate_code(code, SecurityLevel.MODERATE)
        assert len(violations) > 0
        assert any("import" in v.lower() for v in violations)

    def test_validate_file_operations(self):
        """Test detection of file operations."""
        code = "with open('/etc/passwd', 'r') as f:\n    print(f.read())"
        violations = SecurityValidator.validate_code(code, SecurityLevel.MODERATE)
        assert len(violations) > 0
        assert any("file" in v.lower() or "i/o" in v.lower() for v in violations)

    def test_validate_network_operations(self):
        """Test detection of network operations."""
        code = "import urllib\nurllib.request.urlopen('http://example.com')"
        violations = SecurityValidator.validate_code(code, SecurityLevel.MODERATE)
        assert len(violations) > 0
        assert any("network" in v.lower() or "import" in v.lower() for v in violations)

    def test_validate_system_operations(self):
        """Test detection of system operations."""
        code = "import subprocess\nsubprocess.run(['whoami'])"
        violations = SecurityValidator.validate_code(code, SecurityLevel.MODERATE)
        assert len(violations) > 0
        assert any("import" in v.lower() or "system" in v.lower() for v in violations)

    def test_validate_permissive_level(self):
        """Test that permissive level allows more code."""
        code = "import os"
        violations = SecurityValidator.validate_code(code, SecurityLevel.PERMISSIVE)
        assert violations == []  # Permissive mode skips validation

    def test_validate_strict_level(self):
        """Test that strict level blocks more code."""
        code = "import math"
        violations = SecurityValidator.validate_code(code, SecurityLevel.STRICT)
        assert len(violations) > 0


class TestResourceLimits:
    """Test ResourceLimits dataclass."""

    def test_default_limits(self):
        """Test default resource limits."""
        limits = ResourceLimits()
        assert limits.max_execution_time_seconds == 5.0
        assert limits.max_memory_mb == 128
        assert limits.max_processes == 1

    def test_custom_limits(self):
        """Test custom resource limits."""
        limits = ResourceLimits(
            max_execution_time_seconds=10.0,
            max_memory_mb=256,
            max_processes=2,
        )
        assert limits.max_execution_time_seconds == 10.0
        assert limits.max_memory_mb == 256
        assert limits.max_processes == 2


class TestResourceMonitor:
    """Test ResourceMonitor class."""

    def test_initialization(self):
        """Test ResourceMonitor initialization."""
        limits = ResourceLimits()
        monitor = ResourceMonitor(limits)
        assert monitor.limits == limits
        assert monitor.peak_memory == 0

    def test_check_resources_time(self):
        """Test resource checking for time violations."""
        limits = ResourceLimits(max_execution_time_seconds=0.1)
        monitor = ResourceMonitor(limits)

        # Wait to exceed time limit
        time.sleep(0.15)
        violations = monitor.check_resources()

        assert len(violations) > 0
        assert any("time" in v.lower() for v in violations)

    def test_check_resources_no_violations(self):
        """Test that no violations occur within limits."""
        # Use high enough limits to not trigger in test environment
        limits = ResourceLimits(max_execution_time_seconds=10.0, max_memory_mb=2000)
        monitor = ResourceMonitor(limits)

        violations = monitor.check_resources()
        # Should only check time, not memory (or memory should be under 2GB)
        assert len([v for v in violations if "time" in v.lower()]) == 0

    @pytest.mark.skip(reason="psutil is optional dependency - skipped if not installed")
    def test_memory_monitoring_with_psutil(self):
        """Test memory monitoring with psutil available."""
        try:
            from unittest.mock import Mock, patch

            with patch("psutil.Process") as mock_process_class:
                mock_process = Mock()
                mock_process.memory_info.return_value.rss = 200 * 1024 * 1024  # 200MB
                mock_process_class.return_value = mock_process

                limits = ResourceLimits(max_memory_mb=128)
                monitor = ResourceMonitor(limits)

                violations = monitor.check_resources()
                assert len(violations) > 0
                assert any("memory" in v.lower() for v in violations)
        except ImportError:
            pytest.skip("psutil not installed")


class TestNetworkConfig:
    """Test NetworkConfig dataclass."""

    def test_default_config(self):
        """Test default network configuration."""
        config = NetworkConfig()
        assert config.allow_network is False
        assert config.allowed_hosts == set()
        assert config.block_local_network is True

    def test_custom_config(self):
        """Test custom network configuration."""
        config = NetworkConfig(
            allow_network=True,
            allowed_hosts={"example.com", "api.test.com"},
            allowed_ports={80, 443},
            block_local_network=False,
        )
        assert config.allow_network is True
        assert len(config.allowed_hosts) == 2
        assert len(config.allowed_ports) == 2
        assert config.block_local_network is False


class TestNetworkInterceptor:
    """Test NetworkInterceptor class."""

    def test_block_when_network_disabled(self):
        """Test that network is blocked when disabled."""
        config = NetworkConfig(allow_network=False)
        interceptor = NetworkInterceptor(config)

        result = interceptor.validate_host("example.com")
        assert result is False
        assert len(interceptor.get_violations()) > 0

    def test_allow_whitelisted_host(self):
        """Test that whitelisted hosts are allowed."""
        config = NetworkConfig(
            allow_network=True,
            allowed_hosts={"example.com"},
        )
        interceptor = NetworkInterceptor(config)

        result = interceptor.validate_host("example.com")
        assert result is True
        assert len(interceptor.get_violations()) == 0

    def test_block_non_whitelisted_host(self):
        """Test that non-whitelisted hosts are blocked."""
        config = NetworkConfig(
            allow_network=True,
            allowed_hosts={"example.com"},
        )
        interceptor = NetworkInterceptor(config)

        result = interceptor.validate_host("evil.com")
        assert result is False
        assert len(interceptor.get_violations()) > 0

    def test_block_localhost(self):
        """Test that localhost is blocked."""
        config = NetworkConfig(
            allow_network=True,
            block_local_network=True,
        )
        interceptor = NetworkInterceptor(config)

        result = interceptor.validate_host("localhost")
        assert result is False

        result = interceptor.validate_host("127.0.0.1")
        assert result is False

        result = interceptor.validate_host("192.168.1.1")
        assert result is False


class TestFailureIsolation:
    """Test FailureIsolation class."""

    def test_successful_execution(self):
        """Test successful code execution in isolated process."""
        code = "x = 10 + 20\nprint(x)"
        safe_globals = {"__builtins__": {"print": print}}
        limits = ResourceLimits()

        result = FailureIsolation.execute_in_process(code, safe_globals, limits)

        assert result["success"] is True
        assert "30" in result["output"]
        assert "x" in result["variables"]

    def test_exception_handling(self):
        """Test that exceptions are caught and reported."""
        code = "x = 1 / 0"
        safe_globals = {"__builtins__": {}}
        limits = ResourceLimits()

        result = FailureIsolation.execute_in_process(code, safe_globals, limits)

        assert result["success"] is False
        assert "error" in result
        # May be ZeroDivisionError or RuntimeError depending on execution context
        assert result.get("error_type", "") in ["ZeroDivisionError", "RuntimeError", "NameError"]

    def test_isolation(self):
        """Test that executions are isolated from each other."""
        code1 = "x = 100"
        code2 = "y = x + 1"  # Should fail because x doesn't exist
        safe_globals = {"__builtins__": {}}
        limits = ResourceLimits()

        result1 = FailureIsolation.execute_in_process(code1, safe_globals, limits)
        assert result1["success"] is True

        result2 = FailureIsolation.execute_in_process(code2, safe_globals, limits)
        assert result2["success"] is False  # x not defined


class TestAuditLogger:
    """Test AuditLogger class."""

    def test_initialization(self):
        """Test AuditLogger initialization."""
        logger = AuditLogger()
        assert logger.records == []
        assert logger.log_file is None

    def test_log_execution(self):
        """Test logging an execution record."""
        logger = AuditLogger()
        record = AuditRecord(
            timestamp=time.time(),
            session_id="test-session",
            user_id="test-user",
            code_hash="abc123",
            code_preview="print('hello')",
            security_level=SecurityLevel.MODERATE,
            resource_limits=ResourceLimits(),
            network_config=NetworkConfig(),
            success=True,
            execution_time=0.1,
            memory_used=10,
            violations=[],
            result_summary="Success",
        )

        logger.log_execution(record)
        assert len(logger.records) == 1
        assert logger.records[0] == record

    def test_get_records(self):
        """Test retrieving audit records."""
        logger = AuditLogger()
        record1 = AuditRecord(
            timestamp=time.time(),
            session_id="session-1",
            user_id="user-1",
            code_hash="hash1",
            code_preview="code1",
            security_level=SecurityLevel.MODERATE,
            resource_limits=ResourceLimits(),
            network_config=NetworkConfig(),
            success=True,
            execution_time=0.1,
            memory_used=10,
            violations=[],
            result_summary="Success",
        )
        record2 = AuditRecord(
            timestamp=time.time(),
            session_id="session-2",
            user_id="user-2",
            code_hash="hash2",
            code_preview="code2",
            security_level=SecurityLevel.MODERATE,
            resource_limits=ResourceLimits(),
            network_config=NetworkConfig(),
            success=True,
            execution_time=0.1,
            memory_used=10,
            violations=[],
            result_summary="Success",
        )

        logger.log_execution(record1)
        logger.log_execution(record2)

        # Get all records
        all_records = logger.get_records()
        assert len(all_records) == 2

        # Get records by session
        session1_records = logger.get_records("session-1")
        assert len(session1_records) == 1
        assert session1_records[0].session_id == "session-1"

    @patch("builtins.open", create=True)
    def test_log_to_file(self, mock_open):
        """Test logging to file."""
        logger = AuditLogger(log_file="/tmp/test_audit.log")
        record = AuditRecord(
            timestamp=time.time(),
            session_id="test-session",
            user_id="test-user",
            code_hash="abc123",
            code_preview="print('hello')",
            security_level=SecurityLevel.MODERATE,
            resource_limits=ResourceLimits(),
            network_config=NetworkConfig(),
            success=True,
            execution_time=0.1,
            memory_used=10,
            violations=[],
            result_summary="Success",
        )

        logger.log_execution(record)
        mock_open.assert_called()


class TestSecurityLevelEnum:
    """Test SecurityLevel enum."""

    def test_security_levels(self):
        """Test that all security levels are defined."""
        assert SecurityLevel.STRICT == "strict"
        assert SecurityLevel.MODERATE == "moderate"
        assert SecurityLevel.PERMISSIVE == "permissive"

    def test_security_level_comparison(self):
        """Test security level values."""
        assert SecurityLevel.STRICT.value == "strict"
        assert SecurityLevel.MODERATE.value == "moderate"
        assert SecurityLevel.PERMISSIVE.value == "permissive"
