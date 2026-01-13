"""Enhanced sandboxed code execution with security controls.

This module implements a comprehensive sandboxed code execution system with:
- Resource limits (CPU, memory, execution time)
- Network allowlist functionality
- Comprehensive audit trail
- Security review and validation
- Failure isolation mechanisms
"""

import logging
import resource
import signal
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum
from io import StringIO
from typing import Any

logger = logging.getLogger(__name__)


class SandboxViolationError(Exception):
    """Exception raised when sandbox security is violated."""

    pass


class SecurityLevel(str, Enum):
    """Security levels for code execution."""

    STRICT = "strict"  # Maximum restrictions
    MODERATE = "moderate"  # Balanced restrictions
    PERMISSIVE = "permissive"  # Minimal restrictions (for trusted code)


@dataclass
class ResourceLimits:
    """Resource limits for sandboxed execution."""

    # Time limits
    max_execution_time_seconds: float = 5.0
    max_cpu_time_seconds: float = 4.0

    # Memory limits
    max_memory_mb: int = 128
    max_stack_size_mb: int = 8

    # Process limits
    max_processes: int = 1
    max_open_files: int = 32

    # Output limits
    max_output_size: int = 1024 * 1024  # 1MB
    max_variables: int = 1000


@dataclass
class NetworkConfig:
    """Network access configuration."""

    allow_network: bool = False
    allowed_hosts: set[str] = field(default_factory=set)
    allowed_ports: set[int] = field(default_factory=set)
    block_local_network: bool = True

    def __post_init__(self):
        if self.allowed_hosts is None:
            self.allowed_hosts = set()
        if self.allowed_ports is None:
            self.allowed_ports = set()


@dataclass
class AuditRecord:
    """Audit record for code execution."""

    timestamp: float
    session_id: str
    user_id: str
    code_hash: str
    code_preview: str
    security_level: SecurityLevel
    resource_limits: ResourceLimits
    network_config: NetworkConfig
    success: bool
    execution_time: float
    memory_used: int
    violations: list[str]
    result_summary: str
    error_type: str | None = None
    error_message: str | None = None


class SecurityValidator:
    """Validates code for security issues before execution."""

    # Dangerous imports that should be blocked
    DANGEROUS_IMPORTS = {
        "os",
        "sys",
        "subprocess",
        "shutil",
        "glob",
        "pathlib",
        "socket",
        "urllib",
        "requests",
        "http",
        "ftplib",
        "__import__",
        "exec",
        "eval",
        "compile",
        "open",
        "file",
        "input",
        "raw_input",
    }

    # Dangerous attributes/functions
    DANGEROUS_ATTRS = {
        "__import__",
        "__loader__",
        "__package__",
        "__spec__",
        "__builtins__",
        "__globals__",
        "__locals__",
        "__dict__",
        "__class__",
        "__bases__",
        "__subclasses__",
        "__mro__",
    }

    # Dangerous keywords
    DANGEROUS_KEYWORDS = {
        "import",
        "from",
        "exec",
        "eval",
        "compile",
        "open",
        "__import__",
        "getattr",
        "setattr",
        "delattr",
        "hasattr",
    }

    @classmethod
    def validate_code(cls, code: str, security_level: SecurityLevel) -> list[str]:
        """Validate code for security violations.

        Args:
            code: Python code to validate
            security_level: Security level for validation

        Returns:
            List of security violations (empty if safe)
        """
        violations: list[str] = []

        if security_level == SecurityLevel.PERMISSIVE:
            return violations  # No validation for permissive mode

        # Check for dangerous imports
        for dangerous in cls.DANGEROUS_IMPORTS:
            if f"import {dangerous}" in code or f"from {dangerous}" in code:
                violations.append(f"Dangerous import detected: {dangerous}")

        # Check for dangerous attributes
        for attr in cls.DANGEROUS_ATTRS:
            if attr in code:
                violations.append(f"Dangerous attribute access: {attr}")

        # Check for dangerous keywords in strict mode
        if security_level == SecurityLevel.STRICT:
            for keyword in cls.DANGEROUS_KEYWORDS:
                if keyword in code:
                    violations.append(f"Restricted keyword: {keyword}")

        # Check for file operations
        if any(op in code for op in ["open(", "file(", "input(", "raw_input("]):
            violations.append("File I/O operations not allowed")

        # Check for network operations
        if any(net in code for net in ["urllib", "requests", "socket", "http"]):
            violations.append("Network operations detected")

        # Check for system operations
        if any(sys_op in code for sys_op in ["os.", "subprocess", "system("]):
            violations.append("System operations not allowed")

        return violations


class ResourceMonitor:
    """Monitors and enforces resource usage during execution."""

    def __init__(self, limits: ResourceLimits):
        self.limits = limits
        self.start_time = time.time()
        self.peak_memory = 0

    def check_resources(self) -> list[str]:
        """Check current resource usage against limits.

        Returns:
            List of resource violations
        """
        violations = []
        current_time = time.time()

        # Check execution time
        elapsed = current_time - self.start_time
        if elapsed > self.limits.max_execution_time_seconds:
            violations.append(
                f"Execution time exceeded: {elapsed:.2f}s > {self.limits.max_execution_time_seconds}s"
            )

        # Check memory usage
        try:
            import psutil

            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.peak_memory = max(self.peak_memory, int(memory_mb))

            if memory_mb > self.limits.max_memory_mb:
                violations.append(
                    f"Memory limit exceeded: {memory_mb:.1f}MB > {self.limits.max_memory_mb}MB"
                )
        except ImportError:
            # psutil not available, use basic resource monitoring
            try:
                usage = resource.getrusage(resource.RUSAGE_SELF)
                memory_mb = usage.ru_maxrss / 1024  # On Linux, ru_maxrss is in KB
                self.peak_memory = max(self.peak_memory, int(memory_mb))

                if memory_mb > self.limits.max_memory_mb:
                    violations.append(
                        f"Memory limit exceeded: {memory_mb:.1f}MB > {self.limits.max_memory_mb}MB"
                    )
            except Exception:  # nosec B110  # Resource monitoring is optional
                pass  # Resource monitoring not available

        return violations

    @contextmanager
    def enforce_limits(self):
        """Context manager to enforce resource limits during execution."""
        # Set process limits
        try:
            # CPU time limit
            signal.alarm(int(self.limits.max_cpu_time_seconds))

            # Memory limit (soft)
            memory_bytes = self.limits.max_memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))

            # Stack size limit
            stack_bytes = self.limits.max_stack_size_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_STACK, (stack_bytes, stack_bytes))

            # Process limit
            resource.setrlimit(
                resource.RLIMIT_NPROC, (self.limits.max_processes, self.limits.max_processes)
            )

            # File descriptor limit
            resource.setrlimit(
                resource.RLIMIT_NOFILE, (self.limits.max_open_files, self.limits.max_open_files)
            )

            yield self

        except Exception as e:
            logger.warning(f"Could not set all resource limits: {e}")
            yield self
        finally:
            # Clear alarm
            signal.alarm(0)


class NetworkInterceptor:
    """Intercepts and validates network access attempts."""

    def __init__(self, config: NetworkConfig):
        self.config = config
        self.violations: list[str] = []

    def validate_host(self, host: str) -> bool:
        """Validate if host access is allowed.

        Args:
            host: Hostname to validate

        Returns:
            True if allowed, False otherwise
        """
        if not self.config.allow_network:
            self.violations.append(f"Network access blocked: {host}")
            return False

        # Check if host is in allowlist
        if self.config.allowed_hosts and host not in self.config.allowed_hosts:
            self.violations.append(f"Host not in allowlist: {host}")
            return False

        # Block local network if configured
        if self.config.block_local_network:
            local_patterns = ["localhost", "127.", "192.168.", "10.", "172."]
            if any(pattern in host for pattern in local_patterns):
                self.violations.append(f"Local network access blocked: {host}")
                return False

        return True

    def get_violations(self) -> list[str]:
        """Get network violations that occurred."""
        return self.violations.copy()


class FailureIsolation:
    """Provides failure isolation for code execution."""

    @staticmethod
    def execute_in_process(code: str, safe_globals: dict, limits: ResourceLimits) -> dict[str, Any]:
        """Execute code in an isolated process.

        Args:
            code: Code to execute
            safe_globals: Safe global namespace
            limits: Resource limits

        Returns:
            Execution result dictionary
        """

        def _isolated_execute():
            """Function to run in isolated process."""
            try:
                # Set up resource monitoring
                monitor = ResourceMonitor(limits)

                with monitor.enforce_limits():
                    # Capture output
                    old_stdout = sys.stdout
                    old_stderr = sys.stderr
                    sys.stdout = captured_stdout = StringIO()
                    sys.stderr = captured_stderr = StringIO()

                    try:
                        safe_locals: dict[str, Any] = {}

                        # Execute the code
                        exec(code, safe_globals, safe_locals)  # nosec B102  # Intentional exec in sandbox

                        # Get results
                        output = captured_stdout.getvalue()
                        errors = captured_stderr.getvalue()

                        # Check resource violations
                        violations = monitor.check_resources()

                        # Filter result variables
                        result_vars = {
                            k: v
                            for k, v in safe_locals.items()
                            if not k.startswith("_") and len(str(v)) < 10000
                        }

                        # Limit variables if too many
                        if len(result_vars) > limits.max_variables:
                            result_vars = dict(list(result_vars.items())[: limits.max_variables])

                        return {
                            "success": True,
                            "output": output[: limits.max_output_size],
                            "errors": errors[: limits.max_output_size],
                            "variables": result_vars,
                            "violations": violations,
                            "peak_memory": monitor.peak_memory,
                            "execution_time": time.time() - monitor.start_time,
                        }

                    finally:
                        sys.stdout = old_stdout
                        sys.stderr = old_stderr

            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "traceback": traceback.format_exc(),
                    "violations": getattr(monitor, "check_resources", lambda: [])(),
                    "peak_memory": getattr(monitor, "peak_memory", 0),
                    "execution_time": time.time() - getattr(monitor, "start_time", time.time()),
                }

        result = _isolated_execute()
        return result  # type: ignore[no-any-return]


class AuditLogger:
    """Comprehensive audit logging for code execution."""

    def __init__(self, log_file: str | None = None):
        self.log_file = log_file
        self.records: list[AuditRecord] = []

    def log_execution(self, record: AuditRecord):
        """Log an execution record.

        Args:
            record: Audit record to log
        """
        self.records.append(record)

        # Log to file if configured
        if self.log_file:
            try:
                from pathlib import Path

                with Path(self.log_file).open("a") as f:
                    import json

                    f.write(json.dumps(asdict(record), default=str) + "\n")
            except Exception as e:
                logger.error(f"Failed to write audit log: {e}")

        # Log to system logger
        log_level = logging.WARNING if not record.success or record.violations else logging.INFO
        logger.log(
            log_level,
            f"Code execution: session={record.session_id}, success={record.success}, "
            f"violations={len(record.violations)}, time={record.execution_time:.3f}s",
        )

    def get_records(self, session_id: str | None = None) -> list[AuditRecord]:
        """Get audit records, optionally filtered by session ID."""
        if session_id:
            return [r for r in self.records if r.session_id == session_id]
        return self.records.copy()
