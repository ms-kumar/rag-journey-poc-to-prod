"""Main sandboxed code executor implementation.

This module provides the main SandboxedCodeExecutor class that integrates
all security components for safe code execution.
"""

import hashlib
import logging
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any

from .sandbox import (
    AuditLogger,
    AuditRecord,
    FailureIsolation,
    NetworkConfig,
    ResourceLimits,
    SecurityLevel,
    SecurityValidator,
)

logger = logging.getLogger(__name__)


class SandboxedCodeExecutor:
    """Main class for executing Python code in a secure sandbox.

    This class integrates all security components:
    - Resource limits and monitoring
    - Network access control
    - Security validation
    - Comprehensive audit logging
    - Process isolation
    """

    def __init__(
        self,
        security_level: SecurityLevel = SecurityLevel.MODERATE,
        resource_limits: ResourceLimits | None = None,
        network_config: NetworkConfig | None = None,
        audit_log_file: str | None = None,
        max_workers: int = 2,
    ):
        """Initialize the sandboxed code executor.

        Args:
            security_level: Security level for code validation
            resource_limits: Resource limits for execution
            network_config: Network access configuration
            audit_log_file: Optional file path for audit logging
            max_workers: Maximum concurrent executions
        """
        self.security_level = security_level
        self.resource_limits = resource_limits or ResourceLimits()
        self.network_config = network_config or NetworkConfig()

        # Initialize components
        self.audit_logger = AuditLogger(audit_log_file)
        self.process_pool = ProcessPoolExecutor(max_workers=max_workers)

        # Session tracking
        self.session_id = str(uuid.uuid4())

        logger.info(
            f"SandboxedCodeExecutor initialized: "
            f"security_level={security_level}, session={self.session_id}"
        )

    def _get_safe_globals(self) -> dict[str, Any]:
        """Get safe global namespace for code execution.

        Returns:
            Dictionary of safe globals based on security level
        """
        if self.security_level == SecurityLevel.PERMISSIVE:
            # More permissive globals (but still restricted)
            safe_globals = {
                "__builtins__": {
                    "abs": abs,
                    "all": all,
                    "any": any,
                    "bool": bool,
                    "dict": dict,
                    "enumerate": enumerate,
                    "filter": filter,
                    "float": float,
                    "int": int,
                    "len": len,
                    "list": list,
                    "map": map,
                    "max": max,
                    "min": min,
                    "print": print,
                    "range": range,
                    "round": round,
                    "set": set,
                    "sorted": sorted,
                    "str": str,
                    "sum": sum,
                    "tuple": tuple,
                    "zip": zip,
                    "type": type,
                    "isinstance": isinstance,
                    "hasattr": hasattr,
                }
            }

            # Add some safe modules
            import datetime
            import json
            import math
            import random

            safe_globals.update(
                {
                    "math": math,  # type: ignore[dict-item]
                    "random": random,  # type: ignore[dict-item]
                    "datetime": datetime,  # type: ignore[dict-item]
                    "json": json,  # type: ignore[dict-item]
                }
            )

        elif self.security_level == SecurityLevel.MODERATE:
            # Moderate restrictions
            safe_globals = {
                "__builtins__": {
                    "abs": abs,
                    "all": all,
                    "any": any,
                    "bool": bool,
                    "dict": dict,
                    "enumerate": enumerate,
                    "filter": filter,
                    "float": float,
                    "int": int,
                    "len": len,
                    "list": list,
                    "map": map,
                    "max": max,
                    "min": min,
                    "print": print,
                    "range": range,
                    "round": round,
                    "set": set,
                    "sorted": sorted,
                    "str": str,
                    "sum": sum,
                    "tuple": tuple,
                    "zip": zip,
                }
            }

            # Add safe math module only
            import math

            safe_globals["math"] = math  # type: ignore[assignment]

        else:  # STRICT
            # Minimal globals for strict security
            safe_globals = {
                "__builtins__": {
                    "abs": abs,
                    "all": all,
                    "any": any,
                    "bool": bool,
                    "dict": dict,
                    "enumerate": enumerate,
                    "float": float,
                    "int": int,
                    "len": len,
                    "list": list,
                    "max": max,
                    "min": min,
                    "print": print,
                    "range": range,
                    "round": round,
                    "set": set,
                    "sorted": sorted,
                    "str": str,
                    "sum": sum,
                    "tuple": tuple,
                    "zip": zip,
                }
            }

        return safe_globals

    def _extract_code_from_query(self, query: str) -> str:
        """Extract Python code from a query string.

        Args:
            query: User query that may contain code

        Returns:
            Extracted Python code
        """
        # Handle markdown code blocks
        if "```python" in query:
            parts = query.split("```python")
            if len(parts) > 1:
                code = parts[1].split("```")[0].strip()
                return code
        elif "```" in query:
            parts = query.split("```")
            if len(parts) >= 3:
                code = parts[1].strip()
                return code

        # Check if it looks like direct code
        code_indicators = ["=", "def ", "for ", "if ", "while ", "import ", "print("]
        if any(indicator in query for indicator in code_indicators):
            return query.strip()

        # Otherwise, assume it's a request to generate code
        return f"# Request: {query}\\nprint('No executable code found in query')"

    async def execute(
        self, query: str, user_id: str = "anonymous", timeout: float | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Execute Python code safely in the sandbox.

        Args:
            query: Query containing Python code or code request
            user_id: User identifier for audit logging
            timeout: Optional timeout override
            **kwargs: Additional execution parameters

        Returns:
            Dictionary containing execution results and metadata
        """
        execution_start = time.time()
        code = ""
        audit_record = None

        try:
            # Extract code from query
            code = self._extract_code_from_query(query)

            if not code or code.strip() == "":
                return {
                    "success": False,
                    "result": None,
                    "error": "No executable Python code found in query",
                    "metadata": {"query": query},
                }

            # Generate code hash for audit
            code_hash = hashlib.sha256(code.encode()).hexdigest()[:16]

            logger.info(f"Executing code (hash: {code_hash}): {code[:100]}...")

            # Security validation
            violations = SecurityValidator.validate_code(code, self.security_level)
            if violations:
                error_msg = f"Security violations detected: {', '.join(violations)}"
                logger.warning(f"Code blocked due to security violations: {violations}")

                # Create audit record for blocked execution
                audit_record = AuditRecord(
                    timestamp=time.time(),
                    session_id=self.session_id,
                    user_id=user_id,
                    code_hash=code_hash,
                    code_preview=code[:200],
                    security_level=self.security_level,
                    resource_limits=self.resource_limits,
                    network_config=self.network_config,
                    success=False,
                    execution_time=time.time() - execution_start,
                    memory_used=0,
                    violations=violations,
                    result_summary="Blocked by security validation",
                    error_type="SandboxViolation",
                    error_message=error_msg,
                )
                self.audit_logger.log_execution(audit_record)

                return {
                    "success": False,
                    "result": None,
                    "error": error_msg,
                    "metadata": {
                        "code": code,
                        "code_hash": code_hash,
                        "violations": violations,
                        "security_level": self.security_level.value,
                    },
                }

            # Set up execution timeout
            execution_timeout = timeout or self.resource_limits.max_execution_time_seconds

            # Get safe execution environment
            safe_globals = self._get_safe_globals()

            # Execute in isolated process
            try:
                future = self.process_pool.submit(
                    FailureIsolation.execute_in_process, code, safe_globals, self.resource_limits
                )

                # Wait for execution with timeout
                exec_result = future.result(timeout=execution_timeout)

            except FutureTimeoutError:
                future.cancel()
                raise TimeoutError(f"Code execution timed out after {execution_timeout}s")

            # Process execution result
            execution_time = time.time() - execution_start
            success = exec_result.get("success", False)

            if success:
                result = {
                    "output": exec_result.get("output", ""),
                    "variables": exec_result.get("variables", {}),
                    "code": code,
                }

                # Check for resource violations
                resource_violations = exec_result.get("violations", [])
                if resource_violations:
                    logger.warning(f"Resource violations during execution: {resource_violations}")

                result_summary = f"Success: {len(result['output'])} chars output, {len(result['variables'])} variables"

            else:
                result = None
                resource_violations = exec_result.get("violations", [])
                result_summary = f"Failed: {exec_result.get('error_type', 'Unknown error')}"

            # Create audit record
            audit_record = AuditRecord(
                timestamp=time.time(),
                session_id=self.session_id,
                user_id=user_id,
                code_hash=code_hash,
                code_preview=code[:200],
                security_level=self.security_level,
                resource_limits=self.resource_limits,
                network_config=self.network_config,
                success=success,
                execution_time=execution_time,
                memory_used=int(exec_result.get("peak_memory", 0)),
                violations=resource_violations,
                result_summary=result_summary,
                error_type=exec_result.get("error_type") if not success else None,
                error_message=exec_result.get("error") if not success else None,
            )
            self.audit_logger.log_execution(audit_record)

            if success:
                logger.info(f"Code execution successful (hash: {code_hash})")
                return {
                    "success": True,
                    "result": result,
                    "error": None,
                    "metadata": {
                        "execution_time": execution_time,
                        "peak_memory_mb": exec_result.get("peak_memory", 0),
                        "code_hash": code_hash,
                        "violations": resource_violations,
                        "security_level": self.security_level.value,
                    },
                }
            logger.error(f"Code execution failed (hash: {code_hash}): {exec_result.get('error')}")
            return {
                "success": False,
                "result": None,
                "error": exec_result.get("error", "Unknown execution error"),
                "metadata": {
                    "execution_time": execution_time,
                    "code_hash": code_hash,
                    "error_type": exec_result.get("error_type"),
                    "violations": resource_violations,
                    "traceback": exec_result.get("traceback"),
                },
            }

        except Exception as e:
            execution_time = time.time() - execution_start
            error_msg = str(e)
            error_type = type(e).__name__

            logger.error(f"Sandbox execution error: {error_msg}")

            # Create audit record for exception
            if not audit_record:  # Only if we haven't created one already
                code_hash = hashlib.sha256(code.encode()).hexdigest()[:16] if code else "no_code"
                audit_record = AuditRecord(
                    timestamp=time.time(),
                    session_id=self.session_id,
                    user_id=user_id,
                    code_hash=code_hash,
                    code_preview=code[:200] if code else "No code",
                    security_level=self.security_level,
                    resource_limits=self.resource_limits,
                    network_config=self.network_config,
                    success=False,
                    execution_time=execution_time,
                    memory_used=0,
                    violations=[],
                    result_summary=f"Exception: {error_type}",
                    error_type=error_type,
                    error_message=error_msg,
                )
                self.audit_logger.log_execution(audit_record)

            return {
                "success": False,
                "result": None,
                "error": error_msg,
                "metadata": {
                    "execution_time": execution_time,
                    "error_type": error_type,
                    "code": code if code else "No code extracted",
                },
            }

    def get_session_audit_records(self) -> list[AuditRecord]:
        """Get audit records for the current session.

        Returns:
            List of audit records for this session
        """
        return self.audit_logger.get_records(self.session_id)

    def get_security_info(self) -> dict[str, Any]:
        """Get current security configuration information.

        Returns:
            Dictionary containing security configuration details
        """
        return {
            "session_id": self.session_id,
            "security_level": self.security_level.value,
            "resource_limits": {
                "max_execution_time": self.resource_limits.max_execution_time_seconds,
                "max_memory_mb": self.resource_limits.max_memory_mb,
                "max_processes": self.resource_limits.max_processes,
                "max_open_files": self.resource_limits.max_open_files,
            },
            "network_config": {
                "allow_network": self.network_config.allow_network,
                "allowed_hosts": list(self.network_config.allowed_hosts),
                "block_local_network": self.network_config.block_local_network,
            },
            "audit_records_count": len(self.audit_logger.records),
        }

    def close(self):
        """Clean up resources."""
        if hasattr(self, 'process_pool') and self.process_pool is not None:
            try:
                self.process_pool.shutdown(wait=False)
                self.process_pool = None
                logger.info(f"SandboxedCodeExecutor session {self.session_id} closed")
            except Exception as e:
                logger.warning(f"Error closing SandboxedCodeExecutor: {e}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Factory function for easy creation
def create_sandbox(
    security_level: SecurityLevel = SecurityLevel.MODERATE, **kwargs
) -> SandboxedCodeExecutor:
    """Factory function to create a sandboxed code executor.

    Args:
        security_level: Security level for the sandbox
        **kwargs: Additional configuration options

    Returns:
        Configured SandboxedCodeExecutor instance
    """
    return SandboxedCodeExecutor(security_level=security_level, **kwargs)
