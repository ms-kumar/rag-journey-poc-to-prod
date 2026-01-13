"""Enhanced safe code executor tool with comprehensive sandbox."""

import logging
from typing import Any, Optional

from src.config import get_settings
from src.services.agent.tools.base import BaseTool, ToolCategory, ToolMetadata
from src.services.agent.tools.hybrid.sandbox_config import (
    get_network_config,
    get_resource_limits,
    get_security_level,
)
from src.services.agent.tools.hybrid.sandboxed_executor import (
    SandboxedCodeExecutor,
    SecurityLevel,
)

logger = logging.getLogger(__name__)


class CodeExecutorTool(BaseTool):
    """Tool for safe Python code execution with comprehensive security sandbox."""

    def __init__(
        self, 
        timeout: Optional[int] = None,
        security_level: Optional[SecurityLevel] = None,
        max_memory_mb: Optional[int] = None,
        max_processes: Optional[int] = None,
        allow_network: Optional[bool] = None,
        audit_log_file: Optional[str] = None,
    ):
        """Initialize enhanced code executor tool.

        Args:
            timeout: Execution timeout in seconds (uses config if None)
            security_level: Security level (uses config if None)
            max_memory_mb: Maximum memory usage in MB (uses config if None)
            max_processes: Maximum number of processes (uses config if None)
            allow_network: Whether to allow network access (uses config if None)
            audit_log_file: Optional path to audit log file (uses config if None)
        """
        metadata = ToolMetadata(
            name="code_executor",
            description="Execute Python code safely in a comprehensive security sandbox with resource limits, audit logging, and isolation",
            category=ToolCategory.HYBRID,
            capabilities=[
                "python execution",
                "code running",
                "calculations",
                "data analysis", 
                "mathematical operations",
                "resource monitoring",
                "security validation",
                "audit logging",
                "process isolation",
            ],
            cost_per_call=0.0,
            avg_latency_ms=400.0,  # Slightly higher due to enhanced security
            success_rate=0.90,     # Higher due to better error handling
            requires_api_key=False,
        )
        super().__init__(metadata)
        
        # Load settings
        settings = get_settings()
        sandbox_settings = settings.sandbox
        
        # Get resource limits from config (with overrides)
        self.resource_limits = get_resource_limits()
        if timeout is not None:
            self.resource_limits.max_execution_time_seconds = float(timeout)
            self.resource_limits.max_cpu_time_seconds = float(timeout - 0.5)
        if max_memory_mb is not None:
            self.resource_limits.max_memory_mb = max_memory_mb
        if max_processes is not None:
            self.resource_limits.max_processes = max_processes
        
        # Get network config from config (with overrides)
        self.network_config = get_network_config()
        if allow_network is not None:
            self.network_config.allow_network = allow_network
        
        # Get security level from config (with override)
        self.security_level = security_level if security_level is not None else get_security_level()
        
        # Get audit log file from config (with override)
        self.audit_log_file = (
            audit_log_file if audit_log_file is not None 
            else (sandbox_settings.audit_log_file if sandbox_settings.enable_audit_logging else None)
        )
        
        # Initialize sandbox (will be created per execution for isolation)
        self._sandbox: Optional[SandboxedCodeExecutor] = None
        
        self.logger.info(
            f"CodeExecutorTool initialized: security_level={self.security_level.value}, "
            f"timeout={self.resource_limits.max_execution_time_seconds}s, "
            f"memory_limit={self.resource_limits.max_memory_mb}MB, "
            f"network_allowed={self.network_config.allow_network}"
        )

    async def execute(self, query: str, **kwargs: Any) -> dict[str, Any]:
        """Execute Python code safely in the enhanced sandbox.

        Args:
            query: Query containing Python code
            **kwargs: Optional parameters:
                - user_id: User identifier for audit logging
                - timeout: Override default timeout
                - security_level: Override default security level

        Returns:
            Dictionary with execution results and comprehensive metadata
        """
        try:
            if not self.validate_input(query, **kwargs):
                return {
                    "success": False,
                    "result": None,
                    "error": "Invalid input parameters",
                    "metadata": {},
                }

            # Extract parameters
            user_id = kwargs.get("user_id", "anonymous")
            timeout = kwargs.get("timeout", self.resource_limits.max_execution_time_seconds)
            security_level = kwargs.get("security_level", self.security_level)

            self.logger.info(f"Executing code for user {user_id}: {query[:100]}...")

            # Create fresh sandbox for this execution (for maximum isolation)
            with SandboxedCodeExecutor(
                security_level=security_level,
                resource_limits=self.resource_limits,
                network_config=self.network_config,
                audit_log_file=self.audit_log_file,
                max_workers=1,  # Single worker for this execution
            ) as sandbox:
                
                # Execute in sandbox
                result = await sandbox.execute(
                    query=query,
                    user_id=user_id,
                    timeout=timeout,
                )
                
                # Add sandbox-specific metadata
                security_info = sandbox.get_security_info()
                audit_records = sandbox.get_session_audit_records()
                
                # Enhance result with additional metadata
                if result["success"]:
                    result["metadata"].update({
                        "session_id": security_info["session_id"],
                        "audit_records_count": len(audit_records),
                        "resource_limits": security_info["resource_limits"],
                        "network_config": security_info["network_config"],
                    })
                    
                    self.logger.info(f"Code execution successful: {result['metadata'].get('code_hash', 'unknown')}")
                else:
                    result["metadata"].update({
                        "session_id": security_info["session_id"],
                        "audit_records_count": len(audit_records),
                    })
                    
                    self.logger.warning(f"Code execution failed: {result.get('error', 'Unknown error')}")

                return result

        except Exception as e:
            self.logger.error(f"Code executor tool error: {e}")
            return {
                "success": False,
                "result": None,
                "error": f"Code executor error: {str(e)}",
                "metadata": {
                    "error_type": type(e).__name__,
                    "query": query[:200],
                },
            }

    def validate_input(self, query: str, **kwargs: Any) -> bool:
        """Validate input parameters with enhanced checks.

        Args:
            query: User query
            **kwargs: Additional parameters

        Returns:
            True if valid, False otherwise
        """
        if not query or not isinstance(query, str):
            self.logger.error("Invalid query: must be a non-empty string")
            return False
            
        if len(query) > 50000:  # Reasonable limit for code size
            self.logger.error("Query too long: maximum 50,000 characters")
            return False
            
        return True
        
    def get_security_status(self) -> dict[str, Any]:
        """Get current security configuration status.
        
        Returns:
            Dictionary with security configuration details
        """
        return {
            "tool_name": self.metadata.name,
            "security_level": self.security_level.value,
            "resource_limits": {
                "max_execution_time": self.resource_limits.max_execution_time_seconds,
                "max_memory_mb": self.resource_limits.max_memory_mb,
                "max_processes": self.resource_limits.max_processes,
            },
            "network_config": {
                "allow_network": self.network_config.allow_network,
                "block_local_network": self.network_config.block_local_network,
            },
            "audit_log_file": self.audit_log_file,
        }
