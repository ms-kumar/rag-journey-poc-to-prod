"""Configuration settings for sandboxed code execution.

This module provides helper functions to work with sandbox configuration
from the main application settings.
"""

import os
from typing import List

from src.config import get_settings

from .sandbox import NetworkConfig, ResourceLimits, SecurityLevel


def get_security_level_from_string(level_str: str) -> SecurityLevel:
    """Convert string security level to SecurityLevel enum.
    
    Args:
        level_str: Security level as string (strict, moderate, permissive)
        
    Returns:
        SecurityLevel enum value
    """
    level_map = {
        "strict": SecurityLevel.STRICT,
        "moderate": SecurityLevel.MODERATE,
        "permissive": SecurityLevel.PERMISSIVE,
    }
    return level_map.get(level_str.lower(), SecurityLevel.MODERATE)


def get_resource_limits() -> ResourceLimits:
    """Get resource limits from application settings.
    
    Returns:
        ResourceLimits configured from settings
    """
    settings = get_settings()
    sandbox = settings.sandbox
    
    return ResourceLimits(
        max_execution_time_seconds=sandbox.max_execution_time,
        max_cpu_time_seconds=sandbox.max_cpu_time,
        max_memory_mb=sandbox.max_memory_mb,
        max_stack_size_mb=sandbox.max_stack_size_mb,
        max_processes=sandbox.max_processes,
        max_open_files=sandbox.max_open_files,
        max_output_size=sandbox.max_output_size,
        max_variables=sandbox.max_variables,
    )


def get_network_config() -> NetworkConfig:
    """Get network configuration from application settings.
    
    Returns:
        NetworkConfig configured from settings
    """
    settings = get_settings()
    sandbox = settings.sandbox
    
    return NetworkConfig(
        allow_network=sandbox.allow_network,
        allowed_hosts=sandbox.get_allowed_hosts_set(),
        allowed_ports=sandbox.get_allowed_ports_set(),
        block_local_network=sandbox.block_local_network,
    )


def get_security_level() -> SecurityLevel:
    """Get security level from application settings.
    
    Returns:
        SecurityLevel enum value
    """
    settings = get_settings()
    return get_security_level_from_string(settings.sandbox.security_level)


def validate_config() -> List[str]:
    """Validate sandbox configuration from application settings.
    
    Returns:
        List of validation errors (empty if valid)
    """
    settings = get_settings()
    sandbox = settings.sandbox
    errors = []
    
    # Validate time limits
    if sandbox.max_execution_time <= 0:
        errors.append("max_execution_time must be positive")
    
    if sandbox.max_cpu_time >= sandbox.max_execution_time:
        errors.append("max_cpu_time should be less than max_execution_time")
    
    # Validate memory limits
    if sandbox.max_memory_mb <= 0:
        errors.append("max_memory_mb must be positive")
    
    if sandbox.max_memory_mb > 2048:  # 2GB seems reasonable as upper limit
        errors.append("max_memory_mb should not exceed 2048 MB")
    
    # Validate process limits
    if sandbox.max_processes <= 0 or sandbox.max_processes > 10:
        errors.append("max_processes must be between 1 and 10")
    
    # Validate file limits
    if sandbox.max_open_files <= 0 or sandbox.max_open_files > 1024:
        errors.append("max_open_files must be between 1 and 1024")
    
    # Validate output limits
    if sandbox.max_output_size <= 0:
        errors.append("max_output_size must be positive")
    
    if sandbox.max_output_size > 10 * 1024 * 1024:  # 10MB limit
        errors.append("max_output_size should not exceed 10MB")
    
    # Validate network config
    security_level = get_security_level()
    if (sandbox.allow_network and 
        not sandbox.get_allowed_hosts_set() and 
        security_level == SecurityLevel.STRICT):
        errors.append("In STRICT mode, allowed_hosts must be specified when network is enabled")
    
    return errors


# Legacy compatibility - environment-based configs
def get_config(environment: str = None):
    """Get sandbox configuration.
    
    This function now redirects to the main application settings.
    The environment parameter is kept for backward compatibility but
    configuration should be set via environment variables or .env file.
    
    Args:
        environment: Legacy parameter (ignored, use ENVIRONMENT env var instead)
        
    Returns:
        Main application settings object
    """
    return get_settings()


# Re-export for convenience
__all__ = [
    "get_security_level",
    "get_resource_limits", 
    "get_network_config",
    "validate_config",
    "get_config",
    "get_security_level_from_string",
]