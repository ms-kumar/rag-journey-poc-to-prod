"""
Health check models for the RAG system.

Provides comprehensive status information about system dependencies
and service health.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ServiceStatus(str, Enum):
    """Status of a service component."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ComponentHealth(BaseModel):
    """Health information for a single component."""

    name: str = Field(..., description="Name of the component")
    status: ServiceStatus = Field(..., description="Current status of the component")
    message: str | None = Field(None, description="Optional status message")
    details: dict[str, Any] | None = Field(None, description="Additional details")
    response_time_ms: float | None = Field(None, description="Response time in milliseconds")


class HealthCheckResponse(BaseModel):
    """Complete health check response."""

    status: ServiceStatus = Field(..., description="Overall system status")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Timestamp of health check (ISO format)")
    components: list[ComponentHealth] = Field(
        default_factory=list, description="Health status of individual components"
    )
    uptime_seconds: float | None = Field(None, description="System uptime in seconds")


class DetailedHealthResponse(HealthCheckResponse):
    """Extended health check with detailed system information."""

    system_info: dict[str, Any] | None = Field(
        None, description="System information (OS, Python version, etc.)"
    )
    dependencies: dict[str, str] | None = Field(None, description="Installed dependency versions")
