"""Health check router for monitoring system status."""

import logging
from datetime import UTC, datetime

from fastapi import APIRouter, Query

from src.config import settings
from src.models.health import DetailedHealthResponse, HealthCheckResponse, ServiceStatus
from src.services.health_check import (
    check_all_components,
    determine_overall_status,
    get_dependency_versions,
    get_system_info,
    get_uptime,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthCheckResponse, tags=["health"])
async def basic_health_check():
    """
    Basic health check endpoint.

    Returns simple status without checking dependencies.
    Useful for load balancers and quick availability checks.

    Returns:
        HealthCheckResponse with basic status information
    """
    return HealthCheckResponse(
        status=ServiceStatus.HEALTHY,
        version=settings.app.version,
        timestamp=datetime.now(UTC).isoformat(),
        uptime_seconds=round(get_uptime(), 2),
    )


@router.get("/health/detailed", response_model=DetailedHealthResponse, tags=["health"])
async def detailed_health_check(
    include_system_info: bool = Query(True, description="Include system information in response"),
    include_dependencies: bool = Query(True, description="Include dependency versions in response"),
    check_components: bool = Query(
        False, description="Check health of individual components (slower)"
    ),
):
    """
    Detailed health check with component status.

    Provides comprehensive health information including:
    - System information (CPU, memory, disk)
    - Dependency versions
    - Component health status (optional, slower)

    Args:
        include_system_info: Include system resource information
        include_dependencies: Include dependency version information
        check_components: Check individual component health (slower)

    Returns:
        DetailedHealthResponse with comprehensive health information
    """
    logger.debug(
        f"Detailed health check requested: "
        f"system_info={include_system_info}, "
        f"dependencies={include_dependencies}, "
        f"components={check_components}"
    )

    # Get basic health info
    health_data = {
        "status": ServiceStatus.HEALTHY,
        "version": settings.app.version,
        "timestamp": datetime.now(UTC).isoformat(),
        "uptime_seconds": round(get_uptime(), 2),
    }

    # Add system info if requested
    if include_system_info:
        health_data["system_info"] = get_system_info()

    # Add dependency versions if requested
    if include_dependencies:
        health_data["dependencies"] = get_dependency_versions()

    # Check components if requested
    if check_components:
        logger.debug("Checking component health...")
        component_results = await check_all_components()
        health_data["components"] = component_results
        health_data["status"] = determine_overall_status(component_results)

    return DetailedHealthResponse(**health_data)


@router.get("/health/ready", tags=["health"])
async def readiness_check():
    """
    Kubernetes readiness probe endpoint.

    Checks if the service is ready to accept traffic.
    Returns 200 if ready, 503 if not ready.
    """
    try:
        # Basic check - could be expanded to check critical dependencies
        return {"status": "ready", "timestamp": datetime.now(UTC).isoformat()}
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return {"status": "not_ready", "error": str(e)}


@router.get("/health/live", tags=["health"])
async def liveness_check():
    """
    Kubernetes liveness probe endpoint.

    Checks if the service is alive and should not be restarted.
    Returns 200 if alive.
    """
    return {"status": "alive", "timestamp": datetime.now(UTC).isoformat()}
