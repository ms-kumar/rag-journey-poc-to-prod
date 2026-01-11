"""
Health check endpoints for monitoring system status.
"""

import logging
from datetime import UTC, datetime

from fastapi import APIRouter, Query

from src.config import settings
from src.schemas.api.health import DetailedHealthResponse, HealthCheckResponse, ServiceStatus
from src.services.health_check import (
    check_all_components,
    determine_overall_status,
    get_dependency_versions,
    get_system_info,
    get_uptime,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthCheckResponse)
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


@router.get("/health/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check(
    include_system_info: bool = Query(True, description="Include system information in response"),
    include_dependencies: bool = Query(True, description="Include dependency versions in response"),
    check_components: bool = Query(
        False, description="Check health of individual components (slower)"
    ),
):
    """
    Detailed health check endpoint with component status.

    Provides comprehensive health information including:
    - Overall system status
    - Component health (vectorstore, embeddings, generation)
    - System information (OS, Python version)
    - Dependency versions
    - Uptime

    Query Parameters:
        include_system_info: Include system/OS information
        include_dependencies: Include installed package versions
        check_components: Run health checks on components (slower, more thorough)

    Returns:
        DetailedHealthResponse with full health information

    Example:
        GET /health/detailed?check_components=true
    """
    timestamp = datetime.now(UTC).isoformat()
    uptime = get_uptime()

    # Optional: check component health
    components = []
    if check_components:
        try:
            # Try to get pipeline components for health checks
            # Note: In production, you'd want to pass actual clients here
            from src.services.pipeline.naive_pipeline.factory import get_naive_pipeline

            try:
                pipeline = get_naive_pipeline()
                components = await check_all_components(
                    vectorstore_client=pipeline.vectorstore,
                    embeddings_client=pipeline.lc_embeddings,
                    generator_client=pipeline.generator,
                )
            except Exception as e:
                logger.warning(f"Could not check components: {e}")
                components = []
        except Exception as e:
            logger.warning(f"Could not import pipeline for health check: {e}")

    # Determine overall status
    if components:
        status = determine_overall_status(components)
    else:
        status = ServiceStatus.HEALTHY

    # Build response
    response = DetailedHealthResponse(
        status=status,
        version=settings.app.version,
        timestamp=timestamp,
        components=components,
        uptime_seconds=round(uptime, 2),
        system_info=get_system_info() if include_system_info else None,
        dependencies=get_dependency_versions() if include_dependencies else None,
    )

    return response


@router.get("/health/ready")
async def readiness_check():
    """
    Readiness check for Kubernetes/orchestration systems.

    Checks if the application is ready to serve traffic by verifying
    that critical components are initialized and accessible.

    Returns:
        200 OK if ready, 503 Service Unavailable if not ready
    """
    try:
        # Try to import pipeline to ensure dependencies are available
        from src.services.pipeline.naive_pipeline.factory import get_naive_pipeline

        pipeline = get_naive_pipeline()

        # Run quick component checks
        components = await check_all_components(
            vectorstore_client=pipeline.vectorstore,
            embeddings_client=pipeline.lc_embeddings,
            generator_client=pipeline.generator,
        )

        status = determine_overall_status(components)

        if status in [ServiceStatus.HEALTHY, ServiceStatus.DEGRADED]:
            return {
                "ready": True,
                "status": status.value,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        return {
            "ready": False,
            "status": status.value,
            "timestamp": datetime.now(UTC).isoformat(),
            "components": [
                {"name": c.name, "status": c.status.value, "message": c.message} for c in components
            ],
        }

    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return {
            "ready": False,
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now(UTC).isoformat(),
        }


@router.get("/health/live")
async def liveness_check():
    """
    Liveness check for Kubernetes/orchestration systems.

    Simple check to verify the application is running.
    Does not check dependencies - just confirms the process is alive.

    Returns:
        200 OK if alive
    """
    return {
        "alive": True,
        "timestamp": datetime.now(UTC).isoformat(),
        "uptime_seconds": round(get_uptime(), 2),
    }
