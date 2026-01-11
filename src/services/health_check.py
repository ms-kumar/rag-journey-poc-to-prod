"""
Health check service for monitoring system and dependency health.

Provides utilities to check the status of various system components
including vectorstore, embeddings, and generation services.
"""

import asyncio
import logging
import platform
import time
from typing import Any

from src.schemas.api.health import ComponentHealth, ServiceStatus

logger = logging.getLogger(__name__)

# Track application start time
_start_time = time.time()


def get_uptime() -> float:
    """Get application uptime in seconds."""
    return time.time() - _start_time


def get_system_info() -> dict[str, str]:
    """Get basic system information."""
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "architecture": platform.machine(),
        "processor": platform.processor() or "unknown",
    }


def get_dependency_versions() -> dict[str, str]:
    """Get versions of key dependencies."""
    versions = {}

    try:
        import fastapi

        versions["fastapi"] = fastapi.__version__
    except (ImportError, AttributeError):
        versions["fastapi"] = "not installed"

    try:
        import qdrant_client

        versions["qdrant-client"] = getattr(qdrant_client, "__version__", "unknown")
    except ImportError:
        versions["qdrant-client"] = "not installed"

    try:
        import langchain

        versions["langchain"] = langchain.__version__
    except (ImportError, AttributeError):
        versions["langchain"] = "not installed"

    try:
        import transformers

        versions["transformers"] = transformers.__version__
    except (ImportError, AttributeError):
        versions["transformers"] = "not installed"

    try:
        import torch

        versions["torch"] = torch.__version__
    except (ImportError, AttributeError):
        versions["torch"] = "not installed"

    try:
        import pydantic

        versions["pydantic"] = pydantic.__version__
    except (ImportError, AttributeError):
        versions["pydantic"] = "not installed"

    return versions


async def check_vectorstore_health(
    vectorstore_client: Any | None = None,
) -> ComponentHealth:
    """
    Check health of vectorstore (Qdrant) connection.

    Args:
        vectorstore_client: Optional QdrantVectorStoreClient instance

    Returns:
        ComponentHealth status
    """
    start = time.time()

    if vectorstore_client is None:
        return ComponentHealth(
            name="vectorstore",
            status=ServiceStatus.UNKNOWN,
            message="Vectorstore client not initialized",
            details=None,
            response_time_ms=None,
        )

    try:
        # Try to get collection info
        collection_name = vectorstore_client.config.collection_name
        collection_info = vectorstore_client.qdrant_client.get_collection(collection_name)

        elapsed_ms = (time.time() - start) * 1000

        # Extract useful info
        vector_count = collection_info.points_count if collection_info else 0
        vector_size = collection_info.config.params.vectors.size if collection_info else 0

        return ComponentHealth(
            name="vectorstore",
            status=ServiceStatus.HEALTHY,
            message=f"Collection '{collection_name}' accessible",
            details={
                "collection": collection_name,
                "points_count": vector_count,
                "vector_size": vector_size,
            },
            response_time_ms=round(elapsed_ms, 2),
        )

    except Exception as e:
        elapsed_ms = (time.time() - start) * 1000
        logger.error(f"Vectorstore health check failed: {e}")
        return ComponentHealth(
            name="vectorstore",
            status=ServiceStatus.UNHEALTHY,
            message=f"Failed to connect: {str(e)}",
            details=None,
            response_time_ms=round(elapsed_ms, 2),
        )


async def check_embeddings_health(embeddings_client: Any | None = None) -> ComponentHealth:
    """
    Check health of embeddings service.

    Args:
        embeddings_client: Optional embeddings client instance

    Returns:
        ComponentHealth status
    """
    start = time.time()

    if embeddings_client is None:
        return ComponentHealth(
            name="embeddings",
            status=ServiceStatus.UNKNOWN,
            message="Embeddings client not initialized",
            details=None,
            response_time_ms=None,
        )

    try:
        # Try a simple embedding test
        test_text = "health check"
        embedding = embeddings_client.embed_query(test_text)

        elapsed_ms = (time.time() - start) * 1000

        return ComponentHealth(
            name="embeddings",
            status=ServiceStatus.HEALTHY,
            message="Embeddings service responsive",
            details={
                "embedding_dim": len(embedding) if embedding else 0,
                "test_query": test_text,
            },
            response_time_ms=round(elapsed_ms, 2),
        )

    except Exception as e:
        elapsed_ms = (time.time() - start) * 1000
        logger.error(f"Embeddings health check failed: {e}")
        return ComponentHealth(
            name="embeddings",
            status=ServiceStatus.UNHEALTHY,
            message=f"Failed to generate embeddings: {str(e)}",
            details=None,
            response_time_ms=round(elapsed_ms, 2),
        )


async def check_generation_health(generator_client: Any | None = None) -> ComponentHealth:
    """
    Check health of text generation service.

    Args:
        generator_client: Optional generator client instance

    Returns:
        ComponentHealth status
    """
    start = time.time()

    if generator_client is None:
        return ComponentHealth(
            name="generation",
            status=ServiceStatus.UNKNOWN,
            message="Generation client not initialized",
            details=None,
            response_time_ms=None,
        )

    try:
        # Try a simple generation test
        test_prompt = "Health check:"
        result = generator_client.generate(
            test_prompt, overrides={"max_new_tokens": 10, "do_sample": False}
        )

        elapsed_ms = (time.time() - start) * 1000

        return ComponentHealth(
            name="generation",
            status=ServiceStatus.HEALTHY,
            message="Generation service responsive",
            details={
                "model": generator_client.config.model_name,
                "test_output_length": len(result) if result else 0,
            },
            response_time_ms=round(elapsed_ms, 2),
        )

    except Exception as e:
        elapsed_ms = (time.time() - start) * 1000
        logger.error(f"Generation health check failed: {e}")
        return ComponentHealth(
            name="generation",
            status=ServiceStatus.UNHEALTHY,
            message=f"Failed to generate text: {str(e)}",
            details=None,
            response_time_ms=round(elapsed_ms, 2),
        )


async def check_all_components(
    vectorstore_client: Any | None = None,
    embeddings_client: Any | None = None,
    generator_client: Any | None = None,
) -> list[ComponentHealth]:
    """
    Check health of all system components in parallel.

    Args:
        vectorstore_client: Optional QdrantVectorStoreClient instance
        embeddings_client: Optional embeddings client instance
        generator_client: Optional generator client instance

    Returns:
        List of ComponentHealth status for each component
    """
    # Run all checks in parallel
    results = await asyncio.gather(
        check_vectorstore_health(vectorstore_client),
        check_embeddings_health(embeddings_client),
        check_generation_health(generator_client),
        return_exceptions=True,
    )

    components = []
    for result in results:
        if isinstance(result, ComponentHealth):
            components.append(result)
        elif isinstance(result, Exception):
            logger.error(f"Component health check raised exception: {result}")
            components.append(
                ComponentHealth(
                    name="unknown",
                    status=ServiceStatus.UNHEALTHY,
                    message=f"Check failed with exception: {str(result)}",
                    details=None,
                    response_time_ms=None,
                )
            )

    return components


def determine_overall_status(components: list[ComponentHealth]) -> ServiceStatus:
    """
    Determine overall system status based on component health.

    Rules:
    - UNHEALTHY if any component is unhealthy
    - DEGRADED if any component is degraded or unknown
    - HEALTHY if all components are healthy

    Args:
        components: List of component health statuses

    Returns:
        Overall ServiceStatus
    """
    if not components:
        return ServiceStatus.UNKNOWN

    statuses = [c.status for c in components]

    if ServiceStatus.UNHEALTHY in statuses:
        return ServiceStatus.UNHEALTHY
    if ServiceStatus.DEGRADED in statuses or ServiceStatus.UNKNOWN in statuses:
        return ServiceStatus.DEGRADED
    return ServiceStatus.HEALTHY
