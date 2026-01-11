"""
FastAPI dependency injection functions.

This module provides reusable dependencies for API endpoints,
including service instances, configuration access, and request validation.
Uses app state pattern for service lifecycle management.
"""

import logging
from typing import Annotated, Any

from fastapi import Depends, Header, HTTPException, Query, Request, status

from src.config import Settings, settings
from src.services.cache.client import CacheClient
from src.services.embeddings.cached_client import CachedEmbeddingClient
from src.services.generation.client import HFGenerator
from src.services.guardrails.client import GuardrailsClient
from src.services.query_understanding.client import QueryUnderstandingClient
from src.services.reranker.client import CrossEncoderReranker
from src.services.vectorstore.client import QdrantVectorStoreClient

logger = logging.getLogger(__name__)


# ==============================================================================
# Configuration Dependencies
# ==============================================================================


def get_settings() -> Settings:
    """Get application settings."""
    return settings


def get_request_settings(request: Request) -> Settings:
    """Get settings from the request state."""
    return request.app.state.settings  # type: ignore[no-any-return]


# ==============================================================================
# Service Dependencies (App State Pattern)
# ==============================================================================


def get_embedding_service(request: Request) -> CachedEmbeddingClient:
    """Get embedding service from the request state."""
    return request.app.state.embedding_service  # type: ignore[no-any-return]


def get_cache_service(request: Request) -> CacheClient:
    """Get cache service from the request state."""
    return request.app.state.cache_service  # type: ignore[no-any-return]


def get_vector_store_service(request: Request) -> QdrantVectorStoreClient:
    """Get vector store service from the request state."""
    return request.app.state.vector_store_service  # type: ignore[no-any-return]


def get_reranker_service(request: Request) -> CrossEncoderReranker:
    """Get reranker service from the request state."""
    return request.app.state.reranker_service  # type: ignore[no-any-return]


def get_generation_service(request: Request) -> HFGenerator:
    """Get generation service from the request state."""
    return request.app.state.generation_service  # type: ignore[no-any-return]


def get_query_understanding_service(request: Request) -> QueryUnderstandingClient:
    """Get query understanding service from the request state."""
    return request.app.state.query_understanding_service  # type: ignore[no-any-return]


def get_guardrails_service(request: Request) -> GuardrailsClient:
    """Get guardrails coordinator from the request state."""
    return request.app.state.guardrails_service  # type: ignore[no-any-return]


# ==============================================================================
# Type Aliases for Dependency Injection
# ==============================================================================

# Use Annotated to define reusable dependency types
SettingsDep = Annotated[Settings, Depends(get_settings)]
RequestSettingsDep = Annotated[Settings, Depends(get_request_settings)]
EmbeddingServiceDep = Annotated[CachedEmbeddingClient, Depends(get_embedding_service)]
CacheServiceDep = Annotated[CacheClient, Depends(get_cache_service)]
VectorStoreServiceDep = Annotated[QdrantVectorStoreClient, Depends(get_vector_store_service)]
RerankerServiceDep = Annotated[CrossEncoderReranker, Depends(get_reranker_service)]
GenerationServiceDep = Annotated[HFGenerator, Depends(get_generation_service)]
QueryUnderstandingServiceDep = Annotated[
    QueryUnderstandingClient, Depends(get_query_understanding_service)
]
GuardrailsServiceDep = Annotated[GuardrailsClient, Depends(get_guardrails_service)]


# ==============================================================================
# Request Validation Dependencies
# ==============================================================================


async def verify_api_key(
    x_api_key: Annotated[str | None, Header()] = None,
) -> str | None:
    """Verify API key from request header."""
    # If no API key is configured, skip validation
    api_key = getattr(settings.app, "api_key", None)
    if not api_key:
        return None

    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if x_api_key != api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )

    return x_api_key


async def get_request_id(
    x_request_id: Annotated[str | None, Header()] = None,
) -> str | None:
    """Extract request ID from headers for tracing."""
    return x_request_id


async def validate_query_params(
    query: Annotated[str, Query(min_length=1, max_length=1000)],
    top_k: Annotated[int, Query(ge=1, le=100)] = 5,
    include_metadata: Annotated[bool, Query()] = True,
) -> dict[str, Any]:
    """Validate common query parameters for RAG endpoints."""
    if not query.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Query cannot be empty or whitespace only",
        )

    return {
        "query": query.strip(),
        "top_k": top_k,
        "include_metadata": include_metadata,
    }


# ==============================================================================
# Pagination Dependencies
# ==============================================================================


async def get_pagination(
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 10,
) -> dict[str, int]:
    """Get pagination parameters."""
    return {"skip": skip, "limit": limit}


# ==============================================================================
# Feature Flag Dependencies
# ==============================================================================


async def check_feature_enabled(feature_name: str) -> Any:
    """Check if a feature is enabled."""

    async def _check() -> None:
        # Check if feature is enabled in settings
        feature_flags = getattr(settings.app, "feature_flags", {})
        if not feature_flags.get(feature_name, True):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Feature '{feature_name}' is currently disabled",
            )

    return _check


# ==============================================================================
# Rate Limiting Dependencies (placeholder for future implementation)
# ==============================================================================


async def check_rate_limit(
    x_forwarded_for: Annotated[str | None, Header()] = None,
) -> None:
    """Check rate limiting for requests (placeholder for future implementation)."""
    # TODO: Implement rate limiting
    pass


# ==============================================================================
# Health Check Dependencies
# ==============================================================================


async def check_system_health(request: Request) -> dict[str, Any]:
    """Perform basic system health checks."""
    health_status: dict[str, Any] = {
        "api": "healthy",
        "services": {},
    }

    services: dict[str, str] = {}

    # Check services are initialized in app state
    try:
        if hasattr(request.app.state, "embedding_service"):
            services["embedding"] = "initialized"
        else:
            services["embedding"] = "not_initialized"
    except Exception as e:
        logger.error(f"Embedding service health check failed: {e}")
        services["embedding"] = "error"

    try:
        if hasattr(request.app.state, "vector_store_service"):
            services["vectorstore"] = "initialized"
        else:
            services["vectorstore"] = "not_initialized"
    except Exception as e:
        logger.error(f"Vector store service health check failed: {e}")
        services["vectorstore"] = "error"

    try:
        if hasattr(request.app.state, "cache_service"):
            services["cache"] = "initialized"
        else:
            services["cache"] = "not_initialized"
    except Exception as e:
        logger.error(f"Cache service health check failed: {e}")
        services["cache"] = "error"

    health_status["services"] = services
    return health_status
