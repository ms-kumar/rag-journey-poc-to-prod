"""Tests for dependency injection functions."""

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from src.dependencies import (
    check_rate_limit,
    get_cache_service,
    get_embedding_service,
    get_generation_service,
    get_guardrails_service,
    get_pagination,
    get_query_understanding_service,
    get_request_id,
    get_reranker_service,
    get_settings,
    get_vector_store_service,
    validate_query_params,
    verify_api_key,
)


class TestSettingsDependency:
    """Test settings dependency."""

    def test_get_settings(self):
        """Test getting settings."""
        settings = get_settings()

        assert settings is not None
        assert hasattr(settings, "app")
        assert hasattr(settings, "embedding")
        assert hasattr(settings, "vectorstore")


@pytest.fixture
def mock_request():
    """Create a mock request with app state."""
    request = MagicMock()
    request.app.state.embedding_service = MagicMock()
    request.app.state.cache_service = MagicMock()
    request.app.state.vector_store_service = MagicMock()
    request.app.state.reranker_service = MagicMock()
    request.app.state.generation_service = MagicMock()
    request.app.state.query_understanding_service = MagicMock()
    request.app.state.guardrails_service = MagicMock()
    return request


class TestServiceDependencies:
    """Test service dependencies."""

    def test_get_embedding_service(self, mock_request):
        """Test getting embedding service."""
        service = get_embedding_service(mock_request)
        assert service is not None

    def test_get_cache_service(self, mock_request):
        """Test getting cache service."""
        service = get_cache_service(mock_request)
        assert service is not None

    def test_get_vector_store_service(self, mock_request):
        """Test getting vector store service."""
        service = get_vector_store_service(mock_request)
        assert service is not None

    def test_get_reranker_service(self, mock_request):
        """Test getting reranker service."""
        service = get_reranker_service(mock_request)
        assert service is not None

    def test_get_generation_service(self, mock_request):
        """Test getting generation service."""
        service = get_generation_service(mock_request)
        assert service is not None

    def test_get_query_understanding_service(self, mock_request):
        """Test getting query understanding service."""
        service = get_query_understanding_service(mock_request)
        assert service is not None

    def test_get_guardrails_service(self, mock_request):
        """Test getting guardrails service."""
        service = get_guardrails_service(mock_request)
        assert service is not None


class TestAPIKeyVerification:
    """Test API key verification."""

    @pytest.mark.asyncio
    async def test_verify_api_key_no_key_configured(self):
        """Test when no API key is configured."""
        # When no API key is set, should return None
        result = await verify_api_key(x_api_key=None)
        assert result is None

    # Note: Skipping tests that modify frozen settings
    # In production, API keys would be configured via environment variables


class TestRequestID:
    """Test request ID extraction."""

    @pytest.mark.asyncio
    async def test_get_request_id_present(self):
        """Test getting request ID when present."""
        result = await get_request_id(x_request_id="test-request-123")
        assert result == "test-request-123"

    @pytest.mark.asyncio
    async def test_get_request_id_missing(self):
        """Test getting request ID when missing."""
        result = await get_request_id(x_request_id=None)
        assert result is None


class TestQueryValidation:
    """Test query parameter validation."""

    @pytest.mark.asyncio
    async def test_validate_query_params_valid(self):
        """Test validating valid query parameters."""
        result = await validate_query_params(
            query="What is machine learning?", top_k=10, include_metadata=True
        )

        assert result["query"] == "What is machine learning?"
        assert result["top_k"] == 10
        assert result["include_metadata"] is True

    @pytest.mark.asyncio
    async def test_validate_query_params_strips_whitespace(self):
        """Test that query is stripped of whitespace."""
        result = await validate_query_params(
            query="  test query  ", top_k=5, include_metadata=False
        )

        assert result["query"] == "test query"

    @pytest.mark.asyncio
    async def test_validate_query_params_empty_string(self):
        """Test validation fails for empty query."""
        with pytest.raises(HTTPException) as exc_info:
            await validate_query_params(query="   ", top_k=5, include_metadata=True)

        assert exc_info.value.status_code == 422
        assert "empty" in exc_info.value.detail.lower()


class TestPagination:
    """Test pagination dependency."""

    @pytest.mark.asyncio
    async def test_get_pagination_default(self):
        """Test pagination with default values."""
        result = await get_pagination()

        assert result["skip"] == 0
        assert result["limit"] == 10

    @pytest.mark.asyncio
    async def test_get_pagination_custom(self):
        """Test pagination with custom values."""
        result = await get_pagination(skip=20, limit=50)

        assert result["skip"] == 20
        assert result["limit"] == 50


class TestRateLimiting:
    """Test rate limiting dependency."""

    @pytest.mark.asyncio
    async def test_check_rate_limit(self):
        """Test rate limiting check (currently a pass-through)."""
        # Should not raise any exception
        await check_rate_limit(x_forwarded_for="192.168.1.1")
        await check_rate_limit(x_forwarded_for=None)


class TestFeatureFlags:
    """Test feature flag checking."""

    @pytest.mark.asyncio
    async def test_check_feature_enabled_no_flags(self):
        """Test feature check when no flags configured."""
        from src.dependencies import check_feature_enabled

        checker = await check_feature_enabled("test_feature")
        # Should not raise exception when no feature flags configured
        await checker()

    # Note: Skipping tests that modify frozen settings
    # In production, feature flags would be configured via environment variables


class TestHealthCheck:
    """Test health check dependency."""

    @pytest.mark.asyncio
    async def test_check_system_health(self, mock_request):
        """Test system health check."""
        from src.dependencies import check_system_health

        result = await check_system_health(mock_request)

        assert "api" in result
        assert result["api"] == "healthy"
        assert "services" in result
        assert isinstance(result["services"], dict)
