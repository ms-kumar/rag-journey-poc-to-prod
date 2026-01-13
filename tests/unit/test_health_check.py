"""
Tests for health check functionality.
"""

import time
from unittest.mock import Mock

import pytest

from src.schemas.api.health import ComponentHealth, ServiceStatus
from src.services.health_check import (
    check_all_components,
    check_embeddings_health,
    check_generation_health,
    check_vectorstore_health,
    determine_overall_status,
    get_dependency_versions,
    get_system_info,
    get_uptime,
)


class TestGetUptime:
    """Tests for uptime tracking."""

    def test_uptime_increases(self):
        """Test that uptime increases over time."""
        uptime1 = get_uptime()
        time.sleep(0.01)
        uptime2 = get_uptime()
        assert uptime2 > uptime1

    def test_uptime_is_positive(self):
        """Test that uptime is always positive."""
        uptime = get_uptime()
        assert uptime >= 0


class TestGetSystemInfo:
    """Tests for system information retrieval."""

    def test_system_info_contains_required_keys(self):
        """Test that system info contains expected keys."""
        info = get_system_info()
        assert "platform" in info
        assert "python_version" in info
        assert "architecture" in info
        assert "processor" in info

    def test_system_info_values_are_strings(self):
        """Test that all system info values are strings."""
        info = get_system_info()
        for value in info.values():
            assert isinstance(value, str)


class TestGetDependencyVersions:
    """Tests for dependency version retrieval."""

    def test_dependency_versions_includes_key_packages(self):
        """Test that key dependencies are included."""
        versions = get_dependency_versions()
        expected_packages = [
            "fastapi",
            "qdrant-client",
            "langchain",
            "transformers",
            "torch",
            "pydantic",
        ]
        for package in expected_packages:
            assert package in versions

    def test_dependency_versions_are_strings(self):
        """Test that all versions are strings."""
        versions = get_dependency_versions()
        for version in versions.values():
            assert isinstance(version, str)


class TestCheckVectorstoreHealth:
    """Tests for vectorstore health checks."""

    @pytest.mark.asyncio
    async def test_vectorstore_health_no_client(self):
        """Test health check when vectorstore client is None."""
        result = await check_vectorstore_health(None)
        assert result.name == "vectorstore"
        assert result.status == ServiceStatus.UNKNOWN
        assert "not initialized" in result.message

    @pytest.mark.asyncio
    async def test_vectorstore_health_success(self):
        """Test successful vectorstore health check."""
        # Mock vectorstore client
        mock_collection_info = Mock()
        mock_collection_info.points_count = 100
        mock_collection_info.config.params.vectors.size = 384

        mock_client = Mock()
        mock_client.config.collection_name = "test_collection"
        mock_client.qdrant_client.get_collection.return_value = mock_collection_info

        result = await check_vectorstore_health(mock_client)

        assert result.name == "vectorstore"
        assert result.status == ServiceStatus.HEALTHY
        assert "accessible" in result.message
        assert result.details["collection"] == "test_collection"
        assert result.details["points_count"] == 100
        assert result.details["vector_size"] == 384
        assert result.response_time_ms is not None
        assert result.response_time_ms >= 0

    @pytest.mark.asyncio
    async def test_vectorstore_health_failure(self):
        """Test vectorstore health check when connection fails."""
        mock_client = Mock()
        mock_client.config.collection_name = "test_collection"
        mock_client.qdrant_client.get_collection.side_effect = ConnectionError("Connection refused")

        result = await check_vectorstore_health(mock_client)

        assert result.name == "vectorstore"
        assert result.status == ServiceStatus.UNHEALTHY
        assert "Failed to connect" in result.message
        assert result.response_time_ms is not None


class TestCheckEmbeddingsHealth:
    """Tests for embeddings health checks."""

    @pytest.mark.asyncio
    async def test_embeddings_health_no_client(self):
        """Test health check when embeddings client is None."""
        result = await check_embeddings_health(None)
        assert result.name == "embeddings"
        assert result.status == ServiceStatus.UNKNOWN
        assert "not initialized" in result.message

    @pytest.mark.asyncio
    async def test_embeddings_health_success(self):
        """Test successful embeddings health check."""
        mock_client = Mock()
        mock_client.embed_query.return_value = [0.1] * 384

        result = await check_embeddings_health(mock_client)

        assert result.name == "embeddings"
        assert result.status == ServiceStatus.HEALTHY
        assert "responsive" in result.message
        assert result.details["embedding_dim"] == 384
        assert result.response_time_ms is not None
        assert result.response_time_ms >= 0

    @pytest.mark.asyncio
    async def test_embeddings_health_failure(self):
        """Test embeddings health check when service fails."""
        mock_client = Mock()
        mock_client.embed_query.side_effect = RuntimeError("Model not loaded")

        result = await check_embeddings_health(mock_client)

        assert result.name == "embeddings"
        assert result.status == ServiceStatus.UNHEALTHY
        assert "Failed to generate embeddings" in result.message
        assert result.response_time_ms is not None


class TestCheckGenerationHealth:
    """Tests for generation health checks."""

    @pytest.mark.asyncio
    async def test_generation_health_no_client(self):
        """Test health check when generation client is None."""
        result = await check_generation_health(None)
        assert result.name == "generation"
        assert result.status == ServiceStatus.UNKNOWN
        assert "not initialized" in result.message

    @pytest.mark.asyncio
    async def test_generation_health_success(self):
        """Test successful generation health check."""
        mock_client = Mock()
        mock_client.config.model_name = "gpt2"
        mock_client.generate.return_value = "test output"

        result = await check_generation_health(mock_client)

        assert result.name == "generation"
        assert result.status == ServiceStatus.HEALTHY
        assert "responsive" in result.message
        assert result.details["model"] == "gpt2"
        assert result.details["test_output_length"] == 11
        assert result.response_time_ms is not None
        assert result.response_time_ms >= 0

    @pytest.mark.asyncio
    async def test_generation_health_failure(self):
        """Test generation health check when service fails."""
        mock_client = Mock()
        mock_client.config.model_name = "gpt2"
        mock_client.generate.side_effect = RuntimeError("CUDA out of memory")

        result = await check_generation_health(mock_client)

        assert result.name == "generation"
        assert result.status == ServiceStatus.UNHEALTHY
        assert "Failed to generate text" in result.message
        assert result.response_time_ms is not None


class TestCheckAllComponents:
    """Tests for parallel component health checks."""

    @pytest.mark.asyncio
    async def test_check_all_components_all_healthy(self):
        """Test checking all components when all are healthy."""
        # Mock clients
        mock_vectorstore = Mock()
        mock_vectorstore.config.collection_name = "test"
        mock_vectorstore.qdrant_client.get_collection.return_value = Mock(
            points_count=10, config=Mock(params=Mock(vectors=Mock(size=384)))
        )

        mock_embeddings = Mock()
        mock_embeddings.embed_query.return_value = [0.1] * 384

        mock_generation = Mock()
        mock_generation.config.model_name = "gpt2"
        mock_generation.generate.return_value = "output"

        components = await check_all_components(mock_vectorstore, mock_embeddings, mock_generation)

        assert len(components) == 3
        assert all(c.status == ServiceStatus.HEALTHY for c in components)

    @pytest.mark.asyncio
    async def test_check_all_components_some_unhealthy(self):
        """Test checking components when some are unhealthy."""
        # Mock clients - vectorstore fails, others succeed
        mock_vectorstore = Mock()
        mock_vectorstore.config.collection_name = "test"
        mock_vectorstore.qdrant_client.get_collection.side_effect = ConnectionError("Failed")

        mock_embeddings = Mock()
        mock_embeddings.embed_query.return_value = [0.1] * 384

        mock_generation = Mock()
        mock_generation.config.model_name = "gpt2"
        mock_generation.generate.return_value = "output"

        components = await check_all_components(mock_vectorstore, mock_embeddings, mock_generation)

        assert len(components) == 3
        # Find vectorstore component
        vectorstore_component = next(c for c in components if c.name == "vectorstore")
        assert vectorstore_component.status == ServiceStatus.UNHEALTHY

        # Others should be healthy
        embeddings_component = next(c for c in components if c.name == "embeddings")
        generation_component = next(c for c in components if c.name == "generation")
        assert embeddings_component.status == ServiceStatus.HEALTHY
        assert generation_component.status == ServiceStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_check_all_components_none_provided(self):
        """Test checking components when none are provided."""
        components = await check_all_components(None, None, None)

        assert len(components) == 3
        assert all(c.status == ServiceStatus.UNKNOWN for c in components)


class TestDetermineOverallStatus:
    """Tests for overall status determination."""

    def test_determine_status_all_healthy(self):
        """Test status when all components are healthy."""
        components = [
            ComponentHealth(name="test1", status=ServiceStatus.HEALTHY),
            ComponentHealth(name="test2", status=ServiceStatus.HEALTHY),
            ComponentHealth(name="test3", status=ServiceStatus.HEALTHY),
        ]
        status = determine_overall_status(components)
        assert status == ServiceStatus.HEALTHY

    def test_determine_status_one_unhealthy(self):
        """Test status when one component is unhealthy."""
        components = [
            ComponentHealth(name="test1", status=ServiceStatus.HEALTHY),
            ComponentHealth(name="test2", status=ServiceStatus.UNHEALTHY),
            ComponentHealth(name="test3", status=ServiceStatus.HEALTHY),
        ]
        status = determine_overall_status(components)
        assert status == ServiceStatus.UNHEALTHY

    def test_determine_status_one_degraded(self):
        """Test status when one component is degraded."""
        components = [
            ComponentHealth(name="test1", status=ServiceStatus.HEALTHY),
            ComponentHealth(name="test2", status=ServiceStatus.DEGRADED),
            ComponentHealth(name="test3", status=ServiceStatus.HEALTHY),
        ]
        status = determine_overall_status(components)
        assert status == ServiceStatus.DEGRADED

    def test_determine_status_one_unknown(self):
        """Test status when one component is unknown."""
        components = [
            ComponentHealth(name="test1", status=ServiceStatus.HEALTHY),
            ComponentHealth(name="test2", status=ServiceStatus.UNKNOWN),
            ComponentHealth(name="test3", status=ServiceStatus.HEALTHY),
        ]
        status = determine_overall_status(components)
        assert status == ServiceStatus.DEGRADED

    def test_determine_status_unhealthy_takes_precedence(self):
        """Test that unhealthy takes precedence over degraded."""
        components = [
            ComponentHealth(name="test1", status=ServiceStatus.DEGRADED),
            ComponentHealth(name="test2", status=ServiceStatus.UNHEALTHY),
            ComponentHealth(name="test3", status=ServiceStatus.HEALTHY),
        ]
        status = determine_overall_status(components)
        assert status == ServiceStatus.UNHEALTHY

    def test_determine_status_empty_list(self):
        """Test status with empty component list."""
        status = determine_overall_status([])
        assert status == ServiceStatus.UNKNOWN


class TestHealthCheckModels:
    """Tests for health check Pydantic models."""

    def test_component_health_creation(self):
        """Test creating ComponentHealth model."""
        component = ComponentHealth(
            name="test_service",
            status=ServiceStatus.HEALTHY,
            message="All good",
            details={"version": "1.0"},
            response_time_ms=10.5,
        )
        assert component.name == "test_service"
        assert component.status == ServiceStatus.HEALTHY
        assert component.message == "All good"
        assert component.details["version"] == "1.0"
        assert component.response_time_ms == 10.5

    def test_component_health_minimal(self):
        """Test creating ComponentHealth with minimal fields."""
        component = ComponentHealth(name="test", status=ServiceStatus.UNKNOWN)
        assert component.name == "test"
        assert component.status == ServiceStatus.UNKNOWN
        assert component.message is None
        assert component.details is None
        assert component.response_time_ms is None

    def test_service_status_enum(self):
        """Test ServiceStatus enum values."""
        assert ServiceStatus.HEALTHY.value == "healthy"
        assert ServiceStatus.DEGRADED.value == "degraded"
        assert ServiceStatus.UNHEALTHY.value == "unhealthy"
        assert ServiceStatus.UNKNOWN.value == "unknown"
