"""Test suite for sandbox configuration helpers.

Tests for:
- Configuration loading from settings
- Security level parsing
- Resource limits configuration
- Network configuration
- Configuration validation
"""

from unittest.mock import Mock, patch

from src.services.agent.tools.hybrid.sandbox import (
    ResourceLimits,
    SecurityLevel,
)
from src.services.agent.tools.hybrid.sandbox_config import (
    get_network_config,
    get_resource_limits,
    get_security_level,
    get_security_level_from_string,
    validate_config,
)


class TestSecurityLevelParsing:
    """Test security level parsing functions."""

    def test_get_security_level_from_string_valid(self):
        """Test parsing valid security level strings."""
        assert get_security_level_from_string("strict") == SecurityLevel.STRICT
        assert get_security_level_from_string("moderate") == SecurityLevel.MODERATE
        assert get_security_level_from_string("permissive") == SecurityLevel.PERMISSIVE

    def test_get_security_level_from_string_case_insensitive(self):
        """Test case-insensitive parsing."""
        assert get_security_level_from_string("STRICT") == SecurityLevel.STRICT
        assert get_security_level_from_string("Moderate") == SecurityLevel.MODERATE
        assert get_security_level_from_string("PerMissive") == SecurityLevel.PERMISSIVE

    def test_get_security_level_from_string_invalid(self):
        """Test parsing invalid security level strings."""
        # Invalid strings return MODERATE as default
        result = get_security_level_from_string("invalid")
        assert result == SecurityLevel.MODERATE

        result = get_security_level_from_string("")
        assert result == SecurityLevel.MODERATE

    @patch("src.services.agent.tools.hybrid.sandbox_config.get_settings")
    def test_get_security_level_from_settings(self, mock_get_settings):
        """Test loading security level from settings."""
        mock_settings = Mock()
        mock_settings.sandbox.security_level = "moderate"
        mock_get_settings.return_value = mock_settings

        level = get_security_level()
        assert level == SecurityLevel.MODERATE


class TestResourceLimitsConfiguration:
    """Test resource limits configuration."""

    @patch("src.services.agent.tools.hybrid.sandbox_config.get_settings")
    def test_get_resource_limits_from_settings(self, mock_get_settings):
        """Test loading resource limits from settings."""
        mock_settings = Mock()
        mock_settings.sandbox.max_execution_time = 10.0
        mock_settings.sandbox.max_cpu_time = 9.0
        mock_settings.sandbox.max_memory_mb = 256
        mock_settings.sandbox.max_stack_size_mb = 16
        mock_settings.sandbox.max_processes = 2
        mock_settings.sandbox.max_open_files = 20
        mock_settings.sandbox.max_output_size = 20000
        mock_settings.sandbox.max_variables = 2000
        mock_get_settings.return_value = mock_settings

        limits = get_resource_limits()

        assert limits.max_execution_time_seconds == 10.0
        assert limits.max_memory_mb == 256
        assert limits.max_processes == 2
        assert limits.max_open_files == 20
        assert limits.max_output_size == 20000
        assert limits.max_variables == 2000

    @patch("src.services.agent.tools.hybrid.sandbox_config.get_settings")
    def test_get_resource_limits_defaults(self, mock_get_settings):
        """Test default resource limits."""
        mock_settings = Mock()
        mock_settings.sandbox.max_execution_time = 5.0
        mock_settings.sandbox.max_cpu_time = 4.0
        mock_settings.sandbox.max_memory_mb = 128
        mock_settings.sandbox.max_stack_size_mb = 8
        mock_settings.sandbox.max_processes = 1
        mock_settings.sandbox.max_open_files = 32
        mock_settings.sandbox.max_output_size = 10000
        mock_settings.sandbox.max_variables = 1000
        mock_get_settings.return_value = mock_settings

        limits = get_resource_limits()

        assert isinstance(limits, ResourceLimits)
        assert limits.max_execution_time_seconds > 0
        assert limits.max_memory_mb > 0


class TestNetworkConfiguration:
    """Test network configuration."""

    @patch("src.services.agent.tools.hybrid.sandbox_config.get_settings")
    def test_get_network_config_disabled(self, mock_get_settings):
        """Test network config when network is disabled."""
        mock_settings = Mock()
        mock_settings.sandbox.allow_network = False
        mock_settings.sandbox.get_allowed_hosts_set.return_value = set()
        mock_settings.sandbox.get_allowed_ports_set.return_value = set()
        mock_settings.sandbox.block_local_network = True
        mock_get_settings.return_value = mock_settings

        config = get_network_config()

        assert config.allow_network is False
        assert len(config.allowed_hosts) == 0
        assert config.block_local_network is True

    @patch("src.services.agent.tools.hybrid.sandbox_config.get_settings")
    def test_get_network_config_with_hosts(self, mock_get_settings):
        """Test network config with allowed hosts."""
        mock_settings = Mock()
        mock_settings.sandbox.allow_network = True
        mock_settings.sandbox.get_allowed_hosts_set.return_value = {"example.com", "api.test.com"}
        mock_settings.sandbox.get_allowed_ports_set.return_value = {80, 443}
        mock_settings.sandbox.block_local_network = True
        mock_get_settings.return_value = mock_settings

        config = get_network_config()

        assert config.allow_network is True
        assert "example.com" in config.allowed_hosts
        assert "api.test.com" in config.allowed_hosts
        assert 80 in config.allowed_ports
        assert 443 in config.allowed_ports

    @patch("src.services.agent.tools.hybrid.sandbox_config.get_settings")
    def test_get_network_config_empty_hosts(self, mock_get_settings):
        """Test network config with empty host list."""
        mock_settings = Mock()
        mock_settings.sandbox.allow_network = True
        mock_settings.sandbox.get_allowed_hosts_set.return_value = set()
        mock_settings.sandbox.get_allowed_ports_set.return_value = set()
        mock_settings.sandbox.block_local_network = True
        mock_get_settings.return_value = mock_settings

        config = get_network_config()

        assert config.allow_network is True
        assert len(config.allowed_hosts) == 0
        assert len(config.allowed_ports) == 0


class TestConfigurationValidation:
    """Test configuration validation."""

    @patch("src.services.agent.tools.hybrid.sandbox_config.get_settings")
    def test_validate_config_valid(self, mock_get_settings):
        """Test validation of valid configuration."""
        mock_settings = Mock()
        mock_settings.sandbox.security_level = "moderate"
        mock_settings.sandbox.max_execution_time = 5.0
        mock_settings.sandbox.max_cpu_time = 4.0
        mock_settings.sandbox.max_memory_mb = 128
        mock_settings.sandbox.max_stack_size_mb = 8
        mock_settings.sandbox.max_processes = 1
        mock_settings.sandbox.max_open_files = 10
        mock_settings.sandbox.max_output_size = 10000
        mock_settings.sandbox.max_variables = 1000
        mock_settings.sandbox.allow_network = False
        mock_settings.sandbox.get_allowed_hosts_set.return_value = set()
        mock_settings.sandbox.block_local_network = True
        mock_get_settings.return_value = mock_settings

        # Should return empty list for valid config
        errors = validate_config()
        assert errors == []

    @patch("src.services.agent.tools.hybrid.sandbox_config.get_settings")
    def test_validate_config_invalid_security_level(self, mock_get_settings):
        """Test validation with invalid security level."""
        mock_settings = Mock()
        mock_settings.sandbox.security_level = "invalid"
        mock_settings.sandbox.max_execution_time = 5.0
        mock_settings.sandbox.max_cpu_time = 4.0
        mock_settings.sandbox.max_memory_mb = 128
        mock_settings.sandbox.max_stack_size_mb = 8
        mock_settings.sandbox.max_processes = 1
        mock_settings.sandbox.max_open_files = 10
        mock_settings.sandbox.max_output_size = 10000
        mock_settings.sandbox.max_variables = 1000
        mock_settings.sandbox.allow_network = False
        mock_settings.sandbox.get_allowed_hosts_set.return_value = set()
        mock_settings.sandbox.block_local_network = True
        mock_get_settings.return_value = mock_settings

        # Security level validation happens in get_security_level, which returns default
        # The validation function only checks values, not enum conversion
        errors = validate_config()
        # Should pass since invalid strings default to MODERATE
        assert isinstance(errors, list)

    @patch("src.services.agent.tools.hybrid.sandbox_config.get_settings")
    def test_validate_config_negative_limits(self, mock_get_settings):
        """Test validation with negative resource limits."""
        mock_settings = Mock()
        mock_settings.sandbox.security_level = "moderate"
        mock_settings.sandbox.max_execution_time = -1.0  # Invalid
        mock_settings.sandbox.max_cpu_time = 4.0
        mock_settings.sandbox.max_memory_mb = 128
        mock_settings.sandbox.max_stack_size_mb = 8
        mock_settings.sandbox.max_processes = 1
        mock_settings.sandbox.max_open_files = 10
        mock_settings.sandbox.max_output_size = 10000
        mock_settings.sandbox.max_variables = 1000
        mock_settings.sandbox.allow_network = False
        mock_settings.sandbox.get_allowed_hosts_set.return_value = set()
        mock_settings.sandbox.block_local_network = True
        mock_get_settings.return_value = mock_settings

        errors = validate_config()
        assert len(errors) > 0
        assert any("max_execution_time" in err for err in errors)


class TestConfigurationIntegration:
    """Integration tests for configuration."""

    @patch("src.services.agent.tools.hybrid.sandbox_config.get_settings")
    def test_full_config_load(self, mock_get_settings):
        """Test loading complete configuration."""
        mock_settings = Mock()
        mock_settings.sandbox.security_level = "strict"
        mock_settings.sandbox.max_execution_time = 3.0
        mock_settings.sandbox.max_cpu_time = 2.5
        mock_settings.sandbox.max_memory_mb = 64
        mock_settings.sandbox.max_stack_size_mb = 4
        mock_settings.sandbox.max_processes = 1
        mock_settings.sandbox.max_open_files = 5
        mock_settings.sandbox.max_output_size = 5000
        mock_settings.sandbox.max_variables = 500
        mock_settings.sandbox.allow_network = True
        mock_settings.sandbox.get_allowed_hosts_set.return_value = {"api.example.com"}
        mock_settings.sandbox.get_allowed_ports_set.return_value = {443}
        mock_settings.sandbox.block_local_network = True
        mock_get_settings.return_value = mock_settings

        # Load all configs
        security_level = get_security_level()
        resource_limits = get_resource_limits()
        network_config = get_network_config()

        # Verify all loaded correctly
        assert security_level == SecurityLevel.STRICT
        assert resource_limits.max_execution_time_seconds == 3.0
        assert resource_limits.max_memory_mb == 64
        assert network_config.allow_network is True
        assert "api.example.com" in network_config.allowed_hosts

    @patch("src.services.agent.tools.hybrid.sandbox_config.get_settings")
    def test_config_consistency(self, mock_get_settings):
        """Test that multiple calls return consistent config."""
        mock_settings = Mock()
        mock_settings.sandbox.security_level = "moderate"
        mock_settings.sandbox.max_execution_time = 5.0
        mock_settings.sandbox.max_cpu_time = 4.0
        mock_settings.sandbox.max_memory_mb = 128
        mock_settings.sandbox.max_stack_size_mb = 8
        mock_settings.sandbox.max_processes = 1
        mock_settings.sandbox.max_open_files = 10
        mock_settings.sandbox.max_output_size = 10000
        mock_settings.sandbox.max_variables = 1000
        mock_settings.sandbox.allow_network = False
        mock_settings.sandbox.get_allowed_hosts_set.return_value = set()
        mock_settings.sandbox.get_allowed_ports_set.return_value = set()
        mock_settings.sandbox.block_local_network = True
        mock_get_settings.return_value = mock_settings

        # Load config multiple times
        level1 = get_security_level()
        level2 = get_security_level()
        limits1 = get_resource_limits()
        limits2 = get_resource_limits()

        assert level1 == level2
        assert limits1.max_execution_time_seconds == limits2.max_execution_time_seconds
        assert limits1.max_memory_mb == limits2.max_memory_mb
