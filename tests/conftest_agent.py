"""Pytest configuration for agent tests."""

import pytest


@pytest.fixture(autouse=True)
def reset_global_registry():
    """Reset global registry before each test."""
    # Reset the global registry singleton
    import src.services.agent.tools.registry as registry_module

    registry_module._global_registry = None
    yield
    # Clean up after test
    registry_module._global_registry = None


@pytest.fixture(autouse=True)
def reset_global_tracker():
    """Reset global metrics tracker before each test."""
    # Reset the global tracker singleton
    import src.services.agent.metrics.tracker as tracker_module

    tracker_module._global_tracker = None
    yield
    # Clean up after test
    tracker_module._global_tracker = None
