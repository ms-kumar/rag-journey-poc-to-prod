"""Pytest configuration for integration tests."""

import gc

import pytest


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup resources after each test to prevent memory leaks."""
    yield
    # Force garbage collection to clean up any unclosed ProcessPoolExecutors
    gc.collect()
