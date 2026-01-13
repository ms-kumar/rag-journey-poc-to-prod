"""Pytest configuration for performance tests."""

import time

import pytest


@pytest.fixture
def performance_tracker():
    """Provide a simple performance tracker."""

    class PerformanceTracker:
        def __init__(self):
            self.metrics = {}

        def start(self, name):
            self.metrics[name] = {"start": time.time()}

        def end(self, name):
            if name in self.metrics:
                self.metrics[name]["end"] = time.time()
                self.metrics[name]["duration"] = (
                    self.metrics[name]["end"] - self.metrics[name]["start"]
                )

        def get_duration(self, name):
            return self.metrics.get(name, {}).get("duration", 0)

    return PerformanceTracker()


@pytest.fixture
def performance_config():
    """Provide performance configuration."""
    return {
        "enable_profiling": True,
        "track_memory": True,
        "track_latency": True,
    }
