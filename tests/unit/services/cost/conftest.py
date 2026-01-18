"""Fixtures for cost module tests."""

import pytest

from src.services.cost.autoscaler import Autoscaler, AutoscalingPolicy, LoadMetrics
from src.services.cost.model_selector import ModelCandidate, ModelSelector, ModelTier
from src.services.cost.tracker import CostTracker


@pytest.fixture
def cost_tracker() -> CostTracker:
    """Create a fresh cost tracker."""
    return CostTracker()


@pytest.fixture
def autoscaler() -> Autoscaler:
    """Create autoscaler with default policy."""
    return Autoscaler()


@pytest.fixture
def autoscaler_with_policy() -> Autoscaler:
    """Create autoscaler with custom policy."""
    policy = AutoscalingPolicy(
        min_instances=1,
        max_instances=5,
        scale_up_cpu_threshold=60,
        scale_down_cpu_threshold=20,
        cooldown_period=0,  # No cooldown for testing
    )
    return Autoscaler(policy=policy)


@pytest.fixture
def model_selector() -> ModelSelector:
    """Create model selector with default models."""
    return ModelSelector()


@pytest.fixture
def custom_models() -> list[ModelCandidate]:
    """Create custom model candidates for testing."""
    return [
        ModelCandidate(
            name="cheap-model",
            tier=ModelTier.BUDGET,
            cost_per_1k=0.10,
            avg_latency_ms=50,
            quality_score=0.60,
        ),
        ModelCandidate(
            name="balanced-model",
            tier=ModelTier.BALANCED,
            cost_per_1k=0.50,
            avg_latency_ms=100,
            quality_score=0.75,
        ),
        ModelCandidate(
            name="premium-model",
            tier=ModelTier.PREMIUM,
            cost_per_1k=2.00,
            avg_latency_ms=200,
            quality_score=0.90,
        ),
    ]


@pytest.fixture
def normal_load_metrics() -> LoadMetrics:
    """Normal load metrics (no scaling needed)."""
    return LoadMetrics(
        cpu_usage=50.0,
        queue_size=10,
        active_requests=5,
        avg_latency_ms=100.0,
        p95_latency_ms=200.0,
        error_rate=1.0,
        requests_per_second=50.0,
    )


@pytest.fixture
def high_load_metrics() -> LoadMetrics:
    """High load metrics (scale up needed)."""
    return LoadMetrics(
        cpu_usage=85.0,
        queue_size=100,
        active_requests=20,
        avg_latency_ms=500.0,
        p95_latency_ms=1500.0,
        error_rate=5.0,
        requests_per_second=10.0,
    )


@pytest.fixture
def low_load_metrics() -> LoadMetrics:
    """Low load metrics (scale down possible)."""
    return LoadMetrics(
        cpu_usage=15.0,
        queue_size=2,
        active_requests=1,
        avg_latency_ms=20.0,
        p95_latency_ms=50.0,
        error_rate=0.0,
        requests_per_second=5.0,
    )
