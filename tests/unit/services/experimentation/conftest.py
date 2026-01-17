"""Shared fixtures for experimentation tests."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def sample_experiment() -> dict[str, Any]:
    """Create a sample experiment definition."""
    return {
        "id": "test-experiment-001",
        "name": "Test Reranker Model",
        "description": "A/B test for reranker model performance",
        "status": "running",
        "variants": [
            {"id": "control", "name": "cross-encoder-v1", "weight": 50},
            {"id": "treatment", "name": "cross-encoder-v2", "weight": 50},
        ],
        "start_date": datetime.now(UTC).isoformat(),
        "end_date": None,
        "metrics": ["latency_ms", "relevance_score", "user_satisfaction"],
    }


@pytest.fixture
def sample_feature_flag() -> dict[str, Any]:
    """Create a sample feature flag."""
    return {
        "id": "new-chunking-strategy",
        "name": "New Chunking Strategy",
        "description": "Enable new semantic chunking",
        "enabled": True,
        "rollout_percentage": 25,
        "targeting_rules": {
            "user_groups": ["beta_testers"],
            "user_ids": ["user-123", "user-456"],
        },
    }


@pytest.fixture
def sample_experiment_results() -> dict[str, Any]:
    """Create sample experiment results."""
    return {
        "experiment_id": "test-experiment-001",
        "control": {
            "sample_size": 500,
            "latency_ms": {"mean": 150.0, "std": 30.0},
            "relevance_score": {"mean": 0.75, "std": 0.1},
            "conversion_rate": 0.12,
        },
        "treatment": {
            "sample_size": 500,
            "latency_ms": {"mean": 120.0, "std": 25.0},
            "relevance_score": {"mean": 0.82, "std": 0.08},
            "conversion_rate": 0.15,
        },
    }


@pytest.fixture
def sample_canary_config() -> dict[str, Any]:
    """Create a sample canary configuration."""
    return {
        "deployment_id": "canary-001",
        "traffic_percentage": 5,
        "health_thresholds": {
            "error_rate": 0.05,
            "latency_p99_ms": 500,
            "min_requests": 100,
        },
        "promotion_rules": {
            "auto_promote": True,
            "min_duration_seconds": 300,
            "success_rate_threshold": 0.99,
        },
        "rollback_rules": {
            "auto_rollback": True,
            "error_rate_threshold": 0.10,
        },
    }


@pytest.fixture
def mock_experiment_manager() -> MagicMock:
    """Create a mock experiment manager."""
    manager = MagicMock()
    manager.get_variant.return_value = {"id": "control", "name": "default"}
    manager.record_exposure.return_value = None
    manager.record_outcome.return_value = None
    return manager


@pytest.fixture
def mock_flag_manager() -> MagicMock:
    """Create a mock feature flag manager."""
    manager = MagicMock()
    manager.is_enabled.return_value = True
    manager.get_value.return_value = "default_value"
    return manager


@pytest.fixture
def sample_analysis_result() -> dict[str, Any]:
    """Create a sample statistical analysis result."""
    return {
        "metric": "latency_ms",
        "test_type": "t_test",
        "p_value": 0.023,
        "is_significant": True,
        "confidence_level": 0.95,
        "effect_size": 0.35,
        "control_mean": 150.0,
        "treatment_mean": 120.0,
        "relative_improvement": -0.20,
        "recommendation": "Treatment shows significant improvement",
    }


@pytest.fixture
def sample_report_data() -> dict[str, Any]:
    """Create sample report data."""
    return {
        "experiment_id": "test-experiment-001",
        "experiment_name": "Test Reranker Model",
        "status": "completed",
        "duration_days": 14,
        "total_participants": 1000,
        "summary": {
            "winner": "treatment",
            "confidence": 0.95,
            "key_findings": [
                "20% latency improvement",
                "9% relevance score increase",
                "25% conversion rate lift",
            ],
        },
        "metrics": {
            "latency_ms": {"winner": "treatment", "p_value": 0.023},
            "relevance_score": {"winner": "treatment", "p_value": 0.008},
            "conversion_rate": {"winner": "treatment", "p_value": 0.041},
        },
    }
