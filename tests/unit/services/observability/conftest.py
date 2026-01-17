"""Shared fixtures for observability tests."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_tracer() -> MagicMock:
    """Create a mock tracer."""
    tracer = MagicMock()
    tracer.service_name = "test-service"
    tracer.start_span.return_value.__enter__ = MagicMock()
    tracer.start_span.return_value.__exit__ = MagicMock()
    return tracer


@pytest.fixture
def sample_span_context() -> dict[str, Any]:
    """Create a sample span context."""
    return {
        "trace_id": "abc123",
        "span_id": "def456",
        "parent_span_id": None,
        "service_name": "test-service",
        "operation_name": "test-operation",
    }


@pytest.fixture
def sample_log_entry() -> dict[str, Any]:
    """Create a sample log entry."""
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "level": "INFO",
        "message": "Test log message",
        "correlation_id": "corr-123",
        "service": "test-service",
        "extra": {"key": "value"},
    }


@pytest.fixture
def sample_metrics() -> dict[str, Any]:
    """Create sample metrics data."""
    return {
        "latency": {
            "p50": 50.0,
            "p95": 150.0,
            "p99": 300.0,
            "count": 1000,
        },
        "cost": {
            "total_tokens": 50000,
            "total_cost_usd": 0.05,
            "api_calls": 100,
        },
        "quality": {
            "relevance_scores": [0.8, 0.9, 0.85],
            "error_rate": 0.01,
        },
    }


@pytest.fixture
def sample_slo_definition() -> dict[str, Any]:
    """Create a sample SLO definition."""
    return {
        "name": "availability",
        "target": 0.999,
        "window_seconds": 86400,  # 24 hours
        "description": "Service availability SLO",
    }


@pytest.fixture
def sample_golden_trace() -> dict[str, Any]:
    """Create a sample golden trace."""
    return {
        "trace_id": "golden-123",
        "name": "standard_rag_query",
        "spans": [
            {
                "name": "retrieval",
                "duration_ms": 45.0,
                "attributes": {"num_results": 5},
            },
            {
                "name": "reranking",
                "duration_ms": 30.0,
                "attributes": {"reranked_count": 3},
            },
            {
                "name": "generation",
                "duration_ms": 200.0,
                "attributes": {"tokens": 150},
            },
        ],
        "metadata": {
            "created_at": datetime.now(UTC).isoformat(),
            "description": "Standard RAG query golden trace",
        },
    }
