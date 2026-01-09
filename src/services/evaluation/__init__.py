"""
Evaluation harness for RAG system quality assurance.

Provides comprehensive metrics, dataset management, and evaluation workflows
for continuous integration and weekly reporting.
"""

from .dataset import EvalDataset, EvalExample
from .harness import EvalResult, EvaluationHarness, ThresholdConfig
from .metrics import MetricType, RAGMetrics

__all__ = [
    "EvaluationHarness",
    "EvalResult",
    "ThresholdConfig",
    "RAGMetrics",
    "MetricType",
    "EvalDataset",
    "EvalExample",
]
