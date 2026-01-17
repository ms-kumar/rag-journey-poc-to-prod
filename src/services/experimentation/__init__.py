"""
Experimentation module for RAG pipeline.

Provides A/B testing, feature flags, experiment management,
outcome analysis, canary deployments, and reporting.

Components:
- Experiments: Define and manage A/B experiments for prompts/models
- Feature Flags: Toggle features with gradual rollout support
- Analysis: Statistical analysis of experiment outcomes
- Canary: Canary deployment with automatic rollback
- Reports: Generate experiment reports and insights
"""

from src.services.experimentation.analysis import (
    ExperimentAnalysis,
    ExperimentOutcome,
    StatisticalTest,
    analyze_experiment,
)
from src.services.experimentation.canary import (
    CanaryConfig,
    CanaryDeployment,
    CanaryManager,
    CanaryStatus,
)
from src.services.experimentation.experiments import (
    Assignment,
    Experiment,
    ExperimentManager,
    ExperimentStatus,
    Variant,
)
from src.services.experimentation.feature_flags import (
    FeatureFlag,
    FlagManager,
    FlagType,
    RolloutConfig,
)
from src.services.experimentation.reports import (
    ExperimentReport,
    ReportFormat,
    ReportGenerator,
)

__all__ = [
    # Experiments
    "Assignment",
    "Experiment",
    "ExperimentManager",
    "ExperimentStatus",
    "Variant",
    # Feature Flags
    "FeatureFlag",
    "FlagManager",
    "FlagType",
    "RolloutConfig",
    # Analysis
    "ExperimentAnalysis",
    "ExperimentOutcome",
    "StatisticalTest",
    "analyze_experiment",
    # Canary
    "CanaryConfig",
    "CanaryDeployment",
    "CanaryManager",
    "CanaryStatus",
    # Reports
    "ExperimentReport",
    "ReportFormat",
    "ReportGenerator",
]
