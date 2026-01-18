"""
Statistical analysis for A/B experiment outcomes.

Provides statistical tests, confidence intervals, and
outcome analysis for experiment evaluation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class StatisticalTest(str, Enum):
    """Type of statistical test."""

    T_TEST = "t_test"
    Z_TEST = "z_test"
    CHI_SQUARE = "chi_square"
    MANN_WHITNEY = "mann_whitney"


class SignificanceLevel(str, Enum):
    """Significance level for statistical tests."""

    P_01 = "0.01"
    P_05 = "0.05"
    P_10 = "0.10"


@dataclass
class ExperimentOutcome:
    """Outcome data for a single variant."""

    variant_name: str
    sample_size: int
    metric_values: list[float]
    mean: float = 0.0
    std_dev: float = 0.0
    median: float = 0.0
    min_value: float = 0.0
    max_value: float = 0.0
    conversion_count: int = 0
    conversion_rate: float = 0.0

    def __post_init__(self) -> None:
        """Calculate statistics from metric values."""
        if self.metric_values:
            self.sample_size = len(self.metric_values)
            self.mean = sum(self.metric_values) / self.sample_size
            self.min_value = min(self.metric_values)
            self.max_value = max(self.metric_values)

            # Calculate standard deviation
            if self.sample_size > 1:
                variance = sum((x - self.mean) ** 2 for x in self.metric_values) / (
                    self.sample_size - 1
                )
                self.std_dev = math.sqrt(variance)

            # Calculate median
            sorted_values = sorted(self.metric_values)
            mid = self.sample_size // 2
            if self.sample_size % 2 == 0:
                self.median = (sorted_values[mid - 1] + sorted_values[mid]) / 2
            else:
                self.median = sorted_values[mid]

    def to_dict(self) -> dict[str, Any]:
        """Convert outcome to dictionary."""
        return {
            "variant_name": self.variant_name,
            "sample_size": self.sample_size,
            "mean": self.mean,
            "std_dev": self.std_dev,
            "median": self.median,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "conversion_count": self.conversion_count,
            "conversion_rate": self.conversion_rate,
        }


@dataclass
class StatTestResult:
    """Result of a statistical test."""

    test_type: StatisticalTest
    statistic: float
    p_value: float
    confidence_level: float
    is_significant: bool
    effect_size: float = 0.0
    confidence_interval: tuple[float, float] = (0.0, 0.0)

    def to_dict(self) -> dict[str, Any]:
        """Convert test result to dictionary."""
        return {
            "test_type": self.test_type.value,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "confidence_level": self.confidence_level,
            "is_significant": self.is_significant,
            "effect_size": self.effect_size,
            "confidence_interval": list(self.confidence_interval),
        }


@dataclass
class ExperimentAnalysis:
    """Complete analysis of an experiment."""

    experiment_id: str
    experiment_name: str
    control_outcome: ExperimentOutcome
    treatment_outcomes: list[ExperimentOutcome]
    test_results: list[StatTestResult]
    winner: str | None = None
    recommendation: str = ""
    analyzed_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert analysis to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "control_outcome": self.control_outcome.to_dict(),
            "treatment_outcomes": [t.to_dict() for t in self.treatment_outcomes],
            "test_results": [t.to_dict() for t in self.test_results],
            "winner": self.winner,
            "recommendation": self.recommendation,
            "analyzed_at": self.analyzed_at.isoformat(),
            "metadata": self.metadata,
        }


class StatisticalAnalyzer:
    """Performs statistical analysis on experiment data."""

    # Z-scores for common confidence levels
    Z_SCORES = {
        0.90: 1.645,
        0.95: 1.96,
        0.99: 2.576,
    }

    # T-distribution critical values (approximate for large samples)
    T_CRITICAL = {
        0.90: 1.645,
        0.95: 1.96,
        0.99: 2.576,
    }

    def __init__(self, confidence_level: float = 0.95):
        """Initialize analyzer with confidence level."""
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

    def t_test(
        self,
        control: ExperimentOutcome,
        treatment: ExperimentOutcome,
    ) -> StatTestResult:
        """
        Perform independent two-sample t-test.

        Tests whether the means of two groups are significantly different.
        """
        n1, n2 = control.sample_size, treatment.sample_size
        m1, m2 = control.mean, treatment.mean
        s1, s2 = control.std_dev, treatment.std_dev

        # Check for sufficient sample size
        if n1 < 2 or n2 < 2:
            return StatTestResult(
                test_type=StatisticalTest.T_TEST,
                statistic=0.0,
                p_value=1.0,
                confidence_level=self.confidence_level,
                is_significant=False,
            )

        # Pooled standard error
        se = math.sqrt((s1**2 / n1) + (s2**2 / n2))
        if se == 0:
            return StatTestResult(
                test_type=StatisticalTest.T_TEST,
                statistic=0.0,
                p_value=1.0,
                confidence_level=self.confidence_level,
                is_significant=False,
            )

        # T-statistic
        t_stat = (m2 - m1) / se

        # Degrees of freedom (Welch's approximation)
        df_num = ((s1**2 / n1) + (s2**2 / n2)) ** 2
        df_denom = ((s1**2 / n1) ** 2 / (n1 - 1)) + ((s2**2 / n2) ** 2 / (n2 - 1))
        _df = df_num / df_denom if df_denom > 0 else n1 + n2 - 2  # noqa: F841

        # Approximate p-value using normal distribution for large samples
        p_value = self._approximate_p_value(abs(t_stat))

        # Effect size (Cohen's d)
        pooled_std = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        effect_size = (m2 - m1) / pooled_std if pooled_std > 0 else 0

        # Confidence interval for difference
        t_critical = self.T_CRITICAL.get(self.confidence_level, 1.96)
        margin = t_critical * se
        ci = (m2 - m1 - margin, m2 - m1 + margin)

        return StatTestResult(
            test_type=StatisticalTest.T_TEST,
            statistic=t_stat,
            p_value=p_value,
            confidence_level=self.confidence_level,
            is_significant=p_value < self.alpha,
            effect_size=effect_size,
            confidence_interval=ci,
        )

    def z_test_proportions(
        self,
        control: ExperimentOutcome,
        treatment: ExperimentOutcome,
    ) -> StatTestResult:
        """
        Perform z-test for proportions (conversion rates).

        Tests whether conversion rates are significantly different.
        """
        n1, n2 = control.sample_size, treatment.sample_size
        p1, p2 = control.conversion_rate, treatment.conversion_rate

        if n1 == 0 or n2 == 0:
            return StatTestResult(
                test_type=StatisticalTest.Z_TEST,
                statistic=0.0,
                p_value=1.0,
                confidence_level=self.confidence_level,
                is_significant=False,
            )

        # Pooled proportion
        p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)

        # Standard error
        se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
        if se == 0:
            return StatTestResult(
                test_type=StatisticalTest.Z_TEST,
                statistic=0.0,
                p_value=1.0,
                confidence_level=self.confidence_level,
                is_significant=False,
            )

        # Z-statistic
        z_stat = (p2 - p1) / se

        # P-value
        p_value = self._approximate_p_value(abs(z_stat))

        # Confidence interval for difference
        z_critical = self.Z_SCORES.get(self.confidence_level, 1.96)
        se_diff = math.sqrt((p1 * (1 - p1) / n1) + (p2 * (1 - p2) / n2))
        margin = z_critical * se_diff
        ci = (p2 - p1 - margin, p2 - p1 + margin)

        return StatTestResult(
            test_type=StatisticalTest.Z_TEST,
            statistic=z_stat,
            p_value=p_value,
            confidence_level=self.confidence_level,
            is_significant=p_value < self.alpha,
            effect_size=p2 - p1,  # Absolute difference in proportions
            confidence_interval=ci,
        )

    def _approximate_p_value(self, z: float) -> float:
        """
        Approximate two-tailed p-value from z-score.

        Uses the error function approximation.
        """
        # Approximation of the cumulative normal distribution
        # P(Z > z) ≈ erfc(z/sqrt(2))/2
        return 2 * (1 - self._normal_cdf(abs(z)))

    def _normal_cdf(self, x: float) -> float:
        """Approximate cumulative normal distribution."""
        # Approximation using error function
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def calculate_sample_size(
        self,
        baseline_rate: float,
        minimum_detectable_effect: float,
        power: float = 0.8,
    ) -> int:
        """
        Calculate required sample size per variant.

        Args:
            baseline_rate: Expected baseline conversion rate
            minimum_detectable_effect: Minimum effect size to detect (absolute)
            power: Statistical power (default 0.8)

        Returns:
            Required sample size per variant
        """
        # Z-scores for alpha and power
        z_alpha = self.Z_SCORES.get(self.confidence_level, 1.96)
        z_beta = 0.84 if power == 0.8 else 1.28  # Approximate for common powers

        p1 = baseline_rate
        p2 = baseline_rate + minimum_detectable_effect

        # Pooled variance estimate
        p_avg = (p1 + p2) / 2
        variance = 2 * p_avg * (1 - p_avg)

        # Sample size formula
        effect = abs(p2 - p1)
        if effect == 0:
            return 10000  # Return large number if no effect

        n = ((z_alpha + z_beta) ** 2 * variance) / (effect**2)
        return int(math.ceil(n))


def analyze_experiment(
    experiment_id: str,
    experiment_name: str,
    control_values: list[float],
    treatment_values: dict[str, list[float]],
    metric_type: str = "continuous",
    confidence_level: float = 0.95,
) -> ExperimentAnalysis:
    """
    Analyze an A/B experiment.

    Args:
        experiment_id: Experiment identifier
        experiment_name: Human-readable name
        control_values: Metric values for control variant
        treatment_values: Dict of variant name -> metric values
        metric_type: 'continuous' or 'binary' (conversion)
        confidence_level: Confidence level for statistical tests

    Returns:
        Complete experiment analysis
    """
    analyzer = StatisticalAnalyzer(confidence_level=confidence_level)

    # Create outcomes
    control_outcome = ExperimentOutcome(
        variant_name="control",
        sample_size=len(control_values),
        metric_values=control_values,
    )

    # For binary metrics, calculate conversion rate
    if metric_type == "binary":
        control_outcome.conversion_count = sum(1 for v in control_values if v > 0)
        control_outcome.conversion_rate = (
            control_outcome.conversion_count / control_outcome.sample_size
            if control_outcome.sample_size > 0
            else 0
        )

    treatment_outcomes = []
    test_results = []

    for variant_name, values in treatment_values.items():
        treatment_outcome = ExperimentOutcome(
            variant_name=variant_name,
            sample_size=len(values),
            metric_values=values,
        )

        if metric_type == "binary":
            treatment_outcome.conversion_count = sum(1 for v in values if v > 0)
            treatment_outcome.conversion_rate = (
                treatment_outcome.conversion_count / treatment_outcome.sample_size
                if treatment_outcome.sample_size > 0
                else 0
            )

        treatment_outcomes.append(treatment_outcome)

        # Run appropriate statistical test
        if metric_type == "binary":
            result = analyzer.z_test_proportions(control_outcome, treatment_outcome)
        else:
            result = analyzer.t_test(control_outcome, treatment_outcome)

        test_results.append(result)

    # Determine winner
    winner = None
    recommendation = "No significant difference detected."

    for i, result in enumerate(test_results):
        if result.is_significant and result.effect_size > 0:
            variant_name = treatment_outcomes[i].variant_name
            if winner is None or result.effect_size > test_results[i - 1].effect_size:
                winner = variant_name
                recommendation = (
                    f"Recommend {variant_name} variant. "
                    f"Effect size: {result.effect_size:.4f}, "
                    f"p-value: {result.p_value:.4f}"
                )

    return ExperimentAnalysis(
        experiment_id=experiment_id,
        experiment_name=experiment_name,
        control_outcome=control_outcome,
        treatment_outcomes=treatment_outcomes,
        test_results=test_results,
        winner=winner,
        recommendation=recommendation,
    )


def calculate_confidence_interval(
    values: list[float],
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """Calculate confidence interval for a sample mean."""
    if not values:
        return (0.0, 0.0)

    n = len(values)
    mean = sum(values) / n

    if n < 2:
        return (mean, mean)

    std_dev = math.sqrt(sum((x - mean) ** 2 for x in values) / (n - 1))
    se = std_dev / math.sqrt(n)

    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_scores.get(confidence_level, 1.96)

    margin = z * se
    return (mean - margin, mean + margin)


def calculate_lift(control_mean: float, treatment_mean: float) -> float:
    """Calculate percentage lift from control to treatment."""
    if control_mean == 0:
        return 0.0 if treatment_mean == 0 else float("inf")
    return ((treatment_mean - control_mean) / control_mean) * 100
