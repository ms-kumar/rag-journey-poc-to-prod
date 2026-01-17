"""Tests for analysis module."""

import pytest

from src.services.experimentation.analysis import (
    ExperimentAnalysis,
    ExperimentOutcome,
    StatisticalAnalyzer,
    StatisticalTest,
    TestResult,
    analyze_experiment,
    calculate_confidence_interval,
    calculate_lift,
)


class TestExperimentOutcome:
    """Tests for ExperimentOutcome."""

    def test_outcome_statistics(self):
        """Test automatic statistics calculation."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        outcome = ExperimentOutcome(
            variant_name="test",
            sample_size=0,
            metric_values=values,
        )

        assert outcome.sample_size == 5
        assert outcome.mean == 3.0
        assert outcome.min_value == 1.0
        assert outcome.max_value == 5.0
        assert outcome.median == 3.0

    def test_outcome_std_dev(self):
        """Test standard deviation calculation."""
        values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        outcome = ExperimentOutcome(
            variant_name="test",
            sample_size=0,
            metric_values=values,
        )

        # Mean is 5.0, std dev should be ~2.0
        assert outcome.mean == 5.0
        assert 1.9 < outcome.std_dev < 2.2

    def test_outcome_median_even(self):
        """Test median calculation with even sample size."""
        values = [1.0, 2.0, 3.0, 4.0]
        outcome = ExperimentOutcome(
            variant_name="test",
            sample_size=0,
            metric_values=values,
        )

        assert outcome.median == 2.5

    def test_outcome_to_dict(self):
        """Test outcome serialization."""
        outcome = ExperimentOutcome(
            variant_name="control",
            sample_size=0,
            metric_values=[1.0, 2.0, 3.0],
        )
        data = outcome.to_dict()

        assert data["variant_name"] == "control"
        assert data["sample_size"] == 3
        assert data["mean"] == 2.0


class TestTestResult:
    """Tests for TestResult."""

    def test_test_result_to_dict(self):
        """Test result serialization."""
        result = TestResult(
            test_type=StatisticalTest.T_TEST,
            statistic=2.5,
            p_value=0.01,
            confidence_level=0.95,
            is_significant=True,
            effect_size=0.5,
            confidence_interval=(-0.1, 0.3),
        )
        data = result.to_dict()

        assert data["test_type"] == "t_test"
        assert data["statistic"] == 2.5
        assert data["is_significant"] is True


class TestStatisticalAnalyzer:
    """Tests for StatisticalAnalyzer."""

    def test_t_test_significant(self):
        """Test t-test with significant difference."""
        analyzer = StatisticalAnalyzer(confidence_level=0.95)

        control = ExperimentOutcome(
            variant_name="control",
            sample_size=0,
            metric_values=[1.0, 2.0, 3.0, 4.0, 5.0] * 20,  # Mean ~3
        )
        treatment = ExperimentOutcome(
            variant_name="treatment",
            sample_size=0,
            metric_values=[4.0, 5.0, 6.0, 7.0, 8.0] * 20,  # Mean ~6
        )

        result = analyzer.t_test(control, treatment)

        assert result.test_type == StatisticalTest.T_TEST
        assert result.is_significant is True
        assert result.effect_size > 0

    def test_t_test_not_significant(self):
        """Test t-test with no significant difference."""
        analyzer = StatisticalAnalyzer(confidence_level=0.95)

        control = ExperimentOutcome(
            variant_name="control",
            sample_size=0,
            metric_values=[3.0, 3.1, 2.9, 3.0, 3.1] * 10,
        )
        treatment = ExperimentOutcome(
            variant_name="treatment",
            sample_size=0,
            metric_values=[3.0, 3.1, 2.9, 3.0, 3.05] * 10,
        )

        result = analyzer.t_test(control, treatment)

        # Very similar means should not be significant
        assert result.p_value > 0.05 or result.effect_size < 0.1

    def test_t_test_small_sample(self):
        """Test t-test with insufficient sample."""
        analyzer = StatisticalAnalyzer()

        control = ExperimentOutcome(
            variant_name="control",
            sample_size=0,
            metric_values=[1.0],
        )
        treatment = ExperimentOutcome(
            variant_name="treatment",
            sample_size=0,
            metric_values=[2.0],
        )

        result = analyzer.t_test(control, treatment)
        assert result.is_significant is False

    def test_z_test_proportions(self):
        """Test z-test for proportions."""
        analyzer = StatisticalAnalyzer(confidence_level=0.95)

        control = ExperimentOutcome(
            variant_name="control",
            sample_size=1000,
            metric_values=[],
            conversion_count=100,
            conversion_rate=0.10,
        )
        treatment = ExperimentOutcome(
            variant_name="treatment",
            sample_size=1000,
            metric_values=[],
            conversion_count=150,
            conversion_rate=0.15,
        )

        result = analyzer.z_test_proportions(control, treatment)

        assert result.test_type == StatisticalTest.Z_TEST
        assert result.is_significant is True
        assert result.effect_size == pytest.approx(0.05, abs=0.01)

    def test_calculate_sample_size(self):
        """Test sample size calculation."""
        analyzer = StatisticalAnalyzer(confidence_level=0.95)

        sample_size = analyzer.calculate_sample_size(
            baseline_rate=0.10,
            minimum_detectable_effect=0.02,
            power=0.8,
        )

        # Should need a reasonable sample size
        assert sample_size > 100
        assert sample_size < 100000


class TestAnalyzeExperiment:
    """Tests for analyze_experiment function."""

    def test_analyze_continuous_metric(self):
        """Test analyzing experiment with continuous metric."""
        control_values = [1.0, 2.0, 3.0, 4.0, 5.0] * 20
        treatment_values = {
            "treatment_a": [3.0, 4.0, 5.0, 6.0, 7.0] * 20,
        }

        analysis = analyze_experiment(
            experiment_id="exp-123",
            experiment_name="Test Experiment",
            control_values=control_values,
            treatment_values=treatment_values,
            metric_type="continuous",
        )

        assert analysis.experiment_id == "exp-123"
        assert analysis.control_outcome.mean == 3.0
        assert len(analysis.treatment_outcomes) == 1
        assert len(analysis.test_results) == 1

    def test_analyze_binary_metric(self):
        """Test analyzing experiment with binary metric."""
        # Control: 10% conversion
        control_values = [1.0] * 10 + [0.0] * 90
        # Treatment: 15% conversion
        treatment_values = {
            "treatment_a": [1.0] * 15 + [0.0] * 85,
        }

        analysis = analyze_experiment(
            experiment_id="exp-123",
            experiment_name="Conversion Test",
            control_values=control_values,
            treatment_values=treatment_values,
            metric_type="binary",
        )

        assert analysis.control_outcome.conversion_rate == 0.10
        assert analysis.treatment_outcomes[0].conversion_rate == 0.15

    def test_analyze_determines_winner(self):
        """Test that analysis determines winner when significant."""
        control_values = [1.0, 2.0, 3.0] * 100
        treatment_values = {
            "treatment_a": [5.0, 6.0, 7.0] * 100,
        }

        analysis = analyze_experiment(
            experiment_id="exp-123",
            experiment_name="Test",
            control_values=control_values,
            treatment_values=treatment_values,
        )

        assert analysis.winner == "treatment_a"
        assert "treatment_a" in analysis.recommendation

    def test_analyze_multiple_treatments(self):
        """Test analyzing experiment with multiple treatments."""
        control_values = [2.0, 3.0, 4.0] * 50
        treatment_values = {
            "treatment_a": [3.0, 4.0, 5.0] * 50,
            "treatment_b": [4.0, 5.0, 6.0] * 50,
        }

        analysis = analyze_experiment(
            experiment_id="exp-123",
            experiment_name="Multi-arm Test",
            control_values=control_values,
            treatment_values=treatment_values,
        )

        assert len(analysis.treatment_outcomes) == 2
        assert len(analysis.test_results) == 2


class TestExperimentAnalysis:
    """Tests for ExperimentAnalysis."""

    def test_analysis_to_dict(self):
        """Test analysis serialization."""
        control = ExperimentOutcome(
            variant_name="control",
            sample_size=0,
            metric_values=[1.0, 2.0, 3.0],
        )
        treatment = ExperimentOutcome(
            variant_name="treatment",
            sample_size=0,
            metric_values=[2.0, 3.0, 4.0],
        )
        test_result = TestResult(
            test_type=StatisticalTest.T_TEST,
            statistic=1.5,
            p_value=0.1,
            confidence_level=0.95,
            is_significant=False,
        )

        analysis = ExperimentAnalysis(
            experiment_id="exp-123",
            experiment_name="Test",
            control_outcome=control,
            treatment_outcomes=[treatment],
            test_results=[test_result],
        )

        data = analysis.to_dict()
        assert data["experiment_id"] == "exp-123"
        assert "control_outcome" in data
        assert len(data["treatment_outcomes"]) == 1


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_calculate_confidence_interval(self):
        """Test confidence interval calculation."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        ci = calculate_confidence_interval(values, confidence_level=0.95)

        # Mean is 3.0, CI should contain it
        assert ci[0] < 3.0 < ci[1]

    def test_calculate_confidence_interval_empty(self):
        """Test confidence interval with empty values."""
        ci = calculate_confidence_interval([])
        assert ci == (0.0, 0.0)

    def test_calculate_confidence_interval_single(self):
        """Test confidence interval with single value."""
        ci = calculate_confidence_interval([5.0])
        assert ci == (5.0, 5.0)

    def test_calculate_lift_positive(self):
        """Test positive lift calculation."""
        lift = calculate_lift(control_mean=100.0, treatment_mean=120.0)
        assert lift == pytest.approx(20.0)

    def test_calculate_lift_negative(self):
        """Test negative lift calculation."""
        lift = calculate_lift(control_mean=100.0, treatment_mean=80.0)
        assert lift == pytest.approx(-20.0)

    def test_calculate_lift_zero_control(self):
        """Test lift with zero control."""
        lift = calculate_lift(control_mean=0.0, treatment_mean=10.0)
        assert lift == float("inf")

        lift = calculate_lift(control_mean=0.0, treatment_mean=0.0)
        assert lift == 0.0
