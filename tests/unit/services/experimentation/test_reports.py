"""Tests for reports module."""

import json

import pytest

from src.services.experimentation.analysis import (
    ExperimentAnalysis,
    ExperimentOutcome,
    StatisticalTest,
    StatTestResult,
)
from src.services.experimentation.canary import (
    CanaryConfig,
    CanaryDeployment,
    CanaryStatus,
)
from src.services.experimentation.experiments import (
    Experiment,
    ExperimentStatus,
    Variant,
)
from src.services.experimentation.reports import (
    ExperimentReport,
    ReportFormat,
    ReportGenerator,
    ReportSection,
    ReportType,
    generate_quick_report,
)


class TestReportFormat:
    """Tests for ReportFormat enum."""

    def test_format_values(self):
        """Test format enum values."""
        assert ReportFormat.TEXT.value == "text"
        assert ReportFormat.MARKDOWN.value == "markdown"
        assert ReportFormat.JSON.value == "json"
        assert ReportFormat.HTML.value == "html"


class TestReportType:
    """Tests for ReportType enum."""

    def test_type_values(self):
        """Test report type values."""
        assert ReportType.EXPERIMENT.value == "experiment"
        assert ReportType.CANARY.value == "canary"
        assert ReportType.SUMMARY.value == "summary"


class TestReportSection:
    """Tests for ReportSection."""

    def test_section_creation(self):
        """Test creating a report section."""
        section = ReportSection(
            title="Overview",
            content="This is the overview.",
        )

        assert section.title == "Overview"
        assert section.content == "This is the overview."

    def test_section_with_data(self):
        """Test section with data."""
        section = ReportSection(
            title="Metrics",
            content="Metrics summary",
            data={"conversion_rate": 0.15},
        )

        assert section.data["conversion_rate"] == 0.15

    def test_section_to_dict(self):
        """Test section serialization."""
        section = ReportSection(
            title="Summary",
            content="Summary content",
            data={"key": "value"},
        )
        data = section.to_dict()

        assert data["title"] == "Summary"
        assert data["content"] == "Summary content"
        assert data["data"]["key"] == "value"


class TestExperimentReport:
    """Tests for ExperimentReport."""

    def test_report_creation(self):
        """Test creating a report."""
        report = ExperimentReport(
            experiment_id="exp-123",
            experiment_name="Test Experiment",
            report_type=ReportType.EXPERIMENT,
        )

        assert report.experiment_id == "exp-123"
        assert report.experiment_name == "Test Experiment"
        assert report.generated_at is not None

    def test_report_add_section(self):
        """Test adding sections to report."""
        report = ExperimentReport(
            experiment_id="exp-123",
            experiment_name="Results",
            report_type=ReportType.EXPERIMENT,
        )

        report.add_section(
            title="Overview",
            content="Overview content",
        )

        assert len(report.sections) == 1
        assert report.sections[0].title == "Overview"

    def test_report_to_dict(self):
        """Test report serialization."""
        report = ExperimentReport(
            experiment_id="exp-123",
            experiment_name="Results",
            report_type=ReportType.EXPERIMENT,
            metadata={"author": "test"},
        )
        data = report.to_dict()

        assert data["experiment_id"] == "exp-123"
        assert data["report_type"] == "experiment"
        assert data["metadata"]["author"] == "test"

    def test_report_to_text(self):
        """Test report to text conversion."""
        report = ExperimentReport(
            experiment_id="exp-123",
            experiment_name="Test Experiment",
            report_type=ReportType.EXPERIMENT,
        )
        report.add_section("Results", "Some results here")

        text = report.to_text()

        assert "Test Experiment" in text
        assert "RESULTS" in text  # Section title is uppercased in to_text()
        assert "exp-123" in text

    def test_report_to_markdown(self):
        """Test report to markdown conversion."""
        report = ExperimentReport(
            experiment_id="exp-123",
            experiment_name="Test Experiment",
            report_type=ReportType.EXPERIMENT,
        )
        report.add_section("Results", "Some results here")

        md = report.to_markdown()

        assert "#" in md  # Should have markdown headers
        assert "Test Experiment" in md

    def test_report_to_html(self):
        """Test report to HTML conversion."""
        report = ExperimentReport(
            experiment_id="exp-123",
            experiment_name="Test Experiment",
            report_type=ReportType.EXPERIMENT,
        )
        report.add_section("Results", "Some results here")

        html = report.to_html()

        assert "<" in html  # Should have HTML tags
        assert "Test Experiment" in html


class TestReportGenerator:
    """Tests for ReportGenerator."""

    @pytest.fixture
    def sample_experiment(self):
        """Create sample experiment."""
        from src.services.experimentation.experiments import ExperimentType

        control = Variant(name="Control", config={"model": "gpt-3.5"})
        treatment = Variant(name="Treatment", config={"model": "gpt-4"})

        return Experiment(
            id="exp-123",
            name="Test Experiment",
            experiment_type=ExperimentType.MODEL,
            description="A test experiment",
            variants=[control, treatment],
            status=ExperimentStatus.COMPLETED,
        )

    @pytest.fixture
    def sample_analysis(self):
        """Create sample experiment analysis."""
        control = ExperimentOutcome(
            variant_name="control",
            sample_size=0,
            metric_values=[1.0, 2.0, 3.0, 4.0, 5.0],
        )
        treatment = ExperimentOutcome(
            variant_name="treatment",
            sample_size=0,
            metric_values=[2.0, 3.0, 4.0, 5.0, 6.0],
        )
        test_result = StatTestResult(
            test_type=StatisticalTest.T_TEST,
            statistic=2.5,
            p_value=0.02,
            confidence_level=0.95,
            is_significant=True,
            effect_size=0.8,
        )

        return ExperimentAnalysis(
            experiment_id="exp-123",
            experiment_name="Test Experiment",
            control_outcome=control,
            treatment_outcomes=[treatment],
            test_results=[test_result],
            winner="treatment",
            recommendation="Deploy treatment",
        )

    @pytest.fixture
    def sample_deployment(self):
        """Create sample canary deployment."""
        deployment = CanaryDeployment(
            id="deploy-123",
            name="API v2 Canary",
            baseline_version="v1.0",
            canary_version="v2.0",
            config=CanaryConfig(),
        )
        deployment.canary_metrics.request_count = 1000
        deployment.canary_metrics.error_count = 5
        deployment.canary_metrics.latencies_ms = [50.0] * 1000
        deployment.baseline_metrics.request_count = 10000
        deployment.baseline_metrics.error_count = 100
        deployment.baseline_metrics.latencies_ms = [50.0] * 10000
        deployment.status = CanaryStatus.RUNNING
        deployment.current_percentage = 25.0
        return deployment

    def test_generate_experiment_report(self, sample_experiment, sample_analysis, tmp_path):
        """Test generating experiment report."""
        generator = ReportGenerator(output_dir=tmp_path)
        report = generator.generate_experiment_report(
            experiment=sample_experiment,
            analysis=sample_analysis,
        )

        assert report.experiment_id == "exp-123"
        assert report.experiment_name == "Test Experiment"
        assert len(report.sections) > 0

    def test_generate_experiment_report_no_analysis(self, sample_experiment, tmp_path):
        """Test generating experiment report without analysis."""
        generator = ReportGenerator(output_dir=tmp_path)
        report = generator.generate_experiment_report(
            experiment=sample_experiment,
            analysis=None,
        )

        assert report.experiment_id == "exp-123"
        assert len(report.sections) > 0

    def test_generate_canary_report(self, sample_deployment, tmp_path):
        """Test generating canary deployment report."""
        generator = ReportGenerator(output_dir=tmp_path)
        report = generator.generate_canary_report(
            deployment=sample_deployment,
        )

        assert report.experiment_id == "deploy-123"
        assert "API v2 Canary" in report.experiment_name
        assert len(report.sections) > 0

    def test_save_report_text(self, sample_experiment, tmp_path):
        """Test saving report as text."""
        generator = ReportGenerator(output_dir=tmp_path)
        report = generator.generate_experiment_report(
            experiment=sample_experiment,
        )

        filepath = generator.save_report(report, format=ReportFormat.TEXT)

        assert filepath.exists()
        assert filepath.suffix == ".txt"
        content = filepath.read_text()
        assert "Test Experiment" in content

    def test_save_report_markdown(self, sample_experiment, tmp_path):
        """Test saving report as markdown."""
        generator = ReportGenerator(output_dir=tmp_path)
        report = generator.generate_experiment_report(
            experiment=sample_experiment,
        )

        filepath = generator.save_report(report, format=ReportFormat.MARKDOWN)

        assert filepath.exists()
        assert filepath.suffix == ".md"
        content = filepath.read_text()
        assert "#" in content  # Markdown headers

    def test_save_report_json(self, sample_experiment, tmp_path):
        """Test saving report as JSON."""
        generator = ReportGenerator(output_dir=tmp_path)
        report = generator.generate_experiment_report(
            experiment=sample_experiment,
        )

        filepath = generator.save_report(report, format=ReportFormat.JSON)

        assert filepath.exists()
        assert filepath.suffix == ".json"
        content = filepath.read_text()
        data = json.loads(content)
        assert data["experiment_id"] == "exp-123"

    def test_save_report_html(self, sample_experiment, tmp_path):
        """Test saving report as HTML."""
        generator = ReportGenerator(output_dir=tmp_path)
        report = generator.generate_experiment_report(
            experiment=sample_experiment,
        )

        filepath = generator.save_report(report, format=ReportFormat.HTML)

        assert filepath.exists()
        assert filepath.suffix == ".html"
        content = filepath.read_text()
        assert "<" in content  # HTML tags


class TestGenerateQuickReport:
    """Tests for generate_quick_report helper function."""

    @pytest.fixture
    def sample_experiment(self):
        """Create sample experiment."""
        from src.services.experimentation.experiments import ExperimentType

        control = Variant(name="Control", config={"prompt": "v1"})
        treatment = Variant(name="Treatment", config={"prompt": "v2"})

        return Experiment(
            id="exp-quick",
            name="Quick Test",
            experiment_type=ExperimentType.PROMPT,
            description="Quick test experiment",
            variants=[control, treatment],
        )

    def test_quick_report_text(self, sample_experiment):
        """Test quick report generation as text."""
        report = generate_quick_report(
            experiment=sample_experiment,
            format=ReportFormat.TEXT,
        )

        assert "Quick Test" in report
        assert isinstance(report, str)

    def test_quick_report_markdown(self, sample_experiment):
        """Test quick report generation as markdown."""
        report = generate_quick_report(
            experiment=sample_experiment,
            format=ReportFormat.MARKDOWN,
        )

        assert "#" in report  # Markdown headers
        assert "Quick Test" in report

    def test_quick_report_json(self, sample_experiment):
        """Test quick report generation as JSON."""
        report = generate_quick_report(
            experiment=sample_experiment,
            format=ReportFormat.JSON,
        )

        data = json.loads(report)
        assert data["experiment_id"] == "exp-quick"

    def test_quick_report_html(self, sample_experiment):
        """Test quick report generation as HTML."""
        report = generate_quick_report(
            experiment=sample_experiment,
            format=ReportFormat.HTML,
        )

        assert "<" in report  # HTML tags

    def test_quick_report_with_analysis(self, sample_experiment):
        """Test quick report with analysis data."""
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
        analysis = ExperimentAnalysis(
            experiment_id="exp-quick",
            experiment_name="Quick Test",
            control_outcome=control,
            treatment_outcomes=[treatment],
            test_results=[],
        )

        report = generate_quick_report(
            experiment=sample_experiment,
            analysis=analysis,
            format=ReportFormat.TEXT,
        )

        assert "Quick Test" in report
