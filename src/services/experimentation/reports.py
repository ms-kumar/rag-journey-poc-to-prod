"""
Report generation for experiments and A/B tests.

Provides comprehensive reports including:
- Statistical summaries
- Visualizations (text-based)
- Recommendations
- Export to multiple formats
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from src.services.experimentation.analysis import ExperimentAnalysis, calculate_lift
from src.services.experimentation.canary import CanaryDeployment, CanaryStatus
from src.services.experimentation.experiments import Experiment, ExperimentStatus


class ReportFormat(str, Enum):
    """Output format for reports."""

    TEXT = "text"
    MARKDOWN = "markdown"
    JSON = "json"
    HTML = "html"


class ReportType(str, Enum):
    """Type of report."""

    EXPERIMENT = "experiment"
    FEATURE_FLAG = "feature_flag"
    CANARY = "canary"
    SUMMARY = "summary"


@dataclass
class ReportSection:
    """A section in a report."""

    title: str
    content: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert section to dictionary."""
        return {
            "title": self.title,
            "content": self.content,
            "data": self.data,
        }


@dataclass
class ExperimentReport:
    """Complete experiment report."""

    experiment_id: str
    experiment_name: str
    report_type: ReportType
    sections: list[ReportSection] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_section(self, title: str, content: str, data: dict[str, Any] | None = None) -> None:
        """Add a section to the report."""
        self.sections.append(ReportSection(title=title, content=content, data=data or {}))

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "report_type": self.report_type.value,
            "sections": [s.to_dict() for s in self.sections],
            "generated_at": self.generated_at.isoformat(),
            "metadata": self.metadata,
        }

    def to_text(self) -> str:
        """Convert report to plain text."""
        lines = [
            "=" * 60,
            f"EXPERIMENT REPORT: {self.experiment_name}",
            "=" * 60,
            f"ID: {self.experiment_id}",
            f"Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]

        for section in self.sections:
            lines.extend(
                [
                    "-" * 40,
                    section.title.upper(),
                    "-" * 40,
                    section.content,
                    "",
                ]
            )

        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Convert report to Markdown."""
        lines = [
            f"# Experiment Report: {self.experiment_name}",
            "",
            f"**ID:** {self.experiment_id}",
            f"**Generated:** {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]

        for section in self.sections:
            lines.extend(
                [
                    f"## {section.title}",
                    "",
                    section.content,
                    "",
                ]
            )

        return "\n".join(lines)

    def to_html(self) -> str:
        """Convert report to HTML."""
        sections_html = ""
        for section in self.sections:
            content_html = section.content.replace("\n", "<br>")
            sections_html += f"""
            <div class="section">
                <h2>{section.title}</h2>
                <p>{content_html}</p>
            </div>
            """

        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Experiment Report: {self.experiment_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; border-bottom: 1px solid #ccc; }}
        .section {{ margin: 20px 0; }}
        .meta {{ color: #888; font-size: 0.9em; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f4f4f4; }}
        .winner {{ color: green; font-weight: bold; }}
        .significant {{ color: blue; }}
    </style>
</head>
<body>
    <h1>Experiment Report: {self.experiment_name}</h1>
    <p class="meta">
        <strong>ID:</strong> {self.experiment_id}<br>
        <strong>Generated:</strong> {self.generated_at.strftime("%Y-%m-%d %H:%M:%S")}
    </p>
    {sections_html}
</body>
</html>
        """


class ReportGenerator:
    """Generates reports for experiments and deployments."""

    def __init__(self, output_dir: str | None = None):
        """Initialize report generator."""
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_experiment_report(
        self,
        experiment: Experiment,
        analysis: ExperimentAnalysis | None = None,
        stats: dict[str, Any] | None = None,
    ) -> ExperimentReport:
        """Generate a report for an experiment."""
        report = ExperimentReport(
            experiment_id=experiment.id,
            experiment_name=experiment.name,
            report_type=ReportType.EXPERIMENT,
        )

        # Overview section
        overview = self._generate_overview(experiment)
        report.add_section("Overview", overview, {"experiment": experiment.to_dict()})

        # Variants section
        variants = self._generate_variants_section(experiment)
        report.add_section("Variants", variants)

        # Results section (if analysis available)
        if analysis:
            results = self._generate_results_section(analysis)
            report.add_section("Results", results, {"analysis": analysis.to_dict()})

        # Statistics section (if stats available)
        if stats:
            statistics = self._generate_statistics_section(stats)
            report.add_section("Statistics", statistics, {"stats": stats})

        # Recommendations section
        recommendations = self._generate_recommendations(experiment, analysis)
        report.add_section("Recommendations", recommendations)

        return report

    def generate_canary_report(
        self,
        deployment: CanaryDeployment,
    ) -> ExperimentReport:
        """Generate a report for a canary deployment."""
        report = ExperimentReport(
            experiment_id=deployment.id,
            experiment_name=deployment.name,
            report_type=ReportType.CANARY,
        )

        # Overview
        overview = self._generate_canary_overview(deployment)
        report.add_section("Deployment Overview", overview)

        # Metrics comparison
        comparison = self._generate_canary_comparison(deployment)
        report.add_section("Metrics Comparison", comparison)

        # Status and recommendations
        status = self._generate_canary_status(deployment)
        report.add_section("Status & Recommendations", status)

        return report

    def generate_summary_report(
        self,
        experiments: list[Experiment],
        deployments: list[CanaryDeployment] | None = None,
    ) -> ExperimentReport:
        """Generate a summary report across experiments."""
        report = ExperimentReport(
            experiment_id="summary",
            experiment_name="Experimentation Summary",
            report_type=ReportType.SUMMARY,
        )

        # Experiments summary
        exp_summary = self._generate_experiments_summary(experiments)
        report.add_section("Experiments Summary", exp_summary)

        # Canary summary (if provided)
        if deployments:
            canary_summary = self._generate_canary_summary(deployments)
            report.add_section("Canary Deployments Summary", canary_summary)

        return report

    def _generate_overview(self, experiment: Experiment) -> str:
        """Generate overview section."""
        duration = ""
        if experiment.started_at and experiment.ended_at:
            delta = experiment.ended_at - experiment.started_at
            duration = f"\nDuration: {delta.days} days, {delta.seconds // 3600} hours"
        elif experiment.started_at:
            delta = datetime.utcnow() - experiment.started_at
            duration = f"\nRunning for: {delta.days} days, {delta.seconds // 3600} hours"

        return f"""
Name: {experiment.name}
Type: {experiment.experiment_type.value}
Status: {experiment.status.value}
Owner: {experiment.owner or "Not specified"}
Description: {experiment.description or "No description"}

Created: {experiment.created_at.strftime("%Y-%m-%d %H:%M")}
Started: {experiment.started_at.strftime("%Y-%m-%d %H:%M") if experiment.started_at else "Not started"}
Ended: {experiment.ended_at.strftime("%Y-%m-%d %H:%M") if experiment.ended_at else "Still running"}{duration}

Target Sample Size: {experiment.target_sample_size}
Minimum Runtime: {experiment.min_runtime_hours} hours
        """.strip()

    def _generate_variants_section(self, experiment: Experiment) -> str:
        """Generate variants section."""
        lines = []
        weights = experiment.get_variant_weights()

        for variant in experiment.variants:
            lines.append(f"• {variant.name}")
            lines.append(f"  Traffic: {weights[variant.name] * 100:.1f}%")
            if variant.description:
                lines.append(f"  Description: {variant.description}")
            lines.append(f"  Config: {json.dumps(variant.config, indent=4)}")
            lines.append("")

        return "\n".join(lines)

    def _generate_results_section(self, analysis: ExperimentAnalysis) -> str:
        """Generate results section."""
        lines = [
            "CONTROL:",
            f"  Sample Size: {analysis.control_outcome.sample_size}",
            f"  Mean: {analysis.control_outcome.mean:.4f}",
            f"  Std Dev: {analysis.control_outcome.std_dev:.4f}",
            "",
        ]

        for i, treatment in enumerate(analysis.treatment_outcomes):
            test_result = analysis.test_results[i] if i < len(analysis.test_results) else None
            lift = calculate_lift(analysis.control_outcome.mean, treatment.mean)

            lines.extend(
                [
                    f"TREATMENT ({treatment.variant_name}):",
                    f"  Sample Size: {treatment.sample_size}",
                    f"  Mean: {treatment.mean:.4f}",
                    f"  Std Dev: {treatment.std_dev:.4f}",
                    f"  Lift vs Control: {lift:+.2f}%",
                ]
            )

            if test_result:
                lines.extend(
                    [
                        f"  Statistical Test: {test_result.test_type.value}",
                        f"  P-value: {test_result.p_value:.4f}",
                        f"  Significant: {'Yes' if test_result.is_significant else 'No'}",
                        f"  Effect Size: {test_result.effect_size:.4f}",
                        f"  CI: [{test_result.confidence_interval[0]:.4f}, {test_result.confidence_interval[1]:.4f}]",
                    ]
                )
            lines.append("")

        if analysis.winner:
            lines.extend(
                [
                    "WINNER:",
                    f"  {analysis.winner}",
                    "",
                ]
            )

        return "\n".join(lines)

    def _generate_statistics_section(self, stats: dict[str, Any]) -> str:
        """Generate statistics section."""
        lines = [
            f"Total Assignments: {stats.get('total_assignments', 0)}",
            f"Total Outcomes: {stats.get('total_outcomes', 0)}",
            "",
            "Assignments per Variant:",
        ]

        for variant, count in stats.get("variant_counts", {}).items():
            lines.append(f"  • {variant}: {count}")

        lines.append("")
        lines.append("Metrics per Variant:")

        for variant, metrics in stats.get("variant_metrics", {}).items():
            lines.append(f"  {variant}:")
            for metric, values in metrics.items():
                if values:
                    avg = sum(values) / len(values)
                    lines.append(f"    • {metric}: avg={avg:.4f}, n={len(values)}")

        return "\n".join(lines)

    def _generate_recommendations(
        self,
        experiment: Experiment,
        analysis: ExperimentAnalysis | None,
    ) -> str:
        """Generate recommendations section."""
        lines = []

        if experiment.status == ExperimentStatus.DRAFT:
            lines.append("• Experiment is in draft. Start when ready to collect data.")
        elif experiment.status == ExperimentStatus.RUNNING:
            lines.append("• Experiment is running. Continue collecting data.")
            if analysis and not analysis.winner:
                lines.append("• No significant winner yet. Consider extending runtime.")
        elif experiment.status == ExperimentStatus.COMPLETED:
            if analysis:
                lines.append(f"• {analysis.recommendation}")
            else:
                lines.append("• Run analysis to determine winner.")

        if analysis:
            for i, result in enumerate(analysis.test_results):
                treatment = analysis.treatment_outcomes[i]
                if result.is_significant:
                    if result.effect_size > 0:
                        lines.append(
                            f"• {treatment.variant_name} shows significant improvement "
                            f"(effect size: {result.effect_size:.4f})"
                        )
                    else:
                        lines.append(
                            f"• {treatment.variant_name} shows significant degradation "
                            f"(effect size: {result.effect_size:.4f})"
                        )
                else:
                    lines.append(f"• {treatment.variant_name} shows no significant difference")

        return "\n".join(lines) if lines else "No recommendations available."

    def _generate_canary_overview(self, deployment: CanaryDeployment) -> str:
        """Generate canary overview."""
        duration = ""
        if deployment.started_at:
            delta = (deployment.ended_at or datetime.utcnow()) - deployment.started_at
            duration = (
                f"Duration: {delta.days}d {delta.seconds // 3600}h {(delta.seconds % 3600) // 60}m"
            )

        return f"""
Name: {deployment.name}
Status: {deployment.status.value}
Current Traffic: {deployment.current_percentage:.1f}%

Baseline Version: {deployment.baseline_version}
Canary Version: {deployment.canary_version}

{duration}
Created: {deployment.created_at.strftime("%Y-%m-%d %H:%M")}
Started: {deployment.started_at.strftime("%Y-%m-%d %H:%M") if deployment.started_at else "Not started"}
        """.strip()

    def _generate_canary_comparison(self, deployment: CanaryDeployment) -> str:
        """Generate canary metrics comparison."""
        baseline = deployment.baseline_metrics
        canary = deployment.canary_metrics

        return f"""
                    BASELINE        CANARY          DIFF
Requests:           {baseline.request_count:<15} {canary.request_count:<15} -
Error Rate:         {baseline.error_rate * 100:.2f}%          {canary.error_rate * 100:.2f}%          {(canary.error_rate - baseline.error_rate) * 100:+.2f}%
Latency P50:        {baseline.latency_p50:.1f}ms         {canary.latency_p50:.1f}ms         {canary.latency_p50 - baseline.latency_p50:+.1f}ms
Latency P99:        {baseline.latency_p99:.1f}ms         {canary.latency_p99:.1f}ms         {canary.latency_p99 - baseline.latency_p99:+.1f}ms
Avg Quality:        {baseline.avg_quality:.4f}          {canary.avg_quality:.4f}          {canary.avg_quality - baseline.avg_quality:+.4f}
        """.strip()

    def _generate_canary_status(self, deployment: CanaryDeployment) -> str:
        """Generate canary status and recommendations."""
        lines = [f"Status: {deployment.status.value}"]

        if deployment.rollback_reason:
            lines.append(f"Rollback Reason: {deployment.rollback_reason.value}")

        should_roll, reason = deployment.should_rollback()
        if should_roll:
            lines.append(f"⚠️ ROLLBACK RECOMMENDED: {reason.value if reason else 'unknown'}")
        elif deployment.should_promote():
            lines.append("✅ READY FOR PROMOTION")
        elif deployment.status == CanaryStatus.RUNNING:
            lines.append("🔄 Deployment in progress...")

        # Thresholds check
        config = deployment.config
        canary = deployment.canary_metrics

        lines.extend(
            [
                "",
                "Threshold Checks:",
                f"  Error Rate: {canary.error_rate * 100:.2f}% / {config.max_error_rate * 100:.1f}% max "
                f"{'✅' if canary.error_rate <= config.max_error_rate else '❌'}",
                f"  Latency P99: {canary.latency_p99:.1f}ms / {config.max_latency_p99_ms:.1f}ms max "
                f"{'✅' if canary.latency_p99 <= config.max_latency_p99_ms else '❌'}",
            ]
        )

        if canary.quality_scores:
            lines.append(
                f"  Quality: {canary.avg_quality:.4f} / {config.min_quality_score:.4f} min "
                f"{'✅' if canary.avg_quality >= config.min_quality_score else '❌'}"
            )

        return "\n".join(lines)

    def _generate_experiments_summary(self, experiments: list[Experiment]) -> str:
        """Generate summary of all experiments."""
        by_status: dict[str, int] = {}
        by_type: dict[str, int] = {}

        for exp in experiments:
            by_status[exp.status.value] = by_status.get(exp.status.value, 0) + 1
            by_type[exp.experiment_type.value] = by_type.get(exp.experiment_type.value, 0) + 1

        lines = [
            f"Total Experiments: {len(experiments)}",
            "",
            "By Status:",
        ]
        for status, count in sorted(by_status.items()):
            lines.append(f"  • {status}: {count}")

        lines.extend(["", "By Type:"])
        for exp_type, count in sorted(by_type.items()):
            lines.append(f"  • {exp_type}: {count}")

        # Recent experiments
        recent = sorted(experiments, key=lambda x: x.created_at, reverse=True)[:5]
        if recent:
            lines.extend(["", "Recent Experiments:"])
            for exp in recent:
                lines.append(f"  • {exp.name} ({exp.status.value})")

        return "\n".join(lines)

    def _generate_canary_summary(self, deployments: list[CanaryDeployment]) -> str:
        """Generate summary of canary deployments."""
        by_status: dict[str, int] = {}

        for dep in deployments:
            by_status[dep.status.value] = by_status.get(dep.status.value, 0) + 1

        lines = [
            f"Total Deployments: {len(deployments)}",
            "",
            "By Status:",
        ]
        for status, count in sorted(by_status.items()):
            lines.append(f"  • {status}: {count}")

        # Active deployments
        active = [d for d in deployments if d.status == CanaryStatus.RUNNING]
        if active:
            lines.extend(["", "Active Deployments:"])
            for dep in active:
                lines.append(f"  • {dep.name}: {dep.current_percentage:.1f}% traffic")

        return "\n".join(lines)

    def save_report(
        self,
        report: ExperimentReport,
        format: ReportFormat = ReportFormat.MARKDOWN,
        filename: str | None = None,
    ) -> Path:
        """Save report to file."""
        if not self.output_dir:
            raise ValueError("Output directory not configured")

        if not filename:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"{report.experiment_id}_{timestamp}"

        # Generate content based on format
        if format == ReportFormat.TEXT:
            content = report.to_text()
            ext = ".txt"
        elif format == ReportFormat.MARKDOWN:
            content = report.to_markdown()
            ext = ".md"
        elif format == ReportFormat.JSON:
            content = json.dumps(report.to_dict(), indent=2)
            ext = ".json"
        elif format == ReportFormat.HTML:
            content = report.to_html()
            ext = ".html"
        else:
            raise ValueError(f"Unsupported format: {format}")

        filepath = self.output_dir / f"{filename}{ext}"
        filepath.write_text(content)
        return filepath


# Convenience function for quick report generation
def generate_quick_report(
    experiment: Experiment,
    analysis: ExperimentAnalysis | None = None,
    format: ReportFormat = ReportFormat.TEXT,
) -> str:
    """Generate a quick report without saving to file."""
    generator = ReportGenerator()
    report = generator.generate_experiment_report(experiment, analysis)

    if format == ReportFormat.TEXT:
        return report.to_text()
    if format == ReportFormat.MARKDOWN:
        return report.to_markdown()
    if format == ReportFormat.JSON:
        return json.dumps(report.to_dict(), indent=2)
    if format == ReportFormat.HTML:
        return report.to_html()
    return report.to_text()
