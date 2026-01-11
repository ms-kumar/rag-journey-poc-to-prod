"""
Performance reporting and visualization.

Generates comprehensive performance reports in various formats:
- Console output
- JSON export
- HTML reports
- Markdown summaries
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PerformanceReporter:
    """Generate performance reports in various formats."""

    def __init__(self):
        """Initialize reporter."""
        pass

    def print_summary(self, stats: dict[str, Any]) -> None:
        """
        Print performance summary to console.

        Args:
            stats: Performance statistics from profiler
        """
        overall = stats["overall"]
        operations = stats["operations"]

        print("\n" + "=" * 80)
        print("PERFORMANCE REPORT")
        print("=" * 80)
        print(f"Timestamp: {stats['timestamp']}")
        print(f"Total Requests: {overall['total_requests']:,}")
        print(f"Elapsed Time: {overall['elapsed_time_s']:.2f}s")
        print(f"Throughput: {overall['throughput_rps']:.2f} requests/second")
        print(f"Success Rate: {overall['success_rate']:.2%}")

        print("\n" + "-" * 80)
        print("OVERALL LATENCY PERCENTILES")
        print("-" * 80)
        latency = overall["latency"]
        print(f"  P50: {latency['p50']:>8.2f}ms")
        print(f"  P90: {latency['p90']:>8.2f}ms")
        print(f"  P95: {latency['p95']:>8.2f}ms")
        print(f"  P99: {latency['p99']:>8.2f}ms")
        print(f"  Mean: {latency['mean']:>7.2f}ms")
        print(f"  Min: {latency['min']:>8.2f}ms")
        print(f"  Max: {latency['max']:>8.2f}ms")

        if operations:
            print("\n" + "-" * 80)
            print("PER-OPERATION STATS")
            print("-" * 80)
            print(
                f"{'Operation':<25} {'Count':>8} {'Success':>8} "
                f"{'P50':>8} {'P95':>8} {'Throughput':>12}"
            )
            print("-" * 80)

            for op_name, op_stats in sorted(operations.items()):
                op_latency = op_stats["latency"]
                print(
                    f"{op_name:<25} "
                    f"{op_stats['count']:>8} "
                    f"{op_stats['success_rate']:>7.1%} "
                    f"{op_latency['p50']:>7.1f}ms "
                    f"{op_latency['p95']:>7.1f}ms "
                    f"{op_stats['throughput_rps']:>11.2f} RPS"
                )

        print("=" * 80 + "\n")

    def print_sla_result(self, result: Any) -> None:
        """
        Print SLA compliance result.

        Args:
            result: SLAResult from profiler.check_sla()
        """
        print("\n" + "=" * 80)
        print("SLA COMPLIANCE CHECK")
        print("=" * 80)
        print(f"Timestamp: {result.timestamp}")
        print(f"Status: {'✓ PASSED' if result.passed else '✗ FAILED'}")

        if result.violations:
            print(f"\nViolations ({len(result.violations)}):")
            for i, violation in enumerate(result.violations, 1):
                print(f"  {i}. {violation}")
        else:
            print("\nAll SLA targets met! 🎉")

        print("=" * 80 + "\n")

    def export_json(self, stats: dict[str, Any], output_path: str | Path) -> None:
        """
        Export statistics to JSON file.

        Args:
            stats: Performance statistics
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w") as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Performance stats exported to {output_path}")

    def export_markdown(self, stats: dict[str, Any], output_path: str | Path) -> None:
        """
        Export statistics to Markdown file.

        Args:
            stats: Performance statistics
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        overall = stats["overall"]
        operations = stats["operations"]

        md = []
        md.append("# Performance Report\n")
        md.append(f"**Generated:** {stats['timestamp']}\n")
        md.append("## Overall Metrics\n")
        md.append(f"- **Total Requests:** {overall['total_requests']:,}")
        md.append(f"- **Elapsed Time:** {overall['elapsed_time_s']:.2f}s")
        md.append(f"- **Throughput:** {overall['throughput_rps']:.2f} requests/second")
        md.append(f"- **Success Rate:** {overall['success_rate']:.2%}\n")

        md.append("## Latency Percentiles\n")
        latency = overall["latency"]
        md.append("| Percentile | Latency (ms) |")
        md.append("|------------|--------------|")
        md.append(f"| P50        | {latency['p50']:.2f} |")
        md.append(f"| P90        | {latency['p90']:.2f} |")
        md.append(f"| P95        | {latency['p95']:.2f} |")
        md.append(f"| P99        | {latency['p99']:.2f} |")
        md.append(f"| Mean       | {latency['mean']:.2f} |")
        md.append(f"| Min        | {latency['min']:.2f} |")
        md.append(f"| Max        | {latency['max']:.2f} |\n")

        if operations:
            md.append("## Per-Operation Statistics\n")
            md.append(
                "| Operation | Count | Success Rate | P50 (ms) | P95 (ms) | Throughput (RPS) |"
            )
            md.append(
                "|-----------|-------|--------------|----------|----------|------------------|"
            )

            for op_name, op_stats in sorted(operations.items()):
                op_latency = op_stats["latency"]
                md.append(
                    f"| {op_name} | {op_stats['count']} | "
                    f"{op_stats['success_rate']:.1%} | "
                    f"{op_latency['p50']:.1f} | "
                    f"{op_latency['p95']:.1f} | "
                    f"{op_stats['throughput_rps']:.2f} |"
                )

        with output_path.open("w") as f:
            f.write("\n".join(md))

        logger.info(f"Performance report exported to {output_path}")

    def export_html(self, stats: dict[str, Any], output_path: str | Path) -> None:
        """
        Export statistics to HTML file.

        Args:
            stats: Performance statistics
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        overall = stats["overall"]
        operations = stats["operations"]
        latency = overall["latency"]

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Performance Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 40px auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
            border-bottom: 2px solid #ddd;
            padding-bottom: 8px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #f9f9f9;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #4CAF50;
        }}
        .metric-label {{
            color: #666;
            font-size: 14px;
            font-weight: 500;
            margin-bottom: 5px;
        }}
        .metric-value {{
            color: #333;
            font-size: 24px;
            font-weight: bold;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #f1f1f1;
            font-weight: 600;
            color: #555;
        }}
        tr:hover {{
            background: #f9f9f9;
        }}
        .timestamp {{
            color: #999;
            font-size: 14px;
        }}
        .status-good {{
            color: #4CAF50;
        }}
        .status-warning {{
            color: #FF9800;
        }}
        .status-bad {{
            color: #F44336;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>⚡ Performance Report</h1>
        <p class="timestamp">Generated: {stats["timestamp"]}</p>

        <h2>Overall Metrics</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Total Requests</div>
                <div class="metric-value">{overall["total_requests"]:,}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Throughput</div>
                <div class="metric-value">{overall["throughput_rps"]:.1f} <span style="font-size: 16px;">RPS</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Success Rate</div>
                <div class="metric-value {"status-good" if overall["success_rate"] >= 0.95 else "status-warning"}">
                    {overall["success_rate"]:.1%}
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Elapsed Time</div>
                <div class="metric-value">{overall["elapsed_time_s"]:.1f}<span style="font-size: 16px;">s</span></div>
            </div>
        </div>

        <h2>Latency Percentiles</h2>
        <table>
            <thead>
                <tr>
                    <th>Percentile</th>
                    <th>Latency (ms)</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>P50 (Median)</td>
                    <td>{latency["p50"]:.2f}</td>
                    <td class="{"status-good" if latency["p50"] < 500 else "status-warning"}">
                        {"✓ Good" if latency["p50"] < 500 else "⚠ Acceptable"}
                    </td>
                </tr>
                <tr>
                    <td>P90</td>
                    <td>{latency["p90"]:.2f}</td>
                    <td class="{"status-good" if latency["p90"] < 800 else "status-warning"}">
                        {"✓ Good" if latency["p90"] < 800 else "⚠ Acceptable"}
                    </td>
                </tr>
                <tr>
                    <td>P95</td>
                    <td>{latency["p95"]:.2f}</td>
                    <td class="{"status-good" if latency["p95"] < 1000 else "status-warning"}">
                        {"✓ Good" if latency["p95"] < 1000 else "⚠ Acceptable"}
                    </td>
                </tr>
                <tr>
                    <td>P99</td>
                    <td>{latency["p99"]:.2f}</td>
                    <td class="{"status-good" if latency["p99"] < 2000 else "status-warning"}">
                        {"✓ Good" if latency["p99"] < 2000 else "⚠ Acceptable"}
                    </td>
                </tr>
                <tr>
                    <td>Mean</td>
                    <td>{latency["mean"]:.2f}</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td>Min</td>
                    <td>{latency["min"]:.2f}</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td>Max</td>
                    <td>{latency["max"]:.2f}</td>
                    <td>-</td>
                </tr>
            </tbody>
        </table>
"""

        if operations:
            html += """
        <h2>Per-Operation Statistics</h2>
        <table>
            <thead>
                <tr>
                    <th>Operation</th>
                    <th>Count</th>
                    <th>Success Rate</th>
                    <th>P50 (ms)</th>
                    <th>P95 (ms)</th>
                    <th>Throughput (RPS)</th>
                </tr>
            </thead>
            <tbody>
"""

            for op_name, op_stats in sorted(operations.items()):
                op_latency = op_stats["latency"]
                html += f"""
                <tr>
                    <td><strong>{op_name}</strong></td>
                    <td>{op_stats["count"]}</td>
                    <td class="{"status-good" if op_stats["success_rate"] >= 0.95 else "status-warning"}">
                        {op_stats["success_rate"]:.1%}
                    </td>
                    <td>{op_latency["p50"]:.1f}</td>
                    <td>{op_latency["p95"]:.1f}</td>
                    <td>{op_stats["throughput_rps"]:.2f}</td>
                </tr>
"""

            html += """
            </tbody>
        </table>
"""

        html += """
    </div>
</body>
</html>
"""

        with output_path.open("w") as f:
            f.write(html)

        logger.info(f"HTML report exported to {output_path}")
