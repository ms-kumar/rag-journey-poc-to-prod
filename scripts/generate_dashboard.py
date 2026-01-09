"""
Weekly evaluation dashboard generator.

Generates HTML dashboard with evaluation metrics, trends, and visualizations.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_evaluation_results(results_dir: Path) -> list[dict[str, Any]]:
    """
    Load all evaluation results from directory.

    Args:
        results_dir: Directory containing evaluation result files

    Returns:
        List of evaluation results sorted by timestamp
    """
    results = []

    if not results_dir.exists():
        logger.warning(f"Results directory not found: {results_dir}")
        return results

    # Look for both eval_*.json and demo_eval_result.json
    result_files = list(results_dir.glob("eval_*.json"))
    demo_file = results_dir / "demo_eval_result.json"
    if demo_file.exists():
        result_files.append(demo_file)
    
    for result_file in result_files:
        try:
            with open(result_file) as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            logger.warning(f"Failed to load {result_file}: {e}")

    # Sort by timestamp
    results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return results


def calculate_trends(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Calculate metric trends from historical results.

    Args:
        results: List of evaluation results

    Returns:
        Dict with trend analysis
    """
    if len(results) < 2:
        return {"status": "insufficient_data", "message": "Need at least 2 data points"}

    latest = results[0]["metrics"]
    previous = results[1]["metrics"]

    trends = {
        "retrieval": {},
        "performance": {},
        "overall_trend": "stable",
    }

    # Compare retrieval metrics
    def calc_change(latest_val: float, prev_val: float) -> dict:
        delta = latest_val - prev_val
        pct_change = (delta / prev_val * 100) if prev_val != 0 else 0
        return {
            "current": latest_val,
            "previous": prev_val,
            "delta": delta,
            "pct_change": pct_change,
            "trend": "up" if delta > 0 else "down" if delta < 0 else "stable",
        }

    trends["retrieval"]["precision@5"] = calc_change(
        latest["retrieval"]["precision@k"]["5"],
        previous["retrieval"]["precision@k"]["5"],
    )

    trends["retrieval"]["recall@10"] = calc_change(
        latest["retrieval"]["recall@k"]["10"],
        previous["retrieval"]["recall@k"]["10"],
    )

    trends["retrieval"]["mrr"] = calc_change(
        latest["retrieval"]["mrr"], previous["retrieval"]["mrr"]
    )

    trends["performance"]["latency_p95"] = calc_change(
        latest["performance"]["latency_p95_ms"],
        previous["performance"]["latency_p95_ms"],
    )

    return trends


def generate_html_dashboard(
    results: list[dict[str, Any]],
    trends: dict[str, Any],
    output_path: Path,
) -> None:
    """
    Generate HTML dashboard.

    Args:
        results: List of evaluation results
        trends: Trend analysis
        output_path: Path to save HTML file
    """
    if not results:
        logger.warning("No results to generate dashboard")
        return

    latest = results[0]
    metrics = latest["metrics"]

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Evaluation Dashboard</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: #f5f5f5;
            padding: 20px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 2.5em;
        }}
        .subtitle {{
            color: #7f8c8d;
            margin-bottom: 30px;
            font-size: 1.1em;
        }}
        .status {{
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            margin-bottom: 20px;
        }}
        .status.passed {{
            background: #d4edda;
            color: #155724;
        }}
        .status.failed {{
            background: #f8d7da;
            color: #721c24;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }}
        .metric-card h3 {{
            color: #2c3e50;
            font-size: 0.9em;
            text-transform: uppercase;
            margin-bottom: 10px;
            font-weight: 600;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
            margin: 10px 0;
        }}
        .metric-trend {{
            font-size: 0.9em;
            color: #7f8c8d;
        }}
        .trend-up {{
            color: #27ae60;
        }}
        .trend-down {{
            color: #e74c3c;
        }}
        .section {{
            margin: 40px 0;
        }}
        .section h2 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 20px;
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
            background: #3498db;
            color: white;
            font-weight: 600;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #7f8c8d;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 RAG Evaluation Dashboard</h1>
        <div class="subtitle">Weekly evaluation metrics and trends</div>
        
        <div class="status {'passed' if latest.get('passed') else 'failed'}">
            {'✅ PASSED' if latest.get('passed') else '❌ FAILED'}
        </div>
        
        <div class="section">
            <h2>📊 Latest Metrics</h2>
            <p><strong>Evaluation Date:</strong> {datetime.fromisoformat(latest['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Duration:</strong> {latest['duration_seconds']:.2f}s</p>
            <p><strong>Queries Evaluated:</strong> {metrics['metadata']['num_queries']}</p>
        </div>

        <div class="section">
            <h2>🔍 Retrieval Metrics</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>Precision@5</h3>
                    <div class="metric-value">{metrics['retrieval']['precision@k']['5']:.3f}</div>
                    {get_trend_html(trends, 'retrieval', 'precision@5')}
                </div>
                <div class="metric-card">
                    <h3>Recall@10</h3>
                    <div class="metric-value">{metrics['retrieval']['recall@k']['10']:.3f}</div>
                    {get_trend_html(trends, 'retrieval', 'recall@10')}
                </div>
                <div class="metric-card">
                    <h3>MRR</h3>
                    <div class="metric-value">{metrics['retrieval']['mrr']:.3f}</div>
                    {get_trend_html(trends, 'retrieval', 'mrr')}
                </div>
                <div class="metric-card">
                    <h3>NDCG@10</h3>
                    <div class="metric-value">{metrics['retrieval']['ndcg@k']['10']:.3f}</div>
                </div>
                <div class="metric-card">
                    <h3>MAP</h3>
                    <div class="metric-value">{metrics['retrieval']['map']:.3f}</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>⚡ Performance Metrics</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>Latency P50</h3>
                    <div class="metric-value">{metrics['performance']['latency_p50_ms']:.1f}ms</div>
                </div>
                <div class="metric-card">
                    <h3>Latency P95</h3>
                    <div class="metric-value">{metrics['performance']['latency_p95_ms']:.1f}ms</div>
                    {get_trend_html(trends, 'performance', 'latency_p95')}
                </div>
                <div class="metric-card">
                    <h3>Latency P99</h3>
                    <div class="metric-value">{metrics['performance']['latency_p99_ms']:.1f}ms</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>📈 Historical Results</h2>
            <table>
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Status</th>
                        <th>Precision@5</th>
                        <th>Recall@10</th>
                        <th>MRR</th>
                        <th>Latency P95</th>
                    </tr>
                </thead>
                <tbody>
                    {generate_history_rows(results[:10])}
                </tbody>
            </table>
        </div>

        <div class="footer">
            Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
            RAG Journey - Production Evaluation System
        </div>
    </div>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)

    logger.info(f"Dashboard generated: {output_path}")


def get_trend_html(trends: dict, category: str, metric: str) -> str:
    """Generate HTML for metric trend."""
    if trends.get("status") == "insufficient_data":
        return '<div class="metric-trend">No trend data</div>'

    trend_data = trends.get(category, {}).get(metric)
    if not trend_data:
        return '<div class="metric-trend">-</div>'

    delta = trend_data["delta"]
    pct = trend_data["pct_change"]
    trend_class = "trend-up" if delta > 0 else "trend-down" if delta < 0 else ""

    # For latency, down is good
    if "latency" in metric.lower():
        trend_class = "trend-down" if delta > 0 else "trend-up" if delta < 0 else ""

    symbol = "▲" if delta > 0 else "▼" if delta < 0 else "="

    return f'<div class="metric-trend {trend_class}">{symbol} {abs(pct):.1f}% vs last week</div>'


def generate_history_rows(results: list[dict[str, Any]]) -> str:
    """Generate HTML rows for historical results."""
    rows = []
    for result in results:
        metrics = result["metrics"]
        status = "✅" if result.get("passed") else "❌"
        timestamp = datetime.fromisoformat(result["timestamp"])

        row = f"""
                    <tr>
                        <td>{timestamp.strftime('%Y-%m-%d')}</td>
                        <td>{status}</td>
                        <td>{metrics['retrieval']['precision@k']['5']:.3f}</td>
                        <td>{metrics['retrieval']['recall@k']['10']:.3f}</td>
                        <td>{metrics['retrieval']['mrr']:.3f}</td>
                        <td>{metrics['performance']['latency_p95_ms']:.1f}ms</td>
                    </tr>
        """
        rows.append(row)

    return "".join(rows)


def main():
    """Generate weekly evaluation dashboard."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate evaluation dashboard")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing evaluation results",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/dashboard.html",
        help="Output path for HTML dashboard",
    )

    args = parser.parse_args()

    # Load results
    results_dir = Path(args.results_dir)
    results = load_evaluation_results(results_dir)

    if not results:
        logger.error("No evaluation results found")
        logger.info("Run evaluations first:")
        logger.info("  python scripts/ci_eval_gate.py")
        return

    logger.info(f"Loaded {len(results)} evaluation results")

    # Calculate trends
    trends = calculate_trends(results)

    # Generate dashboard
    output_path = Path(args.output)
    generate_html_dashboard(results, trends, output_path)

    print(f"\n✅ Dashboard generated: {output_path}")
    print(f"   Open with: open {output_path}")


if __name__ == "__main__":
    main()
