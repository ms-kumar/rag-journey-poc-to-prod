"""
Create sample evaluation results for dashboard demonstration.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path


def create_sample_results():
    """Create sample evaluation results showing improvement over time."""
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Create 5 sample results showing gradual improvement
    base_date = datetime.now()

    samples = [
        {
            "days_ago": 28,
            "metrics": {
                "precision@5": 0.58,
                "recall@10": 0.65,
                "mrr": 0.52,
                "ndcg@10": 0.62,
                "map": 0.58,
                "latency_p95": 2200.0,
            },
            "passed": False,
        },
        {
            "days_ago": 21,
            "metrics": {
                "precision@5": 0.62,
                "recall@10": 0.68,
                "mrr": 0.56,
                "ndcg@10": 0.64,
                "map": 0.61,
                "latency_p95": 2100.0,
            },
            "passed": True,
        },
        {
            "days_ago": 14,
            "metrics": {
                "precision@5": 0.65,
                "recall@10": 0.72,
                "mrr": 0.60,
                "ndcg@10": 0.67,
                "map": 0.63,
                "latency_p95": 1950.0,
            },
            "passed": True,
        },
        {
            "days_ago": 7,
            "metrics": {
                "precision@5": 0.68,
                "recall@10": 0.75,
                "mrr": 0.63,
                "ndcg@10": 0.70,
                "map": 0.66,
                "latency_p95": 1850.0,
            },
            "passed": True,
        },
        {
            "days_ago": 0,
            "metrics": {
                "precision@5": 0.72,
                "recall@10": 0.78,
                "mrr": 0.67,
                "ndcg@10": 0.73,
                "map": 0.69,
                "latency_p95": 1750.0,
            },
            "passed": True,
        },
    ]

    for i, sample in enumerate(samples):
        timestamp = base_date - timedelta(days=sample["days_ago"])
        m = sample["metrics"]

        result = {
            "metrics": {
                "retrieval": {
                    "precision@k": {
                        "1": m["precision@5"] - 0.15,
                        "3": m["precision@5"] - 0.08,
                        "5": m["precision@5"],
                        "10": m["precision@5"] - 0.05,
                        "20": m["precision@5"] - 0.10,
                    },
                    "recall@k": {
                        "1": m["recall@10"] - 0.40,
                        "3": m["recall@10"] - 0.25,
                        "5": m["recall@10"] - 0.15,
                        "10": m["recall@10"],
                        "20": m["recall@10"] + 0.05,
                    },
                    "mrr": m["mrr"],
                    "ndcg@k": {
                        "1": m["ndcg@10"] - 0.08,
                        "3": m["ndcg@10"] - 0.05,
                        "5": m["ndcg@10"] - 0.03,
                        "10": m["ndcg@10"],
                        "20": m["ndcg@10"] + 0.02,
                    },
                    "map": m["map"],
                },
                "generation": {
                    "faithfulness": 0.0,
                    "relevance": 0.0,
                    "answer_quality": 0.0,
                },
                "performance": {
                    "latency_p50_ms": m["latency_p95"] * 0.45,
                    "latency_p95_ms": m["latency_p95"],
                    "latency_p99_ms": m["latency_p95"] * 1.15,
                },
                "metadata": {
                    "num_queries": 21,
                    "timestamp": timestamp.isoformat(),
                },
            },
            "passed": sample["passed"],
            "failed_checks": [] if sample["passed"] else ["Precision@5 below threshold"],
            "timestamp": timestamp.isoformat(),
            "duration_seconds": 12.5 + i * 0.5,
            "metadata": {
                "dataset_name": "rag_default_eval",
                "dataset_size": 21,
                "k_values": [1, 3, 5, 10, 20],
                "include_generation": False,
            },
        }

        filename = f"eval_week{5-i}_result.json"
        filepath = results_dir / filename

        with open(filepath, "w") as f:
            json.dump(result, f, indent=2)

        print(f"✓ Created {filename} - {'PASSED' if sample['passed'] else 'FAILED'}")

    print(f"\n✅ Created {len(samples)} sample evaluation results in {results_dir}/")
    print(f"   Showing improvement from {samples[0]['metrics']['precision@5']:.2f} to {samples[-1]['metrics']['precision@5']:.2f} precision")
    print(f"\n   Run: make dashboard")


if __name__ == "__main__":
    create_sample_results()
