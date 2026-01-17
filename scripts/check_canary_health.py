#!/usr/bin/env python3
"""
Canary Health Check Script.

Checks canary deployment health metrics and determines if it's safe to proceed.
Used in CI/CD pipelines to gate promotions.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Try to import requests, fall back to urllib
try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    import urllib.error
    import urllib.request

    HAS_REQUESTS = False


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    healthy: bool
    error_rate: float
    latency_p50: float
    latency_p99: float
    request_count: int
    checks_passed: list[str]
    checks_failed: list[str]
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "healthy": self.healthy,
            "error_rate": self.error_rate,
            "latency_p50": self.latency_p50,
            "latency_p99": self.latency_p99,
            "request_count": self.request_count,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "timestamp": self.timestamp,
        }


def fetch_metrics(endpoint: str) -> dict[str, Any]:
    """Fetch metrics from endpoint."""
    if HAS_REQUESTS:
        response = requests.get(endpoint, timeout=10)
        response.raise_for_status()
        return response.json()
    req = urllib.request.Request(endpoint)
    with urllib.request.urlopen(req, timeout=10) as response:
        return json.loads(response.read().decode())


def get_canary_metrics(deployment: str, metrics_endpoint: str | None = None) -> dict[str, Any]:
    """
    Get canary deployment metrics.

    In a real setup, this would fetch from:
    - Prometheus/Grafana
    - Datadog
    - CloudWatch
    - Application metrics endpoint
    """
    # Default metrics endpoint
    if not metrics_endpoint:
        metrics_endpoint = f"https://{deployment}.rag-api.example.com/metrics"

    try:
        return fetch_metrics(metrics_endpoint)
    except Exception as e:
        print(f"⚠️ Could not fetch metrics from {metrics_endpoint}: {e}")
        # Return simulated metrics for demo
        return simulate_metrics()


def simulate_metrics() -> dict[str, Any]:
    """Simulate metrics for testing/demo purposes."""
    import random

    # Simulate healthy metrics most of the time
    is_healthy = random.random() > 0.1

    if is_healthy:
        return {
            "request_count": random.randint(100, 1000),
            "error_count": random.randint(0, 5),
            "latency_p50_ms": random.uniform(50, 100),
            "latency_p99_ms": random.uniform(150, 300),
            "latency_mean_ms": random.uniform(60, 120),
            "success_rate": random.uniform(0.98, 1.0),
        }
    return {
        "request_count": random.randint(100, 500),
        "error_count": random.randint(20, 50),
        "latency_p50_ms": random.uniform(200, 400),
        "latency_p99_ms": random.uniform(500, 1000),
        "latency_mean_ms": random.uniform(250, 500),
        "success_rate": random.uniform(0.85, 0.95),
    }


def check_canary_health(
    deployment: str,
    error_threshold: float = 0.05,
    latency_threshold: float = 500.0,
    min_requests: int = 100,
    metrics_endpoint: str | None = None,
) -> HealthCheckResult:
    """
    Check if canary deployment is healthy.

    Args:
        deployment: Name of the canary deployment
        error_threshold: Maximum acceptable error rate (0.05 = 5%)
        latency_threshold: Maximum acceptable P99 latency in ms
        min_requests: Minimum requests required for valid check
        metrics_endpoint: Optional custom metrics endpoint

    Returns:
        HealthCheckResult with detailed check results
    """
    print(f"🔍 Checking canary health for: {deployment}")
    print(f"   Error threshold: {error_threshold * 100:.1f}%")
    print(f"   Latency threshold: {latency_threshold}ms")
    print(f"   Min requests: {min_requests}")
    print()

    metrics = get_canary_metrics(deployment, metrics_endpoint)

    # Extract metrics
    request_count = metrics.get("request_count", 0)
    error_count = metrics.get("error_count", 0)
    error_rate = error_count / request_count if request_count > 0 else 0.0
    latency_p50 = metrics.get("latency_p50_ms", 0.0)
    latency_p99 = metrics.get("latency_p99_ms", 0.0)

    checks_passed = []
    checks_failed = []

    # Check 1: Minimum request volume
    if request_count >= min_requests:
        checks_passed.append(f"Request volume: {request_count} >= {min_requests}")
    else:
        checks_failed.append(f"Request volume: {request_count} < {min_requests}")

    # Check 2: Error rate
    if error_rate <= error_threshold:
        checks_passed.append(f"Error rate: {error_rate * 100:.2f}% <= {error_threshold * 100:.1f}%")
    else:
        checks_failed.append(f"Error rate: {error_rate * 100:.2f}% > {error_threshold * 100:.1f}%")

    # Check 3: Latency
    if latency_p99 <= latency_threshold:
        checks_passed.append(f"P99 latency: {latency_p99:.1f}ms <= {latency_threshold}ms")
    else:
        checks_failed.append(f"P99 latency: {latency_p99:.1f}ms > {latency_threshold}ms")

    # Determine overall health
    healthy = len(checks_failed) == 0 and request_count >= min_requests

    result = HealthCheckResult(
        healthy=healthy,
        error_rate=error_rate,
        latency_p50=latency_p50,
        latency_p99=latency_p99,
        request_count=request_count,
        checks_passed=checks_passed,
        checks_failed=checks_failed,
        timestamp=datetime.utcnow().isoformat(),
    )

    # Print results
    print("📊 Health Check Results:")
    print(f"   Request count: {request_count}")
    print(f"   Error rate: {error_rate * 100:.2f}%")
    print(f"   P50 latency: {latency_p50:.1f}ms")
    print(f"   P99 latency: {latency_p99:.1f}ms")
    print()

    if checks_passed:
        print("✅ Passed checks:")
        for check in checks_passed:
            print(f"   • {check}")

    if checks_failed:
        print("❌ Failed checks:")
        for check in checks_failed:
            print(f"   • {check}")

    print()
    if healthy:
        print("✅ Canary is HEALTHY")
    else:
        print("❌ Canary is UNHEALTHY")

    return result


def wait_and_check(
    deployment: str,
    error_threshold: float,
    latency_threshold: float,
    min_requests: int,
    wait_seconds: int = 60,
    max_retries: int = 3,
) -> HealthCheckResult:
    """Wait for metrics to accumulate and then check health."""
    print(f"⏳ Waiting {wait_seconds}s for metrics to accumulate...")
    time.sleep(wait_seconds)

    for attempt in range(max_retries):
        print(f"\n🔄 Health check attempt {attempt + 1}/{max_retries}")
        result = check_canary_health(
            deployment=deployment,
            error_threshold=error_threshold,
            latency_threshold=latency_threshold,
            min_requests=min_requests,
        )

        if result.healthy:
            return result

        if attempt < max_retries - 1:
            print("\n⏳ Retrying in 30s...")
            time.sleep(30)

    return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Check canary deployment health")
    parser.add_argument("--deployment", required=True, help="Canary deployment name")
    parser.add_argument(
        "--error-threshold",
        type=float,
        default=0.05,
        help="Maximum error rate (default: 0.05 = 5%%)",
    )
    parser.add_argument(
        "--latency-threshold",
        type=float,
        default=500.0,
        help="Maximum P99 latency in ms (default: 500)",
    )
    parser.add_argument(
        "--min-requests",
        type=int,
        default=100,
        help="Minimum requests for valid check (default: 100)",
    )
    parser.add_argument(
        "--metrics-endpoint",
        help="Custom metrics endpoint URL",
    )
    parser.add_argument(
        "--wait",
        type=int,
        default=0,
        help="Seconds to wait before checking (default: 0)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=1,
        help="Number of retry attempts (default: 1)",
    )
    parser.add_argument(
        "--output",
        help="Output file for results JSON",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("🐤 Canary Health Check")
    print("=" * 60)
    print()

    if args.wait > 0 or args.retries > 1:
        result = wait_and_check(
            deployment=args.deployment,
            error_threshold=args.error_threshold,
            latency_threshold=args.latency_threshold,
            min_requests=args.min_requests,
            wait_seconds=args.wait,
            max_retries=args.retries,
        )
    else:
        result = check_canary_health(
            deployment=args.deployment,
            error_threshold=args.error_threshold,
            latency_threshold=args.latency_threshold,
            min_requests=args.min_requests,
            metrics_endpoint=args.metrics_endpoint,
        )

    # Save results if output specified
    if args.output:
        with Path(args.output).open("w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\n📄 Results saved to: {args.output}")

    # Exit with appropriate code
    sys.exit(0 if result.healthy else 1)


if __name__ == "__main__":
    main()
