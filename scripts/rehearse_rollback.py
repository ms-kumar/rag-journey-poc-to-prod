#!/usr/bin/env python3
"""
Rollback Rehearsal Script.

Simulates and validates rollback procedures in a safe environment.
Ensures team is prepared for real rollback scenarios.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class ScenarioType(str, Enum):
    """Types of rollback scenarios."""

    CANARY_FAILURE = "canary_failure"
    ERROR_SPIKE = "error_spike"
    LATENCY_DEGRADATION = "latency_degradation"
    HEALTH_CHECK_FAILURE = "health_check_failure"
    MEMORY_LEAK = "memory_leak"
    DEPENDENCY_FAILURE = "dependency_failure"


@dataclass
class RehearsalStep:
    """A single step in the rehearsal."""

    name: str
    description: str
    command: str | None = None
    expected_outcome: str = ""
    actual_outcome: str = ""
    passed: bool | None = None
    duration_seconds: float = 0.0


@dataclass
class RehearsalResult:
    """Result of a rollback rehearsal."""

    scenario: ScenarioType
    environment: str
    steps: list[RehearsalStep] = field(default_factory=list)
    started_at: str = ""
    completed_at: str = ""
    total_duration_seconds: float = 0.0
    success: bool = False
    lessons_learned: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scenario": self.scenario.value,
            "environment": self.environment,
            "steps": [
                {
                    "name": s.name,
                    "description": s.description,
                    "command": s.command,
                    "expected_outcome": s.expected_outcome,
                    "actual_outcome": s.actual_outcome,
                    "passed": s.passed,
                    "duration_seconds": s.duration_seconds,
                }
                for s in self.steps
            ],
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "total_duration_seconds": self.total_duration_seconds,
            "success": self.success,
            "lessons_learned": self.lessons_learned,
        }


def print_header(text: str) -> None:
    """Print formatted header."""
    print()
    print("=" * 70)
    print(f"  {text}")
    print("=" * 70)
    print()


def print_step(step_num: int, total: int, name: str) -> None:
    """Print step header."""
    print(f"\n📋 Step {step_num}/{total}: {name}")
    print("-" * 50)


def simulate_command(command: str, dry_run: bool = True) -> tuple[bool, str]:
    """
    Simulate or execute a command.

    Args:
        command: Command to run
        dry_run: If True, simulate; if False, actually run

    Returns:
        Tuple of (success, output)
    """
    if dry_run:
        # Simulate command execution
        print(f"  [DRY RUN] Would execute: {command}")
        time.sleep(random.uniform(0.5, 1.5))  # Simulate execution time
        return True, "Simulated success"

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return result.returncode == 0, result.stdout or result.stderr
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, str(e)


def prompt_user(question: str, default: bool = True) -> bool:
    """Prompt user for yes/no confirmation."""
    default_str = "Y/n" if default else "y/N"
    response = input(f"  {question} [{default_str}]: ").strip().lower()

    if not response:
        return default
    return response in ("y", "yes")


def get_scenario_steps(scenario: ScenarioType, environment: str) -> list[RehearsalStep]:
    """Get steps for a specific scenario."""
    base_steps = [
        RehearsalStep(
            name="Detect Issue",
            description="Monitor alerts and identify the problem",
            expected_outcome="Issue identified within 2 minutes",
        ),
        RehearsalStep(
            name="Assess Impact",
            description="Determine scope and severity of the issue",
            expected_outcome="Impact assessment complete",
        ),
        RehearsalStep(
            name="Decide to Rollback",
            description="Make go/no-go decision based on decision matrix",
            expected_outcome="Rollback decision made within 5 minutes",
        ),
    ]

    rollback_steps = {
        ScenarioType.CANARY_FAILURE: [
            RehearsalStep(
                name="Scale Down Canary",
                description="Remove canary deployment from rotation",
                command=f"kubectl -n {environment} scale deployment/rag-api-canary --replicas=0",
                expected_outcome="Canary scaled to 0 replicas",
            ),
            RehearsalStep(
                name="Remove Traffic Split",
                description="Route all traffic to stable deployment",
                command=f'kubectl -n {environment} patch virtualservice rag-api --type=merge -p \'{{"spec":{{"http":[{{"route":[{{"destination":{{"host":"rag-api"}}}}]}}]}}}}\'',
                expected_outcome="All traffic routed to stable",
            ),
        ],
        ScenarioType.ERROR_SPIKE: [
            RehearsalStep(
                name="Check Current Version",
                description="Identify current running version",
                command=f"kubectl -n {environment} get deployment rag-api -o jsonpath='{{.spec.template.spec.containers[0].image}}'",
                expected_outcome="Current version identified",
            ),
            RehearsalStep(
                name="Identify Last Good Version",
                description="Find the last known good version from rollout history",
                command=f"kubectl -n {environment} rollout history deployment/rag-api",
                expected_outcome="Last good version identified",
            ),
            RehearsalStep(
                name="Execute Rollback",
                description="Roll back to previous version",
                command=f"kubectl -n {environment} rollout undo deployment/rag-api",
                expected_outcome="Rollback initiated",
            ),
            RehearsalStep(
                name="Monitor Rollout",
                description="Watch rollback progress",
                command=f"kubectl -n {environment} rollout status deployment/rag-api --timeout=5m",
                expected_outcome="Rollback completed successfully",
            ),
        ],
        ScenarioType.LATENCY_DEGRADATION: [
            RehearsalStep(
                name="Analyze Latency Source",
                description="Identify if latency is from app or dependencies",
                command=f"kubectl -n {environment} logs -l app=rag-api --tail=100 | grep -i latency",
                expected_outcome="Latency source identified",
            ),
            RehearsalStep(
                name="Check Resource Usage",
                description="Verify pod resource consumption",
                command=f"kubectl -n {environment} top pods -l app=rag-api",
                expected_outcome="Resource usage assessed",
            ),
            RehearsalStep(
                name="Execute Rollback",
                description="Roll back deployment",
                command=f"kubectl -n {environment} rollout undo deployment/rag-api",
                expected_outcome="Rollback initiated",
            ),
        ],
        ScenarioType.HEALTH_CHECK_FAILURE: [
            RehearsalStep(
                name="Check Pod Status",
                description="Identify failing pods",
                command=f"kubectl -n {environment} get pods -l app=rag-api",
                expected_outcome="Failing pods identified",
            ),
            RehearsalStep(
                name="Check Pod Logs",
                description="Gather error information from logs",
                command=f"kubectl -n {environment} logs -l app=rag-api --tail=50",
                expected_outcome="Error cause identified",
            ),
            RehearsalStep(
                name="Execute Rollback",
                description="Roll back to healthy version",
                command=f"kubectl -n {environment} rollout undo deployment/rag-api",
                expected_outcome="Rollback initiated",
            ),
        ],
        ScenarioType.MEMORY_LEAK: [
            RehearsalStep(
                name="Identify Memory Usage Trend",
                description="Check memory consumption over time",
                command=f"kubectl -n {environment} top pods -l app=rag-api",
                expected_outcome="Memory leak pattern confirmed",
            ),
            RehearsalStep(
                name="Restart Pods (Temporary)",
                description="Restart pods as immediate mitigation",
                command=f"kubectl -n {environment} rollout restart deployment/rag-api",
                expected_outcome="Pods restarted, memory freed",
            ),
            RehearsalStep(
                name="Execute Rollback",
                description="Roll back to non-leaking version",
                command=f"kubectl -n {environment} rollout undo deployment/rag-api",
                expected_outcome="Rollback completed",
            ),
        ],
        ScenarioType.DEPENDENCY_FAILURE: [
            RehearsalStep(
                name="Check Dependency Status",
                description="Verify status of Qdrant, Redis, etc.",
                command=f"kubectl -n {environment} get pods | grep -E 'qdrant|redis'",
                expected_outcome="Dependency status determined",
            ),
            RehearsalStep(
                name="Assess if Rollback Will Help",
                description="Determine if issue is in app or dependency",
                expected_outcome="Root cause categorized",
            ),
            RehearsalStep(
                name="Execute Rollback if Applicable",
                description="Roll back if issue is in application code",
                command=f"kubectl -n {environment} rollout undo deployment/rag-api",
                expected_outcome="Rollback completed if applicable",
            ),
        ],
    }

    verification_steps = [
        RehearsalStep(
            name="Verify Pods Running",
            description="Confirm all pods are running and ready",
            command=f"kubectl -n {environment} get pods -l app=rag-api",
            expected_outcome="All pods in Running state",
        ),
        RehearsalStep(
            name="Verify Health Endpoint",
            description="Check application health endpoint",
            command=f"curl -s https://{environment}.rag-api.example.com/health | jq .",
            expected_outcome="Health check returns 200 OK",
        ),
        RehearsalStep(
            name="Verify Metrics",
            description="Confirm error rate and latency are normal",
            expected_outcome="Metrics within acceptable thresholds",
        ),
        RehearsalStep(
            name="Send Communication",
            description="Notify stakeholders of rollback completion",
            expected_outcome="Communication sent to appropriate channels",
        ),
    ]

    return base_steps + rollback_steps.get(scenario, []) + verification_steps


def run_rehearsal(
    scenario: ScenarioType,
    environment: str,
    dry_run: bool = True,
    interactive: bool = True,
) -> RehearsalResult:
    """
    Run a rollback rehearsal.

    Args:
        scenario: Type of scenario to simulate
        environment: Target environment
        dry_run: If True, simulate commands; if False, actually run
        interactive: If True, prompt user at each step

    Returns:
        RehearsalResult with detailed results
    """
    print_header(f"Rollback Rehearsal: {scenario.value.replace('_', ' ').title()}")

    print(f"Environment: {environment}")
    print(f"Mode: {'DRY RUN (simulated)' if dry_run else 'LIVE EXECUTION'}")
    print(f"Interactive: {'Yes' if interactive else 'No'}")

    if not dry_run:
        print("\n⚠️  WARNING: LIVE EXECUTION MODE")
        print("    Commands will actually be executed!")
        if not prompt_user("Continue?", default=False):
            print("Rehearsal cancelled.")
            sys.exit(0)

    result = RehearsalResult(
        scenario=scenario,
        environment=environment,
        started_at=datetime.utcnow().isoformat(),
    )

    steps = get_scenario_steps(scenario, environment)
    start_time = time.time()

    print(f"\nScenario: {scenario.value}")
    print(f"Total steps: {len(steps)}")

    for i, step in enumerate(steps, 1):
        print_step(i, len(steps), step.name)
        print(f"  Description: {step.description}")

        if step.command:
            print(f"  Command: {step.command}")

        step_start = time.time()

        if interactive and not prompt_user("Execute this step?"):
            step.actual_outcome = "Skipped by user"
            step.passed = None
            result.steps.append(step)
            continue

        if step.command:
            success, output = simulate_command(step.command, dry_run)
            step.actual_outcome = output
            step.passed = success
        else:
            # Manual step - ask user to confirm
            if interactive:
                step.passed = prompt_user("Mark step as complete?")
                step.actual_outcome = "Completed" if step.passed else "Not completed"
            else:
                step.passed = True
                step.actual_outcome = "Simulated as complete"

        step.duration_seconds = time.time() - step_start

        # Show result
        if step.passed:
            print(f"  ✅ Result: {step.actual_outcome}")
        elif step.passed is None:
            print("  ⏭️  Skipped")
        else:
            print(f"  ❌ Failed: {step.actual_outcome}")

        result.steps.append(step)

    result.completed_at = datetime.utcnow().isoformat()
    result.total_duration_seconds = time.time() - start_time

    # Determine success
    executed_steps = [s for s in result.steps if s.passed is not None]
    passed_steps = [s for s in executed_steps if s.passed]
    result.success = len(passed_steps) == len(executed_steps) and len(executed_steps) > 0

    return result


def print_summary(result: RehearsalResult) -> None:
    """Print rehearsal summary."""
    print_header("Rehearsal Summary")

    print(f"Scenario: {result.scenario.value}")
    print(f"Environment: {result.environment}")
    print(f"Duration: {result.total_duration_seconds:.1f} seconds")
    print(f"Status: {'✅ SUCCESS' if result.success else '❌ FAILED'}")

    print("\n📊 Step Results:")
    for step in result.steps:
        if step.passed:
            status = "✅"
        elif step.passed is None:
            status = "⏭️"
        else:
            status = "❌"
        print(f"  {status} {step.name} ({step.duration_seconds:.1f}s)")

    # Calculate stats
    total = len(result.steps)
    passed = len([s for s in result.steps if s.passed])
    failed = len([s for s in result.steps if s.passed is False])
    skipped = len([s for s in result.steps if s.passed is None])

    print("\n📈 Statistics:")
    print(f"  Total steps: {total}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Skipped: {skipped}")


def collect_lessons_learned(result: RehearsalResult, interactive: bool = True) -> None:
    """Collect lessons learned from the rehearsal."""
    if not interactive:
        result.lessons_learned = ["Rehearsal completed in non-interactive mode"]
        return

    print_header("Lessons Learned")

    print("Please share any lessons learned from this rehearsal.")
    print("Enter each lesson on a new line. Enter empty line to finish.\n")

    while True:
        lesson = input("Lesson: ").strip()
        if not lesson:
            break
        result.lessons_learned.append(lesson)

    if not result.lessons_learned:
        result.lessons_learned = ["No lessons recorded"]


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run rollback rehearsal")
    parser.add_argument(
        "--scenario",
        type=str,
        choices=[s.value for s in ScenarioType],
        default=ScenarioType.ERROR_SPIKE.value,
        help="Rollback scenario to rehearse",
    )
    parser.add_argument(
        "--environment",
        type=str,
        default="staging",
        help="Target environment (default: staging)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Execute commands for real (default: dry run)",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Run without user prompts",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for results JSON",
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="List available scenarios and exit",
    )

    args = parser.parse_args()

    if args.list_scenarios:
        print("Available scenarios:")
        for scenario in ScenarioType:
            print(f"  - {scenario.value}")
        sys.exit(0)

    print_header("🎭 Rollback Rehearsal Tool")
    print("This tool helps teams practice rollback procedures.")
    print("Regular rehearsals improve incident response times.\n")

    scenario = ScenarioType(args.scenario)
    dry_run = not args.live
    interactive = not args.non_interactive

    result = run_rehearsal(
        scenario=scenario,
        environment=args.environment,
        dry_run=dry_run,
        interactive=interactive,
    )

    print_summary(result)

    if interactive:
        collect_lessons_learned(result, interactive)

    # Save results
    if args.output:
        with Path(args.output).open("w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\n📄 Results saved to: {args.output}")

    print("\n" + "=" * 70)
    if result.success:
        print("✅ Rehearsal completed successfully!")
        print("   Team is prepared for this rollback scenario.")
    else:
        print("⚠️  Rehearsal had issues. Review failed steps.")
        print("   Consider scheduling additional practice.")
    print("=" * 70 + "\n")

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
