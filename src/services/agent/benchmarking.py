"""Benchmarking module for complex task evaluation."""

import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TaskBenchmark:
    """Benchmark result for a single task.

    Attributes:
        task_id: Unique task identifier
        task_description: Task description
        query: Original query
        execution_time: Time taken in seconds
        success: Whether task succeeded
        quality_score: Quality of result (0.0 to 1.0)
        tool_used: Tool that was used
        tool_confidence: Tool selection confidence
        retry_count: Number of retries
        error_message: Error message if failed
        result_metadata: Additional result metadata
        timestamp: When benchmark was recorded
    """

    task_id: str
    task_description: str
    query: str
    execution_time: float
    success: bool
    quality_score: float = 0.0
    tool_used: str = ""
    tool_confidence: float = 0.0
    retry_count: int = 0
    error_message: str = ""
    result_metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class BenchmarkSuite:
    """A suite of benchmark results.

    Attributes:
        name: Suite name
        description: Suite description
        benchmarks: List of task benchmarks
        total_time: Total execution time
        success_rate: Overall success rate
        average_quality: Average quality score
        created_at: When suite was created
    """

    name: str
    description: str
    benchmarks: list[TaskBenchmark] = field(default_factory=list)
    total_time: float = 0.0
    success_rate: float = 0.0
    average_quality: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ComplexQueryBenchmark:
    """Benchmark for a complete complex query execution.

    Attributes:
        query: Original complex query
        plan_complexity: Planned complexity level
        num_tasks: Number of tasks in plan
        total_execution_time: Total time for all tasks
        planning_time: Time spent planning
        task_benchmarks: Individual task benchmarks
        overall_success: Whether query was successfully answered
        final_quality_score: Final answer quality
        needs_improvement: List of improvement areas
        timestamp: When benchmark was recorded
    """

    query: str
    plan_complexity: str
    num_tasks: int
    total_execution_time: float
    planning_time: float
    task_benchmarks: list[TaskBenchmark] = field(default_factory=list)
    overall_success: bool = True
    final_quality_score: float = 0.0
    needs_improvement: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class TaskBenchmarker:
    """Benchmarks complex task execution."""

    def __init__(self, storage_path: str | None = None):
        """Initialize benchmarker.

        Args:
            storage_path: Optional path to store benchmarks
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self.current_suite: BenchmarkSuite | None = None
        self.benchmark_history: list[ComplexQueryBenchmark] = []
        self.logger = logging.getLogger(__name__)

    def start_benchmark(
        self, query: str, plan_complexity: str, num_tasks: int
    ) -> ComplexQueryBenchmark:
        """Start benchmarking a complex query.

        Args:
            query: The query
            plan_complexity: Complexity level
            num_tasks: Number of tasks

        Returns:
            New ComplexQueryBenchmark
        """
        benchmark = ComplexQueryBenchmark(
            query=query,
            plan_complexity=plan_complexity,
            num_tasks=num_tasks,
            total_execution_time=0.0,
            planning_time=0.0,
        )

        self.logger.info(f"Started benchmark for: {query[:50]}...")
        return benchmark

    def record_task(
        self,
        benchmark: ComplexQueryBenchmark,
        task_id: str,
        task_description: str,
        execution_time: float,
        success: bool,
        tool_used: str = "",
        quality_score: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Record a task execution.

        Args:
            benchmark: The benchmark object
            task_id: Task identifier
            task_description: Task description
            execution_time: Time taken
            success: Whether successful
            tool_used: Tool used
            quality_score: Quality score
            **kwargs: Additional metadata
        """
        task_benchmark = TaskBenchmark(
            task_id=task_id,
            task_description=task_description,
            query=benchmark.query,
            execution_time=execution_time,
            success=success,
            tool_used=tool_used,
            quality_score=quality_score,
            tool_confidence=kwargs.get("tool_confidence", 0.0),
            retry_count=kwargs.get("retry_count", 0),
            error_message=kwargs.get("error_message", ""),
            result_metadata=kwargs.get("result_metadata", {}),
        )

        benchmark.task_benchmarks.append(task_benchmark)
        benchmark.total_execution_time += execution_time

        self.logger.debug(
            f"Recorded task {task_id}: {execution_time:.2f}s, "
            f"success={success}, quality={quality_score:.2f}"
        )

    def complete_benchmark(
        self, benchmark: ComplexQueryBenchmark, final_quality_score: float = 0.0
    ) -> None:
        """Complete and store a benchmark.

        Args:
            benchmark: The benchmark object
            final_quality_score: Final answer quality
        """
        # Calculate metrics
        successful_tasks = sum(1 for t in benchmark.task_benchmarks if t.success)
        benchmark.overall_success = successful_tasks == benchmark.num_tasks

        benchmark.final_quality_score = final_quality_score

        # Identify improvement areas
        benchmark.needs_improvement = self._identify_improvements(benchmark)

        # Store
        self.benchmark_history.append(benchmark)

        # Save to disk
        if self.storage_path:
            self._save_benchmark(benchmark)

        self.logger.info(
            f"Benchmark complete: {benchmark.total_execution_time:.2f}s, "
            f"success={benchmark.overall_success}, "
            f"quality={benchmark.final_quality_score:.2f}"
        )

    def get_statistics(self, recent_n: int | None = None) -> dict[str, Any]:
        """Get benchmark statistics.

        Args:
            recent_n: Optional limit to recent N benchmarks

        Returns:
            Statistics dictionary
        """
        benchmarks = self.benchmark_history
        if recent_n:
            benchmarks = benchmarks[-recent_n:]

        if not benchmarks:
            return {"error": "No benchmarks available"}

        total = len(benchmarks)
        successful = sum(1 for b in benchmarks if b.overall_success)

        avg_time = sum(b.total_execution_time for b in benchmarks) / total
        avg_quality = sum(b.final_quality_score for b in benchmarks) / total
        avg_tasks = sum(b.num_tasks for b in benchmarks) / total

        # Complexity breakdown
        complexity_counts: dict[str, int] = {}
        for b in benchmarks:
            complexity_counts[b.plan_complexity] = complexity_counts.get(b.plan_complexity, 0) + 1

        # Tool usage
        tool_counts: dict[str, int] = {}
        for b in benchmarks:
            for task in b.task_benchmarks:
                if task.tool_used:
                    tool_counts[task.tool_used] = tool_counts.get(task.tool_used, 0) + 1

        # Common improvement areas
        improvement_counts: dict[str, int] = {}
        for b in benchmarks:
            for improvement in b.needs_improvement:
                improvement_counts[improvement] = improvement_counts.get(improvement, 0) + 1

        return {
            "total_benchmarks": total,
            "success_rate": successful / total,
            "average_execution_time": avg_time,
            "average_quality_score": avg_quality,
            "average_tasks_per_query": avg_tasks,
            "complexity_distribution": complexity_counts,
            "tool_usage": dict(sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            "common_improvements": dict(
                sorted(improvement_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            ),
        }

    def get_slow_queries(self, threshold: float = 10.0, limit: int = 10) -> list[dict[str, Any]]:
        """Get queries that took longer than threshold.

        Args:
            threshold: Time threshold in seconds
            limit: Maximum number to return

        Returns:
            List of slow queries with details
        """
        slow = [
            {
                "query": b.query,
                "execution_time": b.total_execution_time,
                "num_tasks": b.num_tasks,
                "complexity": b.plan_complexity,
                "timestamp": b.timestamp.isoformat(),
            }
            for b in self.benchmark_history
            if b.total_execution_time > threshold
        ]

        def get_execution_time(item: dict[str, Any]) -> float:
            """Extract execution time for sorting."""
            time_val = item.get("execution_time", 0.0)
            return time_val if isinstance(time_val, (int, float)) else 0.0

        return sorted(slow, key=get_execution_time, reverse=True)[:limit]

    def get_low_quality_queries(
        self, threshold: float = 0.6, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Get queries with low quality scores.

        Args:
            threshold: Quality threshold
            limit: Maximum number to return

        Returns:
            List of low quality queries
        """
        low_quality = [
            {
                "query": b.query,
                "quality_score": b.final_quality_score,
                "success": b.overall_success,
                "improvements": b.needs_improvement,
                "timestamp": b.timestamp.isoformat(),
            }
            for b in self.benchmark_history
            if b.final_quality_score < threshold
        ]

        def get_quality_score(item: dict[str, Any]) -> float:
            """Extract quality score for sorting."""
            score = item.get("quality_score", 0.0)
            return score if isinstance(score, (int, float)) else 0.0

        return sorted(low_quality, key=get_quality_score)[:limit]

    def export_benchmarks(self, export_path: str) -> None:
        """Export benchmarks to JSON file.

        Args:
            export_path: Path to export file
        """
        import json

        export_file = Path(export_path)
        export_file.parent.mkdir(parents=True, exist_ok=True)

        data = []
        for benchmark in self.benchmark_history:
            entry = asdict(benchmark)
            # Convert datetime to string
            entry["timestamp"] = entry["timestamp"].isoformat()
            for task in entry["task_benchmarks"]:
                task["timestamp"] = task["timestamp"].isoformat()
            data.append(entry)

        with export_file.open("w") as f:
            json.dump(data, f, indent=2)

        self.logger.info(f"Exported {len(data)} benchmarks to {export_path}")

    def _identify_improvements(self, benchmark: ComplexQueryBenchmark) -> list[str]:
        """Identify areas for improvement.

        Args:
            benchmark: The benchmark

        Returns:
            List of improvement suggestions
        """
        improvements = []

        # Check execution time
        if benchmark.total_execution_time > 15:
            improvements.append("Execution time too long")

        # Check task success rate
        failed_tasks = sum(1 for t in benchmark.task_benchmarks if not t.success)
        if failed_tasks > 0:
            improvements.append(f"{failed_tasks} task(s) failed")

        # Check quality
        if benchmark.final_quality_score < 0.7:
            improvements.append("Answer quality below target")

        # Check for high retry counts
        max_retries = max((t.retry_count for t in benchmark.task_benchmarks), default=0)
        if max_retries > 2:
            improvements.append("Too many retries needed")

        # Check tool selection confidence
        low_confidence_tasks = sum(1 for t in benchmark.task_benchmarks if t.tool_confidence < 0.6)
        if low_confidence_tasks > benchmark.num_tasks * 0.3:
            improvements.append("Low tool selection confidence")

        return improvements

    def _save_benchmark(self, benchmark: ComplexQueryBenchmark) -> None:
        """Save a single benchmark to disk.

        Args:
            benchmark: The benchmark to save
        """
        if not self.storage_path:
            return

        try:
            import json

            self.storage_path.parent.mkdir(parents=True, exist_ok=True)

            # Append to file (one benchmark per line for easier parsing)
            entry = asdict(benchmark)
            entry["timestamp"] = entry["timestamp"].isoformat()
            for task in entry["task_benchmarks"]:
                task["timestamp"] = task["timestamp"].isoformat()

            with self.storage_path.open("a") as f:
                f.write(json.dumps(entry) + "\n")

        except Exception as e:
            self.logger.error(f"Failed to save benchmark: {e}")


def benchmark_decorator(benchmarker: TaskBenchmarker):
    """Decorator to automatically benchmark a function.

    Args:
        benchmarker: TaskBenchmarker instance

    Returns:
        Decorator function
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = False

            try:
                result = func(*args, **kwargs)
                success = True
                return result
            except Exception:
                raise
            finally:
                execution_time = time.time() - start_time

                logger.info(
                    f"Function {func.__name__} executed in {execution_time:.2f}s, success={success}"
                )

        return wrapper

    return decorator
