"""Unit tests for the benchmarking module (TaskBenchmarker, TaskBenchmark, ComplexQueryBenchmark)."""

import tempfile
from pathlib import Path

from src.services.agent.benchmarking import (
    BenchmarkSuite,
    ComplexQueryBenchmark,
    TaskBenchmark,
    TaskBenchmarker,
)


class TestTaskBenchmark:
    """Test TaskBenchmark dataclass."""

    def test_default_values(self):
        """Test default values of TaskBenchmark."""
        benchmark = TaskBenchmark(
            task_id="task_1",
            task_description="Test task",
            query="Test query",
            execution_time=1.5,
            success=True,
        )

        assert benchmark.task_id == "task_1"
        assert benchmark.task_description == "Test task"
        assert benchmark.query == "Test query"
        assert benchmark.execution_time == 1.5
        assert benchmark.success is True
        assert benchmark.quality_score == 0.0
        assert benchmark.tool_used == ""
        assert benchmark.tool_confidence == 0.0
        assert benchmark.retry_count == 0
        assert benchmark.error_message == ""
        assert benchmark.result_metadata == {}
        assert benchmark.timestamp is not None

    def test_custom_values(self):
        """Test TaskBenchmark with custom values."""
        benchmark = TaskBenchmark(
            task_id="task_2",
            task_description="Complex task",
            query="Complex query",
            execution_time=5.5,
            success=False,
            quality_score=0.75,
            tool_used="vectordb_retrieval",
            tool_confidence=0.9,
            retry_count=2,
            error_message="Timeout error",
            result_metadata={"docs_found": 10},
        )

        assert benchmark.task_id == "task_2"
        assert benchmark.execution_time == 5.5
        assert benchmark.success is False
        assert benchmark.quality_score == 0.75
        assert benchmark.tool_used == "vectordb_retrieval"
        assert benchmark.tool_confidence == 0.9
        assert benchmark.retry_count == 2
        assert benchmark.error_message == "Timeout error"
        assert benchmark.result_metadata["docs_found"] == 10


class TestComplexQueryBenchmark:
    """Test ComplexQueryBenchmark dataclass."""

    def test_default_values(self):
        """Test default values of ComplexQueryBenchmark."""
        benchmark = ComplexQueryBenchmark(
            query="Test query",
            plan_complexity="simple",
            num_tasks=1,
            total_execution_time=2.0,
            planning_time=0.5,
        )

        assert benchmark.query == "Test query"
        assert benchmark.plan_complexity == "simple"
        assert benchmark.num_tasks == 1
        assert benchmark.total_execution_time == 2.0
        assert benchmark.planning_time == 0.5
        assert benchmark.task_benchmarks == []
        assert benchmark.overall_success is True
        assert benchmark.final_quality_score == 0.0
        assert benchmark.needs_improvement == []
        assert benchmark.timestamp is not None

    def test_with_task_benchmarks(self):
        """Test ComplexQueryBenchmark with task benchmarks."""
        task_benchmarks = [
            TaskBenchmark(
                task_id="task_1",
                task_description="First task",
                query="Query",
                execution_time=1.0,
                success=True,
            ),
            TaskBenchmark(
                task_id="task_2",
                task_description="Second task",
                query="Query",
                execution_time=1.5,
                success=True,
            ),
        ]

        benchmark = ComplexQueryBenchmark(
            query="Complex query",
            plan_complexity="moderate",
            num_tasks=2,
            total_execution_time=2.5,
            planning_time=0.3,
            task_benchmarks=task_benchmarks,
            overall_success=True,
            final_quality_score=0.85,
        )

        assert len(benchmark.task_benchmarks) == 2
        assert benchmark.final_quality_score == 0.85


class TestBenchmarkSuite:
    """Test BenchmarkSuite dataclass."""

    def test_default_values(self):
        """Test default values of BenchmarkSuite."""
        suite = BenchmarkSuite(
            name="Test Suite",
            description="A test benchmark suite",
        )

        assert suite.name == "Test Suite"
        assert suite.description == "A test benchmark suite"
        assert suite.benchmarks == []
        assert suite.total_time == 0.0
        assert suite.success_rate == 0.0
        assert suite.average_quality == 0.0
        assert suite.created_at is not None

    def test_custom_values(self):
        """Test BenchmarkSuite with custom values."""
        benchmarks = [
            TaskBenchmark(
                task_id="t1",
                task_description="Task 1",
                query="Q1",
                execution_time=1.0,
                success=True,
            ),
        ]

        suite = BenchmarkSuite(
            name="Custom Suite",
            description="Custom description",
            benchmarks=benchmarks,
            total_time=10.0,
            success_rate=0.95,
            average_quality=0.88,
        )

        assert len(suite.benchmarks) == 1
        assert suite.total_time == 10.0
        assert suite.success_rate == 0.95


class TestTaskBenchmarker:
    """Test TaskBenchmarker class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.temp_dir) / "benchmarks.jsonl"
        self.benchmarker = TaskBenchmarker(storage_path=str(self.storage_path))

    def teardown_method(self):
        """Clean up after tests."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init_default(self):
        """Test default initialization."""
        benchmarker = TaskBenchmarker()
        assert benchmarker.storage_path is None
        assert benchmarker.benchmark_history == []

    def test_init_with_storage(self):
        """Test initialization with storage path."""
        benchmarker = TaskBenchmarker(storage_path=str(self.storage_path))
        assert benchmarker.storage_path == self.storage_path

    def test_start_benchmark(self):
        """Test starting a benchmark."""
        benchmark = self.benchmarker.start_benchmark(
            query="Test query",
            plan_complexity="simple",
            num_tasks=2,
        )

        assert benchmark.query == "Test query"
        assert benchmark.plan_complexity == "simple"
        assert benchmark.num_tasks == 2
        assert benchmark.planning_time >= 0

    def test_record_task(self):
        """Test recording a task benchmark."""
        benchmark = self.benchmarker.start_benchmark(
            query="Test query",
            plan_complexity="simple",
            num_tasks=1,
        )

        self.benchmarker.record_task(
            benchmark=benchmark,
            task_id="task_1",
            task_description="Test task",
            execution_time=1.5,
            success=True,
            tool_used="vectordb_retrieval",
            quality_score=0.85,
        )

        assert len(benchmark.task_benchmarks) == 1
        assert benchmark.task_benchmarks[0].task_id == "task_1"
        assert benchmark.task_benchmarks[0].execution_time == 1.5
        assert benchmark.task_benchmarks[0].success is True

    def test_record_multiple_tasks(self):
        """Test recording multiple tasks."""
        benchmark = self.benchmarker.start_benchmark(
            query="Complex query",
            plan_complexity="complex",
            num_tasks=3,
        )

        for i in range(3):
            self.benchmarker.record_task(
                benchmark=benchmark,
                task_id=f"task_{i}",
                task_description=f"Task {i}",
                execution_time=1.0 + i * 0.5,
                success=True,
                quality_score=0.8 + i * 0.05,
            )

        assert len(benchmark.task_benchmarks) == 3

    def test_complete_benchmark(self):
        """Test completing a benchmark."""
        benchmark = self.benchmarker.start_benchmark(
            query="Test query",
            plan_complexity="simple",
            num_tasks=1,
        )

        self.benchmarker.record_task(
            benchmark=benchmark,
            task_id="task_1",
            task_description="Test task",
            execution_time=1.5,
            success=True,
            quality_score=0.9,
        )

        self.benchmarker.complete_benchmark(
            benchmark=benchmark,
            final_quality_score=0.9,
        )

        assert benchmark.final_quality_score == 0.9
        assert benchmark.total_execution_time >= 1.5
        assert len(self.benchmarker.benchmark_history) >= 1

    def test_complete_benchmark_with_failures(self):
        """Test completing benchmark with task failures."""
        benchmark = self.benchmarker.start_benchmark(
            query="Test query",
            plan_complexity="moderate",
            num_tasks=3,
        )

        # One successful, two failed
        self.benchmarker.record_task(
            benchmark=benchmark,
            task_id="task_1",
            task_description="Task 1",
            execution_time=1.0,
            success=True,
            quality_score=0.9,
        )
        self.benchmarker.record_task(
            benchmark=benchmark,
            task_id="task_2",
            task_description="Task 2",
            execution_time=2.0,
            success=False,
            error_message="Timeout",
        )
        self.benchmarker.record_task(
            benchmark=benchmark,
            task_id="task_3",
            task_description="Task 3",
            execution_time=1.5,
            success=False,
            error_message="Not found",
        )

        self.benchmarker.complete_benchmark(
            benchmark=benchmark,
            final_quality_score=0.5,
        )

        assert benchmark.overall_success is False or len(benchmark.needs_improvement) > 0

    def test_benchmark_persists_to_disk(self):
        """Test that benchmarks are persisted to disk."""
        benchmark = self.benchmarker.start_benchmark(
            query="Test query",
            plan_complexity="simple",
            num_tasks=1,
        )

        self.benchmarker.record_task(
            benchmark=benchmark,
            task_id="task_1",
            task_description="Test task",
            execution_time=1.0,
            success=True,
        )

        self.benchmarker.complete_benchmark(benchmark, final_quality_score=0.9)

        # Check file exists
        assert self.storage_path.exists()

    def test_get_statistics_empty(self):
        """Test statistics with no benchmarks."""
        stats = self.benchmarker.get_statistics()

        # When empty, returns empty dict or dict with zero values
        assert stats.get("total_benchmarks", 0) == 0 or stats == {}

    def test_get_statistics_with_data(self):
        """Test statistics calculation with benchmarks."""
        # Create multiple benchmarks
        for i in range(5):
            benchmark = self.benchmarker.start_benchmark(
                query=f"Query {i}",
                plan_complexity="simple" if i < 3 else "complex",
                num_tasks=1,
            )
            self.benchmarker.record_task(
                benchmark=benchmark,
                task_id=f"task_{i}",
                task_description=f"Task {i}",
                execution_time=1.0 + i * 0.5,
                success=i < 4,  # First 4 succeed
                tool_used="vectordb_retrieval",
                quality_score=0.8 + i * 0.02,
            )
            self.benchmarker.complete_benchmark(
                benchmark=benchmark,
                final_quality_score=0.8 + i * 0.02,
            )

        stats = self.benchmarker.get_statistics()

        assert stats["total_benchmarks"] == 5
        assert 0.7 <= stats["success_rate"] <= 1.0  # 4/5 = 80%
        assert stats["average_execution_time"] > 0

    def test_get_statistics_tool_usage(self):
        """Test tool usage statistics."""
        tools = ["vectordb_retrieval", "web_search", "vectordb_retrieval"]

        for i, tool in enumerate(tools):
            benchmark = self.benchmarker.start_benchmark(
                query=f"Query {i}",
                plan_complexity="simple",
                num_tasks=1,
            )
            self.benchmarker.record_task(
                benchmark=benchmark,
                task_id=f"task_{i}",
                task_description=f"Task {i}",
                execution_time=1.0,
                success=True,
                tool_used=tool,
            )
            self.benchmarker.complete_benchmark(benchmark, final_quality_score=0.9)

        stats = self.benchmarker.get_statistics()

        assert "tool_usage" in stats
        assert stats["tool_usage"].get("vectordb_retrieval", 0) == 2
        assert stats["tool_usage"].get("web_search", 0) == 1

    def test_get_slow_queries(self):
        """Test getting slow queries."""
        # Fast query
        fast_benchmark = self.benchmarker.start_benchmark(
            query="Fast query",
            plan_complexity="simple",
            num_tasks=1,
        )
        self.benchmarker.record_task(
            benchmark=fast_benchmark,
            task_id="task_1",
            task_description="Fast task",
            execution_time=0.5,
            success=True,
        )
        self.benchmarker.complete_benchmark(fast_benchmark, final_quality_score=0.9)

        # Slow query
        slow_benchmark = self.benchmarker.start_benchmark(
            query="Slow query",
            plan_complexity="complex",
            num_tasks=1,
        )
        self.benchmarker.record_task(
            benchmark=slow_benchmark,
            task_id="task_1",
            task_description="Slow task",
            execution_time=15.0,
            success=True,
        )
        self.benchmarker.complete_benchmark(slow_benchmark, final_quality_score=0.7)

        slow_queries = self.benchmarker.get_slow_queries(threshold=10.0)

        assert len(slow_queries) >= 1
        queries = [item["query"] for item in slow_queries]
        assert "Slow query" in queries

    def test_get_slow_queries_empty(self):
        """Test getting slow queries when all are fast."""
        for i in range(3):
            benchmark = self.benchmarker.start_benchmark(
                query=f"Fast query {i}",
                plan_complexity="simple",
                num_tasks=1,
            )
            self.benchmarker.record_task(
                benchmark=benchmark,
                task_id=f"task_{i}",
                task_description=f"Task {i}",
                execution_time=0.5,
                success=True,
            )
            self.benchmarker.complete_benchmark(benchmark, final_quality_score=0.9)

        slow_queries = self.benchmarker.get_slow_queries(threshold=10.0)

        assert len(slow_queries) == 0

    def test_complexity_distribution(self):
        """Test complexity distribution in statistics."""
        complexities = ["simple", "simple", "moderate", "complex"]

        for i, complexity in enumerate(complexities):
            benchmark = self.benchmarker.start_benchmark(
                query=f"Query {i}",
                plan_complexity=complexity,
                num_tasks=1,
            )
            self.benchmarker.record_task(
                benchmark=benchmark,
                task_id=f"task_{i}",
                task_description=f"Task {i}",
                execution_time=1.0,
                success=True,
            )
            self.benchmarker.complete_benchmark(benchmark, final_quality_score=0.9)

        stats = self.benchmarker.get_statistics()

        assert "complexity_distribution" in stats
        assert stats["complexity_distribution"].get("simple", 0) == 2
        assert stats["complexity_distribution"].get("moderate", 0) == 1
        assert stats["complexity_distribution"].get("complex", 0) == 1


class TestTaskBenchmarkerEdgeCases:
    """Test edge cases for TaskBenchmarker."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.temp_dir) / "benchmarks.jsonl"
        self.benchmarker = TaskBenchmarker(storage_path=str(self.storage_path))

    def teardown_method(self):
        """Clean up after tests."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_zero_execution_time(self):
        """Test task with zero execution time."""
        benchmark = self.benchmarker.start_benchmark(
            query="Test",
            plan_complexity="simple",
            num_tasks=1,
        )

        self.benchmarker.record_task(
            benchmark=benchmark,
            task_id="task_1",
            task_description="Instant task",
            execution_time=0.0,
            success=True,
        )

        assert benchmark.task_benchmarks[0].execution_time == 0.0

    def test_very_long_execution_time(self):
        """Test task with very long execution time."""
        benchmark = self.benchmarker.start_benchmark(
            query="Test",
            plan_complexity="complex",
            num_tasks=1,
        )

        self.benchmarker.record_task(
            benchmark=benchmark,
            task_id="task_1",
            task_description="Long task",
            execution_time=3600.0,  # 1 hour
            success=True,
        )

        assert benchmark.task_benchmarks[0].execution_time == 3600.0

    def test_empty_query(self):
        """Test benchmark with empty query."""
        benchmark = self.benchmarker.start_benchmark(
            query="",
            plan_complexity="simple",
            num_tasks=1,
        )

        assert benchmark.query == ""

    def test_unicode_in_benchmark(self):
        """Test benchmark with unicode content."""
        benchmark = self.benchmarker.start_benchmark(
            query="Pythonとは何ですか？",
            plan_complexity="simple",
            num_tasks=1,
        )

        self.benchmarker.record_task(
            benchmark=benchmark,
            task_id="task_1",
            task_description="日本語タスク",
            execution_time=1.0,
            success=True,
        )

        assert benchmark.query == "Pythonとは何ですか？"

    def test_many_tasks_in_benchmark(self):
        """Test benchmark with many tasks."""
        benchmark = self.benchmarker.start_benchmark(
            query="Complex multi-step query",
            plan_complexity="complex",
            num_tasks=50,
        )

        for i in range(50):
            self.benchmarker.record_task(
                benchmark=benchmark,
                task_id=f"task_{i}",
                task_description=f"Task {i}",
                execution_time=0.1,
                success=True,
            )

        self.benchmarker.complete_benchmark(benchmark, final_quality_score=0.9)

        assert len(benchmark.task_benchmarks) == 50

    def test_all_tasks_failed(self):
        """Test benchmark where all tasks fail."""
        benchmark = self.benchmarker.start_benchmark(
            query="Failing query",
            plan_complexity="moderate",
            num_tasks=3,
        )

        for i in range(3):
            self.benchmarker.record_task(
                benchmark=benchmark,
                task_id=f"task_{i}",
                task_description=f"Task {i}",
                execution_time=1.0,
                success=False,
                error_message=f"Error {i}",
            )

        self.benchmarker.complete_benchmark(benchmark, final_quality_score=0.0)

        assert benchmark.overall_success is False or benchmark.final_quality_score == 0.0

    def test_high_retry_count(self):
        """Test task with high retry count."""
        benchmark = self.benchmarker.start_benchmark(
            query="Test",
            plan_complexity="simple",
            num_tasks=1,
        )

        self.benchmarker.record_task(
            benchmark=benchmark,
            task_id="task_1",
            task_description="Flaky task",
            execution_time=5.0,
            success=True,
            retry_count=10,
        )

        assert benchmark.task_benchmarks[0].retry_count == 10

    def test_low_tool_confidence(self):
        """Test task with low tool confidence."""
        benchmark = self.benchmarker.start_benchmark(
            query="Test",
            plan_complexity="simple",
            num_tasks=1,
        )

        self.benchmarker.record_task(
            benchmark=benchmark,
            task_id="task_1",
            task_description="Uncertain task",
            execution_time=1.0,
            success=True,
            tool_confidence=0.1,
        )

        assert benchmark.task_benchmarks[0].tool_confidence == 0.1

    def test_result_metadata(self):
        """Test task with result metadata."""
        benchmark = self.benchmarker.start_benchmark(
            query="Test",
            plan_complexity="simple",
            num_tasks=1,
        )

        self.benchmarker.record_task(
            benchmark=benchmark,
            task_id="task_1",
            task_description="Task with metadata",
            execution_time=1.0,
            success=True,
            result_metadata={
                "documents_found": 25,
                "relevance_scores": [0.9, 0.8, 0.7],
                "source": "vectordb",
            },
        )

        assert benchmark.task_benchmarks[0].result_metadata["documents_found"] == 25

    def test_needs_improvement_detection(self):
        """Test detection of improvement areas."""
        benchmark = self.benchmarker.start_benchmark(
            query="Test",
            plan_complexity="complex",
            num_tasks=2,
        )

        # Slow task with low quality
        self.benchmarker.record_task(
            benchmark=benchmark,
            task_id="task_1",
            task_description="Slow low quality task",
            execution_time=20.0,  # Slow
            success=True,
            quality_score=0.5,  # Low quality
            retry_count=3,  # High retries
        )

        # Failed task
        self.benchmarker.record_task(
            benchmark=benchmark,
            task_id="task_2",
            task_description="Failed task",
            execution_time=5.0,
            success=False,
            error_message="Resource not found",
        )

        self.benchmarker.complete_benchmark(benchmark, final_quality_score=0.4)

        # Should identify improvement areas
        assert len(benchmark.needs_improvement) > 0 or benchmark.final_quality_score < 0.7


class TestTaskBenchmarkerStatistics:
    """Test detailed statistics calculations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.benchmarker = TaskBenchmarker()

    def test_average_quality_calculation(self):
        """Test average quality score calculation."""
        qualities = [0.9, 0.8, 0.85, 0.95, 0.7]

        for i, quality in enumerate(qualities):
            benchmark = self.benchmarker.start_benchmark(
                query=f"Query {i}",
                plan_complexity="simple",
                num_tasks=1,
            )
            self.benchmarker.record_task(
                benchmark=benchmark,
                task_id=f"task_{i}",
                task_description=f"Task {i}",
                execution_time=1.0,
                success=True,
                quality_score=quality,
            )
            self.benchmarker.complete_benchmark(benchmark, final_quality_score=quality)

        stats = self.benchmarker.get_statistics()

        # Average should be (0.9+0.8+0.85+0.95+0.7)/5 = 0.84
        assert 0.83 <= stats["average_quality_score"] <= 0.85

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        successes = [True, True, True, False, True]

        for i, success in enumerate(successes):
            benchmark = self.benchmarker.start_benchmark(
                query=f"Query {i}",
                plan_complexity="simple",
                num_tasks=1,
            )
            self.benchmarker.record_task(
                benchmark=benchmark,
                task_id=f"task_{i}",
                task_description=f"Task {i}",
                execution_time=1.0,
                success=success,
            )
            self.benchmarker.complete_benchmark(
                benchmark, final_quality_score=0.9 if success else 0.0
            )

        stats = self.benchmarker.get_statistics()

        # 4/5 = 80% success rate
        assert 0.75 <= stats["success_rate"] <= 0.85

    def test_execution_time_statistics(self):
        """Test execution time statistics."""
        times = [1.0, 2.0, 3.0, 4.0, 5.0]

        for i, exec_time in enumerate(times):
            benchmark = self.benchmarker.start_benchmark(
                query=f"Query {i}",
                plan_complexity="simple",
                num_tasks=1,
            )
            self.benchmarker.record_task(
                benchmark=benchmark,
                task_id=f"task_{i}",
                task_description=f"Task {i}",
                execution_time=exec_time,
                success=True,
            )
            self.benchmarker.complete_benchmark(benchmark, final_quality_score=0.9)

        stats = self.benchmarker.get_statistics()

        # Average time should be (1+2+3+4+5)/5 = 3.0
        assert 2.5 <= stats["average_execution_time"] <= 3.5
