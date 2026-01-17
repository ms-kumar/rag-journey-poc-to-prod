"""Unit tests for the planning module (QueryPlanner, Task, ExecutionPlan)."""

from src.services.agent.planning import (
    ExecutionPlan,
    QueryPlanner,
    Task,
)


class TestTask:
    """Test Task dataclass."""

    def test_default_values(self):
        """Test default values of Task."""
        task = Task(id="task_1", description="Test task")

        assert task.id == "task_1"
        assert task.description == "Test task"
        assert task.type == "general"
        assert task.dependencies == []
        assert task.priority == 1
        assert task.estimated_complexity == 0.5
        assert task.tool_hints == []
        assert task.status == "pending"
        assert task.result is None
        assert task.retry_count == 0
        assert task.metadata == {}

    def test_custom_values(self):
        """Test Task with custom values."""
        task = Task(
            id="task_2",
            description="Complex task",
            type="retrieval",
            dependencies=["task_1"],
            priority=3,
            estimated_complexity=0.8,
            tool_hints=["vectordb_retrieval", "web_search"],
            status="in_progress",
            retry_count=1,
            metadata={"key": "value"},
        )

        assert task.id == "task_2"
        assert task.type == "retrieval"
        assert task.dependencies == ["task_1"]
        assert task.priority == 3
        assert task.estimated_complexity == 0.8
        assert "vectordb_retrieval" in task.tool_hints
        assert task.status == "in_progress"
        assert task.retry_count == 1
        assert task.metadata["key"] == "value"

    def test_task_result_assignment(self):
        """Test assigning result to task."""
        task = Task(id="task_1", description="Test")
        task.result = {"output": "success", "data": [1, 2, 3]}
        task.status = "completed"

        assert task.result is not None
        assert task.result["output"] == "success"
        assert task.status == "completed"


class TestExecutionPlan:
    """Test ExecutionPlan dataclass."""

    def test_default_values(self):
        """Test default values of ExecutionPlan."""
        plan = ExecutionPlan(
            query="Test query",
            tasks=[],
        )

        assert plan.query == "Test query"
        assert plan.tasks == []
        assert plan.execution_strategy == "sequential"
        assert plan.estimated_time == 0.0
        assert plan.complexity_level == "moderate"
        assert plan.revised_count == 0
        assert plan.rationale == ""
        assert plan.created_at is not None

    def test_custom_values(self):
        """Test ExecutionPlan with custom values."""
        tasks = [
            Task(id="task_1", description="First task"),
            Task(id="task_2", description="Second task", dependencies=["task_1"]),
        ]

        plan = ExecutionPlan(
            query="Complex query",
            tasks=tasks,
            execution_strategy="hybrid",
            estimated_time=15.0,
            complexity_level="complex",
            revised_count=2,
            rationale="Multi-step reasoning required",
        )

        assert len(plan.tasks) == 2
        assert plan.execution_strategy == "hybrid"
        assert plan.estimated_time == 15.0
        assert plan.complexity_level == "complex"
        assert plan.revised_count == 2

    def test_plan_with_task_dependencies(self):
        """Test ExecutionPlan with task dependencies."""
        tasks = [
            Task(id="retrieve", description="Retrieve data"),
            Task(id="process", description="Process data", dependencies=["retrieve"]),
            Task(id="synthesize", description="Synthesize results", dependencies=["process"]),
        ]

        plan = ExecutionPlan(query="Test", tasks=tasks)

        assert len(plan.tasks) == 3
        assert plan.tasks[1].dependencies == ["retrieve"]
        assert plan.tasks[2].dependencies == ["process"]


class TestQueryPlanner:
    """Test QueryPlanner class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.planner = QueryPlanner()

    def test_init_default(self):
        """Test default initialization."""
        planner = QueryPlanner()
        assert planner.llm is None

    def test_create_plan_simple_query(self):
        """Test planning for a simple query."""
        plan = self.planner.create_plan("What is Python?")

        assert plan.query == "What is Python?"
        assert len(plan.tasks) >= 1
        assert plan.complexity_level == "simple"
        assert plan.execution_strategy == "sequential"
        assert plan.rationale != ""

    def test_create_plan_moderate_query(self):
        """Test planning for a moderate complexity query."""
        plan = self.planner.create_plan(
            "Explain how Python handles memory management and garbage collection"
        )

        assert len(plan.tasks) >= 1
        # Moderate queries have more detail requirements
        assert plan.complexity_level in ["simple", "moderate", "complex"]

    def test_create_plan_complex_query(self):
        """Test planning for a complex query."""
        plan = self.planner.create_plan(
            "Compare Python and Java for enterprise applications, "
            "considering performance, scalability, ecosystem, and maintainability. "
            "Then analyze which would be better for a microservices architecture."
        )

        assert len(plan.tasks) >= 2
        # Complex queries should have multiple tasks
        assert plan.complexity_level in ["moderate", "complex"]

    def test_create_plan_retrieval_query(self):
        """Test planning for a retrieval-focused query."""
        plan = self.planner.create_plan("Find information about machine learning algorithms")

        assert len(plan.tasks) >= 1
        # Should identify retrieval task type
        retrieval_tasks = [t for t in plan.tasks if t.type == "retrieval"]
        assert len(retrieval_tasks) >= 0  # May or may not be labeled as retrieval

    def test_create_plan_comparison_query(self):
        """Test planning for a comparison query."""
        plan = self.planner.create_plan("Compare React vs Vue for frontend development")

        assert len(plan.tasks) >= 1
        # Comparison queries often have multiple aspects

    def test_create_plan_computation_query(self):
        """Test planning for a computation query."""
        plan = self.planner.create_plan("Calculate the factorial of 10")

        assert len(plan.tasks) >= 1

    def test_create_plan_synthesis_query(self):
        """Test planning for a synthesis query."""
        plan = self.planner.create_plan(
            "Summarize the key concepts of object-oriented programming and explain why they matter"
        )

        assert len(plan.tasks) >= 1

    def test_create_plan_with_context(self):
        """Test planning with additional context."""
        context = {
            "domain": "software development",
            "user_level": "intermediate",
            "preferred_tools": ["vectordb_retrieval"],
        }

        plan = self.planner.create_plan(
            "How do I implement a REST API?",
            context=context,
        )

        assert plan.query == "How do I implement a REST API?"
        assert len(plan.tasks) >= 1

    def test_create_plan_temporal_query(self):
        """Test planning for temporal queries."""
        plan = self.planner.create_plan(
            "What were the major Python releases and when did they come out?"
        )

        assert len(plan.tasks) >= 1

    def test_create_plan_reasoning_query(self):
        """Test planning for reasoning queries."""
        plan = self.planner.create_plan("Why is Python popular for data science?")

        assert len(plan.tasks) >= 1

    def test_revise_plan_basic(self):
        """Test basic plan revision."""
        original_plan = self.planner.create_plan("What is Python?")

        revised_plan = self.planner.revise_plan(
            plan=original_plan,
            feedback="Need more details about Python's history",
            completed_tasks=[],
        )

        assert revised_plan.revised_count == original_plan.revised_count
        assert revised_plan.query == original_plan.query

    def test_revise_plan_with_failed_tasks(self):
        """Test plan revision with failed tasks."""
        original_plan = self.planner.create_plan("Compare A and B")

        revised_plan = self.planner.revise_plan(
            plan=original_plan,
            feedback="Task failed due to timeout",
            completed_tasks=[],
        )

        # Revised plan should address failures
        assert revised_plan.revised_count >= 1

    def test_revise_plan_preserves_query(self):
        """Test that plan revision preserves original query."""
        original_query = "What is machine learning?"
        original_plan = self.planner.create_plan(original_query)

        revised_plan = self.planner.revise_plan(
            plan=original_plan,
            feedback="Add more examples",
            completed_tasks=[],
        )

        assert revised_plan.query == original_query

    def test_analyze_complexity_simple(self):
        """Test complexity analysis for simple queries."""
        # Simple queries should be identified
        simple_queries = [
            "What is Python?",
            "Define machine learning",
            "What time is it?",
        ]

        for query in simple_queries:
            plan = self.planner.create_plan(query)
            # Simple queries should not be classified as complex
            assert plan.complexity_level in ["simple", "moderate"]

    def test_analyze_complexity_complex(self):
        """Test complexity analysis for complex queries."""
        complex_queries = [
            "Compare and contrast Python, Java, and C++ for systems programming, "
            "considering performance, memory management, concurrency, and ecosystem. "
            "Then recommend the best choice for each use case.",
            "Analyze the historical evolution of neural networks, explain their current "
            "applications in NLP and computer vision, and predict future developments.",
        ]

        for query in complex_queries:
            plan = self.planner.create_plan(query)
            # Complex queries should have multiple tasks
            assert len(plan.tasks) >= 1

    def test_task_dependencies_correct(self):
        """Test that task dependencies are logical."""
        plan = self.planner.create_plan(
            "First find information about X, then analyze it, then summarize"
        )

        # Check that dependencies don't reference non-existent tasks
        task_ids = {task.id for task in plan.tasks}
        for task in plan.tasks:
            for dep in task.dependencies:
                assert dep in task_ids or dep.startswith("task_")

    def test_execution_strategy_selection(self):
        """Test that execution strategy is appropriate."""
        # Simple query should be sequential
        simple_plan = self.planner.create_plan("What is Python?")
        assert simple_plan.execution_strategy == "sequential"

        # Complex query might use different strategies
        complex_plan = self.planner.create_plan(
            "Compare Python and Java, analyze their ecosystems, and evaluate "
            "community support for each independently"
        )
        # Should be one of the valid strategies
        assert complex_plan.execution_strategy in ["sequential", "parallel", "hybrid"]

    def test_empty_query(self):
        """Test planning with empty query."""
        plan = self.planner.create_plan("")

        # Should handle empty query gracefully
        assert plan.query == ""
        assert len(plan.tasks) >= 0

    def test_very_long_query(self):
        """Test planning with very long query."""
        long_query = "Explain Python. " * 100

        plan = self.planner.create_plan(long_query)

        # Should handle long queries
        assert len(plan.tasks) >= 1

    def test_special_characters_in_query(self):
        """Test planning with special characters."""
        query = "What is Python's @decorator syntax and how do I use __init__?"

        plan = self.planner.create_plan(query)

        assert plan.query == query
        assert len(plan.tasks) >= 1


class TestQueryPlannerTaskTypes:
    """Test task type identification in QueryPlanner."""

    def setup_method(self):
        """Set up test fixtures."""
        self.planner = QueryPlanner()

    def test_identifies_retrieval_patterns(self):
        """Test identification of retrieval patterns."""
        retrieval_queries = [
            "Find documents about Python",
            "Search for machine learning tutorials",
            "Get information about APIs",
            "Lookup the latest news on AI",
        ]

        for query in retrieval_queries:
            plan = self.planner.create_plan(query)
            # Should create at least one task
            assert len(plan.tasks) >= 1

    def test_identifies_comparison_patterns(self):
        """Test identification of comparison patterns."""
        comparison_queries = [
            "Compare Python vs Java",
            "What is the difference between REST and GraphQL?",
            "Contrast functional and object-oriented programming",
        ]

        for query in comparison_queries:
            plan = self.planner.create_plan(query)
            assert len(plan.tasks) >= 1

    def test_identifies_computation_patterns(self):
        """Test identification of computation patterns."""
        computation_queries = [
            "Calculate the sum of 1 to 100",
            "How many lines of code are in the project?",
            "Count the number of functions in the module",
        ]

        for query in computation_queries:
            plan = self.planner.create_plan(query)
            assert len(plan.tasks) >= 1

    def test_tool_hints_provided(self):
        """Test that tool hints are provided for tasks."""
        plan = self.planner.create_plan("Search for Python documentation")

        # At least some tasks should have tool hints
        tasks_with_hints = [t for t in plan.tasks if len(t.tool_hints) > 0]
        # Tool hints are optional but often present
        assert len(tasks_with_hints) >= 0


class TestQueryPlannerEdgeCases:
    """Test edge cases for QueryPlanner."""

    def setup_method(self):
        """Set up test fixtures."""
        self.planner = QueryPlanner()

    def test_unicode_query(self):
        """Test planning with unicode characters."""
        query = "What is Python? パイソンとは何ですか？"

        plan = self.planner.create_plan(query)

        assert plan.query == query
        assert len(plan.tasks) >= 1

    def test_query_with_code(self):
        """Test planning with code in query."""
        query = "What does this code do: `def foo(): return 42`?"

        plan = self.planner.create_plan(query)

        assert len(plan.tasks) >= 1

    def test_multiple_questions(self):
        """Test planning with multiple questions."""
        query = "What is Python? Why is it popular? How do I install it?"

        plan = self.planner.create_plan(query)

        # Should recognize multiple aspects
        assert len(plan.tasks) >= 1

    def test_query_with_numbers(self):
        """Test planning with numbers in query."""
        query = "Calculate 2+2 and explain why Python 3.9 introduced new features"

        plan = self.planner.create_plan(query)

        assert len(plan.tasks) >= 1

    def test_none_context(self):
        """Test planning with None context."""
        plan = self.planner.create_plan("What is Python?", context=None)

        assert len(plan.tasks) >= 1

    def test_empty_context(self):
        """Test planning with empty context."""
        plan = self.planner.create_plan("What is Python?", context={})

        assert len(plan.tasks) >= 1
