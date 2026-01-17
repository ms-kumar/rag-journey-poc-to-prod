"""Advanced planning module for complex query decomposition."""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """A single task in a plan.

    Attributes:
        id: Unique task identifier
        description: Task description
        type: Task type (retrieval, computation, synthesis, etc.)
        dependencies: IDs of tasks that must complete first
        priority: Priority level (higher = more important)
        estimated_complexity: Estimated complexity (0.0 to 1.0)
        tool_hints: Suggested tools for this task
        status: Current status (pending, in_progress, completed, failed)
        result: Task result once completed
        retry_count: Number of retries attempted
        metadata: Additional task metadata
    """

    id: str
    description: str
    type: str = "general"
    dependencies: list[str] = field(default_factory=list)
    priority: int = 1
    estimated_complexity: float = 0.5
    tool_hints: list[str] = field(default_factory=list)
    status: str = "pending"
    result: dict[str, Any] | None = None
    retry_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionPlan:
    """A complete execution plan for a query.

    Attributes:
        query: Original query
        tasks: List of tasks to execute
        execution_strategy: Strategy (sequential, parallel, hybrid)
        estimated_time: Estimated execution time
        complexity_level: Overall complexity (simple, moderate, complex)
        created_at: Plan creation timestamp
        revised_count: Number of times plan was revised
        rationale: Explanation of the planning decision
    """

    query: str
    tasks: list[Task]
    execution_strategy: str = "sequential"
    estimated_time: float = 0.0
    complexity_level: str = "moderate"
    created_at: datetime = field(default_factory=datetime.utcnow)
    revised_count: int = 0
    rationale: str = ""


class QueryPlanner:
    """Advanced query planner with decomposition and strategy."""

    # Task type patterns
    TASK_PATTERNS = {
        "retrieval": [
            r"find|search|lookup|get|retrieve|what is|tell me about",
            r"information about|details on",
        ],
        "comparison": [r"compare|contrast|difference|versus|vs\.|better than"],
        "computation": [r"calculate|compute|how many|count|sum|average"],
        "synthesis": [
            r"summarize|explain|describe|analyze|synthesize",
            r"why|how does|reasoning",
        ],
        "reasoning": [r"why|because|reason|cause|effect|implication"],
        "temporal": [r"when|timeline|history|chronology|before|after"],
        "spatial": [r"where|location|place|geographic"],
    }

    def __init__(self, llm: Any | None = None):
        """Initialize query planner.

        Args:
            llm: Optional LLM for advanced planning
        """
        self.llm = llm
        self.logger = logging.getLogger(__name__)

    def create_plan(self, query: str, context: dict[str, Any] | None = None) -> ExecutionPlan:
        """Create an execution plan for a query.

        Args:
            query: User query
            context: Optional context (previous results, user prefs, etc.)

        Returns:
            ExecutionPlan with tasks and strategy
        """
        self.logger.info(f"Creating plan for query: {query[:100]}")

        # Analyze query complexity
        complexity = self._analyze_complexity(query)

        # Decompose into tasks
        tasks = self._decompose_query(query, complexity)

        # Determine execution strategy
        strategy = self._determine_strategy(tasks)

        # Estimate time
        estimated_time = sum(t.estimated_complexity * 2.0 for t in tasks)

        # Generate rationale
        rationale = self._generate_rationale(query, tasks, strategy, complexity)

        plan = ExecutionPlan(
            query=query,
            tasks=tasks,
            execution_strategy=strategy,
            estimated_time=estimated_time,
            complexity_level=complexity,
            rationale=rationale,
        )

        self.logger.info(
            f"Plan created: {len(tasks)} tasks, strategy={strategy}, complexity={complexity}"
        )

        return plan

    def revise_plan(
        self,
        plan: ExecutionPlan,
        feedback: str,
        completed_tasks: list[str] | None = None,
    ) -> ExecutionPlan:
        """Revise a plan based on feedback.

        Args:
            plan: Current plan
            feedback: Feedback about what went wrong
            completed_tasks: IDs of tasks already completed

        Returns:
            Revised ExecutionPlan
        """
        self.logger.info(f"Revising plan: {feedback}")

        completed_tasks = completed_tasks or []

        # Keep completed tasks
        revised_tasks = [t for t in plan.tasks if t.id in completed_tasks]

        # Analyze failure
        failure_type = self._analyze_failure(feedback)

        # Add remediation tasks
        new_tasks = self._create_remediation_tasks(plan.query, failure_type, len(revised_tasks))

        revised_tasks.extend(new_tasks)

        # Update plan
        plan.tasks = revised_tasks
        plan.revised_count += 1
        plan.rationale += f"\n\nRevision {plan.revised_count}: {feedback}"

        self.logger.info(f"Plan revised: {len(revised_tasks)} total tasks")

        return plan

    def _analyze_complexity(self, query: str) -> str:
        """Analyze query complexity.

        Args:
            query: Query text

        Returns:
            Complexity level: simple, moderate, or complex
        """
        # Simple heuristics
        query_length = len(query.split())

        # Check for compound questions
        compound_indicators = ["and", "also", "additionally", "furthermore", "moreover"]
        has_compound = any(ind in query.lower() for ind in compound_indicators)

        # Check for multiple question words
        question_words = ["what", "why", "how", "when", "where", "who", "which"]
        question_count = sum(1 for qw in question_words if qw in query.lower())

        if query_length < 10 and question_count <= 1 and not has_compound:
            return "simple"
        if query_length > 25 or question_count > 2 or has_compound:
            return "complex"
        return "moderate"

    def _decompose_query(self, query: str, complexity: str) -> list[Task]:
        """Decompose query into tasks.

        Args:
            query: Query text
            complexity: Complexity level

        Returns:
            List of Task objects
        """
        tasks = []

        if complexity == "simple":
            # Single task
            task_type = self._identify_task_type(query)
            tool_hints = self._suggest_tools(task_type, query)

            tasks.append(
                Task(
                    id="task_1",
                    description=query,
                    type=task_type,
                    priority=1,
                    estimated_complexity=0.3,
                    tool_hints=tool_hints,
                )
            )
        elif complexity == "moderate":
            # 2-3 tasks
            subtasks = self._split_moderate_query(query)
            for i, subtask in enumerate(subtasks, 1):
                task_type = self._identify_task_type(subtask)
                tool_hints = self._suggest_tools(task_type, subtask)

                tasks.append(
                    Task(
                        id=f"task_{i}",
                        description=subtask,
                        type=task_type,
                        priority=i,
                        estimated_complexity=0.5,
                        tool_hints=tool_hints,
                        dependencies=[f"task_{i - 1}"] if i > 1 else [],
                    )
                )
        else:  # complex
            # 3-5 tasks with dependencies
            complex_subtasks = self._split_complex_query(query)
            for i, subtask_info in enumerate(complex_subtasks, 1):
                subtask = subtask_info["task"]
                task_type = self._identify_task_type(subtask)
                tool_hints = self._suggest_tools(task_type, subtask)

                tasks.append(
                    Task(
                        id=f"task_{i}",
                        description=subtask,
                        type=task_type,
                        priority=int(subtask_info.get("priority", i)),
                        estimated_complexity=0.7,
                        tool_hints=tool_hints,
                        dependencies=subtask_info.get("dependencies", []),
                    )
                )

        return tasks

    def _identify_task_type(self, task_text: str) -> str:
        """Identify task type from text.

        Args:
            task_text: Task description

        Returns:
            Task type
        """
        task_lower = task_text.lower()

        for task_type, patterns in self.TASK_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, task_lower):
                    return task_type

        return "general"

    def _suggest_tools(self, task_type: str, task_text: str) -> list[str]:
        """Suggest tools for a task.

        Args:
            task_type: Type of task
            task_text: Task description

        Returns:
            List of suggested tool names
        """
        suggestions = []

        # Type-based suggestions
        type_to_tools = {
            "retrieval": ["vectordb_retrieval", "web_search"],
            "comparison": ["vectordb_retrieval", "reranker"],
            "computation": ["code_executor", "calculator"],
            "synthesis": ["generator", "reranker"],
            "reasoning": ["generator", "vectordb_retrieval"],
            "temporal": ["web_search", "wikipedia"],
            "spatial": ["web_search", "wikipedia"],
        }

        suggestions.extend(type_to_tools.get(task_type, ["vectordb_retrieval"]))

        # Text-based hints
        task_lower = task_text.lower()

        if (
            "calculate" in task_lower or "compute" in task_lower
        ) and "code_executor" not in suggestions:
            suggestions.append("code_executor")

        if (
            "recent" in task_lower or "latest" in task_lower or "news" in task_lower
        ) and "web_search" not in suggestions:
            suggestions.append("web_search")

        if "compare" in task_lower and "reranker" not in suggestions:
            suggestions.append("reranker")

        return suggestions[:3]  # Top 3 suggestions

    def _split_moderate_query(self, query: str) -> list[str]:
        """Split moderate query into 2-3 subtasks.

        Args:
            query: Query text

        Returns:
            List of subtask descriptions
        """
        # Simple split on conjunctions
        conjunctions = [" and ", " also ", " then ", " plus "]

        for conj in conjunctions:
            if conj in query.lower():
                parts = query.lower().split(conj, 1)
                return [parts[0].strip(), parts[1].strip()]

        # If no split found, treat as 2-phase: retrieve then synthesize
        return [
            f"Find information about: {query}",
            f"Synthesize answer for: {query}",
        ]

    def _split_complex_query(self, query: str) -> list[dict[str, Any]]:
        """Split complex query into 3-5 subtasks with dependencies.

        Args:
            query: Query text

        Returns:
            List of task info dicts
        """
        tasks = []

        # Phase 1: Information gathering
        tasks.append(
            {
                "task": f"Gather relevant information about: {query}",
                "priority": 1,
                "dependencies": [],
            }
        )

        # Phase 2: Analysis
        if "why" in query.lower() or "how" in query.lower():
            tasks.append(
                {
                    "task": f"Analyze the reasoning behind: {query}",
                    "priority": 2,
                    "dependencies": ["task_1"],
                }
            )

        # Phase 3: Comparison (if applicable)
        if "compare" in query.lower() or "versus" in query.lower() or "difference" in query.lower():
            tasks.append(
                {
                    "task": f"Compare key aspects mentioned in: {query}",
                    "priority": 3,
                    "dependencies": ["task_1"],
                }
            )

        # Final phase: Synthesis
        dep_ids = [f"task_{i + 1}" for i in range(len(tasks))]
        tasks.append(
            {
                "task": f"Synthesize comprehensive answer for: {query}",
                "priority": len(tasks) + 1,
                "dependencies": dep_ids,
            }
        )

        return tasks

    def _determine_strategy(self, tasks: list[Task]) -> str:
        """Determine execution strategy.

        Args:
            tasks: List of tasks

        Returns:
            Strategy name
        """
        # Check for dependencies
        has_dependencies = any(t.dependencies for t in tasks)

        if not has_dependencies and len(tasks) > 2:
            return "parallel"
        if has_dependencies:
            return "hybrid"  # Some parallel, some sequential
        return "sequential"

    def _generate_rationale(
        self, query: str, tasks: list[Task], strategy: str, complexity: str
    ) -> str:
        """Generate rationale for the plan.

        Args:
            query: Original query
            tasks: List of tasks
            strategy: Execution strategy
            complexity: Complexity level

        Returns:
            Rationale string
        """
        rationale = f"Query complexity: {complexity}. "
        rationale += f"Decomposed into {len(tasks)} tasks. "
        rationale += f"Execution strategy: {strategy}. "

        task_types = [t.type for t in tasks]
        type_summary = ", ".join(set(task_types))
        rationale += f"Task types: {type_summary}."

        return rationale

    def _analyze_failure(self, feedback: str) -> str:
        """Analyze what type of failure occurred.

        Args:
            feedback: Failure feedback

        Returns:
            Failure type
        """
        feedback_lower = feedback.lower()

        if "not found" in feedback_lower or "no results" in feedback_lower:
            return "no_results"
        if "incomplete" in feedback_lower or "missing" in feedback_lower:
            return "incomplete"
        if "error" in feedback_lower or "failed" in feedback_lower:
            return "execution_error"
        if "irrelevant" in feedback_lower or "wrong" in feedback_lower:
            return "wrong_approach"
        return "unknown"

    def _create_remediation_tasks(self, query: str, failure_type: str, start_id: int) -> list[Task]:
        """Create tasks to remediate a failure.

        Args:
            query: Original query
            failure_type: Type of failure
            start_id: Starting ID for new tasks

        Returns:
            List of remediation tasks
        """
        tasks = []

        if failure_type == "no_results":
            # Try different search strategies
            tasks.append(
                Task(
                    id=f"task_{start_id + 1}",
                    description=f"Search with broader terms: {query}",
                    type="retrieval",
                    tool_hints=["web_search", "wikipedia"],
                    priority=1,
                    estimated_complexity=0.4,
                )
            )
        elif failure_type == "incomplete":
            # Gather more information
            tasks.append(
                Task(
                    id=f"task_{start_id + 1}",
                    description=f"Gather additional details: {query}",
                    type="retrieval",
                    tool_hints=["vectordb_retrieval", "reranker"],
                    priority=1,
                    estimated_complexity=0.5,
                )
            )
        elif failure_type == "wrong_approach":
            # Try different tool
            tasks.append(
                Task(
                    id=f"task_{start_id + 1}",
                    description=f"Alternative approach: {query}",
                    type="general",
                    tool_hints=["web_search", "code_executor"],
                    priority=1,
                    estimated_complexity=0.6,
                )
            )

        return tasks
