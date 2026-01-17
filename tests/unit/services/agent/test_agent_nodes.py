"""Unit tests for agent state and nodes."""

from typing import TYPE_CHECKING

import pytest

from src.services.agent.nodes import AgentNodes
from src.services.agent.tools.base import BaseTool, ToolCategory, ToolMetadata

if TYPE_CHECKING:
    from src.services.agent.state import AgentState
from src.services.agent.tools.registry import ToolRegistry
from src.services.agent.tools.router import AgentRouter


class MockTool(BaseTool):
    """Mock tool for testing."""

    def __init__(self, metadata, should_fail=False):
        super().__init__(metadata)
        self.should_fail = should_fail

    async def execute(self, query: str, **kwargs):
        if self.should_fail:
            return {
                "success": False,
                "result": None,
                "error": "Mock tool failed",
                "metadata": {},
            }
        return {
            "success": True,
            "result": {"output": f"Mock result for: {query}"},
            "error": None,
            "metadata": {},
        }


class TestAgentState:
    """Test AgentState TypedDict."""

    def test_state_structure(self):
        """Test agent state structure."""
        state: AgentState = {
            "query": "test query",
            "plan": ["task1", "task2"],
            "current_task": "task1",
            "tool_history": [],
            "results": [],
            "final_answer": "",
            "confidence": 0.0,
            "needs_replanning": False,
            "max_iterations": 5,
            "iteration_count": 0,
            "messages": [],
        }

        assert state["query"] == "test query"
        assert len(state["plan"]) == 2
        assert state["max_iterations"] == 5


class TestAgentNodes:
    """Test AgentNodes class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.registry = ToolRegistry()
        self.router = AgentRouter(self.registry)
        # Disable reflection/planning to test basic node behavior
        self.nodes = AgentNodes(
            self.registry, self.router, enable_reflection=False, enable_planning=False
        )

        # Register test tools
        self._register_test_tools()

    def _register_test_tools(self):
        """Register test tools."""
        for i in range(3):
            metadata = ToolMetadata(
                name=f"test_tool_{i}",
                description=f"Test tool {i}",
                category=ToolCategory.LOCAL,
                capabilities=["test"],
            )
            self.registry.register_tool(MockTool(metadata))

    @pytest.mark.asyncio
    async def test_plan_node_simple_query(self):
        """Test plan node with simple query."""
        state: AgentState = {
            "query": "What is machine learning?",
            "plan": [],
            "current_task": "",
            "tool_history": [],
            "results": [],
            "final_answer": "",
            "confidence": 0.0,
            "needs_replanning": False,
            "max_iterations": 5,
            "iteration_count": 0,
            "messages": [],
        }

        result = await self.nodes.plan_node(state)

        assert "plan" in result
        assert len(result["plan"]) >= 1
        assert "current_task" in result
        assert result["iteration_count"] == 1

    @pytest.mark.asyncio
    async def test_plan_node_complex_query(self):
        """Test plan node with complex query."""
        state: AgentState = {
            "query": "What is Python and calculate factorial of 5?",
            "plan": [],
            "current_task": "",
            "tool_history": [],
            "results": [],
            "final_answer": "",
            "confidence": 0.0,
            "needs_replanning": False,
            "max_iterations": 5,
            "iteration_count": 0,
            "messages": [],
        }

        result = await self.nodes.plan_node(state)

        assert "plan" in result
        # May create multiple tasks for complex query
        assert len(result["plan"]) >= 1

    @pytest.mark.asyncio
    async def test_route_node(self):
        """Test route node."""
        state: AgentState = {
            "query": "test query",
            "plan": ["task1"],
            "current_task": "search for documents",
            "tool_history": [],
            "results": [],
            "final_answer": "",
            "confidence": 0.0,
            "needs_replanning": False,
            "max_iterations": 5,
            "iteration_count": 0,
            "messages": [],
        }

        result = await self.nodes.route_node(state)

        assert "tool_history" in result
        assert len(result["tool_history"]) == 1
        assert "confidence" in result

        decision = result["tool_history"][0]
        assert "tool" in decision
        assert "confidence" in decision
        assert "status" in decision
        assert decision["status"] == "pending"

    @pytest.mark.asyncio
    async def test_execute_node_success(self):
        """Test execute node with successful execution."""
        # Set up state with routing decision
        state: AgentState = {
            "query": "test query",
            "plan": ["task1"],
            "current_task": "task1",
            "tool_history": [
                {
                    "task": "task1",
                    "tool": "test_tool_0",
                    "confidence": 0.8,
                    "status": "pending",
                }
            ],
            "results": [],
            "final_answer": "",
            "confidence": 0.0,
            "needs_replanning": False,
            "max_iterations": 5,
            "iteration_count": 1,
            "messages": [],
        }

        result = await self.nodes.execute_node(state)

        assert "tool_history" in result
        assert result["tool_history"][0]["status"] == "success"
        assert "results" in result
        assert len(result["results"]) == 1

    @pytest.mark.asyncio
    async def test_execute_node_failure(self):
        """Test execute node with failed execution."""
        # Register a failing tool
        metadata = ToolMetadata(
            name="failing_tool",
            description="Failing test tool",
            category=ToolCategory.LOCAL,
        )
        self.registry.register_tool(MockTool(metadata, should_fail=True))

        state: AgentState = {
            "query": "test query",
            "plan": ["task1"],
            "current_task": "task1",
            "tool_history": [
                {
                    "task": "task1",
                    "tool": "failing_tool",
                    "confidence": 0.8,
                    "status": "pending",
                }
            ],
            "results": [],
            "final_answer": "",
            "confidence": 0.0,
            "needs_replanning": False,
            "max_iterations": 5,
            "iteration_count": 1,
            "messages": [],
        }

        result = await self.nodes.execute_node(state)

        assert result["tool_history"][0]["status"] == "failed"

    @pytest.mark.asyncio
    async def test_execute_node_no_tool(self):
        """Test execute node with no routing decision."""
        state: AgentState = {
            "query": "test query",
            "plan": ["task1"],
            "current_task": "task1",
            "tool_history": [],
            "results": [],
            "final_answer": "",
            "confidence": 0.0,
            "needs_replanning": False,
            "max_iterations": 5,
            "iteration_count": 1,
            "messages": [],
        }

        with pytest.raises(ValueError, match="No tool selected"):
            await self.nodes.execute_node(state)

    @pytest.mark.asyncio
    async def test_reflect_node_all_tasks_complete(self):
        """Test reflect node when all tasks are complete."""
        state: AgentState = {
            "query": "test query",
            "plan": ["task1", "task2"],
            "current_task": "task2",
            "tool_history": [
                {"task": "task1", "tool": "tool1", "status": "success"},
                {"task": "task2", "tool": "tool2", "status": "success"},
            ],
            "results": [
                {"success": True, "result": {"output": "result1"}},
                {"success": True, "result": {"output": "result2"}},
            ],
            "final_answer": "",
            "confidence": 0.8,
            "needs_replanning": False,
            "max_iterations": 5,
            "iteration_count": 2,
            "messages": [],
        }

        result = await self.nodes.reflect_node(state)

        assert "final_answer" in result
        assert len(result["final_answer"]) > 0
        assert result["needs_replanning"] is False

    @pytest.mark.asyncio
    async def test_reflect_node_next_task(self):
        """Test reflect node moving to next task."""
        state: AgentState = {
            "query": "test query",
            "plan": ["task1", "task2"],
            "current_task": "task1",
            "tool_history": [
                {"task": "task1", "tool": "tool1", "status": "success"},
            ],
            "results": [
                {"success": True, "result": {"output": "result1"}},
            ],
            "final_answer": "",
            "confidence": 0.8,
            "needs_replanning": False,
            "max_iterations": 5,
            "iteration_count": 1,
            "messages": [],
        }

        result = await self.nodes.reflect_node(state)

        assert "current_task" in result
        assert result["current_task"] == "task2"
        assert result["needs_replanning"] is False

    @pytest.mark.asyncio
    async def test_reflect_node_max_iterations(self):
        """Test reflect node at max iterations."""
        state: AgentState = {
            "query": "test query",
            "plan": ["task1", "task2"],
            "current_task": "task1",
            "tool_history": [],
            "results": [],
            "final_answer": "",
            "confidence": 0.0,
            "needs_replanning": False,
            "max_iterations": 2,
            "iteration_count": 2,
            "messages": [],
        }

        result = await self.nodes.reflect_node(state)

        assert "final_answer" in result
        assert result["needs_replanning"] is False

    @pytest.mark.asyncio
    async def test_reflect_node_tool_failure(self):
        """Test reflect node with tool failure."""
        state: AgentState = {
            "query": "test query",
            "plan": ["task1"],
            "current_task": "task1",
            "tool_history": [
                {"task": "task1", "tool": "tool1", "status": "failed"},
            ],
            "results": [],
            "final_answer": "",
            "confidence": 0.0,
            "needs_replanning": False,
            "max_iterations": 5,
            "iteration_count": 1,
            "messages": [],
        }

        result = await self.nodes.reflect_node(state)

        assert result["needs_replanning"] is True

    def test_synthesize_answer_with_results(self):
        """Test answer synthesis with results."""
        state: AgentState = {
            "query": "test query",
            "plan": [],
            "current_task": "",
            "tool_history": [],
            "results": [
                {
                    "success": True,
                    "result": {
                        "documents": [
                            {"content": "Document 1"},
                            {"content": "Document 2"},
                        ]
                    },
                }
            ],
            "final_answer": "",
            "confidence": 0.0,
            "needs_replanning": False,
            "max_iterations": 5,
            "iteration_count": 0,
            "messages": [],
        }

        answer = self.nodes._synthesize_answer(state)

        assert len(answer) > 0
        assert "Document 1" in answer or "Document 2" in answer

    def test_synthesize_answer_no_results(self):
        """Test answer synthesis with no results."""
        state: AgentState = {
            "query": "test query",
            "plan": [],
            "current_task": "",
            "tool_history": [],
            "results": [],
            "final_answer": "",
            "confidence": 0.0,
            "needs_replanning": False,
            "max_iterations": 5,
            "iteration_count": 0,
            "messages": [],
        }

        answer = self.nodes._synthesize_answer(state)

        assert "couldn't find" in answer.lower() or "apologize" in answer.lower()
