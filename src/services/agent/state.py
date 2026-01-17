"""Agent state definition for LangGraph."""

from typing import Annotated, Any, TypedDict

from langgraph.graph import add_messages


class AgentState(TypedDict, total=False):
    """State for the Agentic RAG workflow.

    Attributes:
        query: Original user query
        plan: List of decomposed subtasks
        current_task: Current task being executed
        tool_history: History of tool invocations with results
        results: Accumulated results from tools
        final_answer: Final answer to the user query
        confidence: Overall confidence score (0.0 to 1.0)
        needs_replanning: Whether the plan needs to be revised
        max_iterations: Maximum number of iterations allowed
        iteration_count: Current iteration count
        messages: Chat messages for LangGraph
        _benchmark: Internal benchmark state for performance tracking
    """

    query: str
    plan: list[str]
    current_task: str
    tool_history: list[dict[str, Any]]
    results: list[dict[str, Any]]
    final_answer: str
    confidence: float
    needs_replanning: bool
    max_iterations: int
    iteration_count: int
    messages: Annotated[list[Any], add_messages]
    _benchmark: Any
