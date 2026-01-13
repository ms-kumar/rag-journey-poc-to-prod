"""LangGraph state machine for Agentic RAG."""

import logging
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from src.services.agent.nodes import AgentNodes
from src.services.agent.state import AgentState
from src.services.agent.tools.registry import ToolRegistry
from src.services.agent.tools.router import AgentRouter

logger = logging.getLogger(__name__)


def create_agent_graph(
    registry: ToolRegistry,
    router: AgentRouter,
    llm: Any | None = None,
) -> StateGraph:
    """Create LangGraph agent workflow.

    Args:
        registry: Tool registry instance
        router: Agent router instance
        llm: Optional LLM for planning/reflection

    Returns:
        Compiled LangGraph state graph
    """
    logger.info("Creating agent graph...")

    # Initialize nodes
    nodes = AgentNodes(registry, router, llm)

    # Create state graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("plan", nodes.plan_node)
    workflow.add_node("route", nodes.route_node)
    workflow.add_node("execute", nodes.execute_node)
    workflow.add_node("reflect", nodes.reflect_node)

    # Define routing logic
    def should_continue(state: AgentState) -> Literal["route", "end"]:
        """Decide whether to continue or end.

        Args:
            state: Current agent state

        Returns:
            Next node name or "end"
        """
        # Check if we have a final answer
        if state.get("final_answer"):
            return "end"

        # Check if we need replanning
        if state.get("needs_replanning"):
            # Try fallback tool
            tool_history = state.get("tool_history", [])
            failed_tools = [h["tool"] for h in tool_history if h["status"] == "failed"]

            # If too many failures, end
            if len(failed_tools) >= 3:
                logger.warning("Too many tool failures, ending workflow")
                return "end"

            # Otherwise continue with routing
            return "route"

        # Check max iterations
        iteration_count = state.get("iteration_count", 0)
        max_iterations = state.get("max_iterations", 5)
        if iteration_count >= max_iterations:
            logger.warning("Max iterations reached")
            return "end"

        # Continue to next task
        return "route"

    # Build graph flow
    # START -> plan
    workflow.add_edge(START, "plan")

    # plan -> route
    workflow.add_edge("plan", "route")

    # route -> execute
    workflow.add_edge("route", "execute")

    # execute -> reflect
    workflow.add_edge("execute", "reflect")

    # reflect -> [route | end]
    workflow.add_conditional_edges(
        "reflect",
        should_continue,
        {
            "route": "route",
            "end": END,
        },
    )

    # Compile graph
    app = workflow.compile()

    logger.info("Agent graph created successfully")
    return app  # type: ignore[return-value]


async def run_agent(
    graph: Any,
    query: str,
    max_iterations: int = 5,
) -> dict:
    """Run the agent workflow.

    Args:
        graph: Compiled agent graph
        query: User query
        max_iterations: Maximum iterations

    Returns:
        Final agent state
    """
    logger.info(f"Running agent for query: {query[:100]}...")

    # Initialize state
    initial_state = {
        "query": query,
        "plan": [],
        "current_task": "",
        "tool_history": [],
        "results": [],
        "final_answer": "",
        "confidence": 0.0,
        "needs_replanning": False,
        "max_iterations": max_iterations,
        "iteration_count": 0,
        "messages": [],
    }

    # Run graph
    final_state: dict[Any, Any] = await graph.ainvoke(initial_state)

    logger.info("Agent workflow completed")
    return final_state
