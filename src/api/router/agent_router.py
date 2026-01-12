"""Agent router for Agentic RAG with LangGraph."""

import logging
import time
from typing import Optional

from fastapi import APIRouter, HTTPException

from src.schemas.services.agent import (
    AgentRequest,
    AgentResponse,
    MetricsSummary,
    ToolExecutionInfo,
    ToolInfo,
)
from src.services.agent.graph import create_agent_graph, run_agent
from src.services.agent.metrics.tracker import get_metrics_tracker
from src.services.agent.tools.external.web_search_tool import WebSearchTool
from src.services.agent.tools.external.wikipedia_tool import WikipediaTool
from src.services.agent.tools.hybrid.code_executor import CodeExecutorTool
from src.services.agent.tools.local.generator_tool import GeneratorTool
from src.services.agent.tools.local.reranker_tool import RerankerTool
from src.services.agent.tools.local.vectordb_tool import VectorDBTool
from src.services.agent.tools.registry import ToolRegistry, get_tool_registry
from src.services.agent.tools.router import AgentRouter

logger = logging.getLogger(__name__)

router = APIRouter()

# Global instances
_registry: Optional[ToolRegistry] = None
_router: Optional[AgentRouter] = None
_graph: Optional[any] = None
_metrics_tracker = None


def _initialize_agent():
    """Initialize agent components."""
    global _registry, _router, _graph, _metrics_tracker
    
    if _registry is not None:
        return  # Already initialized
    
    logger.info("Initializing Agentic RAG system...")
    
    # Initialize registry
    _registry = get_tool_registry()
    
    # Initialize metrics tracker
    _metrics_tracker = get_metrics_tracker("./logs/agent_metrics.json")
    
    # Get services from pipeline (lazy import to avoid circular dependency)
    try:
        from src.services.pipeline.naive_pipeline.factory import get_naive_pipeline
        pipeline = get_naive_pipeline()
        
        # Register local tools
        logger.info("Registering local tools...")
        
        # VectorDB tool
        vectordb_tool = VectorDBTool(
            vectorstore_client=pipeline.vectorstore,
            top_k=5,
        )
        _registry.register_tool(vectordb_tool)
        
        # Reranker tool (if available)
        if pipeline.reranker:
            reranker_tool = RerankerTool(
                reranker_client=pipeline.reranker,
                top_k=3,
            )
            _registry.register_tool(reranker_tool)
        
        # Generator tool
        generator_tool = GeneratorTool(
            generator_client=pipeline.generator,
        )
        _registry.register_tool(generator_tool)
        
    except Exception as e:
        logger.warning(f"Could not initialize local tools from pipeline: {e}")
        logger.info("Agent will work with external tools only")
    
    # Register external tools
    logger.info("Registering external tools...")
    
    # Wikipedia tool (always available)
    wikipedia_tool = WikipediaTool(max_results=3)
    _registry.register_tool(wikipedia_tool)
    
    # Web search tool (optional, depends on API key)
    try:
        import os
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        web_search_tool = WebSearchTool(api_key=tavily_api_key, max_results=5)
        _registry.register_tool(web_search_tool)
    except Exception as e:
        logger.warning(f"Could not initialize web search tool: {e}")
    
    # Code executor tool (hybrid)
    logger.info("Registering hybrid tools...")
    code_executor_tool = CodeExecutorTool(timeout=5)
    _registry.register_tool(code_executor_tool)
    
    # Initialize router
    _router = AgentRouter(_registry)
    
    # Create graph
    _graph = create_agent_graph(_registry, _router)
    
    logger.info(f"Agent initialized with {len(_registry)} tools")


@router.post("/agent/query", response_model=AgentResponse, tags=["agent"])
async def agent_query(request: AgentRequest):
    """
    Execute agentic RAG query with tool routing and self-reflection.
    
    The agent will:
    1. Plan: Decompose query into subtasks
    2. Route: Select appropriate tools with confidence scoring
    3. Execute: Run selected tools
    4. Reflect: Evaluate results and decide next steps
    
    Supports local tools (vectordb, reranker, generator) and external tools
    (web search, Wikipedia, code execution).
    """
    _initialize_agent()
    
    start_time = time.time()
    
    try:
        logger.info(f"Agent query: {request.query[:100]}...")
        
        # Run agent workflow
        final_state = await run_agent(
            graph=_graph,
            query=request.query,
            max_iterations=request.max_iterations,
        )
        
        # Track tool invocations
        for tool_exec in final_state.get("tool_history", []):
            tool_name = tool_exec.get("tool")
            status = tool_exec.get("status")
            confidence = tool_exec.get("confidence", 0.0)
            error = tool_exec.get("error")
            
            # Estimate latency (simplified)
            tool = _registry.get_tool(tool_name)
            latency = tool.metadata.avg_latency_ms if tool else 100.0
            cost = tool.metadata.cost_per_call if tool else 0.0
            
            _metrics_tracker.track_invocation(
                tool_name=tool_name,
                success=(status == "success"),
                latency_ms=latency,
                cost=cost,
                confidence=confidence,
                error=error,
            )
        
        # Format tool history
        tool_history = [
            ToolExecutionInfo(
                tool_name=h.get("tool"),
                task=h.get("task"),
                confidence=h.get("confidence", 0.0),
                status=h.get("status"),
                reasoning=h.get("reasoning"),
                error=h.get("error"),
            )
            for h in final_state.get("tool_history", [])
        ]
        
        # Calculate total latency
        total_latency_ms = (time.time() - start_time) * 1000
        
        # Build response
        response = AgentResponse(
            query=request.query,
            answer=final_state.get("final_answer", "No answer generated"),
            plan=final_state.get("plan", []),
            tool_history=tool_history,
            confidence=final_state.get("confidence", 0.0),
            iterations=final_state.get("iteration_count", 0),
            total_latency_ms=total_latency_ms,
            metadata={
                "num_results": len(final_state.get("results", [])),
                "max_iterations": request.max_iterations,
            },
        )
        
        logger.info(f"Agent query completed in {total_latency_ms:.0f}ms")
        return response
        
    except Exception as e:
        logger.error(f"Agent query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent query failed: {str(e)}")


@router.get("/agent/tools", response_model=list[ToolInfo], tags=["agent"])
async def list_tools():
    """
    List all registered tools with their metadata.
    
    Returns information about available tools including:
    - Name and description
    - Category (local/external/hybrid)
    - Capabilities
    - Performance metrics
    """
    _initialize_agent()
    
    tools = _registry.list_tools()
    
    return [
        ToolInfo(
            name=tool.metadata.name,
            description=tool.metadata.description,
            category=tool.metadata.category.value,
            capabilities=tool.metadata.capabilities,
            cost_per_call=tool.metadata.cost_per_call,
            avg_latency_ms=tool.metadata.avg_latency_ms,
            success_rate=tool.metadata.success_rate,
            requires_api_key=tool.metadata.requires_api_key,
        )
        for tool in tools
    ]


@router.get("/agent/metrics", response_model=MetricsSummary, tags=["agent"])
async def get_metrics():
    """
    Get metrics summary for all tools.
    
    Returns:
    - Total invocations per tool
    - Success rates
    - Average latency
    - Average confidence scores
    - Total costs
    """
    _initialize_agent()
    
    summary = _metrics_tracker.get_summary()
    return MetricsSummary(**summary)


@router.post("/agent/metrics/reset", tags=["agent"])
async def reset_metrics(tool_name: Optional[str] = None):
    """
    Reset metrics for a specific tool or all tools.
    
    Args:
        tool_name: Optional tool name to reset (resets all if not provided)
    """
    _initialize_agent()
    
    _metrics_tracker.reset_metrics(tool_name)
    
    return {
        "message": f"Metrics reset for {'tool: ' + tool_name if tool_name else 'all tools'}",
    }


@router.get("/agent/status", tags=["agent"])
async def agent_status():
    """
    Get agent system status.
    
    Returns information about:
    - Number of registered tools
    - Tool categories breakdown
    - System readiness
    """
    _initialize_agent()
    
    tools = _registry.list_tools()
    
    categories = {}
    for tool in tools:
        category = tool.metadata.category.value
        categories[category] = categories.get(category, 0) + 1
    
    return {
        "status": "ready",
        "total_tools": len(tools),
        "tools_by_category": categories,
        "tool_names": _registry.get_all_tool_names(),
    }
