"""Factory for creating and initializing agent system."""

import logging
import os
from typing import Any

from src.services.agent.graph import create_agent_graph
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


def create_agent_system(
    vectorstore_client=None,
    reranker_client=None,
    generator_client=None,
    enable_web_search: bool = True,
    enable_wikipedia: bool = True,
    enable_code_executor: bool = True,
    tavily_api_key: str | None = None,
    metrics_storage_path: str = "./logs/agent_metrics.json",
) -> tuple[ToolRegistry, AgentRouter, Any, Any]:
    """Create and initialize the complete agent system.

    Args:
        vectorstore_client: Vectorstore client for retrieval
        reranker_client: Reranker client (optional)
        generator_client: Generator client for text generation
        enable_web_search: Enable web search tool
        enable_wikipedia: Enable Wikipedia tool
        enable_code_executor: Enable code executor tool
        tavily_api_key: Tavily API key (uses DuckDuckGo if not provided)
        metrics_storage_path: Path to store metrics

    Returns:
        Tuple of (registry, router, graph, metrics_tracker)
    """
    logger.info("Creating agent system...")

    # Initialize registry
    registry = get_tool_registry()

    # Initialize metrics tracker
    metrics_tracker = get_metrics_tracker(metrics_storage_path)

    # Register local tools
    logger.info("Registering local tools...")

    if vectorstore_client:
        vectordb_tool = VectorDBTool(
            vectorstore_client=vectorstore_client,
            top_k=5,
        )
        registry.register_tool(vectordb_tool)
        logger.info("✓ VectorDB tool registered")

    if reranker_client:
        reranker_tool = RerankerTool(
            reranker_client=reranker_client,
            top_k=3,
        )
        registry.register_tool(reranker_tool)
        logger.info("✓ Reranker tool registered")

    if generator_client:
        generator_tool = GeneratorTool(
            generator_client=generator_client,
        )
        registry.register_tool(generator_tool)
        logger.info("✓ Generator tool registered")

    # Register external tools
    logger.info("Registering external tools...")

    if enable_wikipedia:
        wikipedia_tool = WikipediaTool(max_results=3)
        registry.register_tool(wikipedia_tool)
        logger.info("✓ Wikipedia tool registered")

    if enable_web_search:
        # Use provided API key or check environment
        api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        web_search_tool = WebSearchTool(api_key=api_key, max_results=5)
        registry.register_tool(web_search_tool)
        logger.info(f"✓ Web search tool registered (using {'Tavily' if api_key else 'DuckDuckGo'})")

    # Register hybrid tools
    if enable_code_executor:
        code_executor_tool = CodeExecutorTool(timeout=5)
        registry.register_tool(code_executor_tool)
        logger.info("✓ Code executor tool registered")

    # Initialize router
    router = AgentRouter(registry)
    logger.info(f"✓ Router initialized with {len(registry)} tools")

    # Create graph
    graph = create_agent_graph(registry, router)
    logger.info("✓ LangGraph workflow created")

    logger.info(f"Agent system ready with {len(registry)} tools")

    return registry, router, graph, metrics_tracker


def get_agent_from_pipeline(pipeline):
    """Create agent system from existing pipeline.

    Args:
        pipeline: Existing RAG pipeline instance

    Returns:
        Tuple of (registry, router, graph, metrics_tracker)
    """
    return create_agent_system(
        vectorstore_client=pipeline.vectorstore,
        reranker_client=pipeline.reranker,
        generator_client=pipeline.generator,
    )
