"""
Demo: Agentic RAG with LangGraph

This demonstrates the Week 7 Agentic RAG implementation with:
- Tool registry & router
- Confidence scoring
- LangGraph state machine
- Multiple tools (local + external)
- Self-reflection & planning
"""

import asyncio
import logging

from src.services.agent.factory import create_agent_system
from src.services.agent.graph import run_agent
from src.services.pipeline.naive_pipeline.factory import get_naive_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def demo_agent_workflow():
    """Demonstrate agent workflow with various queries."""
    
    logger.info("=" * 80)
    logger.info("AGENTIC RAG DEMO - Week 7")
    logger.info("=" * 80)
    
    # Initialize pipeline
    logger.info("\n1. Initializing RAG pipeline...")
    pipeline = get_naive_pipeline()
    
    # Create agent system
    logger.info("\n2. Creating agent system...")
    registry, router, graph, metrics = create_agent_system(
        vectorstore_client=pipeline.vectorstore,
        reranker_client=pipeline.reranker,
        generator_client=pipeline.generator,
        enable_web_search=True,
        enable_wikipedia=True,
        enable_code_executor=True,
    )
    
    logger.info(f"\n✓ Agent system ready with {len(registry)} tools")
    
    # List registered tools
    logger.info("\n3. Registered Tools:")
    logger.info("-" * 80)
    for tool in registry.list_tools():
        logger.info(
            f"  • {tool.metadata.name} ({tool.metadata.category.value}): "
            f"{tool.metadata.description[:60]}..."
        )
    
    # Test queries
    test_queries = [
        {
            "query": "What is machine learning?",
            "description": "Simple knowledge base query (should use vectordb + generator)",
        },
        {
            "query": "What is Python and who created it?",
            "description": "General knowledge query (should use Wikipedia)",
        },
        {
            "query": "Calculate the factorial of 5 using Python",
            "description": "Code execution query (should use code executor)",
        },
    ]
    
    # Run test queries
    logger.info("\n4. Running Test Queries:")
    logger.info("=" * 80)
    
    for i, test in enumerate(test_queries, 1):
        logger.info(f"\n--- Query {i}: {test['description']} ---")
        logger.info(f"Query: {test['query']}")
        logger.info("-" * 80)
        
        try:
            # Run agent
            result = await run_agent(
                graph=graph,
                query=test["query"],
                max_iterations=3,
            )
            
            # Display results
            logger.info(f"\n✓ Plan: {result['plan']}")
            logger.info(f"✓ Iterations: {result['iteration_count']}")
            logger.info(f"✓ Tools Used:")
            for h in result.get("tool_history", []):
                logger.info(
                    f"  • {h['tool']} (confidence: {h['confidence']:.2f}, "
                    f"status: {h['status']})"
                )
            
            logger.info(f"\n✓ Answer:\n{result['final_answer'][:300]}...")
            
        except Exception as e:
            logger.error(f"✗ Query failed: {e}")
    
    # Display metrics
    logger.info("\n" + "=" * 80)
    logger.info("5. Agent Metrics Summary:")
    logger.info("=" * 80)
    
    summary = metrics.get_summary()
    for tool_name, tool_metrics in summary["tools"].items():
        logger.info(f"\n{tool_name}:")
        logger.info(f"  Invocations: {tool_metrics['invocations']}")
        logger.info(f"  Success Rate: {tool_metrics['success_rate']:.1%}")
        logger.info(f"  Avg Latency: {tool_metrics['avg_latency_ms']:.0f}ms")
        logger.info(f"  Avg Confidence: {tool_metrics['avg_confidence']:.2f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("DEMO COMPLETED")
    logger.info("=" * 80)


async def demo_tool_router():
    """Demonstrate tool routing with confidence scoring."""
    
    logger.info("\n" + "=" * 80)
    logger.info("TOOL ROUTER DEMO - Confidence Scoring")
    logger.info("=" * 80)
    
    # Initialize system
    pipeline = get_naive_pipeline()
    registry, router, _, _ = create_agent_system(
        vectorstore_client=pipeline.vectorstore,
        reranker_client=pipeline.reranker,
        generator_client=pipeline.generator,
    )
    
    # Test routing decisions
    test_cases = [
        "Find documents about machine learning",
        "Search the web for latest AI news",
        "Calculate the sum of 1 to 100",
        "What is the definition of RAG according to Wikipedia?",
        "Rerank these documents by relevance",
    ]
    
    logger.info("\nRouting Decisions:")
    logger.info("-" * 80)
    
    for query in test_cases:
        decision = await router.route(query)
        logger.info(f"\nQuery: {query}")
        logger.info(f"  → Tool: {decision.tool_name}")
        logger.info(f"  → Confidence: {decision.confidence:.2f}")
        logger.info(f"  → Category: {decision.category.value}")
        logger.info(f"  → Reasoning: {decision.reasoning[:100]}...")
        logger.info(f"  → Fallbacks: {', '.join(decision.fallback_tools)}")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("AGENTIC RAG DEMONSTRATION")
    print("Week 7: Tool Registry, Router, and LangGraph")
    print("=" * 80)
    
    # Run demos
    print("\n[1] Agent Workflow Demo")
    asyncio.run(demo_agent_workflow())
    
    print("\n\n[2] Tool Router Demo")
    asyncio.run(demo_tool_router())
