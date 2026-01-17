"""Example demonstrating self-reflection and planning in Agentic RAG.

This example shows how to use:
1. Advanced query planning with task decomposition
2. Answer critique and quality assessment
3. Source verification and hallucination detection
4. User feedback logging and analytics
5. Task benchmarking for performance evaluation
"""

import asyncio
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

from src.services.agent.benchmarking import TaskBenchmarker
from src.services.agent.feedback import FeedbackLogger, UserFeedback
from src.services.agent.planning import QueryPlanner
from src.services.agent.reflection import AnswerCritic, SourceVerifier


async def demo_query_planning():
    """Demonstrate advanced query planning."""
    print("\n" + "=" * 80)
    print("DEMO 1: Advanced Query Planning")
    print("=" * 80)

    planner = QueryPlanner()

    # Example 1: Simple query
    simple_query = "What is machine learning?"
    plan = planner.create_plan(simple_query)
    print(f"\nSimple Query: {simple_query}")
    print(f"Complexity: {plan.complexity_level}")
    print(f"Tasks: {len(plan.tasks)}")
    for i, task in enumerate(plan.tasks, 1):
        print(f"  {i}. {task.description}")
        print(f"     Type: {task.type}, Tools: {task.tool_hints}")
    print(f"Rationale: {plan.rationale}")

    # Example 2: Moderate query
    moderate_query = "Compare supervised and unsupervised learning"
    plan = planner.create_plan(moderate_query)
    print(f"\nModerate Query: {moderate_query}")
    print(f"Complexity: {plan.complexity_level}")
    print(f"Execution Strategy: {plan.execution_strategy}")
    print(f"Tasks: {len(plan.tasks)}")
    for i, task in enumerate(plan.tasks, 1):
        print(f"  {i}. {task.description}")
        print(f"     Type: {task.type}, Priority: {task.priority}")

    # Example 3: Complex query
    complex_query = (
        "Explain how neural networks work and why they're better "
        "than traditional algorithms for image recognition"
    )
    plan = planner.create_plan(complex_query)
    print(f"\nComplex Query: {complex_query}")
    print(f"Complexity: {plan.complexity_level}")
    print(f"Execution Strategy: {plan.execution_strategy}")
    print(f"Estimated Time: {plan.estimated_time:.1f}s")
    print(f"Tasks: {len(plan.tasks)}")
    for i, task in enumerate(plan.tasks, 1):
        print(f"  {i}. {task.description}")
        print(f"     Dependencies: {task.dependencies}")

    # Example 4: Plan revision
    print("\n--- Plan Revision ---")
    feedback = "First retrieval failed to find relevant information"
    revised_plan = planner.revise_plan(plan, feedback)
    print(f"Revised to {len(revised_plan.tasks)} tasks")
    print(f"Revision reason: {feedback}")


async def demo_answer_critique():
    """Demonstrate answer critique and quality assessment."""
    print("\n" + "=" * 80)
    print("DEMO 2: Answer Critique & Quality Assessment")
    print("=" * 80)

    critic = AnswerCritic(quality_threshold=0.7)

    # Example 1: High-quality answer
    query1 = "What is Python?"
    answer1 = """Python is a high-level, interpreted programming language known for its \
simplicity and readability. Created by Guido van Rossum in 1991, Python emphasizes code \
readability with its notable use of significant whitespace.

Key features include:
- Dynamic typing and automatic memory management
- Comprehensive standard library
- Support for multiple programming paradigms (OOP, functional, procedural)
- Large ecosystem of third-party packages via PyPI

Python is widely used in web development, data science, artificial intelligence, \
scientific computing, and automation."""

    sources1 = [
        {
            "content": "Python is a high-level programming language...",
            "score": 0.95,
            "metadata": {"source": "python.org"},
        },
        {
            "content": "Python supports multiple paradigms...",
            "score": 0.88,
            "metadata": {"source": "wikipedia"},
        },
    ]

    critique1 = critic.critique_answer(answer1, query1, sources1)

    print(f"\nQuery: {query1}")
    print(f"Answer length: {len(answer1)} chars")
    print("\nQuality Scores:")
    print(f"  Overall: {critique1.overall_score:.2f}")
    print(f"  Completeness: {critique1.completeness_score:.2f}")
    print(f"  Accuracy: {critique1.accuracy_score:.2f}")
    print(f"  Relevance: {critique1.relevance_score:.2f}")
    print(f"  Clarity: {critique1.clarity_score:.2f}")
    print(f"  Source Quality: {critique1.source_quality_score:.2f}")
    print(f"\nNeeds Revision: {critique1.needs_revision}")
    if critique1.issues:
        print(f"Issues: {critique1.issues}")
    if critique1.suggestions:
        print(f"Suggestions: {critique1.suggestions}")

    # Example 2: Low-quality answer
    query2 = "How does quantum computing work?"
    answer2 = "Quantum computing uses quantum mechanics. It's faster."

    critique2 = critic.critique_answer(answer2, query2, sources=None)

    print(f"\n\nQuery: {query2}")
    print(f"Answer length: {len(answer2)} chars")
    print("\nQuality Scores:")
    print(f"  Overall: {critique2.overall_score:.2f}")
    print(f"  Completeness: {critique2.completeness_score:.2f}")
    print(f"  Accuracy: {critique2.accuracy_score:.2f}")
    print(f"\nNeeds Revision: {critique2.needs_revision}")
    print(f"Issues: {critique2.issues}")
    print(f"Suggestions: {critique2.suggestions}")
    print(f"Missing Aspects: {critique2.missing_aspects}")


async def demo_source_verification():
    """Demonstrate source verification and hallucination detection."""
    print("\n" + "=" * 80)
    print("DEMO 3: Source Verification & Hallucination Detection")
    print("=" * 80)

    verifier = SourceVerifier()

    # Example 1: Well-sourced answer
    answer1 = """The human brain contains approximately 86 billion neurons. Each neuron \
can form thousands of connections called synapses. The brain weighs about 1.4 kg in adults \
and consumes roughly 20% of the body's energy despite being only 2% of body weight."""

    sources1 = [
        {
            "content": "The human brain contains around 86 billion neurons...",
            "score": 0.92,
            "metadata": {"source": "Nature Neuroscience", "year": 2020},
        },
        {
            "content": "Brain consumes 20% of energy despite being 2% of mass...",
            "score": 0.88,
            "metadata": {"source": "Scientific American"},
        },
    ]

    verification1 = verifier.verify_sources(answer1, sources1)

    print("\nExample 1: Well-Sourced Answer")
    print(f"Sources Found: {verification1.sources_found}")
    print(f"Sources Verified: {verification1.sources_verified}")
    print(f"Source Diversity: {verification1.source_diversity:.2f}")
    print(f"Citation Quality: {verification1.citation_quality:.2f}")
    print(f"Hallucination Risk: {verification1.hallucination_risk:.2f}")
    if verification1.verified_sources:
        print("\nVerified Sources:")
        for src in verification1.verified_sources:
            print(f"  - {src['id']}: {src['content'][:80]}...")

    # Example 2: Poorly sourced answer with potential hallucination
    answer2 = """The newest iPhone model, released in 2026, has a screen resolution \
of 8K and costs exactly $2,499. It includes a revolutionary quantum processor running \
at 10 THz."""

    sources2 = [
        {
            "content": "Apple announces new features...",
            "score": 0.45,
        }
    ]

    verification2 = verifier.verify_sources(answer2, sources2)

    print("\n\nExample 2: Poorly Sourced Answer")
    print(f"Sources Found: {verification2.sources_found}")
    print(f"Sources Verified: {verification2.sources_verified}")
    print(f"Hallucination Risk: {verification2.hallucination_risk:.2f}")
    if verification2.questionable_claims:
        print("\nQuestionable Claims:")
        for claim in verification2.questionable_claims:
            print(f"  - {claim}")


async def demo_feedback_logging():
    """Demonstrate user feedback logging and analytics."""
    print("\n" + "=" * 80)
    print("DEMO 4: User Feedback Logging & Analytics")
    print("=" * 80)

    logger = FeedbackLogger(storage_path="logs/user_feedback.json")

    # Simulate user feedback
    feedback_examples = [
        {
            "query": "What is Python?",
            "answer": "Python is a programming language...",
            "rating": 5,
            "feedback_text": "Very comprehensive and clear explanation!",
            "issues": [],
        },
        {
            "query": "How does blockchain work?",
            "answer": "Blockchain is a distributed ledger...",
            "rating": 4,
            "feedback_text": "Good explanation but could use more examples",
            "issues": [],
        },
        {
            "query": "Explain quantum computing",
            "answer": "Quantum computing uses qubits...",
            "rating": 2,
            "feedback_text": "Too technical and incomplete",
            "issues": ["incomplete", "unclear"],
        },
        {
            "query": "What is machine learning?",
            "answer": "ML is about algorithms...",
            "rating": 5,
            "feedback_text": "Excellent detailed answer",
            "issues": [],
        },
        {
            "query": "How does GPS work?",
            "answer": "GPS uses satellites...",
            "rating": 3,
            "feedback_text": "Okay but missing important details",
            "issues": ["incomplete"],
        },
    ]

    print("\n--- Logging Feedback ---")
    for fb in feedback_examples:
        feedback = UserFeedback(
            query=fb["query"],
            answer=fb["answer"],
            rating=fb["rating"],
            feedback_text=fb["feedback_text"],
            issues=fb["issues"],
            session_id="demo_session",
        )
        logger.log_feedback(feedback)
        print(f"  [{feedback.rating}/5] {feedback.query[:50]}...")

    # Get analytics
    print("\n--- Analytics ---")
    analytics = logger.get_analytics()
    print(f"Total Feedback: {analytics.total_feedback}")
    print(f"Average Rating: {analytics.average_rating:.2f}/5")
    print(f"Positive Feedback Rate: {analytics.positive_feedback_rate:.1%}")

    if analytics.common_issues:
        print("\nCommon Issues:")
        for issue, count in analytics.common_issues.items():
            print(f"  - {issue}: {count} times")

    if analytics.improvement_areas:
        print("\nImprovement Areas:")
        for area in analytics.improvement_areas:
            print(f"  - {area}")

    if analytics.success_patterns:
        print("\nSuccess Patterns:")
        for pattern in analytics.success_patterns:
            print(f"  - {pattern}")

    # Get low-rated queries
    print("\n--- Low-Rated Queries ---")
    low_rated = logger.get_low_rated_queries(threshold=3, limit=3)
    for entry in low_rated:
        print(f"\nQuery: {entry['query']}")
        print(f"Rating: {entry['rating']}/5")
        print(f"Feedback: {entry['feedback']}")


async def demo_benchmarking():
    """Demonstrate task benchmarking."""
    print("\n" + "=" * 80)
    print("DEMO 5: Task Benchmarking")
    print("=" * 80)

    benchmarker = TaskBenchmarker(storage_path="logs/benchmarks.jsonl")

    # Simulate some benchmark runs
    print("\n--- Running Benchmarks ---")

    # Benchmark 1: Simple query
    query1 = "What is Python?"
    benchmark1 = benchmarker.start_benchmark(query1, "simple", 1)
    benchmarker.record_task(
        benchmark1,
        task_id="task_1",
        task_description="Retrieve information about Python",
        execution_time=0.8,
        success=True,
        tool_used="vectordb_retrieval",
        quality_score=0.92,
        tool_confidence=0.88,
    )
    benchmarker.complete_benchmark(benchmark1, final_quality_score=0.92)
    print(f"✓ Benchmarked: {query1}")

    # Benchmark 2: Complex query
    query2 = "Compare Python and Java for data science applications"
    benchmark2 = benchmarker.start_benchmark(query2, "complex", 4)
    benchmarker.record_task(
        benchmark2,
        "task_1",
        "Retrieve Python info",
        1.2,
        True,
        "vectordb_retrieval",
        0.85,
    )
    benchmarker.record_task(
        benchmark2,
        "task_2",
        "Retrieve Java info",
        1.1,
        True,
        "vectordb_retrieval",
        0.82,
    )
    benchmarker.record_task(
        benchmark2,
        "task_3",
        "Compare features",
        2.3,
        True,
        "reranker",
        0.88,
    )
    benchmarker.record_task(
        benchmark2,
        "task_4",
        "Synthesize answer",
        1.8,
        True,
        "generator",
        0.90,
    )
    benchmarker.complete_benchmark(benchmark2, final_quality_score=0.87)
    print(f"✓ Benchmarked: {query2}")

    # Benchmark 3: Failed query
    query3 = "Latest cryptocurrency prices"
    benchmark3 = benchmarker.start_benchmark(query3, "simple", 1)
    benchmarker.record_task(
        benchmark3,
        "task_1",
        "Search for crypto prices",
        15.5,
        False,
        "web_search",
        0.0,
        error_message="Timeout",
    )
    benchmarker.complete_benchmark(benchmark3, final_quality_score=0.0)
    print(f"✗ Benchmarked (failed): {query3}")

    # Get statistics
    print("\n--- Benchmark Statistics ---")
    stats = benchmarker.get_statistics()
    print(f"Total Benchmarks: {stats['total_benchmarks']}")
    print(f"Success Rate: {stats['success_rate']:.1%}")
    print(f"Avg Execution Time: {stats['average_execution_time']:.2f}s")
    print(f"Avg Quality Score: {stats['average_quality_score']:.2f}")

    if stats["tool_usage"]:
        print("\nTool Usage:")
        for tool, count in stats["tool_usage"].items():
            print(f"  - {tool}: {count} times")

    # Get slow queries
    print("\n--- Slow Queries (>5s) ---")
    slow_queries = benchmarker.get_slow_queries(threshold=5.0)
    for sq in slow_queries:
        print(f"  - {sq['query']}: {sq['execution_time']:.2f}s")


async def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("AGENTIC RAG: SELF-REFLECTION & PLANNING DEMO")
    print("=" * 80)

    await demo_query_planning()
    await demo_answer_critique()
    await demo_source_verification()
    await demo_feedback_logging()
    await demo_benchmarking()

    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print("\nKey Features Demonstrated:")
    print("  1. ✓ Advanced query planning with task decomposition")
    print("  2. ✓ Answer quality critique (completeness, accuracy, clarity)")
    print("  3. ✓ Source verification and hallucination detection")
    print("  4. ✓ User feedback logging with analytics")
    print("  5. ✓ Performance benchmarking for optimization")
    print("\nThese features enable continuous improvement of the RAG system!")


if __name__ == "__main__":
    asyncio.run(main())
