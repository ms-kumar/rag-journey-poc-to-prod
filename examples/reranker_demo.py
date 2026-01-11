"""
Demo script for cross-encoder re-ranking functionality.

Shows:
1. Basic re-ranking with precision@k evaluation
2. Comparison with baseline retrieval
3. Batch re-ranking performance
4. Timeout and fallback handling
"""

import logging
from pathlib import Path

from src.config import get_settings
from src.services.embeddings.factory import get_embed_client, get_langchain_embeddings_adapter
from src.services.reranker.client import RerankerConfig
from src.services.reranker.evaluation import RerankingEvaluator, print_benchmark_results
from src.services.reranker.factory import get_reranker
from src.services.vectorstore.client import QdrantVectorStoreClient
from langchain_core.documents import Document

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_test_data():
    """Create test documents and queries for demonstration."""
    # Sample documents about machine learning topics
    test_docs = [
        Document(
            page_content="Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data and learn for themselves.",
            metadata={"id": "doc1", "topic": "ml_basics", "relevance": "high"}
        ),
        Document(
            page_content="Deep learning is a machine learning technique inspired by the structure of the human brain. It uses artificial neural networks with multiple layers to model and understand complex patterns in data.",
            metadata={"id": "doc2", "topic": "deep_learning", "relevance": "high"}
        ),
        Document(
            page_content="Python is a high-level programming language that is widely used in data science and machine learning projects due to its simplicity and extensive library ecosystem.",
            metadata={"id": "doc3", "topic": "programming", "relevance": "medium"}
        ),
        Document(
            page_content="Supervised learning is a type of machine learning where algorithms learn from labeled training data to make predictions on new, unseen data. Examples include classification and regression tasks.",
            metadata={"id": "doc4", "topic": "ml_types", "relevance": "high"}
        ),
        Document(
            page_content="The weather today is sunny with a temperature of 25 degrees Celsius. It's a perfect day for outdoor activities and spending time in the park.",
            metadata={"id": "doc5", "topic": "weather", "relevance": "none"}
        ),
        Document(
            page_content="Natural language processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret and manipulate human language. It combines computational linguistics with statistical and deep learning models.",
            metadata={"id": "doc6", "topic": "nlp", "relevance": "high"}
        ),
        Document(
            page_content="Reinforcement learning is a type of machine learning where an agent learns to make decisions by taking actions in an environment and receiving rewards or penalties for those actions.",
            metadata={"id": "doc7", "topic": "reinforcement_learning", "relevance": "high"}
        ),
        Document(
            page_content="Linear algebra is a branch of mathematics concerning linear equations, linear maps and their representations in vector spaces and through matrices. It's fundamental to many machine learning algorithms.",
            metadata={"id": "doc8", "topic": "mathematics", "relevance": "medium"}
        ),
    ]
    
    test_queries = [
        "What is machine learning and how does it work?",
        "Explain deep learning and neural networks",
        "What are the different types of machine learning?",
        "How does reinforcement learning work?",
    ]
    
    # Define relevant documents for each query (for evaluation)
    relevant_docs = [
        {"doc1", "doc4", "doc7"},  # Query 0: ML basics
        {"doc2", "doc6"},         # Query 1: Deep learning  
        {"doc1", "doc4", "doc7"}, # Query 2: ML types
        {"doc7"},                 # Query 3: Reinforcement learning
    ]
    
    return test_docs, test_queries, relevant_docs


def example_1_basic_reranking():
    """Example 1: Basic re-ranking demonstration."""
    print("=" * 80)
    print("EXAMPLE 1: BASIC RE-RANKING")
    print("=" * 80)
    
    # Setup test data
    test_docs, test_queries, relevant_docs = setup_test_data()
    query = test_queries[0]  # "What is machine learning and how does it work?"
    relevant_set = relevant_docs[0]
    
    print(f"Query: {query}")
    print(f"Relevant documents: {relevant_set}")
    
    # Create reranker
    logger.info("Initializing cross-encoder reranker...")
    reranker = get_reranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size=8,
        timeout_seconds=15.0
    )
    
    print("\\nOriginal document order:")
    for i, doc in enumerate(test_docs):
        relevance = "✓" if doc.metadata["id"] in relevant_set else "✗"
        print(f"{i+1}. [{relevance}] {doc.page_content[:80]}...")
    
    # Re-rank documents
    logger.info("Re-ranking documents...")
    result = reranker.rerank(query, test_docs)
    
    print(f"\\nRe-ranked documents (execution time: {result.execution_time:.2f}s):")
    for i, doc in enumerate(result.documents):
        relevance = "✓" if doc.metadata["id"] in relevant_set else "✗"
        score = f"({result.scores[i]:.3f})" if result.scores else ""
        original_rank = result.original_ranks[i] + 1
        print(f"{i+1}. [{relevance}] {score} (was #{original_rank}) {doc.page_content[:80]}...")
    
    # Evaluate precision@k
    evaluator = RerankingEvaluator(reranker)
    metrics = evaluator.compare_rankings(query, test_docs, relevant_set)
    
    print("\\nPRECISION@K COMPARISON:")
    print("-" * 40)
    print("K\\tBaseline\\tRe-ranked\\tImprovement")
    print("-" * 40)
    for k in sorted(metrics.improvement.keys()):
        baseline = metrics.baseline_metrics.precision_at_k[k]
        reranked = metrics.reranked_metrics.precision_at_k[k]
        improvement = metrics.improvement[k]
        print(f"{k}\\t{baseline:.3f}\\t\\t{reranked:.3f}\\t\\t{improvement:+.3f}")
    
    print()


def example_2_batch_reranking():
    """Example 2: Batch re-ranking with multiple queries."""
    print("=" * 80)
    print("EXAMPLE 2: BATCH RE-RANKING")
    print("=" * 80)
    
    # Setup test data
    test_docs, test_queries, relevant_docs = setup_test_data()
    
    # Create reranker
    reranker = get_reranker(batch_size=16)
    evaluator = RerankingEvaluator(reranker)
    
    # Create document lists for each query (simulate retrieval results)
    document_lists = [test_docs[:6] for _ in test_queries]  # Top 6 docs for each query
    
    # Run benchmark
    logger.info(f"Running benchmark with {len(test_queries)} queries...")
    results = evaluator.benchmark_multiple_queries(
        queries=test_queries,
        document_lists=document_lists,
        relevant_doc_sets=relevant_docs
    )
    
    # Print results
    print_benchmark_results(results)


def example_3_timeout_and_fallback():
    """Example 3: Timeout and fallback handling."""
    print("=" * 80)
    print("EXAMPLE 3: TIMEOUT AND FALLBACK HANDLING")
    print("=" * 80)
    
    test_docs, test_queries, _ = setup_test_data()
    query = test_queries[0]
    
    # Create reranker with very short timeout to trigger fallback
    config = RerankerConfig(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        timeout_seconds=0.001,  # Very short timeout
        fallback_enabled=True,
        fallback_strategy="score_descending"
    )
    reranker = get_reranker(**config.__dict__)
    
    print(f"Testing with timeout: {config.timeout_seconds}s")
    print(f"Fallback strategy: {config.fallback_strategy}")
    
    # This should trigger fallback due to timeout
    result = reranker.rerank(query, test_docs)
    
    print(f"\\nResult:")
    print(f"Fallback used: {result.fallback_used}")
    print(f"Model: {result.model_used}")
    print(f"Execution time: {result.execution_time:.3f}s")
    print(f"Documents returned: {len(result.documents)}")
    
    print()


def example_4_health_check():
    """Example 4: Health check and model information."""
    print("=" * 80)
    print("EXAMPLE 4: HEALTH CHECK")
    print("=" * 80)
    
    reranker = get_reranker()
    health = reranker.health_check()
    
    print("Reranker Health Status:")
    print("-" * 30)
    for key, value in health.items():
        print(f"{key}: {value}")
    
    print()


def main():
    """Run all examples."""
    logger.info("Starting cross-encoder re-ranking demo")
    
    try:
        example_1_basic_reranking()
        example_2_batch_reranking() 
        example_3_timeout_and_fallback()
        example_4_health_check()
        
        print("✓ All examples completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"✗ Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()