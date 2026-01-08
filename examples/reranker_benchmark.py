"""
Benchmark script for evaluating cross-encoder re-ranking performance.

Measures:
- Precision@k improvements over baseline retrieval
- Latency overhead from re-ranking
- Effect on different search types (vector, BM25, hybrid)
- Timeout and fallback behavior
"""

import json
import time
from pathlib import Path
from typing import Any

from src.config import get_settings
from src.services.reranker.evaluation import RerankingEvaluator, BenchmarkConfig, print_benchmark_results
from src.services.reranker.factory import get_reranker
from src.services.pipeline.naive_pipeline.client import NaivePipeline, NaivePipelineConfig
from langchain_core.documents import Document


def load_benchmark_data() -> tuple[list[str], list[list[Document]], list[set[str]]]:
    """Load benchmark queries and relevance judgments."""
    
    # Create synthetic benchmark data for demonstration
    # In practice, you would load from files like TREC, MS MARCO, etc.
    
    queries = [
        "What is machine learning and how does it work?",
        "Explain deep learning and neural networks",
        "What are different types of machine learning algorithms?", 
        "How does natural language processing work?",
        "What is reinforcement learning?",
        "Explain supervised vs unsupervised learning",
        "What are neural networks and how do they work?",
        "How do recommendation systems work?",
        "What is computer vision and its applications?",
        "Explain the difference between AI and machine learning"
    ]
    
    # Sample documents for each query (simulating retrieval results)
    base_documents = [
        Document(
            page_content="Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
            metadata={"id": "ml_basics_1", "topic": "ml_fundamentals"}
        ),
        Document(
            page_content="Deep learning is a machine learning technique that uses artificial neural networks with multiple layers to model and understand complex patterns in data.",
            metadata={"id": "dl_basics_1", "topic": "deep_learning"}
        ),
        Document(
            page_content="Supervised learning uses labeled training data to learn a mapping from inputs to outputs. Examples include classification and regression.",
            metadata={"id": "supervised_1", "topic": "ml_types"}
        ),
        Document(
            page_content="Unsupervised learning finds patterns in data without labeled examples. Common techniques include clustering and dimensionality reduction.",
            metadata={"id": "unsupervised_1", "topic": "ml_types"}
        ),
        Document(
            page_content="Natural language processing (NLP) is a branch of AI that helps computers understand, interpret and manipulate human language.",
            metadata={"id": "nlp_basics_1", "topic": "nlp"}
        ),
        Document(
            page_content="Reinforcement learning is where an agent learns to make decisions by taking actions in an environment and receiving rewards or penalties.",
            metadata={"id": "rl_basics_1", "topic": "reinforcement_learning"}
        ),
        Document(
            page_content="Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes that process information.",
            metadata={"id": "nn_basics_1", "topic": "neural_networks"}
        ),
        Document(
            page_content="Recommendation systems analyze user behavior and preferences to suggest relevant items or content.",
            metadata={"id": "recsys_1", "topic": "recommendation_systems"}
        ),
        Document(
            page_content="Computer vision enables computers to interpret and understand visual information from images and videos.",
            metadata={"id": "cv_basics_1", "topic": "computer_vision"}
        ),
        Document(
            page_content="Artificial intelligence is the broader concept of machines being able to carry out tasks in a smart way, while machine learning is a subset of AI.",
            metadata={"id": "ai_vs_ml_1", "topic": "ai_fundamentals"}
        ),
        # Add some noise documents
        Document(
            page_content="The weather today is sunny with a temperature of 25 degrees Celsius.",
            metadata={"id": "weather_1", "topic": "weather"}
        ),
        Document(
            page_content="Python is a high-level programming language known for its simplicity and readability.",
            metadata={"id": "python_1", "topic": "programming"}
        ),
        Document(
            page_content="Database systems store and organize data for efficient retrieval and management.",
            metadata={"id": "database_1", "topic": "databases"}
        ),
    ]
    
    # Create document lists for each query (simulating different retrieval results)
    import random
    random.seed(42)  # For reproducible results
    
    document_lists = []
    relevant_sets = []
    
    for i, query in enumerate(queries):
        # Create a realistic mix of relevant and irrelevant documents
        query_docs = base_documents.copy()
        random.shuffle(query_docs)
        
        # Take first 10 documents as retrieval results
        query_doc_list = query_docs[:10]
        document_lists.append(query_doc_list)
        
        # Define relevant documents based on query topic
        if "machine learning" in query.lower():
            relevant_ids = {"ml_basics_1", "supervised_1", "unsupervised_1", "ai_vs_ml_1"}
        elif "deep learning" in query.lower() or "neural" in query.lower():
            relevant_ids = {"dl_basics_1", "nn_basics_1"}
        elif "types" in query.lower() and "algorithm" in query.lower():
            relevant_ids = {"supervised_1", "unsupervised_1", "rl_basics_1"}
        elif "natural language" in query.lower():
            relevant_ids = {"nlp_basics_1"}
        elif "reinforcement" in query.lower():
            relevant_ids = {"rl_basics_1"}
        elif "supervised" in query.lower() or "unsupervised" in query.lower():
            relevant_ids = {"supervised_1", "unsupervised_1"}
        elif "recommendation" in query.lower():
            relevant_ids = {"recsys_1"}
        elif "computer vision" in query.lower():
            relevant_ids = {"cv_basics_1"}
        elif "ai" in query.lower() and "machine learning" in query.lower():
            relevant_ids = {"ai_vs_ml_1", "ml_basics_1"}
        else:
            relevant_ids = {"ml_basics_1"}  # Default
            
        relevant_sets.append(relevant_ids)
    
    return queries, document_lists, relevant_sets


def benchmark_reranking_precision():
    """Benchmark re-ranking precision improvements."""
    print("=" * 80)
    print("CROSS-ENCODER RE-RANKING PRECISION BENCHMARK")
    print("=" * 80)
    
    # Load benchmark data
    queries, document_lists, relevant_sets = load_benchmark_data()
    
    # Initialize reranker
    print("Initializing cross-encoder re-ranker...")
    reranker = get_reranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size=16,
        timeout_seconds=30.0
    )
    
    # Run evaluation
    evaluator = RerankingEvaluator(reranker)
    
    config = BenchmarkConfig(
        k_values=[1, 3, 5, 10],
        include_latency=True,
        include_precision=True
    )
    
    print(f"Running benchmark with {len(queries)} queries...")
    results = evaluator.benchmark_multiple_queries(
        queries=queries,
        document_lists=document_lists,
        relevant_doc_sets=relevant_sets,
        config=config
    )
    
    # Print results
    print_benchmark_results(results)
    
    return results


def benchmark_search_type_comparison():
    """Compare re-ranking effectiveness across different search types."""
    print("=" * 80)
    print("SEARCH TYPE COMPARISON WITH RE-RANKING")
    print("=" * 80)
    
    # This would require a full pipeline with vectorstore
    # For now, show conceptual structure
    
    search_types = ["vector", "bm25", "hybrid", "sparse"]
    
    print("Note: This benchmark requires a full pipeline setup with vectorstore.")
    print("Search types to compare:")
    for search_type in search_types:
        print(f"  - {search_type}: with and without re-ranking")
    
    print("\\nExpected results:")
    print("- Re-ranking should improve precision@k for all search types")
    print("- Hybrid search + re-ranking should show best performance") 
    print("- BM25 + re-ranking might show largest improvement due to semantic gap")


def benchmark_latency_overhead():
    """Benchmark latency overhead from re-ranking."""
    print("=" * 80)
    print("RE-RANKING LATENCY OVERHEAD BENCHMARK")
    print("=" * 80)
    
    queries, document_lists, _ = load_benchmark_data()
    
    # Test different batch sizes
    batch_sizes = [8, 16, 32, 64]
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\\nTesting batch size: {batch_size}")
        
        reranker = get_reranker(
            batch_size=batch_size,
            timeout_seconds=30.0
        )
        
        # Measure latency for different document counts
        doc_counts = [5, 10, 20, 50]
        latencies = []
        
        for doc_count in doc_counts:
            if doc_count <= len(document_lists[0]):
                test_docs = document_lists[0][:doc_count]
                
                # Warm up
                reranker.rerank(queries[0], test_docs)
                
                # Measure
                start_time = time.time()
                result = reranker.rerank(queries[0], test_docs)
                latency = time.time() - start_time
                
                latencies.append(latency * 1000)  # Convert to ms
                print(f"  {doc_count} docs: {latency*1000:.2f}ms")
        
        results[batch_size] = {
            'doc_counts': doc_counts[:len(latencies)],
            'latencies': latencies
        }
    
    print("\\nSUMMARY:")
    print("Batch Size\\tAvg Latency (ms)\\tThroughput (docs/s)")
    print("-" * 50)
    
    for batch_size, data in results.items():
        avg_latency = sum(data['latencies']) / len(data['latencies'])
        avg_docs = sum(data['doc_counts']) / len(data['doc_counts'])
        throughput = avg_docs / (avg_latency / 1000) if avg_latency > 0 else 0
        print(f"{batch_size}\\t\\t{avg_latency:.2f}\\t\\t\\t{throughput:.1f}")


def benchmark_timeout_fallback():
    """Benchmark timeout and fallback behavior."""
    print("=" * 80) 
    print("TIMEOUT AND FALLBACK BENCHMARK")
    print("=" * 80)
    
    queries, document_lists, _ = load_benchmark_data()
    
    # Test different timeout values
    timeout_values = [0.1, 1.0, 5.0, 15.0]
    
    for timeout in timeout_values:
        print(f"\\nTesting timeout: {timeout}s")
        
        reranker = get_reranker(
            timeout_seconds=timeout,
            fallback_enabled=True,
            fallback_strategy="score_descending"
        )
        
        test_docs = document_lists[0][:20]  # Use more docs to increase processing time
        
        start_time = time.time()
        result = reranker.rerank(queries[0], test_docs)
        actual_time = time.time() - start_time
        
        print(f"  Actual time: {actual_time:.3f}s")
        print(f"  Fallback used: {result.fallback_used}")
        print(f"  Documents returned: {len(result.documents)}")
        
        if result.fallback_used:
            print(f"  Fallback triggered after {actual_time:.3f}s")


def save_results(results: dict[str, Any], filename: str = "reranking_benchmark_results.json"):
    """Save benchmark results to file."""
    output_file = Path(filename)
    
    # Convert non-serializable objects
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, (str, int, float, bool, list, dict, type(None))):
            serializable_results[key] = value
        else:
            serializable_results[key] = str(value)
    
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to: {output_file.absolute()}")


def main():
    """Run all benchmarks."""
    print("Starting comprehensive cross-encoder re-ranking benchmarks...")
    print("This may take several minutes depending on model loading and computation.")
    
    try:
        # Run precision benchmark
        precision_results = benchmark_reranking_precision()
        
        # Run latency benchmark
        benchmark_latency_overhead()
        
        # Run timeout benchmark  
        benchmark_timeout_fallback()
        
        # Show search type comparison structure
        benchmark_search_type_comparison()
        
        # Save results
        save_results(precision_results)
        
        print("\\n" + "="*80)
        print("✓ ALL BENCHMARKS COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        # Summary
        print("\\nKEY FINDINGS:")
        avg_improvement = precision_results.get('avg_improvement', {})
        for k, improvement in avg_improvement.items():
            print(f"  - Precision@{k} improvement: {improvement:+.3f}")
        
        avg_latency = precision_results.get('avg_latency_ms', 0)
        if avg_latency:
            print(f"  - Average re-ranking latency: {avg_latency:.2f}ms")
        
    except Exception as e:
        print(f"\\n✗ Benchmark failed: {e}")
        raise


if __name__ == "__main__":
    main()