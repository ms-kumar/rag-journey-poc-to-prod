"""
Script to create and manage evaluation datasets.

Generates default evaluation datasets with queries and relevance judgments.
"""

import logging
from pathlib import Path

from src.services.evaluation.dataset import DatasetBuilder, EvalDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_default_dataset() -> EvalDataset:
    """
    Create default evaluation dataset with RAG-related queries.

    Returns:
        EvalDataset with evaluation examples
    """
    builder = DatasetBuilder(
        name="rag_default_eval",
        description="Default evaluation dataset for RAG system covering core topics",
    )

    # Add RAG basics queries
    builder.dataset.add_example(
        query="What is RAG?",
        relevant_doc_ids=["doc_rag_basics", "doc_rag_intro"],
        expected_answer="RAG (Retrieval-Augmented Generation) is a technique that combines information retrieval with language model generation.",
        metadata={"category": "rag_basics", "difficulty": "easy"},
    )

    builder.dataset.add_example(
        query="How does retrieval augmented generation work?",
        relevant_doc_ids=["doc_rag_basics", "doc_rag_architecture"],
        expected_answer="RAG works by first retrieving relevant documents using vector search, then using those documents as context for a language model to generate answers.",
        metadata={"category": "rag_basics", "difficulty": "medium"},
    )

    # Embeddings queries
    builder.dataset.add_example(
        query="What are vector embeddings?",
        relevant_doc_ids=["doc_embeddings", "doc_vector_search"],
        expected_answer="Vector embeddings are numerical representations of text that capture semantic meaning in high-dimensional space.",
        metadata={"category": "embeddings", "difficulty": "easy"},
    )

    builder.dataset.add_example(
        query="How to create embeddings for text?",
        relevant_doc_ids=["doc_embeddings", "doc_sentence_transformers"],
        expected_answer="Text embeddings can be created using models like sentence-transformers, OpenAI embeddings, or Cohere embeddings.",
        metadata={"category": "embeddings", "difficulty": "medium"},
    )

    # BM25 and sparse retrieval queries
    builder.dataset.add_example(
        query="What is BM25?",
        relevant_doc_ids=["doc_bm25", "doc_sparse_retrieval"],
        expected_answer="BM25 is a probabilistic ranking function used in information retrieval for sparse keyword-based search.",
        metadata={"category": "retrieval", "difficulty": "medium"},
    )

    builder.dataset.add_example(
        query="How to implement BM25 search?",
        relevant_doc_ids=["doc_bm25_implementation", "doc_qdrant_sparse"],
        metadata={"category": "retrieval", "difficulty": "hard"},
    )

    # Qdrant queries
    builder.dataset.add_example(
        query="How to use Qdrant vector database?",
        relevant_doc_ids=["doc_qdrant_notes", "doc_qdrant_basics"],
        metadata={"category": "vectordb", "difficulty": "medium"},
    )

    builder.dataset.add_example(
        query="What are Qdrant filters?",
        relevant_doc_ids=["doc_qdrant_filters", "doc_metadata_filters"],
        metadata={"category": "vectordb", "difficulty": "medium"},
    )

    # FastAPI queries
    builder.dataset.add_example(
        query="How to build a REST API with FastAPI?",
        relevant_doc_ids=["doc_python_fastapi", "doc_fastapi_basics"],
        metadata={"category": "api", "difficulty": "medium"},
    )

    builder.dataset.add_example(
        query="FastAPI async endpoints best practices",
        relevant_doc_ids=["doc_python_fastapi", "doc_async_programming"],
        metadata={"category": "api", "difficulty": "hard"},
    )

    # Python ML queries
    builder.dataset.add_example(
        query="What is machine learning?",
        relevant_doc_ids=["doc_ml_short", "doc_ml_basics"],
        metadata={"category": "ml", "difficulty": "easy"},
    )

    builder.dataset.add_example(
        query="How to train a neural network?",
        relevant_doc_ids=["doc_ml_training", "doc_pytorch_basics"],
        metadata={"category": "ml", "difficulty": "hard"},
    )

    # Reranking queries
    builder.dataset.add_example(
        query="What is cross-encoder reranking?",
        relevant_doc_ids=["doc_reranking", "doc_cross_encoder"],
        metadata={"category": "reranking", "difficulty": "medium"},
    )

    builder.dataset.add_example(
        query="How to improve retrieval with reranking?",
        relevant_doc_ids=["doc_reranking", "doc_retrieval_optimization"],
        metadata={"category": "reranking", "difficulty": "hard"},
    )

    # Query understanding queries
    builder.dataset.add_example(
        query="What is query expansion?",
        relevant_doc_ids=["doc_query_understanding", "doc_query_expansion"],
        metadata={"category": "query_processing", "difficulty": "medium"},
    )

    builder.dataset.add_example(
        query="How to handle misspellings in search queries?",
        relevant_doc_ids=["doc_query_understanding", "doc_spell_correction"],
        metadata={"category": "query_processing", "difficulty": "medium"},
    )

    # Chunking queries
    builder.dataset.add_example(
        query="What is text chunking?",
        relevant_doc_ids=["doc_chunking", "doc_text_splitting"],
        metadata={"category": "preprocessing", "difficulty": "easy"},
    )

    builder.dataset.add_example(
        query="Best practices for chunking documents?",
        relevant_doc_ids=["doc_chunking_strategies", "doc_overlap"],
        metadata={"category": "preprocessing", "difficulty": "hard"},
    )

    # Caching queries
    builder.dataset.add_example(
        query="How to cache embeddings?",
        relevant_doc_ids=["doc_embedding_cache", "doc_caching"],
        metadata={"category": "optimization", "difficulty": "medium"},
    )

    # Performance queries
    builder.dataset.add_example(
        query="How to optimize RAG latency?",
        relevant_doc_ids=["doc_performance", "doc_optimization"],
        metadata={"category": "performance", "difficulty": "hard"},
    )

    builder.dataset.add_example(
        query="What causes high retrieval latency?",
        relevant_doc_ids=["doc_performance", "doc_latency_optimization"],
        metadata={"category": "performance", "difficulty": "medium"},
    )

    logger.info(f"Created dataset with {len(builder.dataset)} examples")
    return builder.dataset


def create_small_test_dataset() -> EvalDataset:
    """
    Create small test dataset for quick evaluation.

    Returns:
        Small EvalDataset for testing
    """
    builder = DatasetBuilder(
        name="rag_test_small",
        description="Small test dataset for quick evaluation (5 queries)",
    )

    builder.dataset.add_example(
        query="What is RAG?",
        relevant_doc_ids=["doc_rag_basics"],
        metadata={"category": "rag_basics"},
    )

    builder.dataset.add_example(
        query="How to use embeddings?",
        relevant_doc_ids=["doc_embeddings"],
        metadata={"category": "embeddings"},
    )

    builder.dataset.add_example(
        query="What is BM25?",
        relevant_doc_ids=["doc_bm25"],
        metadata={"category": "retrieval"},
    )

    builder.dataset.add_example(
        query="How to build FastAPI app?",
        relevant_doc_ids=["doc_python_fastapi"],
        metadata={"category": "api"},
    )

    builder.dataset.add_example(
        query="Explain machine learning",
        relevant_doc_ids=["doc_ml_short"],
        metadata={"category": "ml"},
    )

    return builder.dataset


def main():
    """Main function to create and save evaluation datasets."""
    # Create eval data directory
    eval_dir = Path("data/eval")
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Create and save default dataset
    default_dataset = create_default_dataset()
    default_dataset.save(eval_dir / "rag_default_eval.json")

    # Create and save small test dataset
    small_dataset = create_small_test_dataset()
    small_dataset.save(eval_dir / "rag_test_small.json")

    # Print statistics
    print("\n=== Dataset Statistics ===")
    print("\nDefault Dataset:")
    print(f"  File: {eval_dir / 'rag_default_eval.json'}")
    for key, value in default_dataset.get_statistics().items():
        print(f"  {key}: {value}")

    print("\nSmall Test Dataset:")
    print(f"  File: {eval_dir / 'rag_test_small.json'}")
    for key, value in small_dataset.get_statistics().items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
