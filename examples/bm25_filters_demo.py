"""
Example usage of BM25 query builder and filters.

This script demonstrates the various search types and filtering capabilities.
"""

import logging

from src.config import get_settings, VectorStoreSettings
from src.services.embeddings.factory import get_embed_client, get_langchain_embeddings_adapter
from src.services.vectorstore.client import QdrantVectorStoreClient
from src.services.vectorstore.filters import FilterBuilder, build_filter_from_dict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_vectorstore(enable_bm25: bool = True) -> QdrantVectorStoreClient:
    """Initialize vectorstore with BM25 support."""
    settings = get_settings()

    # Get embeddings
    embed_client = get_embed_client(dim=settings.embedding.dim)
    embeddings = get_langchain_embeddings_adapter(embed_client)

    # Configure vectorstore with BM25
    vectorstore_settings = VectorStoreSettings(
        url=settings.vectorstore.url or "http://localhost:6333",
        collection_name="demo_collection",
        vector_size=settings.embedding.dim,
        enable_bm25=enable_bm25,
    )

    return QdrantVectorStoreClient(embeddings, vectorstore_settings)


def example_1_basic_vector_search():
    """Example 1: Basic semantic vector search."""
    logger.info("\n=== Example 1: Basic Vector Search ===")

    vectorstore = setup_vectorstore(enable_bm25=False)

    # Add sample documents
    texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing enables computers to understand text.",
        "Computer vision allows machines to interpret visual information.",
    ]
    metadatas = [
        {"category": "AI", "year": 2020, "source": "textbook.pdf"},
        {"category": "DL", "year": 2021, "source": "paper.pdf"},
        {"category": "NLP", "year": 2022, "source": "article.md"},
        {"category": "CV", "year": 2023, "source": "blog.txt"},
    ]

    vectorstore.add_texts(texts, metadatas)

    # Perform vector search
    query = "How do neural networks work?"
    results = vectorstore.similarity_search(query, k=2)

    logger.info(f"Query: {query}")
    for i, doc in enumerate(results, 1):
        logger.info(f"Result {i}: {doc.page_content[:80]}... (score: {doc.metadata['score']:.3f})")


def example_2_filtered_search():
    """Example 2: Vector search with metadata filters."""
    logger.info("\n=== Example 2: Filtered Search ===")

    vectorstore = setup_vectorstore(enable_bm25=False)

    # Add documents with rich metadata
    texts = [
        "GPT-3 is a large language model with 175 billion parameters.",
        "BERT revolutionized NLP with bidirectional transformers.",
        "ResNet introduced skip connections for deep networks.",
        "YOLO is a real-time object detection system.",
        "AlphaGo defeated world champions using reinforcement learning.",
    ]
    metadatas = [
        {"category": "NLP", "year": 2020, "author": "OpenAI", "citations": 1500},
        {"category": "NLP", "year": 2018, "author": "Google", "citations": 5000},
        {"category": "CV", "year": 2015, "author": "MSR", "citations": 3000},
        {"category": "CV", "year": 2016, "author": "Redmon", "citations": 2000},
        {"category": "RL", "year": 2016, "author": "DeepMind", "citations": 1000},
    ]

    vectorstore.add_texts(texts, metadatas)

    # Search with simple filter (dict format)
    logger.info("\n--- Filter: NLP papers only ---")
    query = "language models"
    results = vectorstore.similarity_search_with_filter(query, k=3, filter_dict={"category": "NLP"})

    for i, doc in enumerate(results, 1):
        logger.info(f"Result {i}: {doc.page_content[:60]}...")

    # Search with range filter
    logger.info("\n--- Filter: Recent papers (2016+) with many citations ---")
    results = vectorstore.similarity_search_with_filter(
        query="computer vision",
        k=3,
        filter_dict={
            "year$gte": 2016,
            "citations$gte": 2000,
        },
    )

    for i, doc in enumerate(results, 1):
        logger.info(
            f"Result {i}: {doc.page_content[:60]}... "
            f"(year: {doc.metadata['year']}, citations: {doc.metadata['citations']})"
        )


def example_3_complex_filters():
    """Example 3: Complex filters using FilterBuilder."""
    logger.info("\n=== Example 3: Complex Filters ===")

    vectorstore = setup_vectorstore(enable_bm25=False)

    # Build complex filter using FilterBuilder
    filter_obj = (
        FilterBuilder()
        .match_any("category", ["NLP", "CV"])  # NLP or CV papers
        .range("year", gte=2015, lte=2020)  # Published 2015-2020
        .range("citations", gte=1000)  # At least 1000 citations
        .must_not("author", "Unknown")  # Exclude unknown authors
        .build()
    )

    logger.info("Filter: (category=NLP or CV) AND (year 2015-2020) AND (citations≥1000)")

    # This would work with actual documents
    logger.info("Filter built successfully!")


def example_4_bm25_search():
    """Example 4: BM25 keyword search."""
    logger.info("\n=== Example 4: BM25 Search ===")

    vectorstore = setup_vectorstore(enable_bm25=True)

    # Add documents
    texts = [
        "Python is a high-level programming language with dynamic typing.",
        "TensorFlow is a machine learning framework developed by Google.",
        "PyTorch provides tensor computation and automatic differentiation.",
        "NumPy is the fundamental package for scientific computing in Python.",
        "Pandas offers data structures for data analysis in Python.",
    ]
    metadatas = [
        {"type": "language", "popularity": "high"},
        {"type": "framework", "popularity": "high"},
        {"type": "framework", "popularity": "high"},
        {"type": "library", "popularity": "high"},
        {"type": "library", "popularity": "high"},
    ]

    vectorstore.add_texts(texts, metadatas)

    # BM25 search - good for exact keyword matching
    query = "Python framework"
    logger.info(f"\nQuery: '{query}' (BM25)")
    results = vectorstore.bm25_search(query, k=3)

    for i, doc in enumerate(results, 1):
        logger.info(f"Result {i}: {doc.page_content[:70]}...")

    # BM25 with filters
    logger.info("\n--- BM25 with filter: frameworks only ---")
    results = vectorstore.bm25_search(
        query="machine learning", k=3, filter_dict={"type": "framework"}
    )

    for i, doc in enumerate(results, 1):
        logger.info(f"Result {i}: {doc.page_content[:70]}...")


def example_5_hybrid_search():
    """Example 5: Hybrid search combining vector + BM25."""
    logger.info("\n=== Example 5: Hybrid Search ===")

    vectorstore = setup_vectorstore(enable_bm25=True)

    # Add documents
    texts = [
        "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "Attention Is All You Need - introduces the Transformer architecture",
        "GPT-3: Language Models are Few-Shot Learners with 175B parameters",
        "ImageNet Classification with Deep Convolutional Neural Networks (AlexNet)",
        "Generative Adversarial Networks (GANs) for image synthesis",
    ]
    metadatas = [
        {"topic": "NLP", "year": 2018, "model": "BERT"},
        {"topic": "NLP", "year": 2017, "model": "Transformer"},
        {"topic": "NLP", "year": 2020, "model": "GPT-3"},
        {"topic": "CV", "year": 2012, "model": "AlexNet"},
        {"topic": "CV", "year": 2014, "model": "GAN"},
    ]

    vectorstore.add_texts(texts, metadatas)

    query = "transformer language models"

    # Compare different alpha values
    logger.info(f"\nQuery: '{query}'")

    for alpha in [0.3, 0.5, 0.7]:
        logger.info(f"\n--- Hybrid search with alpha={alpha} ---")
        logger.info(f"(alpha: 0.0=BM25 only, 1.0=vector only)")

        results = vectorstore.hybrid_search(
            query=query, k=3, alpha=alpha, filter_dict={"topic": "NLP"}
        )

        for i, doc in enumerate(results, 1):
            logger.info(f"Result {i}: {doc.page_content[:70]}...")


def example_6_dict_filter_syntax():
    """Example 6: Dictionary filter syntax reference."""
    logger.info("\n=== Example 6: Filter Syntax Reference ===")

    # Various filter examples
    filters = {
        "exact_match": {"category": "AI", "status": "published"},
        "range": {"year$gte": 2020, "year$lte": 2023, "score$gt": 0.5},
        "match_any": {"category$in": ["AI", "ML", "DL"]},
        "negation": {"status$not": "deleted"},
        "text_search": {"abstract$text": "neural networks"},
        "complex": {
            "category$in": ["AI", "ML"],
            "year$gte": 2020,
            "citations$gte": 100,
            "status$not": "retracted",
            "venue": "ICML",
        },
    }

    for name, filter_dict in filters.items():
        logger.info(f"\n{name}: {filter_dict}")
        filter_obj = build_filter_from_dict(filter_dict)
        logger.info(f"Built: {filter_obj is not None}")


def main():
    """Run all examples."""
    logger.info("=" * 80)
    logger.info("BM25 Query Builder & Filters Examples")
    logger.info("=" * 80)

    try:
        # Run examples
        example_1_basic_vector_search()
        example_2_filtered_search()
        example_3_complex_filters()
        example_4_bm25_search()
        example_5_hybrid_search()
        example_6_dict_filter_syntax()

        logger.info("\n" + "=" * 80)
        logger.info("All examples completed successfully!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)


if __name__ == "__main__":
    main()
