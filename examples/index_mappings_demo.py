"""
Example demonstrating index mappings for Qdrant.

Shows how to create and manage payload indices for optimized filtering.
"""

import logging

from src.config import get_settings
from src.services.embeddings.factory import get_embed_client, get_langchain_embeddings_adapter
from src.services.vectorstore.client import QdrantVectorStoreClient, VectorStoreConfig
from src.services.vectorstore.index_mappings import IndexMappingBuilder, get_preset_mappings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_vectorstore(collection_name: str = "demo_indices") -> QdrantVectorStoreClient:
    """Initialize vectorstore for index demo."""
    settings = get_settings()

    # Get embeddings
    embed_client = get_embed_client(dim=settings.embedding.dim)
    embeddings = get_langchain_embeddings_adapter(embed_client)

    # Configure vectorstore
    config = VectorStoreConfig(
        qdrant_url=settings.vectorstore.url or "http://localhost:6333",
        collection_name=collection_name,
        vector_size=settings.embedding.dim,
        enable_bm25=True,
    )

    return QdrantVectorStoreClient(embeddings, config)


def example_1_basic_indices():
    """Example 1: Create basic payload indices."""
    logger.info("\n=== Example 1: Basic Indices ===")

    vectorstore = setup_vectorstore("basic_indices")

    # Create simple indices
    mappings = (
        vectorstore.get_index_mapping_builder()
        .add_keyword("category")  # Exact match
        .add_integer("year", range=True)  # Range queries
        .add_float("score", range=True)  # Range queries
        .build()
    )

    logger.info(f"Creating {len(mappings)} indices...")
    results = vectorstore.create_indices_from_mappings(mappings)

    logger.info("Index creation results:")
    for field, success in results.items():
        status = "✓" if success else "✗"
        logger.info(f"  {status} {field}")


def example_2_preset_indices():
    """Example 2: Use predefined preset mappings."""
    logger.info("\n=== Example 2: Preset Indices ===")

    vectorstore = setup_vectorstore("preset_indices")

    # Use document_metadata preset
    logger.info("Using 'document_metadata' preset...")
    mappings = get_preset_mappings("document_metadata")

    logger.info(f"Preset includes {len(mappings)} indices:")
    for mapping in mappings:
        logger.info(f"  - {mapping.field_name} ({mapping.field_type})")

    results = vectorstore.create_indices_from_mappings(mappings)
    logger.info(f"Created {sum(results.values())} indices successfully")


def example_3_complex_indices():
    """Example 3: Create complex index configuration."""
    logger.info("\n=== Example 3: Complex Indices ===")

    vectorstore = setup_vectorstore("complex_indices")

    # Build comprehensive index set
    mappings = (
        IndexMappingBuilder()
        # Keyword indices for exact match
        .add_keyword("source")
        .add_keyword("category")
        .add_keyword("author")
        .add_keyword("status")
        # Integer indices for range queries
        .add_integer("year", range=True, lookup=True)
        .add_integer("citation_count", range=True)
        .add_integer("page_number", range=True)
        .add_integer("chunk_index", range=True)
        # Float indices for scores/ratings
        .add_float("confidence_score", range=True)
        .add_float("relevance_score", range=True)
        # Text index for full-text search
        .add_text("abstract", min_token_len=3, max_token_len=25)
        # Boolean flags
        .add_bool("is_published")
        .add_bool("is_verified")
        # Datetime for timestamps
        .add_datetime("created_at", range=True)
        .add_datetime("updated_at", range=True)
        .build()
    )

    logger.info(f"Creating {len(mappings)} indices...")
    results = vectorstore.create_indices_from_mappings(mappings)

    successful = sum(results.values())
    logger.info(f"Successfully created {successful}/{len(mappings)} indices")


def example_4_list_indices():
    """Example 4: List existing indices."""
    logger.info("\n=== Example 4: List Indices ===")

    vectorstore = setup_vectorstore("complex_indices")

    # List all payload indices
    indices = vectorstore.list_payload_indices()

    logger.info(f"Found {len(indices)} payload indices:")
    for field_name, index_type in indices.items():
        logger.info(f"  - {field_name}: {index_type}")


def example_5_index_management():
    """Example 5: Manage indices (create, list, delete)."""
    logger.info("\n=== Example 5: Index Management ===")

    vectorstore = setup_vectorstore("managed_indices")

    # Create initial indices
    logger.info("Step 1: Creating initial indices...")
    mappings = (
        IndexMappingBuilder()
        .add_keyword("category")
        .add_integer("year", range=True)
        .add_text("description")
        .build()
    )
    vectorstore.create_indices_from_mappings(mappings)

    # List indices
    logger.info("\nStep 2: Listing indices...")
    indices = vectorstore.list_payload_indices()
    logger.info(f"Current indices: {list(indices.keys())}")

    # Add new index
    logger.info("\nStep 3: Adding new index...")
    new_mapping = IndexMappingBuilder().add_float("score", range=True).build()
    vectorstore.create_indices_from_mappings(new_mapping)

    # List again
    updated_indices = vectorstore.list_payload_indices()
    logger.info(f"Updated indices: {list(updated_indices.keys())}")

    # Delete an index
    logger.info("\nStep 4: Deleting 'description' index...")
    success = vectorstore.delete_payload_index("description")
    logger.info(f"Deletion {'succeeded' if success else 'failed'}")

    # Final list
    final_indices = vectorstore.list_payload_indices()
    logger.info(f"Final indices: {list(final_indices.keys())}")


def example_6_with_data():
    """Example 6: Create indices and add data with metadata."""
    logger.info("\n=== Example 6: Indices with Data ===")

    vectorstore = setup_vectorstore("data_with_indices")

    # Create indices first
    logger.info("Creating indices...")
    mappings = (
        IndexMappingBuilder()
        .add_keyword("source")
        .add_keyword("category")
        .add_integer("year", range=True)
        .add_float("confidence", range=True)
        .build()
    )
    vectorstore.create_indices_from_mappings(mappings)

    # Add documents with metadata
    logger.info("Adding documents with metadata...")
    texts = [
        "Machine learning is transforming AI research.",
        "Deep learning uses neural networks with many layers.",
        "Natural language processing enables text understanding.",
    ]

    metadatas = [
        {"source": "paper1.pdf", "category": "ML", "year": 2020, "confidence": 0.95},
        {"source": "paper2.pdf", "category": "DL", "year": 2021, "confidence": 0.88},
        {"source": "paper3.pdf", "category": "NLP", "year": 2022, "confidence": 0.92},
    ]

    ids = vectorstore.add_texts(texts, metadatas)
    logger.info(f"Added {len(ids)} documents")

    # Now filters will be fast!
    logger.info("\nQuerying with filters...")
    docs = vectorstore.similarity_search_with_filter(
        query="neural networks", k=5, filter_dict={"year$gte": 2021}
    )
    logger.info(f"Found {len(docs)} documents with year >= 2021")


def example_7_research_paper_indices():
    """Example 7: Complete research paper index setup."""
    logger.info("\n=== Example 7: Research Paper Indices ===")

    vectorstore = setup_vectorstore("research_papers")

    # Use research paper preset
    logger.info("Setting up research paper indices...")
    mappings = get_preset_mappings("research_paper")

    logger.info("Preset includes:")
    for mapping in mappings:
        options = []
        if mapping.field_type in ["integer", "float"] and mapping.range:
            options.append("range")
        if mapping.field_type == "text":
            options.append(f"tokenizer={mapping.tokenizer}")
        opts_str = f" ({', '.join(options)})" if options else ""
        logger.info(f"  - {mapping.field_name}: {mapping.field_type}{opts_str}")

    results = vectorstore.create_indices_from_mappings(mappings)
    logger.info(f"\nCreated {sum(results.values())} indices")

    # Add sample papers
    papers = [
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "Attention Is All You Need - The Transformer Architecture",
        "GPT-3: Language Models are Few-Shot Learners",
    ]

    metadata_list = [
        {
            "title": papers[0],
            "authors": "Devlin et al.",
            "venue": "NAACL",
            "year": 2018,
            "citations": 5000,
            "category": "NLP",
        },
        {
            "title": papers[1],
            "authors": "Vaswani et al.",
            "venue": "NeurIPS",
            "year": 2017,
            "citations": 8000,
            "category": "NLP",
        },
        {
            "title": papers[2],
            "authors": "Brown et al.",
            "venue": "NeurIPS",
            "year": 2020,
            "citations": 3000,
            "category": "NLP",
        },
    ]

    vectorstore.add_texts(papers, metadata_list)
    logger.info(f"Added {len(papers)} research papers")


def main():
    """Run all examples."""
    logger.info("=" * 80)
    logger.info("Index Mappings Examples")
    logger.info("=" * 80)

    try:
        # Run examples
        example_1_basic_indices()
        example_2_preset_indices()
        example_3_complex_indices()
        example_4_list_indices()
        example_5_index_management()
        example_6_with_data()
        example_7_research_paper_indices()

        logger.info("\n" + "=" * 80)
        logger.info("All examples completed successfully!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)


if __name__ == "__main__":
    main()
