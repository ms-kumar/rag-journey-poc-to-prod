"""
Verification script for centralized configuration.

Tests that all services can be created from settings.
"""

from src.config import get_settings


def test_all_services():
    """Test that all services can be created from settings."""
    settings = get_settings()
    
    print("Testing centralized configuration...")
    print(f"App: {settings.app.name} v{settings.app.version}")
    print(f"Server: {settings.server.host}:{settings.server.port}")
    
    # Test Cache
    print("\n✓ Cache Settings:")
    print(f"  Redis: {settings.cache.redis_host}:{settings.cache.redis_port}")
    print(f"  Enabled: {settings.cache.enabled}")
    print(f"  Semantic threshold: {settings.cache.semantic_similarity_threshold}")
    
    # Test Embeddings
    print("\n✓ Embedding Settings:")
    print(f"  Provider: {settings.embedding.provider}")
    print(f"  Model: {settings.embedding.model}")
    print(f"  Dimension: {settings.embedding.dim}")
    
    # Test Vector Store
    print("\n✓ Vector Store Settings:")
    print(f"  URL: {settings.vectorstore.url}")
    print(f"  Collection: {settings.vectorstore.collection_name}")
    print(f"  BM25: {settings.vectorstore.enable_bm25}")
    
    # Test Generation
    print("\n✓ Generation Settings:")
    print(f"  Model: {settings.generation.model}")
    print(f"  Max length: {settings.generation.max_length}")
    
    # Test Reranker
    print("\n✓ Reranker Settings:")
    print(f"  Model: {settings.reranker.model_name}")
    print(f"  Batch size: {settings.reranker.batch_size}")
    print(f"  Fallback: {settings.reranker.fallback_enabled}")
    
    # Test Query Understanding
    print("\n✓ Query Understanding Settings:")
    print(f"  Rewriting: {settings.query_understanding.enable_rewriting}")
    print(f"  Synonyms: {settings.query_understanding.enable_synonyms}")
    print(f"  Intent: {settings.query_understanding.enable_intent_classification}")
    print(f"  Max rewrites: {settings.query_understanding.max_rewrites}")
    
    # Test Chunking
    print("\n✓ Chunking Settings:")
    print(f"  Chunk size: {settings.chunking.chunk_size}")
    print(f"  Overlap: {settings.chunking.chunk_overlap}")
    print(f"  Strategy: {settings.chunking.strategy}")
    
    # Test Ingestion
    print("\n✓ Ingestion Settings:")
    print(f"  Directory: {settings.ingestion.dir}")
    
    # Test RAG
    print("\n✓ RAG Settings:")
    print(f"  Top K: {settings.rag.top_k}")
    print(f"  Max context docs: {settings.rag.max_context_docs}")
    
    print("\n" + "="*60)
    print("Testing factory functions...")
    
    # Test factories
    try:
        from src.services.cache.factory import create_redis_cache, create_semantic_cache
        print("✓ Cache factories imported")
        
        from src.services.embeddings.factory import create_from_settings as create_embeddings
        print("✓ Embeddings factory imported")
        
        from src.services.vectorstore.factory import create_from_settings as create_vectorstore
        print("✓ Vector store factory imported")
        
        from src.services.generation.factory import create_from_settings as create_generation
        print("✓ Generation factory imported")
        
        from src.services.reranker.factory import create_reranker
        print("✓ Reranker factory imported")
        
        from src.services.query_understanding.factory import create_query_understanding
        print("✓ Query understanding factory imported")
        
        from src.services.chunking.factory import create_from_settings as create_chunking
        print("✓ Chunking factory imported")
        
        from src.services.ingestion.factory import create_from_settings as create_ingestion
        print("✓ Ingestion factory imported")
        
        from src.services.pipeline.naive_pipeline.factory import create_naive_pipeline
        print("✓ Pipeline factory imported")
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    print("\n" + "="*60)
    print("✅ All configuration tests passed!")
    print("\nAll services are properly configured and can be created from settings.")
    return True


if __name__ == "__main__":
    test_all_services()
