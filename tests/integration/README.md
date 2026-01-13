# Integration Tests

This directory contains integration tests that verify components work together correctly.

## Purpose

Integration tests differ from unit tests in that they:
- Test multiple components together
- May use real external dependencies (Redis, databases, APIs)
- Take longer to run
- Verify end-to-end workflows

## Structure

```
integration/
├── test_rag_pipeline.py          # Full RAG pipeline tests
├── test_cache_integration.py     # Cache with embeddings/retrieval
├── test_agent_workflows.py       # Agent with tools integration
└── test_api_endpoints.py         # API endpoint tests
```

## Running Integration Tests

```bash
# Run all integration tests
pytest tests/integration/

# Run with specific markers
pytest tests/integration/ -m "not slow"

# Run specific test file
pytest tests/integration/test_rag_pipeline.py
```

## Requirements

Some integration tests may require:
- Redis running on localhost:6379
- Qdrant running on localhost:6333
- API keys in environment variables
- Docker containers for external services

### Setup

```bash
# Start required services with Docker
docker-compose up -d redis qdrant

# Set environment variables
export OPENAI_API_KEY="your-key"
export REDIS_URL="redis://localhost:6379"
```

## Writing Integration Tests

Mark integration tests appropriately:

```python
import pytest

@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_rag_pipeline():
    """Test complete RAG pipeline with real components."""
    # Setup
    embedder = RealEmbedder()
    vector_store = RealVectorStore()
    
    # Test workflow
    docs = await ingest_documents(embedder, vector_store)
    results = await retrieve_documents(query, embedder, vector_store)
    
    # Verify
    assert len(results) > 0
    assert results[0]["score"] > 0.7
```

## Best Practices

1. **Cleanup**: Always clean up resources after tests
2. **Isolation**: Each test should be independent
3. **Timeouts**: Use timeouts to prevent hanging tests
4. **Markers**: Use markers to categorize tests (slow, requires_redis, etc.)
5. **Documentation**: Document required setup and dependencies
