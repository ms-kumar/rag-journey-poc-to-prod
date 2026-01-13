# Unit Tests - Services

This directory contains unit tests for all service modules.

## Structure

Each service module has its own subdirectory:

- **agent/**: Agent framework tests (tools, nodes, routing, metrics)
- **cache/**: Caching layer tests (Redis, semantic cache, embedding cache)
- **embeddings/**: Embedding generation and provider tests
- **evaluation/**: Evaluation metrics and dataset tests
- **guardrails/**: Safety and guardrails tests (toxicity, PII, adversarial)
- **retrieval/**: Search and retrieval tests (BM25, fusion, reranking)
- **ingestion/**: Document ingestion and chunking tests
- **performance/**: Performance monitoring and optimization tests

## Running Tests

```bash
# Run all service tests
pytest tests/unit/services/

# Run specific service tests
pytest tests/unit/services/agent/
pytest tests/unit/services/cache/

# Run with coverage
pytest tests/unit/services/ --cov=src/services --cov-report=html
```

## Writing Service Tests

Each service test should:
1. Test one specific component or function
2. Use mocks for external dependencies
3. Be fast (< 1 second)
4. Be isolated (no shared state)
5. Follow the AAA pattern (Arrange, Act, Assert)

### Example

```python
import pytest
from unittest.mock import Mock

def test_service_function():
    # Arrange
    mock_dependency = Mock()
    mock_dependency.method.return_value = "expected"
    
    # Act
    result = service_function(mock_dependency)
    
    # Assert
    assert result == "expected"
    mock_dependency.method.assert_called_once()
```
