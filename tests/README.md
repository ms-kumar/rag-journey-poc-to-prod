# Test Suite Documentation

This directory contains the comprehensive test suite for the RAG Journey POC to Production project.

## 🎯 Quick Overview

Your tests are now organized in a **professional, scalable structure** that mirrors your source code:

- **1012 tests** passing across 8 organized modules
- **Modular fixtures** - each module has its own test helpers
- **Fast execution** - run specific modules or all tests
- **Production-ready** - follows industry best practices

## 📁 Directory Structure

```
tests/
├── conftest.py                    # Global fixtures & configuration
├── README.md                      # This file
├── setup.sh                       # Interactive helper script
│
├── unit/                          # Unit tests (isolated, fast)
│   ├── services/
│   │   ├── agent/                # Agent framework tests (6 tests)
│   │   │   ├── conftest.py       # Agent-specific fixtures
│   │   │   └── test_*.py         # Agent test files
│   │   ├── cache/                # Caching tests (5 tests)
│   │   │   ├── conftest.py       # Cache-specific fixtures
│   │   │   └── test_*.py         # Cache test files
│   │   ├── embeddings/           # Embedding tests (2 tests)
│   │   ├── evaluation/           # Evaluation tests (3 tests)
│   │   ├── guardrails/           # Safety tests (6 tests)
│   │   ├── retrieval/            # Retrieval tests (10 tests)
│   │   ├── ingestion/            # Ingestion tests (3 tests)
│   │   └── performance/          # Performance tests (7 tests)
│   └── test_*.py                 # Other unit tests (5 tests)
│
├── integration/                   # Integration tests (end-to-end)
│   ├── conftest.py               # Integration fixtures
│   └── README.md                 # Integration test guide
│
├── fixtures/                      # Shared test data
│   └── sample_data.json          # Sample test fixtures
│
└── helpers/                       # Test utilities
    ├── test_utils.py             # Helper functions & builders
    └── mock_factory.py           # Mock object factories
```

## 🚀 Quick Start

### Run All Tests
```bash
pytest
```

### Run Unit Tests Only
```bash
pytest tests/unit/
```

### Run Specific Module
```bash
pytest tests/unit/services/agent/     # Agent tests
pytest tests/unit/services/cache/     # Cache tests
pytest tests/unit/services/guardrails/ # Safety tests
```

### Run with Coverage
```bash
pytest --cov=src --cov-report=html
open htmlcov/index.html  # View report
```

### Run by Marker
```bash
pytest -m agent              # All agent tests
pytest -m cache              # All cache tests
pytest -m "not slow"         # Skip slow tests
pytest -m "agent or cache"   # Multiple markers
```

### Development Mode
```bash
pytest -x                    # Stop on first failure
pytest -v                    # Verbose output
pytest -k "test_agent"       # Run tests matching pattern
```

## 📊 Test Categories

### Unit Tests (`tests/unit/`)
Fast, isolated tests that test individual components without external dependencies.
- **Run time**: < 1 second per test
- **Mocking**: Heavy use of mocks for external dependencies
- **Purpose**: Verify component behavior in isolation
- **Total**: 1012 tests across 8 modules

### Integration Tests (`tests/integration/`)
Tests that verify components work together correctly with real dependencies.
- **Run time**: Variable (may be slower)
- **Dependencies**: May require Redis, vector stores, external APIs
- **Purpose**: Verify end-to-end workflows

## 🎯 Running Tests

```bash
# Run all tests
pytest

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run specific module tests
pytest tests/unit/services/agent/

# Run with coverage
pytest --cov=src --cov-report=html

# Run with markers
pytest -m "not slow"
pytest -m "integration"
```

## 📝 Test Organization Principles

### 1. Mirror Source Structure
Tests mirror the `src/` directory structure for easy navigation:
- `src/services/agent/` → `tests/unit/services/agent/`
- `src/services/cache/` → `tests/unit/services/cache/`
- Easy to find tests for any source module

### 2. Naming Conventions
- **Test files**: `test_<module_name>.py`
- **Test functions**: `test_<functionality>_<scenario>()`
- **Test classes**: `Test<ComponentName>`
- **Fixtures**: Descriptive names like `sample_document`, `mock_redis_client`

### 3. Fixture Organization
- **Global fixtures**: `tests/conftest.py` (available to all tests)
- **Module fixtures**: `tests/unit/services/<module>/conftest.py` (module-specific)
- **Data fixtures**: `tests/fixtures/` (shared test data)

Example:
```python
# tests/unit/services/agent/conftest.py
@pytest.fixture
def mock_tool():
    """Agent-specific fixture - only available to agent tests."""
    return MockTool()

# tests/conftest.py
@pytest.fixture
def sample_text():
    """Global fixture - available to all tests."""
    return "Sample text for testing"
```

### 4. Test Markers
Use pytest markers to categorize and filter tests:
```python
@pytest.mark.unit          # Unit tests
@pytest.mark.integration   # Integration tests
@pytest.mark.slow          # Slow-running tests
@pytest.mark.asyncio       # Async tests
@pytest.mark.agent         # Agent-related
@pytest.mark.cache         # Cache-related
@pytest.mark.embeddings    # Embeddings-related
@pytest.mark.guardrails    # Safety-related
@pytest.mark.retrieval     # Retrieval-related
```

### 5. Test Helpers & Utilities
Use helper functions from `tests/helpers/` for common operations:
```python
from tests.helpers import TestDataBuilder, MockFactory

# Build test data easily
docs = TestDataBuilder.create_documents(count=5)
embedding = TestDataBuilder.create_embedding()

# Create mocks quickly
mock_llm = MockFactory.create_llm_client()
mock_cache = MockFactory.create_cache_client()
```

## Writing Good Tests

### Test Structure (AAA Pattern)
```python
def test_example():
    # Arrange - Set up test data and conditions
    data = create_test_data()
    
    # Act - Execute the code being tested
    result = function_under_test(data)
    
    # Assert - Verify the results
    assert result == expected_value
```

### Test Naming
```python
# Good: Descriptive and specific
def test_cache_returns_cached_value_when_key_exists()

# Bad: Vague
def test_cache()
```

### Fixture Usage
```python
@pytest.fixture
def sample_document():
    """Provide a sample document for testing."""
    return {"text": "Sample content", "metadata": {}}

def test_with_fixture(sample_document):
    result = process_document(sample_document)
    assert result is not None
```

## Common Patterns

### Mocking External Dependencies
```python
from unittest.mock import Mock, patch

def test_with_mock():
    with patch('src.services.external.api_call') as mock_api:
        mock_api.return_value = {"data": "test"}
        result = function_that_calls_api()
        assert result == expected
```

### Testing Async Code
```python
@pytest.mark.asyncio
async def test_async_function():
    result = await async_function()
    assert result == expected
```

### Parametrized Tests
```python
@pytest.mark.parametrize("input,expected", [
    ("test1", "result1"),
    ("test2", "result2"),
])
def test_multiple_cases(input, expected):
    assert process(input) == expected
```

## Coverage Goals

- **Overall coverage**: > 80%
- **Critical paths**: > 90%
- **New code**: 100%

Run coverage reports:
```bash
pytest --cov=src --cov-report=html --cov-report=term-missing
```

View coverage:
```bash
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## Continuous Integration

Tests run automatically on:
- Every commit (unit tests)
- Pull requests (full suite)
- Pre-merge (integration tests)

## 🎨 Module Architecture

```
                    ┌──────────────┐
                    │ Integration  │
                    │    Tests     │
                    └───────┬──────┘
                            │
                    (uses all modules)
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
   ┌─────────┐         ┌─────────┐       ┌──────────┐
   │ Agent   │◄────────┤  Cache  │       │Embeddings│
   │ Tests   │         │  Tests  │       │  Tests   │
   └────┬────┘         └────┬────┘       └────┬─────┘
        │                   │                   │
        │                   ▼                   │
        │            ┌──────────────┐          │
        └───────────►│  Retrieval   │◄─────────┘
                     │    Tests     │
                     └──────┬───────┘
                            │
                            ▼
                     ┌─────────────┐
                     │ Evaluation  │
                     │   Tests     │
                     └─────────────┘
```

Each module is:
- ✅ **Self-contained** - Independent test suite
- ✅ **Independently testable** - Run individually
- ✅ **Has own fixtures** - Module-specific helpers
- ✅ **Clear boundaries** - No cross-module dependencies

## 🔧 Configuration

### pytest Configuration (in pyproject.toml)
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = ["-v", "--strict-markers", "--cov=src"]
markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "slow: Slow-running tests",
    "agent: Agent-related tests",
    "cache: Cache-related tests",
    # ... more markers
]
```

### Coverage Configuration
```toml
[tool.coverage.run]
source = ["src"]
omit = ["*/tests/*", "*/__pycache__/*"]

[tool.coverage.report]
exclude_lines = ["pragma: no cover", "if __name__ == .__main__."]
```

## 🐛 Troubleshooting

### Tests Running Slowly
```bash
pytest -x                              # Stop on first failure
pytest tests/unit/services/agent/      # Run specific module
pytest -m "not slow"                   # Skip slow tests
```

### Tests Not Found
```bash
# Ensure __init__.py files exist
find tests -type d -exec touch {}/__init__.py \;

# Run from project root
cd /path/to/rag-journey-poc-to-prod
pytest
```

### Import Errors
```python
# Use absolute imports from project root
from src.services.agent import Agent
from tests.helpers import TestDataBuilder

# NOT relative imports like: from ..conftest import fixture
```

### Fixture Not Found
Check fixture location:
1. **Global fixtures**: `tests/conftest.py`
2. **Module fixtures**: `tests/unit/services/<module>/conftest.py`
3. **Fixture scope**: function, class, module, or session

### Coverage Not Working
```bash
# Install coverage
pip install pytest-cov

# Run with coverage
pytest --cov=src --cov-report=html

# View report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## ✍️ Contributing & Adding New Tests

### Adding a New Test

1. **Choose the right location**:
   ```bash
   # Unit test
   tests/unit/services/<module>/test_<feature>.py
   
   # Integration test
   tests/integration/test_<workflow>.py
   ```

2. **Follow the naming convention**:
   ```python
   def test_<component>_<action>_<expected_result>():
       """Clear docstring explaining what is tested."""
       pass
   ```

3. **Use available fixtures**:
   ```python
   def test_with_fixtures(sample_documents, mock_redis_client):
       """Fixtures are auto-discovered from conftest.py files."""
       # Test implementation
   ```

4. **Add appropriate markers**:
   ```python
   @pytest.mark.unit
   @pytest.mark.agent
   def test_agent_feature():
       """Mark tests for organization and filtering."""
       pass
   ```

5. **Follow AAA pattern**:
   ```python
   def test_example():
       # Arrange - Set up test data
       data = create_test_data()
       
       # Act - Execute the code
       result = function_under_test(data)
       
       # Assert - Verify results
       assert result == expected_value
   ```

### Adding Module-Specific Fixtures

Add to `tests/unit/services/<module>/conftest.py`:
```python
import pytest

@pytest.fixture
def module_specific_fixture():
    """Fixture available only to this module's tests."""
    return {"data": "for module tests"}
```

### Adding Global Fixtures

Add to `tests/conftest.py`:
```python
@pytest.fixture
def global_fixture():
    """Fixture available to all tests."""
    return "available everywhere"
```

### Using Test Helpers

```python
from tests.helpers import TestDataBuilder, MockFactory, AssertionHelper

# Build test data
docs = TestDataBuilder.create_documents(count=5)
embedding = TestDataBuilder.create_embedding(dimension=384)

# Create mocks
mock_llm = MockFactory.create_llm_client(response="Test response")
mock_cache = MockFactory.create_cache_client()

# Use assertion helpers
AssertionHelper.assert_valid_embedding(embedding, expected_dim=384)
AssertionHelper.assert_valid_document(docs[0])
```

## 📊 Test Execution Strategies

### Strategy 1: Full Suite (CI/CD)
```bash
pytest --cov=src --cov-report=html
```
- Runs all 1012 tests
- Generates coverage report
- Takes ~2 minutes

### Strategy 2: Module-Specific (Development)
```bash
pytest tests/unit/services/agent/ -v
```
- Runs only agent tests (6 tests)
- Very fast (seconds)
- Perfect for development

### Strategy 3: By Marker (Feature Work)
```bash
pytest -m "agent or cache"
```
- Runs related tests only
- Faster than full suite
- Good for feature development

### Strategy 4: Fast Feedback (TDD)
```bash
pytest -x -m "not slow"
```
- Stops on first failure
- Skips slow tests
- Ideal for test-driven development
