# Contributing to Advanced RAG System

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)

## Code of Conduct

This project adheres to the Contributor Covenant [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/rag-journey-poc-to-prod.git
   cd rag-journey-poc-to-prod
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/ms-kumar/rag-journey-poc-to-prod.git
   ```

## Development Setup

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Make

### Install Dependencies

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Or use uv for faster installs
uv pip install -e ".[dev]"
```

### Start Services

```bash
# Start Qdrant and Redis
docker-compose up -d

# Verify services
make health-check
```

### Run Tests

```bash
# All tests
make test

# Unit tests only
make test-unit

# Integration tests only
make test-integration

# With coverage
make test-coverage
```

## How to Contribute

### Reporting Bugs

- Use the [Bug Report](.github/ISSUE_TEMPLATE/bug_report.md) template
- Include clear reproduction steps
- Provide environment details (OS, Python version, etc.)
- Attach relevant logs or screenshots

### Suggesting Features

- Use the [Feature Request](.github/ISSUE_TEMPLATE/feature_request.md) template
- Explain the problem you're trying to solve
- Describe your proposed solution
- Consider alternative approaches

### Working on Issues

1. **Find an issue** to work on (look for `good-first-issue` or `help-wanted` labels)
2. **Comment** on the issue to claim it
3. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

## Coding Standards

### Python Style

- Follow [PEP 8](https://pep8.org/)
- Use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting
- Maximum line length: 100 characters
- Use type hints for all functions

```bash
# Format code
make format

# Lint code
make lint

# Type check
make type-check
```

### Code Quality

- Write docstrings for all public functions/classes (Google style)
- Keep functions small and focused
- Avoid deeply nested code
- Use meaningful variable names
- Add comments for complex logic

### Example

```python
def retrieve_documents(
    query: str,
    top_k: int = 5,
    filters: dict[str, Any] | None = None,
) -> list[Document]:
    """Retrieve relevant documents for a query.
    
    Args:
        query: Search query text
        top_k: Number of documents to return
        filters: Optional metadata filters
        
    Returns:
        List of retrieved documents
        
    Raises:
        VectorStoreError: If retrieval fails
    """
    ...
```

## Testing

### Writing Tests

- Place unit tests in `tests/unit/`
- Place integration tests in `tests/integration/`
- Use descriptive test names: `test_<what>_<when>_<expected>`
- Use pytest fixtures for common setup
- Mock external dependencies in unit tests

### Test Structure

```python
def test_cache_returns_cached_embeddings_when_available():
    """Test that cache returns embeddings when they exist."""
    # Arrange
    cache = EmbeddingCache(max_size=100)
    text = "test query"
    expected = [0.1, 0.2, 0.3]
    cache.set(text, expected)
    
    # Act
    result = cache.get(text)
    
    # Assert
    assert result == expected
```

### Coverage

- Maintain minimum 75% code coverage
- Write tests for all new features
- Add tests when fixing bugs

```bash
# Run with coverage report
make test-coverage

# View HTML report
open htmlcov/index.html
```

## Pull Request Process

### Before Submitting

1. **Update your branch** with latest changes:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run the full test suite**:
   ```bash
   make test
   make lint
   make type-check
   ```

3. **Update documentation** if needed
4. **Add/update tests** for your changes
5. **Update CHANGELOG.md** under `[Unreleased]` section

### PR Requirements

- ✅ All tests pass
- ✅ Code is formatted and linted
- ✅ Type checks pass
- ✅ Documentation updated
- ✅ CHANGELOG.md updated (for user-facing changes)
- ✅ PR template filled out
- ✅ Descriptive commit messages

### PR Title Format

Use conventional commits format:

```
<type>(<scope>): <description>

Types: feat, fix, docs, style, refactor, test, chore
Examples:
  feat(cache): add Redis cache backend
  fix(embeddings): handle empty input gracefully
  docs(readme): update installation instructions
```

### Review Process

1. A maintainer will review your PR
2. Address any requested changes
3. Once approved, a maintainer will merge your PR
4. Your contribution will be included in the next release! 🎉

## Release Process

Releases are automated via GitHub Actions:

1. Update `CHANGELOG.md` with version and date
2. Create and push a version tag:
   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```
3. GitHub Actions will:
   - Create a GitHub Release
   - Build and push Docker images
   - Generate release notes

## Development Tips

### Useful Make Commands

```bash
make help           # Show all available commands
make format         # Format code with ruff
make lint           # Run linting
make type-check     # Run type checking
make test           # Run all tests
make test-watch     # Run tests in watch mode
make clean          # Clean build artifacts
make docs           # Build documentation
```

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Key variables:
- `VECTORSTORE__HOST` - Qdrant host
- `VECTORSTORE__PORT` - Qdrant port
- `CACHE__REDIS_URL` - Redis connection URL
- `EMBEDDINGS__PROVIDER` - Embedding provider (hash, e5, bge, openai, cohere)

### Debugging

```bash
# Run specific test with debugging
pytest tests/unit/test_cache.py::test_name -vv -s

# Run with pdb on failure
pytest --pdb

# Profile performance
pytest --profile
```

## Questions?

- Open a [Discussion](https://github.com/ms-kumar/rag-journey-poc-to-prod/discussions)
- Check existing [Issues](https://github.com/ms-kumar/rag-journey-poc-to-prod/issues)
- Review [Documentation](README.md)

Thank you for contributing! 🚀
