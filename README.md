# Advanced RAG

A production-ready Retrieval-Augmented Generation (RAG) system built with FastAPI, Qdrant, and HuggingFace, featuring intelligent caching, multiple embedding providers, and comprehensive quality tooling.

### Architecture

```
Documents → Chunking → Embeddings → Vector Store (Qdrant)
                           ↓
                    LRU Cache (83x speedup)
                           ↓
Query → Embedding → Similarity Search → Retrieved Chunks → LLM → Answer
```

### Core Components

| Component | Description |
|-----------|-------------|
| **Ingestion** | Multi-format document loading (TXT, MD, HTML, PDF) with BeautifulSoup and PyPDF2 |
| **Chunking** | Fixed-size and heading-aware chunking with configurable overlap |
| **Embeddings** | Multiple providers with automatic caching (Hash, E5, BGE, OpenAI, Cohere) |
| **Cache** | LRU embedding cache with disk persistence (83x speedup on repeated texts) |
| **Vector Store** | Qdrant integration with efficient similarity search |
| **Generation** | HuggingFace transformers for text generation (GPT-2 default) |
| **API** | FastAPI with async endpoints for ingest and generate |

### Key Features

✨ **Intelligent Caching**: LRU cache with disk persistence reduces redundant computations by up to 83x

🔄 **Batch Processing**: Efficient batch encoding with configurable batch sizes

🔌 **Multiple Providers**: Support for local (E5, BGE) and API-based (OpenAI, Cohere) embeddings

📄 **Multi-Format**: Ingest TXT, Markdown, HTML, and PDF documents with format-specific processing

🧪 **Comprehensive Tests**: 184 tests with 70% coverage across all components

🛠️ **Quality Tooling**: Ruff (lint/format), mypy (type-check), bandit (security), pre-commit hooks

📊 **Performance Benchmarks**: Built-in retrieval@k and latency benchmarking tools

### Project Structure

```
src/
├── main.py                 # FastAPI app entry point
├── config.py               # App configuration
├── api/
│   └── v1/endpoints/
│       └── rag.py          # RAG endpoints (ingest, generate)
├── models/
│   └── rag_request.py      # Pydantic models
└── services/
    ├── chunking/           # Document chunking
    ├── embeddings/         # Text embeddings
    ├── generation/         # LLM generation
    ├── ingestion/          # Document loading
    ├── pipeline/           # RAG orchestration
    └── vectorstore/        # Qdrant integration
```

### Quick Start

1. **Start Qdrant** (using Docker):
   ```bash
   docker compose -f infra/docker/compose.yml up -d
   ```

2. **Install dependencies** (using [uv](https://github.com/astral-sh/uv)):
   ```bash
   # Install uv if not already installed
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Sync dependencies
   uv sync
   
   # Or install all optional dependencies (dev, embeddings, parsers)
   uv sync --all-extras
   ```

3. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

4. **Run the server**:
   ```bash
   make run
   # or with auto-reload for development
   make dev
   # or directly
   uv run python -m src.main
   ```

5. **Ingest documents**:
   ```bash
   make ingest
   # or
   curl -X POST http://localhost:8000/api/v1/rag/ingest
   ```

6. **Query the RAG system**:
   ```bash
   curl -X POST http://localhost:8000/api/v1/rag/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "What is RAG?", "top_k": 3}'
   ```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/v1/rag/ingest` | POST | Ingest and index documents |
| `/api/v1/rag/generate` | POST | Generate answer using RAG |

### Development

#### Code Quality Tools

This project uses modern Python tooling for code quality:

```bash
# Install dev dependencies
make install

# Format code with ruff
make format

# Check formatting without changes
make check

# Lint code with ruff
make lint

# Auto-fix linting issues
make lint-fix

# Type check with mypy
make type-check

# Security scan with bandit
make security

# Run all quality checks
make quality

# Setup pre-commit hooks (runs checks automatically)
make pre-commit
```

#### Testing

```bash
# Run tests (excluding slow tests)
make test

# Run tests with coverage report
make test-cov

# Run all tests including slow ones
make test-all

# Run specific test file
uv run pytest tests/test_embeddings.py -v

# Run tests matching a pattern
uv run pytest -k "test_embedding" -v
```

#### Development Workflow

1. **Setup development environment**:
   ```bash
   uv sync --all-extras
   make pre-commit
   ```

2. **Make changes and test**:
   ```bash
   make format      # Format code
   make test        # Run tests
   ```

3. **Run quality checks before commit**:
   ```bash
   make quality     # Runs lint, format-check, type-check, security
   ```

4. **Development server with auto-reload**:
   ```bash
   make dev         # Runs uvicorn with --reload
   ```

### Examples & Demos

#### Embedding Cache Performance

See the performance benefits of intelligent caching:

```bash
python examples/cache_demo.py
```

Expected output showing **83x speedup**:
```
Benchmark 2: WITH CACHE
First run:  5.13ms (cache miss)
Second run: 0.06ms (cache hit)
Third run:  0.04ms (cache hit)
Speedup:    83.1x
```

For detailed caching documentation, see [docs/embedding-cache.md](docs/embedding-cache.md).

#### Retrieval Benchmarks

Run comprehensive retrieval@k and latency benchmarks:

```bash
python examples/benchmark_retrieval.py
```

This benchmarks:
- Document ingestion and chunking latency
- Embedding generation (cold vs warm cache)
- Vector store indexing performance  
- Retrieval@k latency for k=1,3,5,10
- Answer generation time
- End-to-end pipeline performance

#### Embedding Providers

The system supports multiple embedding providers:

| Provider | Type | Models | Setup |
|----------|------|--------|-------|
| **hash** | Local | Deterministic hash-based | Built-in, no dependencies |
| **e5** | Local | E5-small/base/large | `EMBED_PROVIDER=e5 EMBED_MODEL=e5-small` |
| **bge** | Local | BGE-small/base/large | `EMBED_PROVIDER=bge EMBED_MODEL=bge-small` |
| **openai** | API | text-embedding-3-small/large | `EMBED_PROVIDER=openai EMBED_API_KEY=sk-...` |
| **cohere** | API | embed-english/multilingual | `EMBED_PROVIDER=cohere EMBED_API_KEY=...` |

Example configuration in `.env`:
```bash
# Using E5 embeddings (local)
EMBED_PROVIDER=e5
EMBED_MODEL=intfloat/e5-small-v2
EMBED_DEVICE=cuda

# Using OpenAI embeddings (API)
EMBED_PROVIDER=openai
EMBED_MODEL=text-embedding-3-small
EMBED_API_KEY=sk-proj-...
```

### Configuration

All configuration is managed through environment variables and `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `INGESTION_DIR` | `./data` | Directory for document ingestion |
| `CHUNK_SIZE` | `200` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlapping characters between chunks |
| `CHUNKING_STRATEGY` | `heading_aware` | Chunking strategy: `fixed` or `heading_aware` |
| `EMBED_PROVIDER` | `hash` | Embedding provider |
| `EMBED_MODEL` | `simple-hash` | Model name/identifier |
| `EMBED_DIM` | `64` | Embedding dimension |
| `EMBED_CACHE_ENABLED` | `true` | Enable embedding cache |
| `EMBED_CACHE_MAX_SIZE` | `10000` | Maximum cache entries |
| `EMBED_CACHE_DIR` | `.cache/embeddings` | Cache directory |
| `QDRANT_URL` | `None` | Qdrant server URL |
| `QDRANT_COLLECTION_NAME` | `naive_collection` | Collection name |

See [.env.example](.env.example) for complete configuration options.

### Testing & Quality

The project maintains high code quality standards with automated tooling:

```bash
# Run all quality checks (format, lint, type-check, security)
make quality

# Individual checks
make format      # Format code with ruff
make lint        # Lint with ruff  
make type-check  # Type check with mypy
make security    # Security scan with bandit

# Run tests with coverage
make test-cov

# View coverage report
open htmlcov/index.html
```

**Test Coverage**: 184 tests | 70% coverage

Quality gates enforced:
- ✅ Ruff formatting (100 char line length)
- ✅ Ruff linting (E, W, F, I, N, UP, B, C4, SIM, TCH, Q, RET, PTH rules)
- ✅ Mypy type checking (strict mode)
- ✅ Bandit security scanning
- ✅ Pre-commit hooks for automated checks

### CI/CD

GitHub Actions workflow runs on every push:
- Install dependencies with uv
- Run all quality checks
- Execute full test suite with coverage

See [.github/workflows/ci.yml](.github/workflows/ci.yml) for details.

## Special Thanks

- [uv](https://github.com/astral-sh/uv) for fast, modern Python packaging/workflows.
- [Ruff](https://github.com/astral-sh/ruff) for linting/formatting.
- [Mother of AI project (arxiv-paper-curator)](https://github.com/jamwithai/arxiv-paper-curator.git) for inspiration and learning resources.
- The open-source ecosystem that makes this project possible, including:
   - [FastAPI](https://github.com/fastapi/fastapi) and [Uvicorn](https://github.com/encode/uvicorn)
   - [Qdrant](https://github.com/qdrant/qdrant) and [qdrant-client](https://github.com/qdrant/qdrant-client)
   - [LangChain](https://github.com/langchain-ai/langchain)
   - [Transformers](https://github.com/huggingface/transformers) and [PyTorch](https://github.com/pytorch/pytorch)
   - [Pydantic](https://github.com/pydantic/pydantic)
