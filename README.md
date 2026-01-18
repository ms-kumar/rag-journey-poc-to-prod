<p align="center">
  <h1 align="center">🚀 Advanced RAG System</h1>
  <p align="center">
    <strong>Production-Ready Retrieval-Augmented Generation</strong>
  </p>
  <p align="center">
    Built with FastAPI • Qdrant • HuggingFace • LangGraph
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/tests-1600+-green.svg" alt="1600+ Tests">
  <img src="https://img.shields.io/badge/coverage-79%25-yellow.svg" alt="79% Coverage">
  <img src="https://img.shields.io/badge/code%20style-ruff-purple.svg" alt="Ruff">
  <img src="https://img.shields.io/badge/type%20checked-mypy-blue.svg" alt="Mypy">
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Features](#-features)
- [Quick Start](#-quick-start)
- [API Reference](#-api-reference)
- [Configuration](#-configuration)
- [Examples](#-examples)
- [Testing](#-testing)
- [Documentation](#-documentation)
- [CI/CD & Deployment](#-cicd--deployment)
- [Project Structure](#-project-structure)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

A **production-ready RAG system** featuring intelligent caching, multiple embedding providers, comprehensive safety guardrails, and agentic capabilities. Built for scale with observability, A/B testing, and automated deployment pipelines.

### Core Components

| Component | Description |
|:----------|:------------|
| 📥 **Ingestion** | Multi-format document loading (TXT, MD, HTML, PDF) |
| ✂️ **Chunking** | Fixed-size and heading-aware chunking with overlap |
| 🧮 **Embeddings** | Multiple providers with caching (Hash, E5, BGE, OpenAI, Cohere) |
| 💾 **Cache** | LRU embedding cache with disk persistence (**83x speedup**) |
| 🗄️ **Vector Store** | Qdrant integration with similarity search |
| 🎯 **Re-ranking** | Cross-encoder re-ranking with timeout & fallback |
| 🤖 **Generation** | HuggingFace transformers for text generation |
| 🌐 **API** | FastAPI with async endpoints |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INGESTION PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  Documents → Chunking → Embeddings → Vector Store (Qdrant)                  │
│                              ↓                                               │
│                       LRU Cache (83x speedup)                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                              QUERY PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  Query → Query Understanding → Embedding → Similarity Search                │
│            ↓                                      ↓                          │
│     (Rewriting, Synonyms)              Cross-Encoder Re-ranking             │
│                                                   ↓                          │
│                                        Retrieved Chunks → LLM → Answer      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

### 🔍 Search & Retrieval

| Feature | Description |
|:--------|:------------|
| **Hybrid Search** | BM25 keyword + vector similarity + SPLADE sparse |
| **Metadata Filtering** | Rich filters by source, date, tags with operators (`$in`, `$gte`, `$not`) |
| **Cross-Encoder Re-ranking** | Improve precision@k with configurable timeouts |
| **Query Understanding** | Auto rewriting, synonym expansion, intent classification (<1ms) |
| **Index Optimization** | Payload indices for 10-100x faster filtering |
| **Fusion Orchestration** | RRF or weighted fusion for 33%+ recall uplift |

### ⚡ Performance

| Feature | Description |
|:--------|:------------|
| **Intelligent Caching** | LRU cache with disk persistence (83x speedup) |
| **Batch Processing** | Efficient batch encoding with configurable sizes |
| **Token Budgets** | Comprehensive limits and cost estimation |
| **Smart Truncation** | HEAD/TAIL/MIDDLE strategies with word boundaries |
| **Overflow Protection** | Automatic token limit enforcement |
| **Performance Profiling** | Timers, percentile tracking, SLA monitoring |

### 🛡️ Safety & Reliability

| Feature | Description |
|:--------|:------------|
| **PII Detection** | Email, phone, SSN, credit cards, IP addresses |
| **Toxicity Filtering** | Profanity, threats, harassment, hate speech |
| **Jailbreak Detection** | Prompt injection blocking |
| **Audit Logging** | Structured JSON logs with severity levels |
| **Adversarial Testing** | Red-team prompts, 0% violation on 26 attack vectors |
| **Retry & Backoff** | Exponential backoff with jitter |
| **Health Checks** | K8s-ready readiness/liveness probes |

### 🤖 Agentic RAG

| Feature | Description |
|:--------|:------------|
| **LangGraph Agent** | Autonomous agent with tool routing |
| **Self-Reflection** | Answer critique with quality scoring |
| **Planning** | Query decomposition, adaptive replanning |
| **6 Tools** | Local + External + Hybrid integrations |
| **User Feedback** | Feedback learning and analytics |

### 📡 Production Operations

| Feature | Description |
|:--------|:------------|
| **Observability** | Distributed tracing, structured logging, metrics |
| **SLO Monitoring** | Error budgets, burn rate, severity alerts |
| **A/B Testing** | Feature flags, experiments, statistical analysis |
| **CI/CD Pipeline** | Build → Test → Eval → Staging → Canary → Prod |
| **Rollback** | Automated rollback with rehearsal scripts |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker (for Qdrant)
- [uv](https://github.com/astral-sh/uv) (recommended package manager)

### 1️⃣ Start Qdrant

```bash
docker compose -f infra/docker/compose.yml up -d
```

### 2️⃣ Install Dependencies

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies
uv sync --all-extras
```

### 3️⃣ Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings
```

### 4️⃣ Run the Server

```bash
# Production mode
make run

# Development mode (with auto-reload)
make dev
```

### 5️⃣ Ingest Documents

```bash
make ingest
# or
curl -X POST http://localhost:8000/api/v1/rag/ingest
```

### 6️⃣ Query the System

```bash
curl -X POST http://localhost:8000/api/v1/rag/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is RAG?", "top_k": 3}'
```

---

## 📡 API Reference

### Endpoints

| Endpoint | Method | Description |
|:---------|:-------|:------------|
| `/health` | GET | Health check |
| `/health/ready` | GET | Readiness probe |
| `/health/live` | GET | Liveness probe |
| `/api/v1/rag/ingest` | POST | Ingest and index documents |
| `/api/v1/rag/generate` | POST | Generate answer using RAG |
| `/api/v1/agent/query` | POST | Query with agentic RAG |

### Example Request

```bash
# Generate with RAG
curl -X POST http://localhost:8000/api/v1/rag/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain machine learning",
    "top_k": 5,
    "filters": {"source": "ml_notes.txt"}
  }'
```

---

## ⚙️ Configuration

### Environment Variables

<details>
<summary><b>📁 Ingestion & Chunking</b></summary>

| Variable | Default | Description |
|:---------|:--------|:------------|
| `INGESTION__DIR` | `./data` | Document ingestion directory |
| `CHUNKING__CHUNK_SIZE` | `200` | Characters per chunk |
| `CHUNKING__CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `CHUNKING__STRATEGY` | `heading_aware` | `fixed` or `heading_aware` |

</details>

<details>
<summary><b>🧮 Embeddings</b></summary>

| Variable | Default | Description |
|:---------|:--------|:------------|
| `EMBED__PROVIDER` | `hash` | Provider: hash, e5, bge, openai, cohere |
| `EMBED__MODEL` | `simple-hash` | Model name/identifier |
| `EMBED__DIM` | `64` | Embedding dimension |
| `EMBED__CACHE_ENABLED` | `true` | Enable embedding cache |
| `EMBED__CACHE_MAX_SIZE` | `10000` | Maximum cache entries |
| `EMBED__CACHE_DIR` | `.cache/embeddings` | Cache directory |

</details>

<details>
<summary><b>🗄️ Vector Store (Qdrant)</b></summary>

| Variable | Default | Description |
|:---------|:--------|:------------|
| `QDRANT__URL` | `http://localhost:6333` | Qdrant server URL |
| `QDRANT__COLLECTION_NAME` | `naive_collection` | Collection name |
| `QDRANT__PREFER_GRPC` | `true` | Use gRPC protocol |
| `QDRANT__ENABLE_BM25` | `false` | Enable BM25 indexing |

</details>

<details>
<summary><b>💾 Cache (Redis)</b></summary>

| Variable | Default | Description |
|:---------|:--------|:------------|
| `REDIS__HOST` | `localhost` | Redis host |
| `REDIS__PORT` | `6379` | Redis port |
| `CACHE__ENABLED` | `true` | Enable caching |
| `CACHE__DEFAULT_TTL` | `3600` | Default TTL (seconds) |

</details>

See [.env.example](.env.example) for complete configuration options.

### Embedding Providers

| Provider | Type | Models | Configuration |
|:---------|:-----|:-------|:--------------|
| **hash** | Local | Deterministic hash | Built-in, no deps |
| **e5** | Local | E5-small/base/large | `EMBED__PROVIDER=e5` |
| **bge** | Local | BGE-small/base/large | `EMBED__PROVIDER=bge` |
| **openai** | API | text-embedding-3-* | `EMBED__PROVIDER=openai` |
| **cohere** | API | embed-english-* | `EMBED__PROVIDER=cohere` |

---

## 📚 Examples

### Embedding Cache Performance

```bash
python examples/cache_demo.py
```

```
Benchmark: WITH CACHE
First run:  5.13ms (cache miss)
Second run: 0.06ms (cache hit)
Speedup:    83.1x ⚡
```

### Fusion Search

```python
from src.services.vectorstore.fusion import fuse_results, FusionConfig

# Combine multiple search methods
results = {
    "vector": vector_store.similarity_search("query", k=10),
    "bm25": vector_store.bm25_search("query", k=10),
    "sparse": vector_store.sparse_search("query", k=10)
}

# Reciprocal Rank Fusion
config = FusionConfig(method="rrf", rrf_k=60)
fused = fuse_results(results, config=config)
# → 25-50% recall uplift over single method
```

### Guardrails & Safety

```python
from src.services.guardrails.coordinator import GuardrailsCoordinator

coordinator = GuardrailsCoordinator(
    enable_pii_check=True,
    enable_toxicity_check=True,
    auto_redact_pii=True
)

# Process user query safely
is_safe, processed = coordinator.process_query(
    "My SSN is 123-45-6789",
    user_id="user123"
)
# → PII automatically redacted
```

### Token Budget Management

```python
from src.models.token_budgets import get_embedding_budget, estimate_cost

budget = get_embedding_budget("text-embedding-3-small")
print(f"Max tokens: {budget.max_input_tokens}")

cost = estimate_cost("gpt-4-turbo", input_tokens=5000, output_tokens=1000)
print(f"Estimated: ${cost:.4f}")
```

<details>
<summary><b>More Examples</b></summary>

| Example | Description |
|:--------|:------------|
| `cache_demo.py` | Embedding cache performance |
| `fusion_benchmark.py` | Search fusion comparison |
| `query_understanding_demo.py` | Query rewriting & expansion |
| `reranker_demo.py` | Cross-encoder re-ranking |
| `evaluation_demo.py` | Evaluation harness |
| `agent_demo.py` | Agentic RAG usage |
| `sandbox_demo.py` | Code execution sandbox |

Run any example:
```bash
python examples/<example_name>.py
```

</details>

---

## 🧪 Testing

### Test Suite Overview

```
📊 1600+ Tests | 79% Coverage | Organized by Module
```

### Running Tests

```bash
# All tests
make test

# With coverage
make test-cov

# By module
make test-agent       # Agent tests
make test-cache       # Cache tests
make test-guardrails  # Safety tests
make test-retrieval   # Retrieval tests

# By marker
uv run pytest -m "not slow"    # Skip slow tests
uv run pytest -m agent         # Agent tests only

# Specific file
uv run pytest tests/unit/services/agent/test_reflection.py -v
```

### Test Organization

```
tests/
├── unit/                    # Fast, isolated unit tests
│   └── services/
│       ├── agent/          # 275 tests - Agent framework
│       ├── cache/          # Cache tests
│       ├── cost/           # 74 tests - Cost tracking
│       ├── embeddings/     # Embedding tests
│       ├── evaluation/     # Evaluation tests
│       ├── experimentation/# 180+ tests - A/B testing
│       ├── guardrails/     # Safety tests
│       ├── observability/  # 136 tests - Monitoring
│       ├── retrieval/      # Retrieval tests
│       └── performance/    # Performance tests
├── integration/            # End-to-end tests
├── fixtures/               # Shared test data
└── helpers/                # Test utilities
```

### Quality Checks

```bash
# Run all quality checks
make quality

# Individual checks
make format      # Format with Ruff
make lint        # Lint with Ruff
make type-check  # Type check with Mypy
make security    # Security scan with Bandit
```

---

## 📖 Documentation

### Core Guides

| Document | Description |
|:---------|:------------|
| [🤖 Agent Quickstart](docs/AGENT_QUICKSTART.md) | Agentic RAG getting started |
| [🔍 Query Understanding](docs/query-understanding.md) | Rewriting & expansion |
| [🎯 Reranking](docs/reranking.md) | Cross-encoder re-ranking |
| [💾 Embedding Cache](docs/embedding-cache.md) | Cache with 83x speedup |
| [🔀 BM25 Filters](docs/bm25-filters.md) | Keyword search & filtering |

### Safety & Reliability

| Document | Description |
|:---------|:------------|
| [🛡️ Guardrails](docs/guardrails-implementation.md) | PII, toxicity, audit |
| [🔴 Adversarial Testing](docs/adversarial-testing-runbook.md) | Red-team procedures |
| [🔁 Retry & Backoff](docs/retry-backoff.md) | Resilient service calls |
| [🏥 Health Checks](docs/health-check.md) | K8s probes |

### Performance

| Document | Description |
|:---------|:------------|
| [📊 Performance Profiling](docs/performance-profiling.md) | Timers, SLA monitoring |
| [📏 Token Budgets](docs/token-budgets.md) | Limits & cost estimation |
| [✂️ Truncation](docs/truncation.md) | Text truncation strategies |
| [🛡️ Overflow Guards](docs/overflow-guards.md) | Token limit enforcement |

### Operations

| Document | Description |
|:---------|:------------|
| [📡 Observability](docs/observability.md) | Tracing, logging, metrics |
| [🧪 Experimentation](docs/experimentation.md) | A/B testing, feature flags |
| [🚀 CI/CD Pipeline](docs/ci-cd-pipeline.md) | Deployment strategy |
| [⏪ Rollback Playbook](docs/rollback-playbook.md) | Incident response |

### Development Progress

| Week | Focus | Status |
|:-----|:------|:------:|
| Week 1 | Naive RAG Pipeline | ✅ |
| Week 2 | Caching, Providers, Quality | ✅ |
| Week 3 | Hybrid Retrieval & Fusion | ✅ |
| Week 4 | Metadata Filtering | ✅ |
| Week 5 | Evaluation & Guardrails | ✅ |
| Week 6 | Schema Consolidation | ✅ |
| Week 7 | Agentic RAG | ✅ |
| Week 8 | Production Operations | ✅ |

---

## 🚢 CI/CD & Deployment

### Workflows

| Workflow | Trigger | Purpose |
|:---------|:--------|:--------|
| [ci.yml](.github/workflows/ci.yml) | Push, PR | Tests, linting, security |
| [eval_gate.yml](.github/workflows/eval_gate.yml) | PR | RAG quality evaluation |
| [deploy.yml](.github/workflows/deploy.yml) | Push (main) | Full deployment pipeline |
| [rollback.yml](.github/workflows/rollback.yml) | Manual | Emergency rollback |

### Deployment Pipeline

```
┌─────────┐    ┌──────┐    ┌───────────┐    ┌─────────┐    ┌────────┐    ┌────────────┐
│  Build  │ → │ Test │ → │ Eval Gate │ → │ Staging │ → │ Canary │ → │ Production │
└─────────┘    └──────┘    └───────────┘    └─────────┘    └────────┘    └────────────┘
                                                              5% → 25%        100%
```

### Commands

```bash
make docker-build       # Build Docker image
make deploy-staging     # Deploy to staging
make deploy-canary      # Deploy canary (5%)
make deploy-prod        # Deploy to production
make rollback ENV=prod  # Rollback deployment
make canary-health      # Check canary health
make rehearse-rollback  # Practice rollback
```

---

## 📁 Project Structure

<details>
<summary><b>Click to expand full structure</b></summary>

```
src/
├── main.py                     # FastAPI entry point
├── config.py                   # Configuration management
├── dependencies.py             # Dependency injection
├── exceptions.py               # Custom exceptions
│
├── api/
│   ├── router/
│   │   ├── agent_router.py     # Agentic RAG endpoints
│   │   ├── rag_router.py       # Traditional RAG endpoints
│   │   └── health_router.py    # Health checks
│   └── v1/endpoints/
│       └── rag.py              # RAG endpoints
│
├── schemas/                    # Pydantic schemas
│   ├── api/                    # API request/response
│   └── services/               # Service data structures
│
└── services/
    ├── agent/                  # Agentic RAG
    │   ├── graph.py            # LangGraph state machine
    │   ├── nodes.py            # Agent nodes
    │   ├── reflection.py       # Answer critique
    │   ├── planning.py         # Query decomposition
    │   ├── feedback.py         # User feedback
    │   ├── benchmarking.py     # Task benchmarking
    │   ├── tools/              # Tool implementations
    │   └── metrics/            # Confidence scoring
    │
    ├── observability/          # Production monitoring
    │   ├── tracing.py          # Distributed tracing
    │   ├── logging.py          # Structured logging
    │   ├── metrics.py          # Metrics dashboard
    │   ├── slo.py              # SLO monitoring
    │   └── golden_traces.py    # Regression testing
    │
    ├── experimentation/        # A/B testing
    │   ├── experiments.py      # Experiment management
    │   ├── feature_flags.py    # Feature flags
    │   ├── analysis.py         # Statistical analysis
    │   ├── canary.py           # Canary deployments
    │   └── reports.py          # Experiment reports
    │
    ├── guardrails/             # Safety
    │   ├── pii_detector.py     # PII detection
    │   ├── toxicity_filter.py  # Toxicity filtering
    │   ├── jailbreak_detector.py
    │   └── audit_log.py        # Audit logging
    │
    ├── cache/                  # Caching
    ├── chunking/               # Document chunking
    ├── cost/                   # Cost tracking
    ├── embeddings/             # Text embeddings
    ├── evaluation/             # Evaluation harness
    ├── generation/             # LLM generation
    ├── ingestion/              # Document loading
    ├── performance/            # Performance profiling
    ├── pipeline/               # RAG orchestration
    ├── query_understanding/    # Query processing
    ├── reranker/               # Re-ranking
    ├── retry.py                # Retry logic
    ├── truncation.py           # Text truncation
    └── vectorstore/            # Qdrant integration

tests/
├── unit/                       # Unit tests (1600+)
├── integration/                # Integration tests
├── fixtures/                   # Test data
└── helpers/                    # Test utilities

scripts/
├── ci_eval_gate.py             # CI evaluation
├── check_canary_health.py      # Canary health
├── rehearse_rollback.py        # Rollback practice
└── generate_dashboard.py       # Metrics dashboard

docs/                           # Documentation
config/                         # Configuration files
examples/                       # Example scripts
infra/                          # Infrastructure (Docker)
```

</details>

---

## 🙏 Acknowledgments

Special thanks to:

- **[uv](https://github.com/astral-sh/uv)** - Fast, modern Python packaging
- **[Ruff](https://github.com/astral-sh/ruff)** - Lightning-fast linting/formatting
- **[arxiv-paper-curator](https://github.com/jamwithai/arxiv-paper-curator.git)** - Inspiration

And the amazing open-source ecosystem:

<p align="center">
  <a href="https://github.com/fastapi/fastapi">FastAPI</a> •
  <a href="https://github.com/qdrant/qdrant">Qdrant</a> •
  <a href="https://github.com/langchain-ai/langchain">LangChain</a> •
  <a href="https://github.com/huggingface/transformers">Transformers</a> •
  <a href="https://github.com/pydantic/pydantic">Pydantic</a>
</p>

---

<p align="center">
  <sub>Built with ❤️ for production RAG systems</sub>
</p>
