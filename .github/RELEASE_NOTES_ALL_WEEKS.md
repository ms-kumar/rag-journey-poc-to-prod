# Release Notes - All Weeks

This document provides a comprehensive summary of all weekly releases with their tags, titles, and key features.

## Quick Reference

| Week | Version | Date | Theme |
|------|---------|------|-------|
| Week 1 | v0.1.0 | 2025-11-30 | Naive RAG Pipeline |
| Week 2 | v0.2.0 | 2025-12-07 | Production-Ready RAG Enhancements |
| Week 3 | v0.3.0 | 2025-12-14 | Hybrid Retrieval with Fusion |
| Week 4 | v0.4.0 | 2025-12-21 | Metadata Filtering & Query Refinement |
| Week 5 | v0.5.0 | 2025-12-28 | Evaluations & Guardrails |
| Week 6 | v0.6.0 | 2026-01-04 | Schema Consolidation & Architectural Refinement |
| Week 7 | v0.7.0 | 2026-01-11 | Agentic RAG Implementation |
| Week 8 | v0.8.0 | 2026-01-18 | Observability, Experimentation & Production Readiness |

---

## v0.8.0 - Week 8: Observability, Experimentation & Production Readiness
**Release Date:** 2026-01-18

### 🎯 Key Highlights
- **Distributed Tracing** with correlation ID propagation
- **A/B Experiments** with feature flags
- **CI/CD Pipeline** with automated deployment and rollback
- **Canary Deployment** with health checks

### 📦 What's New
#### Observability
- Distributed tracing (Span, SpanContext, Tracer)
- Structured JSON logging with correlation IDs
- Metrics dashboards (latency P50/P95/P99, cost, quality)
- Golden traces for regression testing
- SLO monitoring with Prometheus export

#### Experimentation
- Feature flag system with percentage rollouts
- Experiment tracking with statistical significance
- Variant comparison and A/B testing

#### CI/CD
- GitHub Actions workflows (ci, deploy, rollback, eval_gate)
- Docker optimization with CPU-only PyTorch
- Evaluation gate with quality checks
- Canary deployment strategy
- Automated rollback on failures

### 🚀 Docker Images
```bash
docker pull ghcr.io/ms-kumar/rag-journey-poc-to-prod:0.8.0
docker pull ghcr.io/ms-kumar/rag-journey-poc-to-prod:latest
```

---

## v0.7.0 - Week 7: Agentic RAG Implementation
**Release Date:** 2026-01-11

### 🎯 Key Highlights
- **Agentic RAG** with LangGraph
- **6 Integrated Tools** (VectorDB, Web Search, Wikipedia, Code Executor, etc.)
- **Smart Routing** with confidence scoring
- **Self-Reflection** and quality assessment

### 📦 What's New
#### Tools & Routing
- Tool registry with dynamic registration
- Smart router with confidence-based selection
- 6 tools: VectorDB, Reranker, Generator, Web Search, Wikipedia, Code Executor
- Category-based routing (local/external/hybrid)

#### Safety & Execution
- RestrictedPython sandbox for code execution
- Timeout protection (5 seconds)
- Limited builtins for security
- Stdout capture

#### Self-Reflection
- Plan node for query decomposition
- Answer critic with quality scoring
- Source verifier for hallucination detection
- Query planner with complexity analysis
- Task benchmarker for performance tracking
- User feedback logger

### 🚀 Docker Images
```bash
docker pull ghcr.io/ms-kumar/rag-journey-poc-to-prod:0.7.0
```

---

## v0.6.0 - Week 6: Schema Consolidation & Architectural Refinement
**Release Date:** 2026-01-04

### 🎯 Key Highlights
- **Schema Consolidation** - Single source of truth
- **Domain Organization** - API vs Service schemas
- **Eliminated Redundancy** - Removed duplicate dataclasses

### 📦 What's New
- Centralized Pydantic schemas in `src/schemas/`
- Domain-based structure (api/ vs services/)
- Factory methods (`from_settings()`) for config schemas
- Type-safe enums and Field constraints
- Comprehensive schema modules for all services

### 🔧 Improvements
- Better code maintainability
- Enhanced type safety
- Improved developer experience

### 🚀 Docker Images
```bash
docker pull ghcr.io/ms-kumar/rag-journey-poc-to-prod:0.6.0
```

---

## v0.5.0 - Week 5: Evaluations & Guardrails
**Release Date:** 2025-12-28

### 🎯 Key Highlights
- **Evaluation Framework** with comprehensive metrics
- **Safety Guardrails** (PII, toxicity, prompt injection)
- **Adversarial Testing** with 26 attack vectors
- **0% Violation Rate** on all adversarial tests

### 📦 What's New
#### Evaluation
- Metrics: Precision@k, Recall@k, MRR, NDCG, MAP
- Evaluation datasets with relevance judgments
- CI evaluation gate
- Threshold configuration
- Weekly metric tracking

#### Guardrails
- PII detection (email, phone, SSN, credit cards)
- Toxicity filter with multi-category detection
- Safe response templates
- Audit logging with JSON events
- Guardrails coordinator

#### Security
- Jailbreak detection
- Prompt injection detection
- 26 adversarial attack vectors
- Canary tests (<30s)
- Comprehensive security runbook

### 🚀 Docker Images
```bash
docker pull ghcr.io/ms-kumar/rag-journey-poc-to-prod:0.5.0
```

---

## v0.4.0 - Week 4: Metadata Filtering & Query Refinement
**Release Date:** 2025-12-21

### 🎯 Key Highlights
- **Enhanced Filtering** with metadata support
- **Source/Date/Tag/Author** filtering
- **API-Friendly Interface** with REST endpoints

### 📦 What's New
- FilterBuilder with convenience methods
- Source filtering (single/multiple/exclude)
- Date range filtering (ISO strings + datetime)
- Tag filtering (single/multiple)
- Author filtering (single/multiple)
- Comprehensive documentation with curl examples
- Backward compatibility maintained

### 🚀 Docker Images
```bash
docker pull ghcr.io/ms-kumar/rag-journey-poc-to-prod:0.4.0
```

---

## v0.3.0 - Week 3: Hybrid Retrieval with Fusion
**Release Date:** 2025-12-14

### 🎯 Key Highlights
- **Hybrid Search** combining dense + sparse retrieval
- **SPLADE** neural sparse retrieval
- **Fusion Methods** (RRF + weighted fusion)
- **97% Test Coverage** for retrieval metrics

### 📦 What's New
#### Retrieval
- Dense retriever with vector search
- SPLADE retriever for neural sparse search
- RRF (Reciprocal Rank Fusion)
- Weighted fusion
- Score normalization

#### Metrics & Performance
- Retrieval metrics (MRR, Precision@k, Recall@k)
- Index persistence (snapshots)
- Latency tracking (P50/P95/P99)
- Cache monitoring (hit/miss rates)
- Performance benchmarking

### 🚀 Docker Images
```bash
docker pull ghcr.io/ms-kumar/rag-journey-poc-to-prod:0.3.0
```

---

## v0.2.0 - Week 2: Production-Ready RAG Enhancements
**Release Date:** 2025-12-07

### 🎯 Key Highlights
- **83x Speedup** with intelligent caching
- **5 Embedding Providers** (HuggingFace, E5, BGE, OpenAI, Cohere)
- **Multi-Format Support** (TXT, MD, HTML, PDF)
- **Advanced Chunking** strategies

### 📦 What's New
#### Embeddings
- Multiple providers with auto-detection
- Provider factory pattern
- Model-specific configuration
- Batch processing support

#### Caching
- LRU cache with disk persistence
- Cache statistics tracking
- Model-aware caching
- 83x speedup on repeated texts

#### Ingestion
- Multi-format support (TXT, MD, HTML, PDF)
- Advanced chunking (fixed-size + overlap, heading-aware)
- BeautifulSoup for HTML parsing
- PyPDF2 for PDF extraction

### 🚀 Docker Images
```bash
docker pull ghcr.io/ms-kumar/rag-journey-poc-to-prod:0.2.0
```

---

## v0.1.0 - Week 1: Naive RAG Pipeline
**Release Date:** 2025-11-30

### 🎯 Key Highlights
- **Foundation** - First working RAG pipeline
- **FastAPI** - REST API with OpenAPI docs
- **Qdrant** - Vector store integration
- **GPT-2** - Text generation

### 📦 What's New
- Project foundation with FastAPI
- Document ingestion (TXT files)
- Fixed-size chunking (512 chars)
- Hash embeddings (64 dimensions)
- Qdrant vector store
- HuggingFace generation (GPT-2)
- API endpoints: `/api/v1/rag/ingest`, `/api/v1/rag/generate`
- Pipeline orchestration
- Docker Compose setup

### 🚀 Docker Images
```bash
docker pull ghcr.io/ms-kumar/rag-journey-poc-to-prod:0.1.0
```

---

## 📚 Documentation

- **CHANGELOG.md** - Detailed version history
- **Week Plans** - See `docs/week-plans/week-*.md` for detailed implementation notes
- **Contributing** - See [CONTRIBUTING.md](../CONTRIBUTING.md)
- **Security** - See [SECURITY.md](../SECURITY.md)

## 🚀 Creating Releases

To create and push all release tags:

```bash
# Run the automated script
./scripts/create_releases.sh

# Or manually create a tag
git tag -a v0.8.0 -m "Week 8: Observability, Experimentation & Production Readiness"
git push origin v0.8.0
```

This will automatically trigger the GitHub Actions release workflow that:
1. Creates a GitHub Release with changelog notes
2. Builds and pushes Docker images with version tags
3. Generates release notes

## 📊 Statistics

- **Total Releases:** 8 weekly releases
- **Total Features:** 150+ features implemented
- **Test Coverage:** 79%+
- **Total Tests:** 1600+
- **Lines of Code:** 15,000+
- **Documentation:** 25+ markdown files

---

**Full Changelog:** [CHANGELOG.md](../CHANGELOG.md)
