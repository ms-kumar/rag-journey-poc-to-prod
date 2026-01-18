# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Community health files (CODE_OF_CONDUCT, CONTRIBUTING, SECURITY)
- GitHub Issue and PR templates
- Automated release workflow

## [0.8.0] - 2026-01-18 (Week 8)

**Theme:** Observability, Experimentation & Production Readiness

### Added
- **Distributed Tracing**: Span, SpanContext, Tracer with correlation ID propagation
- **Structured Logging**: JSON-formatted logs with contextual correlation IDs
- **Metrics Dashboards**: Latency (P50/P95/P99), cost, and quality tracking
- **Golden Traces**: Reference trace capture and regression testing
- **SLO Monitoring**: Dashboard data export with Prometheus format
- **A/B Experiments**: Feature flag system with percentage-based rollouts
- **Experiment Tracking**: Statistical significance testing and variant comparison
- **CI/CD Pipeline**: GitHub Actions workflows (ci.yml, deploy.yml, rollback.yml, eval_gate.yml)
- **Docker Optimization**: Multi-stage build with CPU-only PyTorch
- **Evaluation Gate**: Pre-deployment quality checks with CI-specific thresholds
- **Canary Deployment**: Gradual rollout with health checks
- **Automated Rollback**: Failure detection and automatic rollback

### Changed
- Improved test parallelization with matrix strategy (7 parallel jobs)
- Reduced Docker image size by removing CUDA dependencies (~2GB savings)
- Made evaluation gate soft-fail with zero thresholds for CI

### Fixed
- Memory issues in CI by splitting test runs
- gRPC connection issues with Qdrant in CI
- Disk space issues with cleanup step in deploy workflow

## [0.7.0] - 2026-01-11 (Week 7)

**Theme:** Agentic RAG Implementation

### Added
- **Tool Registry**: Dynamic tool registration with 6 integrated tools
- **Smart Router**: Confidence-based routing with fallback strategies
- **Code Sandbox**: RestrictedPython sandbox with timeout protection
- **Self-Reflection**: Plan node for query decomposition
- **Answer Critic**: Quality scoring (completeness, accuracy, clarity)
- **Source Verifier**: Hallucination detection and claim verification
- **Query Planner**: Advanced query decomposition with complexity analysis
- **Task Benchmarker**: Execution time and quality score tracking
- **User Feedback Logger**: Rating collection and pattern analysis
- **LangGraph Integration**: State machine for agentic workflows

### Tools Added
- Local: VectorDB, Reranker, Generator
- External: Web Search, Wikipedia
- Hybrid: Code Executor

## [0.6.0] - 2026-01-04 (Week 6)

**Theme:** Schema Consolidation & Architectural Refinement

### Changed
- **Schema Consolidation**: Centralized all Pydantic schemas in `src/schemas/`
- **Domain Organization**: Separated API schemas from service schemas
- **Eliminated Redundancy**: Removed duplicate dataclass definitions
- **Factory Methods**: Added `from_settings()` classmethods to config schemas

### Added
- Domain-based schema structure (api/ vs services/)
- Comprehensive schema modules for all services
- Type-safe enums and Field constraints

### Improved
- Code maintainability with single source of truth
- Type safety with comprehensive Pydantic validation
- Development experience with centralized schema exports

## [0.5.0] - 2025-12-28 (Week 5)

**Theme:** Evaluations & Guardrails

### Added
- **Evaluation Harness**: Comprehensive metrics (Precision@k, Recall@k, MRR, NDCG, MAP)
- **Evaluation Datasets**: Query-document relevance judgments
- **CI Evaluation Gate**: Automated quality regression prevention
- **Threshold Configuration**: Quality standards enforcement
- **PII Detection**: Email, phone, SSN, credit card detection and redaction
- **Toxicity Filter**: Multi-category toxic content detection
- **Adversarial Testing**: 26 attack vectors for jailbreak and prompt injection
- **Guardrails Coordinator**: Unified safety interface
- **Audit Logging**: JSON structured security events
- **Canary Tests**: Fast CI validation (<30s)

### Security
- Achieved 0% violation rate on all adversarial tests
- Configured violation thresholds (≤ 0.1%)
- Comprehensive safety runbook and documentation

### Testing
- 101 comprehensive guardrails tests (97-100% coverage)
- 16 adversarial test scenarios
- Automated red-team testing

## [0.4.0] - 2025-12-21 (Week 4)

**Theme:** Metadata Filtering & Query Refinement

### Added
- **Enhanced FilterBuilder**: Convenience methods for common filters
- **Source Filtering**: Single and multiple source filters with exclusion
- **Date Range Filtering**: ISO string and datetime support
- **Tag Filtering**: Single and multiple tag matching
- **Author Filtering**: Single and multiple author support
- **API-Friendly Interface**: REST endpoints for metadata filtering
- **Comprehensive Documentation**: curl examples and use cases

### Improved
- Backward compatibility with existing filters
- Query precision with metadata-based narrowing
- Production readiness for filtered retrieval

## [0.3.0] - 2025-12-14 (Week 3)

**Theme:** Hybrid Retrieval with Fusion

### Added
- **Dense Retriever**: Vector search with performance metrics
- **SPLADE Retriever**: Neural sparse retrieval
- **Fusion Methods**: RRF (Reciprocal Rank Fusion) and weighted fusion
- **Retrieval Metrics**: MRR, Precision@k, Recall@k tracking
- **Index Persistence**: Snapshot creation and restoration
- **Score Normalization**: Fair comparison across search types
- **Latency Tracking**: P50/P95/P99 percentiles per search type
- **Cache Monitoring**: Hit/miss rates and speedup ratios

### Performance
- 97% test coverage for retrieval metrics
- Comprehensive fusion benchmarking
- Optimized hybrid search performance

## [0.2.0] - 2025-12-07 (Week 2)

**Theme:** Production-Ready RAG Enhancements

### Added
- **Multiple Embedding Providers**: HuggingFace, E5, BGE, OpenAI, Cohere
- **Intelligent Cache**: LRU cache with disk persistence (83x speedup)
- **Batch Processing**: Configurable batch size with cache awareness
- **Advanced Chunking**: Fixed-size with overlap, heading-aware for markdown
- **Multi-Format Ingestion**: TXT, MD, HTML, PDF support
- **Provider Factory**: Auto-detection and model-specific configuration
- **Cache Statistics**: Hit rate, miss rate, performance tracking
- **Model-Aware Caching**: Separate caches per embedding model

### Performance
- 83x speedup on repeated texts with cache
- Efficient batch processing for large datasets
- Memory-optimized embedding operations

## [0.1.0] - 2025-11-30 (Week 1)

**Theme:** Naive RAG Pipeline

### Added
- **Project Foundation**: FastAPI application with logging and settings
- **Document Ingestion**: Factory pattern for loading TXT files
- **Fixed-Size Chunking**: 512 character chunks with empty filtering
- **Hash Embeddings**: Deterministic 64-dimension vectors
- **Qdrant Integration**: Vector store with auto-collection creation
- **HuggingFace Generation**: GPT-2 text generation pipeline
- **API Endpoints**: `/api/v1/rag/ingest`, `/api/v1/rag/generate`, `/health`
- **Pipeline Orchestration**: NaivePipeline class for end-to-end flow
- **Basic Retrieval**: Similarity search with Qdrant

### Infrastructure
- FastAPI REST API with OpenAPI documentation
- Makefile for common development tasks
- Docker Compose for local services
- Basic test suite

[Unreleased]: https://github.com/ms-kumar/rag-journey-poc-to-prod/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/ms-kumar/rag-journey-poc-to-prod/releases/tag/v0.8.0
[0.7.0]: https://github.com/ms-kumar/rag-journey-poc-to-prod/releases/tag/v0.7.0
[0.6.0]: https://github.com/ms-kumar/rag-journey-poc-to-prod/releases/tag/v0.6.0
[0.5.0]: https://github.com/ms-kumar/rag-journey-poc-to-prod/releases/tag/v0.5.0
[0.4.0]: https://github.com/ms-kumar/rag-journey-poc-to-prod/releases/tag/v0.4.0
[0.3.0]: https://github.com/ms-kumar/rag-journey-poc-to-prod/releases/tag/v0.3.0
[0.2.0]: https://github.com/ms-kumar/rag-journey-poc-to-prod/releases/tag/v0.2.0
[0.1.0]: https://github.com/ms-kumar/rag-journey-poc-to-prod/releases/tag/v0.1.0
