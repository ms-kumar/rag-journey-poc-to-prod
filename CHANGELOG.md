# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Automated CI/CD pipeline with GitHub Actions
- Docker multi-stage build with CPU-only PyTorch optimization
- Evaluation gate for deployment quality checks
- Canary deployment with health checks
- Automated rollback on deployment failure

### Changed
- Improved test parallelization with matrix strategy
- Reduced Docker image size by removing CUDA dependencies

### Fixed
- Memory issues in CI by splitting test runs
- gRPC connection issues with Qdrant in CI

## [0.1.0] - 2026-01-18

### Added
- Initial release of Advanced RAG System
- Multi-format document ingestion (TXT, MD, HTML, PDF)
- Fixed-size and heading-aware chunking
- Multiple embedding providers (Hash, E5, BGE, OpenAI, Cohere)
- LRU embedding cache with disk persistence (83x speedup)
- Qdrant vector store integration
- BM25 hybrid search support
- Cross-encoder reranking
- Query understanding (rewriting, expansion, intent classification)
- Comprehensive guardrails (prompt injection, PII, jailbreak detection)
- Agentic RAG with LangGraph
- Tool execution sandbox with security controls
- Redis caching layer
- Observability with structured logging and tracing
- A/B testing and experimentation framework
- Cost tracking and model selection
- Health checks and retry mechanisms
- FastAPI REST API with OpenAPI documentation
- 1600+ unit and integration tests
- 79% code coverage

### Security
- Input validation and sanitization
- Prompt injection detection
- PII detection and masking
- Jailbreak attempt detection
- Sandboxed code execution

[Unreleased]: https://github.com/ms-kumar/rag-journey-poc-to-prod/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ms-kumar/rag-journey-poc-to-prod/releases/tag/v0.1.0
