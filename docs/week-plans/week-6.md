# Week 6: Schema Consolidation & Architectural Refinement

**Focus:** Eliminate redundant dataclasses and centralize Pydantic schemas for improved maintainability

## Overview

Week 6 addresses architectural debt by identifying and eliminating redundant dataclass definitions across the codebase. All service-level data structures are now centralized in `src/schemas/` using Pydantic models, following the domain-based organization pattern from the arxiv-paper-curator reference architecture.

## Goals

- ✅ Identify all redundant dataclasses duplicating Pydantic schema definitions
- ✅ Consolidate schemas into centralized domain-based structure (`api/` vs `services/`)
- ✅ Update all service imports to use centralized schemas
- ✅ Add factory methods (`from_settings`) to configuration schemas
- ✅ Fix type mismatches between schemas and actual usage patterns
- ✅ Update test suite to use Pydantic validation patterns
- ✅ Maintain 100% test pass rate and code quality standards

## Implementation

### 1. Schema Organization

**Domain-Based Structure:**
```
src/schemas/
├── __init__.py              # Central exports
├── api/
│   └── rag.py              # API request/response models
└── services/
    ├── __init__.py         # Service schema exports
    ├── cache.py            # Cache service schemas (CacheEntry, CacheStats, SemanticCacheResult)
    ├── cost.py             # Cost tracking & autoscaling schemas (ModelMetrics, CostReport, ScalingDecision)
    ├── embeddings.py       # Embedding service schemas (SparseEncoderConfig, EmbeddingRequest/Response)
    ├── evaluation.py       # Evaluation framework schemas (EvaluationMetrics, EvaluationReport, ThresholdConfig)
    ├── guardrails.py       # Safety & guardrails schemas (PIIMatch, ToxicityScore, AuditEvent)
    ├── performance.py      # Performance monitoring schemas (PerformanceMetrics, PerformanceReport)
    ├── query_understanding.py  # Query understanding schemas (QueryRewriterConfig, SynonymExpanderConfig)
    ├── reranker.py         # Reranker service schemas (RerankResult, PrecisionMetrics, BenchmarkConfig)
    ├── retry.py            # Retry & resilience schemas (RetryConfig)
    ├── token_budgets.py    # Token budget definitions (TokenBudget, model-specific budgets)
    └── vectorstore.py      # Vector store service schemas (FusionConfig, RetrievalMetrics, IndexMapping)
```

**Design Principles:**
- **DRY (Don't Repeat Yourself):** Single source of truth for all data structures
- **Domain Separation:** API schemas vs service schemas cleanly separated
- **Comprehensive Coverage:** All services have dedicated schema modules
- **Pydantic First:** Leverage Pydantic's validation, serialization, and type safety
- **Factory Methods:** Configuration schemas include `from_settings()` classmethods
- **Type Safety:** Enums for categorical data, Field constraints for validation

### 2. Complete Schema Catalog

#### Cache Schemas (`cache.py`)
- **CacheEntry**: Cache key-value storage with TTL
- **CacheStats**: Hit rate, misses, size tracking
- **SemanticCacheResult**: Semantic cache lookup results with similarity scores

#### Cost Tracking Schemas (`cost.py`)
- **ModelMetrics**: Per-model cost, latency, quality tracking with computed fields (avg_latency, cost_per_1k, error_rate)
- **CostReport**: Comprehensive cost analysis across all models
- **ScalingDecision**: Autoscaling decisions based on load
- **AutoscalingPolicy**: Policy configuration for auto-scaling
- **LoadMetrics**: Current load metrics (requests/sec, queue depth)
- **ModelCandidate**: Candidate models for selection

#### Embeddings Schemas (`embeddings.py`)
- **SparseEncoderConfig**: SPLADE encoder configuration (model, device, batch_size)
- **EmbeddingRequest**: Request model for embedding generation
- **EmbeddingResponse**: Response with embeddings and usage stats
- **SparseEmbeddingResponse**: Sparse embeddings (token_id → weight mappings)

#### Evaluation Schemas (`evaluation.py`)
- **EvaluationMetrics**: Precision, recall, F1, quality, relevance, faithfulness
- **EvaluationResult**: Single query evaluation with pass/fail status
- **EvaluationReport**: Complete evaluation report with aggregated metrics
- **ThresholdConfig**: Configurable thresholds for pass/fail criteria

#### Guardrails Schemas (`guardrails.py`)
- **PIIType** (Enum): EMAIL, PHONE, SSN, CREDIT_CARD, IP_ADDRESS, NAME, ADDRESS, DOB
- **PIIMatch**: Detected PII with position and confidence
- **ToxicityLevel** (Enum): NONE, LOW, MEDIUM, HIGH, SEVERE
- **ToxicityCategory** (Enum): PROFANITY, HATE_SPEECH, THREAT, HARASSMENT, SEXUAL, VIOLENCE, SELF_HARM
- **ToxicityMatch**: Detected toxic content with category and level
- **ToxicityScore**: Overall toxicity assessment
- **AuditEventType** (Enum): PII_DETECTED, PII_REDACTED, TOXICITY_DETECTED, TOXICITY_FILTERED, etc.
- **AuditSeverity** (Enum): INFO, WARNING, ERROR, CRITICAL
- **AuditEvent**: Structured audit log events with timestamp
- **GuardrailResult**: Combined guardrail check results
- **ResponseType** (Enum): SAFE, PII_VIOLATION, TOXICITY_VIOLATION, JAILBREAK_DETECTED

#### Performance Schemas (`performance.py`)
- **PerformanceMetrics**: Operation-level performance tracking
- **PerformanceReport**: Aggregated performance with percentiles (p50, p95, p99)
- **OperationStats**: Per-operation statistics (count, avg/min/max duration, success rate)

#### Query Understanding Schemas (`query_understanding.py`)
- **QueryRewriterConfig**: Configuration for query rewriting (acronyms, typos, context)
  - Includes `from_settings()` factory method
- **SynonymExpanderConfig**: Configuration for synonym expansion
  - Includes `from_settings()` factory method

#### Reranker Schemas (`reranker.py`)
- **RerankResult**: Reranked documents with scores and metadata
- **PrecisionMetrics**: Precision@k metrics with improvement tracking
- **ComparisonResult**: Baseline vs reranked comparison with statistical significance
- **BenchmarkConfig**: Configuration for reranker benchmarking

#### Retry Schemas (`retry.py`)
- **RetryConfig**: Exponential backoff configuration (max_retries, delays, jitter, status codes)

#### Token Budget Schemas (`token_budgets.py`)
- **TokenBudget**: Token limits and cost estimates (dataclass, kept for performance)
- **EmbeddingModelBudgets**: Budgets for all embedding models (OpenAI, Cohere, HuggingFace)
- **GenerationModelBudgets**: Budgets for generation models (GPT, Claude, etc.)

#### Vector Store Schemas (`vectorstore.py`)
- **FusionConfig**: Fusion method configuration (RRF/weighted, weights, normalization)
- **FusionResult**: Fusion results with combined scores
- **RetrievalMetrics**: Retrieval performance metrics (latencies, cache hits, per-type stats)
- **IndexMapping**: Qdrant payload index configuration
- **EvaluationMetrics**: Recall@k, precision@k, MRR, MAP, NDCG (aliased as VectorstoreEvaluationMetrics)
- **UpliftMetrics**: Fusion uplift analysis vs baselines

**Total Impact:** Removed 11 redundant dataclass definitions across 8 service files

### 3. Dataclasses Eliminated
**File:** `src/services/reranker/client.py`
- ❌ Removed: `RerankResult` dataclass (8 fields)
- ✅ Replaced with: `schemas.services.reranker.RerankResult` (Pydantic)
- **Benefits:** Field validation, JSON serialization, optional field defaults

**File:** `src/services/reranker/evaluation.py`
- ❌ Removed: `PrecisionMetrics`, `ComparisonResult`, `BenchmarkConfig` dataclasses
- ✅ Replaced with: Corresponding Pydantic models from `schemas.services.reranker`
- **Benefits:** Nested validation, field validators, type coercion

#### Vector Store Service
**File:** `src/services/vectorstore/fusion.py`
- ❌ Removed: `FusionConfig` dataclass
- ✅ Replaced with: `schemas.services.vectorstore.FusionConfig`
- **Type Fix:** Changed `weights` from `list[float]` to `dict[str, float]` to match actual usage

**File:** `src/services/vectorstore/fusion_eval.py`
- ❌ Removed: `EvaluationMetrics`, `UpliftMetrics` dataclasses
- ✅ Replaced with: Pydantic models with alias handling
- **Pattern:** Used `VectorstoreEvaluationMetrics as EvaluationMetrics` to avoid naming conflicts

**File:** `src/services/vectorstore/index_mappings.py`
- ❌ Removed: `IndexMapping` dataclass (8 fields)
- ✅ Replaced with: `schemas.services.vectorstore.IndexMapping`
- **Benefits:** Field-level validation for Qdrant payload configurations

**File:** `src/services/vectorstore/retrieval_metrics.py`
- ⚠️ Kept as wrapper class (not a pure data structure)
- ✅ Uses schema internally for data representation
- **Pattern:** Behavior-rich classes can wrap schemas for methods

#### Query Understanding Service
**File:** `src/services/query_understanding/rewriter.py`
- ❌ Removed: `QueryRewriterConfig` dataclass
- ✅ Replaced with: `schemas.services.query_understanding.QueryRewriterConfig`
- ➕ **Added:** `from_settings()` classmethod for easy instantiation from app config

**File:** `src/services/query_understanding/synonym_expander.py`
- ❌ Removed: `SynonymExpanderConfig` dataclass
- ✅ Replaced with: `schemas.services.query_understanding.SynonymExpanderConfig`
- ➕ **Added:** `from_settings()` classmethod

### 4. Schema Enhancements

**Configuration Factory Methods:**
```python
# Before: Manual field mapping
config = QueryRewriterConfig(
    expand_acronyms=settings.query_understanding.expand_acronyms,
    fix_typos=settings.query_understanding.fix_typos,
    # ... more manual mapping
)

# After: Clean factory method
config = QueryRewriterConfig.from_settings(settings)
```

**Type Safety Improvements:**
- **Weights Field:** Fixed `FusionConfig.weights` type from `list[float]` to `dict[str, float]`
- **Improvement Metrics:** Changed from `float` to `dict[int, float]` for precision@k tracking
- **Scores Structure:** Clarified `RetrievalMetrics.scores` as flat `list[float]` vs nested structure

### 5. Test Suite Updates

**Validation Pattern Change:**
```python
# Before: Custom exception handling
with pytest.raises(ValueError, match="Invalid field"):
    create_config(invalid_field="bad")

# After: Pydantic ValidationError
from pydantic import ValidationError

with pytest.raises(ValidationError) as exc_info:
    create_config(invalid_field="bad")
assert "validation error" in str(exc_info.value)
```

**Files Updated:**
- `tests/test_query_understanding.py` (4 validation tests)
- `tests/test_fusion.py` (2 validation tests)
- `tests/test_index_mappings.py` (validation tests)
- Plus updates to handle schema imports

**Test Coverage:** Maintained 79% coverage with 913 tests passing (2 skipped)

### 6. Code Quality Verification

**All Quality Checks Passing:**
```bash
✅ make format    # Ruff formatting
✅ make lint      # Ruff linting (0 errors)
✅ make type      # mypy type checking (0 errors)
✅ make security  # Bandit security scan (0 issues)
✅ make test      # 913 tests passing
```

**Linting Fixes:**
- Removed unused imports (old dataclass imports)
- Replaced bare `except:` with specific `ValidationError` catches
- Fixed import ordering

## Benefits

### 1. **DRY Principle Enforced**
- **Before:** Duplicate definitions across service files and schemas
- **After:** Single source of truth in `src/schemas/`
- **Impact:** Changes to data structures now require updates in only one location

### 2. **Enhanced Validation**
- **Pydantic Features:** Field validators, type coercion, nested validation
- **Runtime Safety:** Invalid data caught at construction time with detailed error messages
- **Type Safety:** Better IDE support and static type checking

### 3. **Improved Maintainability**
- **Discoverability:** All schemas in central location (`src/schemas/`)
- **Consistency:** Uniform patterns across all services
- **Documentation:** Pydantic models serve as self-documenting code

### 4. **Better Serialization**
- **JSON Export:** Built-in `.model_dump()` and `.model_dump_json()`
- **API Integration:** Seamless FastAPI integration for request/response models
- **Config Files:** Easy export/import for configuration management

### 5. **Factory Methods**
- **Clean Initialization:** `from_settings()` reduces boilerplate
- **Flexibility:** Easy to add validation logic in factory methods
- **Testability:** Simplified test setup with factory patterns

## Migration Examples

### Example 1: Fusion Configuration
```python
# Before: Dataclass in fusion.py
@dataclass
class FusionConfig:
    method: str = "rrf"
    rrf_k: int = 60
    weights: list[float] = field(default_factory=list)  # Wrong type!
    # ...

# After: Pydantic in schemas/services/vectorstore.py
class FusionConfig(BaseModel):
    method: Literal["rrf", "weighted"] = "rrf"
    rrf_k: int = Field(default=60, ge=1)
    weights: dict[str, float] = Field(default_factory=dict)  # Correct type!
    # ...
```

### Example 2: Query Rewriter Configuration
```python
# Before: Dataclass with manual validation
@dataclass
class QueryRewriterConfig:
    expand_acronyms: bool = True
    fix_typos: bool = True
    # No validation, no factory methods

# After: Pydantic with factory and validation
class QueryRewriterConfig(BaseModel):
    expand_acronyms: bool = True
    fix_typos: bool = True
    add_context: bool = True
    max_rewrites: int = Field(default=3, ge=1, le=10)
    min_query_length: int = Field(default=3, ge=1)
    
    @classmethod
    def from_settings(cls, settings) -> "QueryRewriterConfig":
        """Create config from app settings."""
        return cls(
            expand_acronyms=settings.query_understanding.expand_acronyms,
            fix_typos=settings.query_understanding.fix_typos,
            # ...
        )
```

### Example 3: Test Validation Updates
```python
# Before: Expecting generic ValueError
def test_invalid_weights():
    with pytest.raises(ValueError):
        FusionConfig(weights=[1.0, -0.5])  # Negative weight

# After: Expecting Pydantic ValidationError
from pydantic import ValidationError

def test_invalid_weights():
    with pytest.raises(ValidationError) as exc_info:
        FusionConfig(weights={"dense": 1.0, "sparse": -0.5})
    assert "validation error" in str(exc_info.value).lower()
```

## Files Modified

### New/Updated Schema Files
- ✅ `src/schemas/services/__init__.py` (Central exports for all service schemas)
- ✅ `src/schemas/services/cache.py` (3 models: CacheEntry, CacheStats, SemanticCacheResult)
- ✅ `src/schemas/services/cost.py` (6 models: ModelMetrics, CostReport, ScalingDecision, AutoscalingPolicy, LoadMetrics, ModelCandidate)
- ✅ `src/schemas/services/embeddings.py` (4 models: SparseEncoderConfig, EmbeddingRequest, EmbeddingResponse, SparseEmbeddingResponse)
- ✅ `src/schemas/services/evaluation.py` (4 models: EvaluationMetrics, EvaluationResult, EvaluationReport, ThresholdConfig)
- ✅ `src/schemas/services/guardrails.py` (12 models/enums: PIIType, PIIMatch, ToxicityLevel, ToxicityCategory, ToxicityMatch, ToxicityScore, AuditEventType, AuditSeverity, AuditEvent, GuardrailResult, ResponseType)
- ✅ `src/schemas/services/performance.py` (3 models: PerformanceMetrics, PerformanceReport, OperationStats)
- ✅ `src/schemas/services/query_understanding.py` (2 models with factories: QueryRewriterConfig, SynonymExpanderConfig)
- ✅ `src/schemas/services/reranker.py` (4 models: RerankResult, PrecisionMetrics, ComparisonResult, BenchmarkConfig)
- ✅ `src/schemas/services/retry.py` (1 model: RetryConfig)
- ✅ `src/schemas/services/token_budgets.py` (TokenBudget dataclass + model budget classes)
- ✅ `src/schemas/services/vectorstore.py` (6 models: FusionConfig, FusionResult, RetrievalMetrics, IndexMapping, EvaluationMetrics, UpliftMetrics)

### Service Files Updated
- ✅ `src/services/reranker/client.py`
- ✅ `src/services/reranker/evaluation.py`
- ✅ `src/services/vectorstore/fusion.py`
- ✅ `src/services/vectorstore/fusion_eval.py`
- ✅ `src/services/vectorstore/index_mappings.py`
- ✅ `src/services/vectorstore/retrieval_metrics.py`
- ✅ `src/services/query_understanding/rewriter.py`
- ✅ `src/services/query_understanding/synonym_expander.py`

### Test Files Updated
- ✅ `tests/test_query_understanding.py`
- ✅ `tests/test_fusion.py`
- ✅ `tests/test_index_mappings.py`

## Metrics

| Metric | Value |
|--------|-------|
| Schema Files Created | 12 |
| Total Schema Models | 50+ (Pydantic models + Enums) |
| Dataclasses Eliminated | 11 |
| Service Files Updated | 8 |
| Test Files Updated | 6+ |
| Tests Passing | 913 (2 skipped) |
| Code Coverage | 79% |
| Type Errors | 0 |
| Lint Issues | 0 |
| Security Issues | 0 |

## Lessons Learned

1. **Schema-First Design:** Starting with centralized schemas prevents duplication from the beginning
2. **Type Safety Matters:** Type mismatches (like `weights` field) can hide bugs; Pydantic catches these
3. **Computed Fields:** Pydantic's `@computed_field` decorator provides derived metrics without storage overhead
4. **Enums for Safety:** Categorical data (PIIType, ToxicityLevel, etc.) as Enums prevents invalid values
5. **Factory Methods:** Adding `from_settings()` significantly reduces boilerplate in service initialization
6. **Test Validation:** Pydantic's `ValidationError` provides richer error details than generic `ValueError`
7. **Incremental Migration:** Service-by-service migration (reranker → vectorstore → query_understanding) minimizes risk
8. **Strategic Dataclass Use:** Keep dataclasses for performance-critical, immutable configs (TokenBudget) while using Pydantic for everything else

## Key Schema Patterns

### Pattern 1: Enums for Type Safety
```python
class PIIType(str, Enum):
    """Types of PII that can be detected."""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    # Prevents: PIIType = "emial" (typo)
```

### Pattern 2: Field Validation Constraints
```python
class RetryConfig(BaseModel):
    max_retries: int = Field(3, ge=0)  # Must be >= 0
    initial_delay: float = Field(1.0, gt=0.0)  # Must be > 0
    hit_rate: float = Field(0.0, ge=0.0, le=100.0)  # Must be 0-100
```

### Pattern 3: Computed Fields
```python
class ModelMetrics(BaseModel):
    total_requests: int = 0
    total_cost: float = 0.0
    
    @computed_field
    @property
    def cost_per_1k(self) -> float:
        """Automatically calculated field."""
        if self.total_requests == 0:
            return 0.0
        return (self.total_cost / self.total_requests) * 1000
```

### Pattern 4: Alias Handling
```python
# Import with alias to avoid naming conflicts
from src.schemas.services.vectorstore import (
    EvaluationMetrics as VectorstoreEvaluationMetrics
)
```

### Pattern 5: Factory Methods
```python
class QueryRewriterConfig(BaseModel):
    expand_acronyms: bool = True
    # ... other fields
    
    @classmethod
    def from_settings(cls, settings) -> "QueryRewriterConfig":
        """Create from app settings - DRY initialization."""
        return cls(
            expand_acronyms=settings.query_understanding.expand_acronyms,
            # ...
        )
```

## Future Improvements

1. **Schema Versioning:** Consider adding version fields for API evolution
2. **Config Schemas:** Extract all config classes to schemas (beyond service configs)
3. **Custom Validators:** Add more field-level validators for domain-specific constraints
4. **JSON Schema Export:** Generate OpenAPI schemas from Pydantic models for documentation
5. **Schema Evolution Testing:** Add tests to ensure backward compatibility on schema changes

## References

- **Inspiration:** [arxiv-paper-curator](https://github.com/yourusername/arxiv-paper-curator) - Schema organization pattern
- **Pydantic Docs:** https://docs.pydantic.dev/latest/
- **ADR (if created):** `docs/adr/006-schema-consolidation.md`

## Summary

Week 6 successfully established a comprehensive, centralized Pydantic schema architecture covering all system services. Created 12 schema modules with 50+ models, eliminated all redundant dataclass definitions, and consolidated them into a maintainable structure with clear domain separation. The migration provides:

- **Complete Coverage**: Every service now has dedicated schema definitions (cache, cost, embeddings, evaluation, guardrails, performance, query understanding, reranker, retry, token budgets, vectorstore)
- **Type Safety**: Extensive use of Pydantic Field validators, Enums, and computed fields
- **Rich Validation**: Built-in constraints (ge/le for numbers, min_length for lists, regex patterns)
- **Better Maintainability**: Single source of truth for all data structures across 913 tests
- **Zero Regressions**: Maintained 100% test pass rate and zero quality issues

This architectural foundation enables consistent data handling patterns across the entire RAG system, from API requests through service layer to database persistence. All future features can leverage these centralized schemas, ensuring type safety and validation consistency throughout the stack.
