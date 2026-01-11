"""Service-specific schemas."""

from src.schemas.services.cache import CacheEntry, CacheStats, SemanticCacheResult
from src.schemas.services.cost import (
    AutoscalingPolicy,
    CostReport,
    LoadMetrics,
    ModelCandidate,
    ModelMetrics,
    ScalingDecision,
)
from src.schemas.services.embeddings import (
    EmbeddingRequest,
    EmbeddingResponse,
    SparseEmbeddingResponse,
    SparseEncoderConfig,
)
from src.schemas.services.evaluation import (
    EvaluationMetrics,
    EvaluationReport,
    EvaluationResult,
    ThresholdConfig,
)
from src.schemas.services.guardrails import (
    AuditEvent,
    AuditEventType,
    AuditSeverity,
    GuardrailResult,
    PIIMatch,
    PIIType,
    ResponseType,
    ToxicityCategory,
    ToxicityLevel,
    ToxicityMatch,
    ToxicityScore,
)
from src.schemas.services.performance import (
    OperationStats,
    PerformanceMetrics,
    PerformanceReport,
)
from src.schemas.services.query_understanding import (
    QueryRewriterConfig,
    SynonymExpanderConfig,
)
from src.schemas.services.reranker import (
    BenchmarkConfig,
    ComparisonResult,
    PrecisionMetrics,
    RerankResult,
)
from src.schemas.services.retry import RetryConfig
from src.schemas.services.token_budgets import TokenBudget
from src.schemas.services.vectorstore import (
    EvaluationMetrics as VectorstoreEvaluationMetrics,
)
from src.schemas.services.vectorstore import (
    FusionConfig,
    FusionResult,
    IndexMapping,
    RetrievalMetrics,
    UpliftMetrics,
)

__all__ = [
    # Cache schemas
    "CacheEntry",
    "CacheStats",
    "SemanticCacheResult",
    # Cost tracking schemas
    "ModelMetrics",
    "CostReport",
    "ScalingDecision",
    "AutoscalingPolicy",
    "LoadMetrics",
    "ModelCandidate",
    # Embeddings schemas
    "SparseEncoderConfig",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "SparseEmbeddingResponse",
    # Evaluation schemas
    "EvaluationMetrics",
    "EvaluationResult",
    "EvaluationReport",
    "ThresholdConfig",
    # Guardrails schemas
    "AuditEvent",
    "AuditEventType",
    "AuditSeverity",
    "GuardrailResult",
    "PIIMatch",
    "PIIType",
    "ResponseType",
    "ToxicityCategory",
    "ToxicityLevel",
    "ToxicityMatch",
    "ToxicityScore",
    # Performance schemas
    "PerformanceMetrics",
    "PerformanceReport",
    "OperationStats",
    # Query understanding schemas
    "QueryRewriterConfig",
    "SynonymExpanderConfig",
    # Reranker schemas
    "RerankResult",
    "PrecisionMetrics",
    "ComparisonResult",
    "BenchmarkConfig",
    # Retry schemas
    "RetryConfig",
    # Token budget schemas
    "TokenBudget",
    # Vectorstore schemas
    "FusionConfig",
    "FusionResult",
    "RetrievalMetrics",
    "IndexMapping",
    "VectorstoreEvaluationMetrics",
    "UpliftMetrics",
]
