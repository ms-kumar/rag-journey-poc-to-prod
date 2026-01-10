"""
Custom exceptions for the RAG application.

Provides a hierarchical exception structure for better error handling
and debugging across all services.
"""

# ruff: noqa: N818  # Base exception classes don't need Error suffix


# ============================================
# Repository Exceptions
# ============================================
class RepositoryException(Exception):
    """Base exception for repository-related errors."""


class DataNotFound(RepositoryException):
    """Exception raised when data is not found."""


class DataNotSaved(RepositoryException):
    """Exception raised when data cannot be saved."""


# ============================================
# Cache Exceptions
# ============================================
class CacheException(Exception):
    """Base exception for cache-related errors."""


class RedisCacheException(CacheException):
    """Exception raised for Redis cache errors."""


class RedisConnectionError(RedisCacheException):
    """Exception raised when cannot connect to Redis."""


class RedisTimeoutError(RedisCacheException):
    """Exception raised when Redis operation times out."""


class SemanticCacheException(CacheException):
    """Exception raised for semantic cache errors."""


class CacheKeyError(CacheException):
    """Exception raised for invalid cache key operations."""


class CacheSerializationError(CacheException):
    """Exception raised when cache serialization fails."""


# ============================================
# Vector Store Exceptions
# ============================================
class VectorStoreException(Exception):
    """Base exception for vector store errors."""


class QdrantException(VectorStoreException):
    """Exception raised for Qdrant-related errors."""


class QdrantConnectionError(QdrantException):
    """Exception raised when cannot connect to Qdrant."""


class QdrantTimeoutError(QdrantException):
    """Exception raised when Qdrant operation times out."""


class CollectionNotFound(QdrantException):
    """Exception raised when collection does not exist."""


class CollectionCreationError(QdrantException):
    """Exception raised when collection creation fails."""


class VectorSearchError(VectorStoreException):
    """Exception raised when vector search fails."""


class VectorIndexingError(VectorStoreException):
    """Exception raised when vector indexing fails."""


# ============================================
# Embedding Exceptions
# ============================================
class EmbeddingException(Exception):
    """Base exception for embedding-related errors."""


class EmbeddingProviderError(EmbeddingException):
    """Exception raised for embedding provider errors."""


class EmbeddingAPIError(EmbeddingException):
    """Exception raised for embedding API errors."""


class EmbeddingTimeoutError(EmbeddingException):
    """Exception raised when embedding generation times out."""


class EmbeddingDimensionMismatch(EmbeddingException):
    """Exception raised when embedding dimensions don't match."""


class EmbeddingCacheError(EmbeddingException):
    """Exception raised for embedding cache errors."""


class SparseEncodingError(EmbeddingException):
    """Exception raised for sparse encoding errors."""


# ============================================
# Reranking Exceptions
# ============================================
class RerankerException(Exception):
    """Base exception for reranker errors."""


class RerankerModelError(RerankerException):
    """Exception raised when reranker model fails."""


class RerankerTimeoutError(RerankerException):
    """Exception raised when reranking times out."""


class RerankerDeviceError(RerankerException):
    """Exception raised for device-related errors."""


class RerankerFallbackError(RerankerException):
    """Exception raised when fallback strategy fails."""


# ============================================
# Query Understanding Exceptions
# ============================================
class QueryUnderstandingException(Exception):
    """Base exception for query understanding errors."""


class QueryRewriteError(QueryUnderstandingException):
    """Exception raised when query rewriting fails."""


class SynonymExpansionError(QueryUnderstandingException):
    """Exception raised when synonym expansion fails."""


class IntentClassificationError(QueryUnderstandingException):
    """Exception raised when intent classification fails."""


class QueryValidationError(QueryUnderstandingException):
    """Exception raised when query validation fails."""


# ============================================
# Generation Exceptions
# ============================================
class GenerationException(Exception):
    """Base exception for text generation errors."""


class LLMException(GenerationException):
    """Base exception for LLM-related errors."""


class LLMConnectionError(LLMException):
    """Exception raised when cannot connect to LLM service."""


class LLMTimeoutError(LLMException):
    """Exception raised when LLM request times out."""


class LLMRateLimitError(LLMException):
    """Exception raised when LLM rate limit is exceeded."""


class LLMAPIError(LLMException):
    """Exception raised for LLM API errors."""


class PromptTooLongError(GenerationException):
    """Exception raised when prompt exceeds token limit."""


class GenerationTimeoutError(GenerationException):
    """Exception raised when generation times out."""


# ============================================
# Chunking Exceptions
# ============================================
class ChunkingException(Exception):
    """Base exception for chunking errors."""


class ChunkSizeError(ChunkingException):
    """Exception raised when chunk size is invalid."""


class ChunkStrategyError(ChunkingException):
    """Exception raised when chunking strategy fails."""


class TextTruncationError(ChunkingException):
    """Exception raised when text truncation fails."""


# ============================================
# Ingestion Exceptions
# ============================================
class IngestionException(Exception):
    """Base exception for ingestion errors."""


class DocumentParsingError(IngestionException):
    """Exception raised when document parsing fails."""


class FileFormatError(IngestionException):
    """Exception raised for unsupported file formats."""


class FileNotFoundError(IngestionException):
    """Exception raised when file is not found."""


class IngestionValidationError(IngestionException):
    """Exception raised when ingestion validation fails."""


# ============================================
# Pipeline Exceptions
# ============================================
class PipelineException(Exception):
    """Base exception for pipeline errors."""


class PipelineConfigurationError(PipelineException):
    """Exception raised for pipeline configuration errors."""


class PipelineExecutionError(PipelineException):
    """Exception raised during pipeline execution."""


class PipelineTimeoutError(PipelineException):
    """Exception raised when pipeline execution times out."""


class RAGPipelineError(PipelineException):
    """Exception raised for RAG pipeline errors."""


# ============================================
# Retry & Health Check Exceptions
# ============================================
class RetryException(Exception):
    """Base exception for retry mechanism errors."""


class MaxRetriesExceeded(RetryException):
    """Exception raised when max retry attempts exceeded."""


class RetryableError(RetryException):
    """Exception that can be retried."""


class NonRetryableError(RetryException):
    """Exception that should not be retried."""


class HealthCheckException(Exception):
    """Base exception for health check errors."""


class ServiceUnavailableError(HealthCheckException):
    """Exception raised when service is unavailable."""


class ServiceDegradedError(HealthCheckException):
    """Exception raised when service is degraded."""


# ============================================
# Configuration Exceptions
# ============================================
class ConfigurationException(Exception):
    """Base exception for configuration errors."""


class SettingsValidationError(ConfigurationException):
    """Exception raised when settings validation fails."""


class EnvironmentVariableError(ConfigurationException):
    """Exception raised for environment variable errors."""


class ConfigurationFileError(ConfigurationException):
    """Exception raised for configuration file errors."""


# ============================================
# Evaluation Exceptions
# ============================================
class EvaluationException(Exception):
    """Base exception for evaluation errors."""


class MetricCalculationError(EvaluationException):
    """Exception raised when metric calculation fails."""


class ThresholdViolationError(EvaluationException):
    """Exception raised when evaluation threshold is violated."""


class EvaluationDataError(EvaluationException):
    """Exception raised for evaluation data errors."""


# ============================================
# General Application Exceptions
# ============================================
class ApplicationException(Exception):
    """Base exception for application-level errors."""


class InitializationError(ApplicationException):
    """Exception raised during application initialization."""


class ShutdownError(ApplicationException):
    """Exception raised during application shutdown."""


class InvalidInputError(ApplicationException):
    """Exception raised for invalid input data."""


class ResourceNotFoundError(ApplicationException):
    """Exception raised when resource is not found."""


class PermissionError(ApplicationException):
    """Exception raised for permission-related errors."""
