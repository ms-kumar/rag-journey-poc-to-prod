"""
Application configuration using Pydantic Settings.

Reads from environment variables and .env file.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE_PATH = PROJECT_ROOT / ".env"


class BaseConfigSettings(BaseSettings):
    """Base configuration with shared settings."""

    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        extra="ignore",
        frozen=True,
        env_nested_delimiter="__",
        case_sensitive=False,
    )


class AppSettings(BaseConfigSettings):
    """Application-level settings."""

    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="APP__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    name: str = "Advanced RAG API"
    version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"


class ServerSettings(BaseConfigSettings):
    """Server configuration settings."""

    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="SERVER__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    host: str = "0.0.0.0"  # nosec B104
    port: int = 8000


class IngestionSettings(BaseConfigSettings):
    """Document ingestion settings."""

    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="INGESTION__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    dir: str = "./data"

    @field_validator("dir")
    @classmethod
    def validate_ingestion_dir(cls, v: str) -> str:
        Path(v).mkdir(parents=True, exist_ok=True)
        return v


class ChunkingSettings(BaseConfigSettings):
    """Chunking configuration settings."""

    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="CHUNKING__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    chunk_size: int = 200  # Characters per chunk
    chunk_overlap: int = 50  # Overlapping characters
    strategy: str = "heading_aware"  # fixed or heading_aware


class EmbeddingSettings(BaseConfigSettings):
    """Embedding model settings."""

    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="EMBED__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    provider: str = "hash"  # hash, e5, bge, huggingface, openai, cohere
    model: str = "simple-hash"
    dim: int = 64
    device: str | None = None  # cpu, cuda, or None for auto
    normalize: bool = True
    api_key: str | None = None
    batch_size: int = 32

    # Cache settings
    cache_enabled: bool = True
    cache_max_size: int = 10000
    cache_dir: str | None = ".cache/embeddings"

    @field_validator("cache_dir")
    @classmethod
    def validate_cache_dir(cls, v: str | None) -> str | None:
        if v:
            Path(v).mkdir(parents=True, exist_ok=True)
        return v


class VectorStoreSettings(BaseConfigSettings):
    """Qdrant vector store settings."""

    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="QDRANT__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    url: str | None = None
    api_key: str | None = None
    collection_name: str = "naive_collection"
    prefer_grpc: bool = True
    distance: str = "Cosine"  # Cosine, Dot, or Euclid
    vector_size: int = 64  # Default embedding dimension
    enable_bm25: bool = False  # Enable BM25 indexing for hybrid search

    # Advanced features
    enable_metrics: bool = False  # Track retrieval metrics
    normalize_scores: bool = False  # Normalize scores to [0, 1]
    enable_sparse: bool = False  # Enable sparse vector storage (SPLADE)
    sparse_vector_name: str = "sparse"  # Name for sparse vector field


class GenerationSettings(BaseConfigSettings):
    """Text generation model settings."""

    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="GENERATOR__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    model: str = "gpt2"
    device: int | None = None  # -1 for CPU, 0+ for GPU
    max_length: int = 128
    temperature: float = 1.0


class RAGSettings(BaseConfigSettings):
    """RAG pipeline settings."""

    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="RAG__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    top_k: int = 5  # Number of documents to retrieve
    max_context_docs: int = 3  # Max documents in context


class RerankerSettings(BaseConfigSettings):
    """Reranker configuration settings."""

    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="RERANKER__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    max_length: int = 512
    batch_size: int = 32
    timeout_seconds: float = 30.0
    device: str | None = None
    fallback_enabled: bool = True
    fallback_strategy: str = "original_order"  # "original_order" or "score_descending"
    use_fp16: bool = True


class QueryUnderstandingSettings(BaseConfigSettings):
    """Query understanding configuration settings."""

    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="QUERY__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    # Query rewriting settings
    enable_rewriting: bool = True
    expand_acronyms: bool = True
    fix_typos: bool = True
    add_context: bool = True
    max_rewrites: int = 3
    min_query_length: int = 3

    # Synonym expansion settings
    enable_synonyms: bool = True
    max_synonyms_per_term: int = 3
    min_term_length: int = 3
    expand_all_terms: bool = False

    # Intent classification settings
    enable_intent_classification: bool = False


class RedisSettings(BaseConfigSettings):
    """Redis connection settings."""

    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="REDIS__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str | None = None
    max_connections: int = 10
    socket_timeout: int = 5
    decode_responses: bool = True


class CacheSettings(BaseConfigSettings):
    """Cache configuration settings."""

    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="CACHE__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    # Redis connection (nested settings)
    redis: RedisSettings = Field(default_factory=RedisSettings)

    # Cache behavior
    default_ttl: int = 3600  # Default TTL in seconds (1 hour)
    key_prefix: str = "rag:"  # Prefix for all cache keys
    enabled: bool = True  # Enable/disable caching globally

    # Semantic cache settings
    semantic_similarity_threshold: float = 0.95  # Cosine similarity threshold
    semantic_embedding_dim: int = 384  # Embedding dimension
    semantic_max_candidates: int = 100  # Max candidates to check

    # Staleness monitoring
    staleness_check_interval: int = 300  # Check interval in seconds (5 min)
    staleness_threshold: int = 3600  # Staleness threshold in seconds (1 hour)
    staleness_auto_invalidate: bool = False  # Auto-invalidate stale entries

    # Performance targets
    target_hit_rate: float = 0.6  # Target cache hit rate (60%)


class Settings(BaseConfigSettings):
    """Aggregated settings for the entire application."""

    # Nested settings using Field with default_factory
    app: AppSettings = Field(default_factory=AppSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    vectorstore: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    query_understanding: QueryUnderstandingSettings = Field(
        default_factory=QueryUnderstandingSettings
    )
    cache: CacheSettings = Field(default_factory=CacheSettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses lru_cache to ensure settings are loaded only once.
    """
    return Settings()


# Convenience: default settings instance
settings = get_settings()
