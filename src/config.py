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


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses lru_cache to ensure settings are loaded only once.
    """
    return Settings()


# Convenience: default settings instance
settings = get_settings()
