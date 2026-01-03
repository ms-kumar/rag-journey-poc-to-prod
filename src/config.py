"""
Application configuration using Pydantic Settings.

Reads from environment variables and .env file.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- API Settings ---
    app_name: str = Field(default="Advanced RAG API", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")

    # --- Server Settings ---
    host: str = Field(default="0.0.0.0", description="Server host")  # nosec B104
    port: int = Field(default=8000, description="Server port")

    # --- Ingestion Settings ---
    ingestion_dir: str = Field(default="./data", description="Directory for document ingestion")

    # --- Chunking Settings ---
    chunk_size: int = Field(default=200, description="Number of words per chunk")
    chunk_overlap: int = Field(
        default=50, description="Number of overlapping characters between chunks"
    )
    chunking_strategy: str = Field(
        default="heading_aware",
        description="Chunking strategy: fixed|heading_aware",
    )

    # --- Embedding Settings ---
    embed_provider: str = Field(
        default="hash", description="Embedding provider: hash, e5, bge, huggingface, openai, cohere"
    )
    embed_model: str = Field(default="simple-hash", description="Embedding model name/identifier")
    embed_dim: int = Field(default=64, description="Embedding vector dimension (for hash provider)")
    embed_device: str | None = Field(
        default=None, description="Device for local embeddings (cpu, cuda, or None for auto)"
    )
    embed_normalize: bool = Field(default=True, description="Whether to normalize embeddings")
    embed_api_key: str | None = Field(
        default=None, description="API key for external embedding providers (OpenAI, Cohere)"
    )
    embed_batch_size: int = Field(default=32, description="Batch size for embedding operations")

    # --- Embedding Cache Settings ---
    embed_cache_enabled: bool = Field(
        default=True, description="Whether to enable embedding caching"
    )
    embed_cache_max_size: int = Field(
        default=10000, description="Maximum number of embeddings to cache in memory"
    )
    embed_cache_dir: str | None = Field(
        default=".cache/embeddings", description="Directory for disk cache (None to disable)"
    )

    # --- Vectorstore Settings (Qdrant) ---
    qdrant_url: str | None = Field(default=None, description="Qdrant server URL")
    qdrant_api_key: str | None = Field(default=None, description="Qdrant API key")
    qdrant_collection_name: str = Field(
        default="naive_collection", description="Qdrant collection name"
    )
    qdrant_prefer_grpc: bool = Field(default=True, description="Prefer gRPC for Qdrant")

    # --- Generation Settings ---
    generator_model: str = Field(default="gpt2", description="HuggingFace model for generation")
    generator_device: int | None = Field(
        default=None, description="Device ID (-1 for CPU, 0+ for GPU)"
    )
    generator_max_length: int = Field(default=128, description="Max generation length")
    generator_temperature: float = Field(default=1.0, description="Generation temperature")

    # --- RAG Settings ---
    rag_top_k: int = Field(default=5, description="Default number of documents to retrieve")
    rag_max_context_docs: int = Field(default=3, description="Max documents to include in context")


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses lru_cache to ensure settings are loaded only once.
    """
    return Settings()


# Convenience: default settings instance
settings = get_settings()
