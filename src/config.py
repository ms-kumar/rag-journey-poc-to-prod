"""
Application configuration using Pydantic Settings.

Reads from environment variables and .env file.
"""

from functools import lru_cache
from typing import Optional

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
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")

    # --- Ingestion Settings ---
    ingestion_dir: str = Field(
        default="./data", description="Directory for document ingestion"
    )

    # --- Chunking Settings ---
    chunk_size: int = Field(default=200, description="Number of words per chunk")

    # --- Embedding Settings ---
    embed_dim: int = Field(default=64, description="Embedding vector dimension")
    embed_model: str = Field(default="simple-hash", description="Embedding model name")

    # --- Vectorstore Settings (Qdrant) ---
    qdrant_url: Optional[str] = Field(default=None, description="Qdrant server URL")
    qdrant_api_key: Optional[str] = Field(default=None, description="Qdrant API key")
    qdrant_collection_name: str = Field(
        default="naive_collection", description="Qdrant collection name"
    )
    qdrant_prefer_grpc: bool = Field(default=True, description="Prefer gRPC for Qdrant")

    # --- Generation Settings ---
    generator_model: str = Field(
        default="gpt2", description="HuggingFace model for generation"
    )
    generator_device: Optional[int] = Field(
        default=None, description="Device ID (-1 for CPU, 0+ for GPU)"
    )
    generator_max_length: int = Field(default=128, description="Max generation length")
    generator_temperature: float = Field(
        default=1.0, description="Generation temperature"
    )

    # --- RAG Settings ---
    rag_top_k: int = Field(
        default=5, description="Default number of documents to retrieve"
    )
    rag_max_context_docs: int = Field(
        default=3, description="Max documents to include in context"
    )


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses lru_cache to ensure settings are loaded only once.
    """
    return Settings()


# Convenience: default settings instance
settings = get_settings()
