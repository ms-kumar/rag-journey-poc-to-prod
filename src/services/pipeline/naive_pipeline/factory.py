"""
Factory for creating naive pipeline from application settings.
"""

import logging
from typing import TYPE_CHECKING

from src.services.pipeline.naive_pipeline.client import NaivePipeline, NaivePipelineConfig

if TYPE_CHECKING:
    from src.config import Settings

logger = logging.getLogger(__name__)


def make_naive_pipeline(settings: "Settings") -> NaivePipeline:
    """
    Create naive pipeline from application settings.

    Args:
        settings: Application settings

    Returns:
        Configured NaivePipeline instance
    """
    config = NaivePipelineConfig(
        ingestion_dir=settings.ingestion.dir,
        chunk_size=settings.chunking.chunk_size,
        embed_dim=settings.embedding.dim,
        qdrant_url=settings.vectorstore.url,
        collection_name=settings.vectorstore.collection_name,
        generator_model=settings.generation.model,
        generator_device=settings.generation.device,
        enable_reranker=settings.reranker.fallback_enabled,
        reranker_model=settings.reranker.model_name,
        reranker_batch_size=settings.reranker.batch_size,
        reranker_timeout=settings.reranker.timeout_seconds,
    )
    logger.info(f"Naive pipeline created with collection={settings.vectorstore.collection_name}")
    return NaivePipeline(config)


def create_naive_pipeline(settings: "Settings") -> NaivePipeline:
    """
    Create naive pipeline from application settings.

    Deprecated: Use make_naive_pipeline() instead.

    Args:
        settings: Application settings

    Returns:
        Configured NaivePipeline instance
    """
    return make_naive_pipeline(settings)


def get_naive_pipeline(
    ingestion_dir: str | None = None,
    chunk_size: int | None = None,
    embed_dim: int | None = None,
    qdrant_url: str | None = None,
    collection_name: str | None = None,
    generator_model: str | None = None,
    generator_device: int | None = None,
) -> NaivePipeline:
    """
    Factory function to create a NaivePipeline instance (legacy function).

    Uses values from settings (loaded from .env) as defaults.
    Pass explicit values to override.

    Args:
        ingestion_dir: Directory containing .txt files to ingest.
        chunk_size: Number of words per chunk.
        embed_dim: Embedding vector dimension.
        qdrant_url: Qdrant server URL (e.g., "http://localhost:6333").
        collection_name: Qdrant collection name.
        generator_model: HuggingFace model name for generation.
        generator_device: Device ID for generation (-1 for CPU, 0+ for GPU).

    Returns:
        Configured NaivePipeline instance.

    Note:
        For new code, prefer using create_naive_pipeline(settings) instead.
    """
    from src.config import settings

    config = NaivePipelineConfig(
        ingestion_dir=ingestion_dir or settings.ingestion.dir,
        chunk_size=chunk_size or settings.chunking.chunk_size,
        embed_dim=embed_dim or settings.embedding.dim,
        qdrant_url=qdrant_url or settings.vectorstore.url,
        collection_name=collection_name or settings.vectorstore.collection_name,
        generator_model=generator_model or settings.generation.model,
        generator_device=generator_device
        if generator_device is not None
        else settings.generation.device,
    )
    return NaivePipeline(config)


__all__ = ["create_naive_pipeline", "get_naive_pipeline", "NaivePipeline", "NaivePipelineConfig"]
