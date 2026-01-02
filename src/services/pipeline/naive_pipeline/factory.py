from typing import Optional

from src.config import settings
from .client import NaivePipeline, NaivePipelineConfig


def get_naive_pipeline(
    ingestion_dir: Optional[str] = None,
    chunk_size: Optional[int] = None,
    embed_dim: Optional[int] = None,
    qdrant_url: Optional[str] = None,
    collection_name: Optional[str] = None,
    generator_model: Optional[str] = None,
    generator_device: Optional[int] = None,
) -> NaivePipeline:
    """
    Factory function to create a NaivePipeline instance.

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
    """
    config = NaivePipelineConfig(
        ingestion_dir=ingestion_dir or settings.ingestion_dir,
        chunk_size=chunk_size or settings.chunk_size,
        embed_dim=embed_dim or settings.embed_dim,
        qdrant_url=qdrant_url or settings.qdrant_url,
        collection_name=collection_name or settings.qdrant_collection_name,
        generator_model=generator_model or settings.generator_model,
        generator_device=generator_device if generator_device is not None else settings.generator_device,
    )
    return NaivePipeline(config)


__all__ = ["get_naive_pipeline", "NaivePipeline", "NaivePipelineConfig"]