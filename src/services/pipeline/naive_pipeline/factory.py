from src.config import settings

from .client import NaivePipeline, NaivePipelineConfig


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


__all__ = ["get_naive_pipeline", "NaivePipeline", "NaivePipelineConfig"]
