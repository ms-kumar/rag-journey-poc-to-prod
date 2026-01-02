from typing import Optional

from .client import NaivePipeline, NaivePipelineConfig


def get_naive_pipeline(
    ingestion_dir: str = "./data",
    chunk_size: int = 200,
    embed_dim: int = 64,
    qdrant_url: Optional[str] = None,
    collection_name: str = "naive_collection",
    generator_model: str = "gpt2",
    generator_device: Optional[int] = None,
) -> NaivePipeline:
    """
    Factory function to create a NaivePipeline instance.

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
        ingestion_dir=ingestion_dir,
        chunk_size=chunk_size,
        embed_dim=embed_dim,
        qdrant_url=qdrant_url,
        collection_name=collection_name,
        generator_model=generator_model,
        generator_device=generator_device,
    )
    return NaivePipeline(config)


__all__ = ["get_naive_pipeline", "NaivePipeline", "NaivePipelineConfig"]