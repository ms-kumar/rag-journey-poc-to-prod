"""
Factory for creating chunking clients from application settings.
"""

import logging
from typing import TYPE_CHECKING

from src.services.chunking.client import ChunkingClient, HeadingAwareChunker

if TYPE_CHECKING:
    from src.config import ChunkingSettings

logger = logging.getLogger(__name__)


def get_chunking_client(
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    strategy: str | None = None,
) -> ChunkingClient:
    """Create chunking client with optional strategy override."""
    selected = (strategy or "fixed").strip().lower()

    if selected in {"heading_aware", "heading-aware", "markdown"}:
        logger.info(f"Created HeadingAwareChunker with chunk_size={chunk_size}")
        return HeadingAwareChunker(chunk_size=chunk_size)  # type: ignore[return-value]

    logger.info(f"Created ChunkingClient with chunk_size={chunk_size}")
    return ChunkingClient(chunk_size=chunk_size)


def make_chunking_client(settings: "ChunkingSettings") -> ChunkingClient:
    """
    Create chunking client from application settings.

    Args:
        settings: Chunking settings

    Returns:
        Configured chunking client
    """
    return get_chunking_client(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        strategy=settings.strategy,
    )


def create_from_settings(settings: "ChunkingSettings", **overrides) -> ChunkingClient:
    """
    Create chunking client from application settings with optional overrides.

    Deprecated: Use make_chunking_client() instead.
    """
    return get_chunking_client(
        chunk_size=overrides.get("chunk_size", settings.chunk_size),
        chunk_overlap=overrides.get("chunk_overlap", settings.chunk_overlap),
        strategy=overrides.get("strategy", settings.strategy),
    )
