from typing import TYPE_CHECKING

from .client import ChunkingClient, HeadingAwareChunker

if TYPE_CHECKING:
    from src.config import Settings


def get_chunking_client(
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    strategy: str | None = None,
) -> ChunkingClient:
    """Create chunking client with optional strategy override."""
    selected = (strategy or "fixed").strip().lower()

    if selected in {"heading_aware", "heading-aware", "markdown"}:
        return HeadingAwareChunker(chunk_size=chunk_size)  # type: ignore[return-value]

    return ChunkingClient(chunk_size=chunk_size)


def create_from_settings(settings: "Settings", **overrides) -> ChunkingClient:
    """Create chunking client from application settings."""
    chunking_settings = settings.chunking
    return get_chunking_client(
        chunk_size=overrides.get("chunk_size", chunking_settings.chunk_size),
        chunk_overlap=overrides.get("chunk_overlap", chunking_settings.chunk_overlap),
        strategy=overrides.get("strategy", chunking_settings.strategy),
    )
