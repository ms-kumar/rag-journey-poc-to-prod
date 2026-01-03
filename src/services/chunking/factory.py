from src.config import settings

from .client import ChunkingClient, HeadingAwareChunker


def get_chunking_client(chunk_size: int = 512, strategy: str | None = None) -> ChunkingClient:
    selected = (strategy or settings.chunking_strategy or "fixed").strip().lower()

    if selected in {"heading_aware", "heading-aware", "markdown"}:
        return HeadingAwareChunker(chunk_size=chunk_size)  # type: ignore[return-value]

    return ChunkingClient(chunk_size=chunk_size)
