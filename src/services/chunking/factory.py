from .client import ChunkingClient


def get_chunking_client(chunk_size: int = 512) -> ChunkingClient:
    return ChunkingClient(chunk_size=chunk_size)
