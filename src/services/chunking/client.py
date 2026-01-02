from typing import List

class ChunkingClient:
    def __init__(self, chunk_size: int = 512):
        self.chunk_size = chunk_size

    def chunk(self, documents: List[str]) -> List[str]:
        """
        Splits each document into chunks of specified size.
        """
        chunks = []
        for doc in documents:
            for i in range(0, len(doc), self.chunk_size):
                chunks.append(doc[i:i + self.chunk_size])
        return chunks
