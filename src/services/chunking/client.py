from typing import List, Union


class ChunkingClient:
    def __init__(self, chunk_size: int = 512):
        self.chunk_size = chunk_size

    def chunk(self, documents: Union[str, List[str]]) -> List[str]:
        """
        Splits documents into chunks of specified size.
        Accepts either a single string or a list of strings.
        """
        # Handle single string input
        if isinstance(documents, str):
            documents = [documents]

        chunks = []
        for doc in documents:
            # Skip empty documents
            if not doc or not doc.strip():
                continue
            for i in range(0, len(doc), self.chunk_size):
                chunk = doc[i : i + self.chunk_size].strip()
                if chunk:  # Only add non-empty chunks
                    chunks.append(chunk)
        return chunks
