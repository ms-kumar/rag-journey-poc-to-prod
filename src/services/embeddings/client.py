from typing import List, Sequence
import hashlib
import math


class EmbedClient:
    """
    Simple deterministic embedder that maps text -> fixed-size vector.

    Implementation notes:
    - Uses SHA-256 of (text + index) to generate each dimension deterministically.
    - Maps 64-bit slice of the hash to a float in [-1, 1].
    - Optionally L2-normalizes the resulting vector.
    - No external dependencies required.
    """

    def __init__(self, model_name: str = "simple-hash", dim: int = 64, normalize: bool = True):
        self.model_name = model_name
        self.dim = int(dim)
        if self.dim <= 0:
            raise ValueError("dim must be a positive integer")
        self.normalize = bool(normalize)

    def _text_to_vector(self, text: str) -> List[float]:
        vec: List[float] = []
        for i in range(self.dim):
            # Use a unique salt per dimension so each dimension is deterministic but different
            digest = hashlib.sha256(f"{text}\x00{i}".encode("utf-8")).digest()
            # Take first 8 bytes (64 bits) and convert to integer
            num = int.from_bytes(digest[:8], "big", signed=False)
            # Map integer to float in [-1, 1]
            f = (num / (2**64 - 1)) * 2.0 - 1.0
            vec.append(f)

        if self.normalize:
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            vec = [x / norm for x in vec]

        return vec

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """
        Generate embeddings for a sequence of texts.

        Args:
          texts: Sequence[str] - list (or any sequence) of strings.

        Returns:
          List of vectors (list of floats) with length equal to `dim`.
        """
        return [self._text_to_vector(t if t is not None else "") for t in texts]
