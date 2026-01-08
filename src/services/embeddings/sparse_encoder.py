"""
SPLADE sparse encoder for neural sparse retrieval.

SPLADE (Sparse Lexical AnD Expansion model) produces learned sparse representations
that are more effective than traditional BM25 while maintaining efficiency.
"""

import logging

logger = logging.getLogger(__name__)


class SparseEncoderConfig:
    """Configuration for SPLADE encoder."""

    def __init__(
        self,
        model_name: str = "naver/splade-cocondenser-ensembledistil",
        device: str = "cpu",
        batch_size: int = 32,
        max_length: int = 256,
    ):
        """
        Initialize SPLADE encoder config.

        Args:
            model_name: HuggingFace model name for SPLADE
            device: Device to run model on ('cpu' or 'cuda')
            batch_size: Batch size for encoding
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length


class SPLADEEncoder:
    """
    SPLADE encoder for producing sparse representations.

    SPLADE uses a transformer model to produce sparse, interpretable representations
    where each dimension corresponds to a vocabulary term with learned importance.
    """

    def __init__(self, config: SparseEncoderConfig | None = None):
        """
        Initialize SPLADE encoder.

        Args:
            config: Encoder configuration
        """
        self.config = config or SparseEncoderConfig()
        self._model = None
        self._tokenizer = None
        self._vocab_size = None

    def _ensure_loaded(self) -> None:
        """Lazy load model and tokenizer."""
        if self._model is not None:
            return

        try:
            from transformers import AutoModelForMaskedLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "transformers is required for SPLADE. Install with: pip install transformers torch"
            ) from e

        logger.info(f"Loading SPLADE model: {self.config.model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self._model = AutoModelForMaskedLM.from_pretrained(self.config.model_name)
        assert self._model is not None  # type guard
        self._model.to(self.config.device)
        self._model.eval()
        self._vocab_size = self._model.config.vocab_size
        logger.info(f"SPLADE model loaded with vocab size: {self._vocab_size}")

    def encode(self, texts: list[str]) -> list[dict[int, float]]:
        """
        Encode texts into sparse representations.

        Args:
            texts: List of texts to encode

        Returns:
            List of sparse vectors as {token_id: weight} dicts

        Example:
            >>> encoder = SPLADEEncoder()
            >>> sparse_vecs = encoder.encode(["machine learning", "deep learning"])
            >>> # Each vector: {2341: 0.8, 5123: 0.6, ...}
        """
        self._ensure_loaded()
        assert self._model is not None
        assert self._tokenizer is not None

        import torch

        sparse_vectors = []

        # Process in batches
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i : i + self.config.batch_size]

            # Tokenize
            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

            # Forward pass
            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits

            # Compute sparse vectors (max over token positions, apply ReLU)
            # Shape: [batch_size, seq_len, vocab_size] -> [batch_size, vocab_size]
            sparse_repr = torch.max(torch.log(1 + torch.relu(logits)), dim=1).values

            # Convert to sparse format: {token_id: weight}
            for batch_idx in range(sparse_repr.shape[0]):
                vec = sparse_repr[batch_idx]
                # Keep only non-zero weights
                sparse_dict = {
                    int(token_id): float(weight)
                    for token_id, weight in enumerate(vec.cpu().numpy())
                    if weight > 0
                }
                sparse_vectors.append(sparse_dict)

        return sparse_vectors

    def encode_query(self, query: str) -> dict[int, float]:
        """
        Encode a single query into sparse representation.

        Args:
            query: Query text

        Returns:
            Sparse vector as {token_id: weight} dict
        """
        return self.encode([query])[0]

    def encode_documents(self, documents: list[str]) -> list[dict[int, float]]:
        """
        Encode documents into sparse representations.

        Args:
            documents: List of document texts

        Returns:
            List of sparse vectors as {token_id: weight} dicts
        """
        return self.encode(documents)

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        self._ensure_loaded()
        return self._vocab_size or 0

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SPLADEEncoder(model={self.config.model_name}, "
            f"device={self.config.device}, vocab_size={self.vocab_size})"
        )


def create_splade_encoder(
    model_name: str = "naver/splade-cocondenser-ensembledistil",
    device: str = "cpu",
    batch_size: int = 32,
) -> SPLADEEncoder:
    """
    Factory function to create SPLADE encoder.

    Args:
        model_name: HuggingFace model name
        device: Device to run on
        batch_size: Batch size for encoding

    Returns:
        SPLADEEncoder instance

    Example:
        >>> encoder = create_splade_encoder(device="cuda")
        >>> sparse_vecs = encoder.encode(["neural information retrieval"])
    """
    config = SparseEncoderConfig(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
    )
    return SPLADEEncoder(config)
