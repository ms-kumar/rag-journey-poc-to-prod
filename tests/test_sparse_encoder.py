"""
Tests for SPLADE sparse encoder.

Tests cover:
- Configuration
- Lazy loading
- Single query encoding
- Batch document encoding
- Sparse vector format validation
- Error handling
"""

import pytest


class TestSparseEncoderConfig:
    """Test SparseEncoderConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from src.services.embeddings.sparse_encoder import SparseEncoderConfig

        config = SparseEncoderConfig()
        assert config.model_name == "naver/splade-cocondenser-ensembledistil"
        assert config.device == "cpu"
        assert config.batch_size == 32
        assert config.max_length == 256

    def test_custom_config(self):
        """Test custom configuration."""
        from src.services.embeddings.sparse_encoder import SparseEncoderConfig

        config = SparseEncoderConfig(
            model_name="custom-model",
            device="cuda",
            batch_size=16,
            max_length=512,
        )
        assert config.model_name == "custom-model"
        assert config.device == "cuda"
        assert config.batch_size == 16
        assert config.max_length == 512


class TestSPLADEEncoder:
    """Test SPLADEEncoder."""

    def test_initialization(self):
        """Test encoder initialization."""
        from src.services.embeddings.sparse_encoder import SPLADEEncoder

        encoder = SPLADEEncoder()
        assert encoder._model is None
        assert encoder._tokenizer is None
        assert encoder._vocab_size is None

    def test_initialization_with_config(self):
        """Test encoder initialization with custom config."""
        from src.services.embeddings.sparse_encoder import (
            SparseEncoderConfig,
            SPLADEEncoder,
        )

        config = SparseEncoderConfig(batch_size=16)
        encoder = SPLADEEncoder(config)
        assert encoder.config.batch_size == 16

    def test_lazy_loading(self):
        """Test that model is loaded lazily on first use."""
        from src.services.embeddings.sparse_encoder import SPLADEEncoder

        encoder = SPLADEEncoder()
        assert encoder._model is None

        # Trigger loading (will load real model if transformers installed)
        # Just verify lazy loading happens
        encoder._ensure_loaded()
        assert encoder._model is not None
        assert encoder._tokenizer is not None
        assert encoder._vocab_size is not None

    def test_encode_query_format(self, mock_splade_model):
        """Test that encode_query returns correct sparse format."""
        from src.services.embeddings.sparse_encoder import SPLADEEncoder

        encoder = SPLADEEncoder()
        encoder._model = mock_splade_model["model"]
        encoder._tokenizer = mock_splade_model["tokenizer"]
        encoder._vocab_size = 30522

        sparse_vec = encoder.encode_query("machine learning")

        # Check it's a dict with int keys and float values
        assert isinstance(sparse_vec, dict)
        assert len(sparse_vec) > 0
        for token_id, weight in sparse_vec.items():
            assert isinstance(token_id, int)
            assert isinstance(weight, float)
            assert weight > 0.0  # Only non-zero weights

    def test_encode_documents_format(self, mock_splade_model):
        """Test that encode_documents returns list of sparse vectors."""
        from src.services.embeddings.sparse_encoder import SPLADEEncoder

        encoder = SPLADEEncoder()
        encoder._model = mock_splade_model["model"]
        encoder._tokenizer = mock_splade_model["tokenizer"]
        encoder._vocab_size = 30522

        texts = ["machine learning", "deep learning", "neural networks"]
        sparse_vecs = encoder.encode_documents(texts)

        assert len(sparse_vecs) == 3
        for sparse_vec in sparse_vecs:
            assert isinstance(sparse_vec, dict)
            assert len(sparse_vec) > 0
            for token_id, weight in sparse_vec.items():
                assert isinstance(token_id, int)
                assert isinstance(weight, float)
                assert weight > 0.0

    def test_encode_batch_processing(self, mock_splade_model):
        """Test that large batches are processed correctly."""
        from src.services.embeddings.sparse_encoder import SPLADEEncoder

        encoder = SPLADEEncoder()
        encoder._model = mock_splade_model["model"]
        encoder._tokenizer = mock_splade_model["tokenizer"]
        encoder._vocab_size = 30522
        encoder.config.batch_size = 2

        # More texts than batch size
        texts = [f"document {i}" for i in range(5)]
        sparse_vecs = encoder.encode_documents(texts)

        assert len(sparse_vecs) == 5

    def test_vocab_size_property(self, mock_splade_model):
        """Test vocab_size property."""
        from src.services.embeddings.sparse_encoder import SPLADEEncoder

        encoder = SPLADEEncoder()
        encoder._model = mock_splade_model["model"]
        encoder._tokenizer = mock_splade_model["tokenizer"]
        encoder._vocab_size = 30522

        assert encoder.vocab_size == 30522

    def test_repr(self):
        """Test string representation."""
        from src.services.embeddings.sparse_encoder import SPLADEEncoder

        encoder = SPLADEEncoder()
        repr_str = repr(encoder)
        assert "SPLADEEncoder" in repr_str
        assert "naver/splade-cocondenser-ensembledistil" in repr_str
        assert "cpu" in repr_str


class TestCreateSpladeEncoder:
    """Test create_splade_encoder factory function."""

    def test_factory_default(self):
        """Test factory with default parameters."""
        from src.services.embeddings.sparse_encoder import create_splade_encoder

        encoder = create_splade_encoder()
        assert encoder.config.model_name == "naver/splade-cocondenser-ensembledistil"
        assert encoder.config.device == "cpu"
        assert encoder.config.batch_size == 32

    def test_factory_custom(self):
        """Test factory with custom parameters."""
        from src.services.embeddings.sparse_encoder import create_splade_encoder

        encoder = create_splade_encoder(
            model_name="custom-model",
            device="cuda",
            batch_size=16,
        )
        assert encoder.config.model_name == "custom-model"
        assert encoder.config.device == "cuda"
        assert encoder.config.batch_size == 16


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_splade_model(monkeypatch):
    """Mock transformers model and tokenizer."""
    import torch

    class MockTokenizer:
        """Mock tokenizer."""

        def __call__(self, texts, **kwargs):
            """Mock tokenization."""
            batch_size = len(texts) if isinstance(texts, list) else 1
            return {
                "input_ids": torch.randint(0, 30522, (batch_size, 10)),
                "attention_mask": torch.ones(batch_size, 10),
            }

    class MockConfig:
        """Mock model config."""

        vocab_size = 30522

    class MockModel:
        """Mock SPLADE model."""

        config = MockConfig()

        def to(self, device):
            """Mock device placement."""
            return self

        def eval(self):
            """Mock eval mode."""
            return self

        def __call__(self, **kwargs):
            """Mock forward pass."""
            batch_size = kwargs["input_ids"].shape[0]
            seq_len = kwargs["input_ids"].shape[1]

            # Create mock logits with some non-zero values
            logits = torch.randn(batch_size, seq_len, 30522)

            class MockOutput:
                """Mock model output."""

                def __init__(self, logits):
                    self.logits = logits

            return MockOutput(logits)

    # Mock transformers imports
    class MockAutoTokenizer:
        """Mock AutoTokenizer."""

        @staticmethod
        def from_pretrained(model_name):
            """Mock from_pretrained."""
            return MockTokenizer()

    class MockAutoModelForMaskedLM:
        """Mock AutoModelForMaskedLM."""

        @staticmethod
        def from_pretrained(model_name):
            """Mock from_pretrained."""
            return MockModel()

    # Patch transformers
    def mock_import_transformers():
        """Mock transformers import."""
        import sys
        from types import ModuleType

        transformers = ModuleType("transformers")
        transformers.AutoTokenizer = MockAutoTokenizer
        transformers.AutoModelForMaskedLM = MockAutoModelForMaskedLM
        sys.modules["transformers"] = transformers

    # Create mock but don't patch yet - let tests control when to load
    return {
        "model": MockModel(),
        "tokenizer": MockTokenizer(),
    }
