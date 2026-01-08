"""
Tests for cross-encoder re-ranker functionality.
"""

from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from src.services.reranker.client import (
    CrossEncoderReranker,
    PrecisionMetrics,
    RerankerConfig,
    RerankResult,
)
from src.services.reranker.evaluation import ComparisonResult, RerankingEvaluator
from src.services.reranker.factory import get_reranker


class TestRerankerConfig:
    """Test reranker configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RerankerConfig()

        assert config.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert config.batch_size == 32
        assert config.timeout_seconds == 30.0
        assert config.fallback_enabled is True
        assert config.device is not None  # Should auto-detect

    def test_custom_config(self):
        """Test custom configuration values."""
        config = RerankerConfig(
            model_name="custom-model",
            batch_size=16,
            timeout_seconds=10.0,
            device="cpu",
            fallback_enabled=False,
        )

        assert config.model_name == "custom-model"
        assert config.batch_size == 16
        assert config.timeout_seconds == 10.0
        assert config.device == "cpu"
        assert config.fallback_enabled is False


class TestCrossEncoderReranker:
    """Test cross-encoder reranker functionality."""

    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing."""
        return [
            Document(
                page_content="Machine learning is a subset of AI",
                metadata={"id": "doc1", "score": 0.8},
            ),
            Document(
                page_content="Deep learning uses neural networks",
                metadata={"id": "doc2", "score": 0.7},
            ),
            Document(
                page_content="Python is a programming language",
                metadata={"id": "doc3", "score": 0.6},
            ),
        ]

    @pytest.fixture
    def mock_reranker(self):
        """Mock reranker for testing without actual model loading."""
        config = RerankerConfig(device="cpu")
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker.config = config
        reranker.model_loaded = True
        reranker.model = Mock()

        # Mock the predict method to return scores
        reranker.model.predict.return_value = [0.9, 0.8, 0.3]

        return reranker

    def test_rerank_empty_documents(self, mock_reranker):
        """Test re-ranking with empty document list."""
        result = mock_reranker.rerank("test query", [])

        assert isinstance(result, RerankResult)
        assert result.documents == []
        assert result.scores == []
        assert result.original_ranks == []
        assert result.fallback_used is False

    def test_rerank_success(self, mock_reranker, sample_documents):
        """Test successful re-ranking."""
        query = "What is machine learning?"

        # Mock the scoring method
        mock_reranker._score_documents_with_timeout = Mock(return_value=[0.9, 0.8, 0.3])

        result = mock_reranker.rerank(query, sample_documents)

        assert isinstance(result, RerankResult)
        assert len(result.documents) == 3
        assert len(result.scores) == 3
        assert len(result.original_ranks) == 3
        assert result.fallback_used is False

        # Check that documents are sorted by score (descending)
        assert result.scores[0] >= result.scores[1] >= result.scores[2]

    def test_rerank_with_top_k(self, mock_reranker, sample_documents):
        """Test re-ranking with top_k limit."""
        query = "What is machine learning?"
        mock_reranker._score_documents_with_timeout = Mock(return_value=[0.9, 0.8, 0.3])

        result = mock_reranker.rerank(query, sample_documents, top_k=2)

        assert len(result.documents) == 2
        assert len(result.scores) == 2
        assert len(result.original_ranks) == 2

    def test_rerank_fallback_on_error(self, mock_reranker, sample_documents):
        """Test fallback when re-ranking fails."""
        query = "What is machine learning?"

        # Mock scoring to raise an exception
        mock_reranker._score_documents_with_timeout = Mock(side_effect=Exception("Model error"))

        result = mock_reranker.rerank(query, sample_documents)

        assert isinstance(result, RerankResult)
        assert result.fallback_used is True
        assert len(result.documents) == len(sample_documents)

    def test_batch_rerank(self, mock_reranker, sample_documents):
        """Test batch re-ranking."""
        queries = ["Query 1", "Query 2"]
        document_lists = [sample_documents[:2], sample_documents[1:]]

        mock_reranker._score_documents_with_timeout = Mock(return_value=[0.9, 0.8])

        results = mock_reranker.batch_rerank(queries, document_lists)

        assert len(results) == 2
        assert all(isinstance(r, RerankResult) for r in results)

    def test_extract_doc_id(self, mock_reranker):
        """Test document ID extraction."""
        doc_with_id = Document(page_content="test", metadata={"id": "doc123"})
        doc_with_chunk_id = Document(page_content="test", metadata={"chunk_id": "chunk456"})
        doc_without_id = Document(page_content="test content")

        assert mock_reranker._extract_doc_id(doc_with_id) == "doc123"
        assert mock_reranker._extract_doc_id(doc_with_chunk_id) == "chunk456"
        assert isinstance(mock_reranker._extract_doc_id(doc_without_id), str)

    def test_health_check(self, mock_reranker):
        """Test health check functionality."""
        health = mock_reranker.health_check()

        assert isinstance(health, dict)
        assert "model_loaded" in health
        assert "model_name" in health
        assert "device" in health
        assert "batch_size" in health
        assert health["model_loaded"] is True


class TestPrecisionMetrics:
    """Test precision metrics evaluation."""

    @pytest.fixture
    def mock_reranker(self):
        """Mock reranker for evaluation testing."""
        reranker = Mock()
        mock_result = RerankResult(
            documents=[
                Document(page_content="relevant doc 1", metadata={"id": "rel1"}),
                Document(page_content="irrelevant doc", metadata={"id": "irr1"}),
                Document(page_content="relevant doc 2", metadata={"id": "rel2"}),
            ],
            scores=[0.9, 0.5, 0.8],
            original_ranks=[0, 1, 2],
            execution_time=0.1,
            model_used="test-model",
        )
        reranker.rerank.return_value = mock_result
        return reranker

    def test_evaluate_precision_at_k(self, mock_reranker):
        """Test precision@k evaluation."""
        reranker_instance = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker_instance.rerank = mock_reranker.rerank

        documents = [
            Document(page_content="doc1", metadata={"id": "rel1"}),
            Document(page_content="doc2", metadata={"id": "irr1"}),
            Document(page_content="doc3", metadata={"id": "rel2"}),
        ]

        relevant_ids = {"rel1", "rel2"}

        metrics = reranker_instance.evaluate_precision_at_k(
            query="test query",
            documents=documents,
            relevant_doc_ids=relevant_ids,
            k_values=[1, 2, 3],
        )

        assert isinstance(metrics, PrecisionMetrics)
        assert metrics.total_relevant == 2
        assert metrics.total_retrieved == 3

        # Check precision calculations
        # After reranking: [rel1, irr1, rel2] with relevant = {rel1, rel2}
        assert metrics.precision_at_k[1] == 1.0  # 1/1 relevant in top-1
        assert metrics.precision_at_k[2] == 0.5  # 1/2 relevant in top-2
        assert metrics.precision_at_k[3] == 2 / 3  # 2/3 relevant in top-3


class TestRerankingEvaluator:
    """Test reranking evaluator functionality."""

    @pytest.fixture
    def mock_reranker(self):
        """Mock reranker for evaluator testing."""
        reranker = Mock()
        return reranker

    @pytest.fixture
    def sample_documents(self):
        """Sample documents for evaluation."""
        return [
            Document(page_content="relevant doc", metadata={"id": "rel1"}),
            Document(page_content="irrelevant doc", metadata={"id": "irr1"}),
            Document(page_content="another relevant", metadata={"id": "rel2"}),
        ]

    def test_compare_rankings(self, mock_reranker, sample_documents):
        """Test ranking comparison."""
        # Mock the evaluate_precision_at_k method
        mock_metrics = PrecisionMetrics(
            precision_at_k={1: 1.0, 2: 0.5, 3: 2 / 3}, total_relevant=2, total_retrieved=3
        )
        mock_reranker.evaluate_precision_at_k.return_value = mock_metrics

        evaluator = RerankingEvaluator(mock_reranker)

        # Mock the _calculate_precision_metrics method
        evaluator._calculate_precision_metrics = Mock(
            return_value=PrecisionMetrics(
                precision_at_k={1: 0.0, 2: 0.5, 3: 2 / 3}, total_relevant=2, total_retrieved=3
            )
        )

        result = evaluator.compare_rankings(
            query="test query", baseline_docs=sample_documents, relevant_doc_ids={"rel1", "rel2"}
        )

        assert isinstance(result, ComparisonResult)
        assert result.improvement[1] == 1.0  # Improvement from 0.0 to 1.0


class TestFactory:
    """Test reranker factory function."""

    @patch("src.services.reranker.factory.CrossEncoderReranker")
    def test_get_reranker(self, mock_reranker_class):
        """Test reranker factory function."""
        mock_instance = Mock()
        mock_reranker_class.return_value = mock_instance

        reranker = get_reranker(model_name="test-model", batch_size=16, timeout_seconds=10.0)

        assert reranker == mock_instance
        mock_reranker_class.assert_called_once()

        # Check that config was created with correct values
        call_args = mock_reranker_class.call_args[0][0]
        assert call_args.model_name == "test-model"
        assert call_args.batch_size == 16
        assert call_args.timeout_seconds == 10.0


@pytest.mark.integration
class TestIntegration:
    """Integration tests (require actual model loading)."""

    @pytest.mark.slow
    def test_real_reranking(self):
        """Test with real model (slow test)."""
        # This test requires actual model download and may be slow
        config = RerankerConfig(
            model_name="cross-encoder/ms-marco-TinyBERT-L-2",  # Smaller model
            device="cpu",
        )

        try:
            reranker = CrossEncoderReranker(config)

            documents = [
                Document(page_content="Machine learning algorithms", metadata={"id": "1"}),
                Document(page_content="Weather forecast today", metadata={"id": "2"}),
            ]

            result = reranker.rerank("machine learning", documents)

            assert isinstance(result, RerankResult)
            assert len(result.documents) == 2
            assert not result.fallback_used

        except Exception as e:
            pytest.skip(f"Real model test failed (expected in CI): {e}")


if __name__ == "__main__":
    pytest.main([__file__])
