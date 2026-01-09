"""
Tests for evaluation metrics.
"""

import pytest

from src.services.evaluation.metrics import MetricsCalculator, RAGMetrics


class TestMetricsCalculator:
    """Test metrics calculation functions."""

    def test_precision_at_k_perfect(self):
        """Test precision@k with all relevant results."""
        calc = MetricsCalculator()
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc1", "doc2", "doc3", "doc4", "doc5"}

        precision = calc.precision_at_k(retrieved, relevant, k=5)
        assert precision == 1.0

    def test_precision_at_k_half(self):
        """Test precision@k with half relevant results."""
        calc = MetricsCalculator()
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc1", "doc3", "doc5"}

        precision = calc.precision_at_k(retrieved, relevant, k=5)
        assert precision == 0.6  # 3/5

    def test_precision_at_k_none(self):
        """Test precision@k with no relevant results."""
        calc = MetricsCalculator()
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc4", "doc5"}

        precision = calc.precision_at_k(retrieved, relevant, k=3)
        assert precision == 0.0

    def test_recall_at_k_perfect(self):
        """Test recall@k with all relevant retrieved."""
        calc = MetricsCalculator()
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc1", "doc2"}

        recall = calc.recall_at_k(retrieved, relevant, k=5)
        assert recall == 1.0

    def test_recall_at_k_partial(self):
        """Test recall@k with partial retrieval."""
        calc = MetricsCalculator()
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1", "doc4", "doc5"}

        recall = calc.recall_at_k(retrieved, relevant, k=3)
        assert recall == pytest.approx(0.333, rel=0.01)  # 1/3

    def test_mrr_first_position(self):
        """Test MRR with relevant doc at first position."""
        calc = MetricsCalculator()
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1"}

        mrr = calc.mean_reciprocal_rank(retrieved, relevant)
        assert mrr == 1.0

    def test_mrr_third_position(self):
        """Test MRR with relevant doc at third position."""
        calc = MetricsCalculator()
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc3"}

        mrr = calc.mean_reciprocal_rank(retrieved, relevant)
        assert mrr == pytest.approx(0.333, rel=0.01)

    def test_mrr_no_relevant(self):
        """Test MRR with no relevant docs."""
        calc = MetricsCalculator()
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc4"}

        mrr = calc.mean_reciprocal_rank(retrieved, relevant)
        assert mrr == 0.0

    def test_ndcg_at_k_perfect(self):
        """Test NDCG@k with perfect ranking."""
        calc = MetricsCalculator()
        retrieved = ["doc1", "doc2", "doc3", "doc4"]
        relevant = {"doc1", "doc2"}

        ndcg = calc.ndcg_at_k(retrieved, relevant, k=4)
        assert ndcg > 0.9  # Should be close to 1.0

    def test_ndcg_at_k_reversed(self):
        """Test NDCG@k with reversed ranking."""
        calc = MetricsCalculator()
        retrieved = ["doc3", "doc4", "doc1", "doc2"]
        relevant = {"doc1", "doc2"}

        ndcg = calc.ndcg_at_k(retrieved, relevant, k=4)
        assert 0.0 < ndcg < 0.8  # Lower than perfect

    def test_average_precision_perfect(self):
        """Test AP with perfect ranking."""
        calc = MetricsCalculator()
        retrieved = ["doc1", "doc2", "doc3", "doc4"]
        relevant = {"doc1", "doc2"}

        ap = calc.average_precision(retrieved, relevant)
        assert ap == 1.0

    def test_average_precision_mixed(self):
        """Test AP with mixed ranking."""
        calc = MetricsCalculator()
        retrieved = ["doc1", "doc3", "doc2", "doc4"]
        relevant = {"doc1", "doc2"}

        ap = calc.average_precision(retrieved, relevant)
        assert 0.7 < ap < 0.9

    def test_latency_percentiles(self):
        """Test latency percentile calculation."""
        calc = MetricsCalculator()
        latencies = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]

        percentiles = calc.calculate_latency_percentiles(latencies)

        assert percentiles["p50"] == pytest.approx(55.0, rel=0.1)
        assert percentiles["p95"] == pytest.approx(95.0, rel=0.1)
        assert percentiles["p99"] == pytest.approx(99.0, rel=0.1)
        assert percentiles["mean"] == 55.0


class TestRAGMetrics:
    """Test RAGMetrics container."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = RAGMetrics(
            precision_at_k={5: 0.8, 10: 0.7},
            recall_at_k={5: 0.6, 10: 0.75},
            mrr=0.85,
            ndcg_at_k={10: 0.82},
            mean_average_precision=0.78,
            num_queries=100,
        )

        result = metrics.to_dict()

        assert result["retrieval"]["precision@k"][5] == 0.8
        assert result["retrieval"]["mrr"] == 0.85
        assert result["metadata"]["num_queries"] == 100

    def test_get_summary(self):
        """Test summary generation."""
        metrics = RAGMetrics(
            precision_at_k={5: 0.8},
            recall_at_k={10: 0.75},
            mrr=0.85,
            num_queries=100,
        )

        summary = metrics.get_summary()

        assert "RAG Evaluation Metrics" in summary
        assert "0.800" in summary  # Precision
        assert "0.850" in summary  # MRR
