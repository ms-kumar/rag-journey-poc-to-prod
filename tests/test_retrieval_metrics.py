"""
Tests for dense retrieval metrics and utilities.
"""

import pytest

from src.services.vectorstore.retrieval_metrics import (
    RetrievalMetrics,
    RetrievalTimer,
    calculate_mrr,
    calculate_precision_at_k,
    calculate_recall_at_k,
    normalize_scores,
)


class TestRetrievalMetrics:
    """Test RetrievalMetrics tracking."""

    def test_initialization(self):
        """Test metrics initialization."""
        metrics = RetrievalMetrics()
        assert metrics.total_queries == 0
        assert metrics.cache_hits == 0
        assert len(metrics.latencies) == 0
        assert len(metrics.scores) == 0

    def test_record_query(self):
        """Test recording a single query."""
        metrics = RetrievalMetrics()
        metrics.record_query(latency=0.05, scores=[0.9, 0.8, 0.7], search_type="vector")

        assert metrics.total_queries == 1
        assert len(metrics.latencies) == 1
        assert metrics.latencies[0] == 0.05
        assert len(metrics.scores) == 3

    def test_record_multiple_queries(self):
        """Test recording multiple queries."""
        metrics = RetrievalMetrics()
        metrics.record_query(0.05, [0.9, 0.8], search_type="vector")
        metrics.record_query(0.03, [0.95, 0.85], search_type="bm25", cache_hit=True)
        metrics.record_query(0.07, [0.88, 0.75], search_type="hybrid")

        assert metrics.total_queries == 3
        assert metrics.cache_hits == 1
        assert len(metrics.latencies) == 3
        assert len(metrics.scores) == 6

    def test_latency_percentiles(self):
        """Test latency percentile calculations."""
        metrics = RetrievalMetrics()
        # Add latencies: 10ms, 20ms, 30ms, 40ms, 50ms
        for lat in [0.01, 0.02, 0.03, 0.04, 0.05]:
            metrics.record_query(lat, [0.9])

        percentiles = metrics.get_latency_percentiles()
        assert percentiles["p50"] == pytest.approx(30.0, abs=5.0)  # ~30ms
        assert percentiles["p95"] == pytest.approx(50.0, abs=5.0)  # ~50ms
        assert percentiles["mean"] == pytest.approx(30.0, abs=5.0)  # 30ms

    def test_latency_percentiles_empty(self):
        """Test percentiles with no data."""
        metrics = RetrievalMetrics()
        percentiles = metrics.get_latency_percentiles()
        assert percentiles["p50"] == 0.0
        assert percentiles["p95"] == 0.0

    def test_score_statistics(self):
        """Test score statistics calculation."""
        metrics = RetrievalMetrics()
        metrics.record_query(0.01, [0.9, 0.8, 0.7, 0.6])

        stats = metrics.get_score_statistics()
        assert stats["mean"] == pytest.approx(0.75, abs=0.01)
        assert stats["min"] == 0.6
        assert stats["max"] == 0.9

    def test_cache_hit_rate(self):
        """Test cache hit rate calculation."""
        metrics = RetrievalMetrics()
        metrics.record_query(0.01, [0.9], cache_hit=True)
        metrics.record_query(0.02, [0.8], cache_hit=False)
        metrics.record_query(0.03, [0.7], cache_hit=True)

        hit_rate = metrics.get_cache_hit_rate()
        assert hit_rate == pytest.approx(66.67, abs=0.1)  # 2 out of 3

    def test_metrics_by_search_type(self):
        """Test per-search-type metrics tracking."""
        metrics = RetrievalMetrics()
        metrics.record_query(0.05, [0.9], search_type="vector")
        metrics.record_query(0.03, [0.8], search_type="bm25")
        metrics.record_query(0.04, [0.85], search_type="vector")

        assert "vector" in metrics.metrics_by_type
        assert "bm25" in metrics.metrics_by_type

        vector_metrics = metrics.metrics_by_type["vector"]
        assert vector_metrics.total_queries == 2

        bm25_metrics = metrics.metrics_by_type["bm25"]
        assert bm25_metrics.total_queries == 1

    def test_get_summary(self):
        """Test comprehensive summary generation."""
        metrics = RetrievalMetrics()
        metrics.record_query(0.05, [0.9, 0.8], search_type="vector")
        metrics.record_query(0.03, [0.95], search_type="bm25", cache_hit=True)

        summary = metrics.get_summary()
        assert "total_queries" in summary
        assert "cache_hit_rate" in summary
        assert "latency" in summary
        assert "scores" in summary
        assert "by_search_type" in summary
        assert summary["total_queries"] == 2

    def test_reset(self):
        """Test metrics reset."""
        metrics = RetrievalMetrics()
        metrics.record_query(0.05, [0.9], search_type="vector")
        assert metrics.total_queries == 1

        metrics.reset()
        assert metrics.total_queries == 0
        assert len(metrics.latencies) == 0
        assert len(metrics.scores) == 0
        assert len(metrics.metrics_by_type) == 0

    def test_repr(self):
        """Test string representation."""
        metrics = RetrievalMetrics()
        metrics.record_query(0.05, [0.9])
        repr_str = repr(metrics)
        assert "RetrievalMetrics" in repr_str
        assert "queries=1" in repr_str


class TestRetrievalTimer:
    """Test RetrievalTimer context manager."""

    def test_timer_basic(self):
        """Test basic timer usage."""
        metrics = RetrievalMetrics()

        with RetrievalTimer(metrics, scores=[0.9, 0.8], search_type="vector"):
            # Simulate some work
            pass

        assert metrics.total_queries == 1
        assert len(metrics.latencies) == 1
        assert metrics.latencies[0] > 0  # Some time passed

    def test_timer_set_scores(self):
        """Test setting scores after timer start."""
        metrics = RetrievalMetrics()

        with RetrievalTimer(metrics, search_type="vector") as timer:
            # Scores not known yet
            timer.set_scores([0.9, 0.8, 0.7])

        assert metrics.total_queries == 1
        assert len(metrics.scores) == 3

    def test_timer_cache_hit(self):
        """Test timer with cache hit."""
        metrics = RetrievalMetrics()

        with RetrievalTimer(metrics, scores=[0.9], search_type="vector", cache_hit=True):
            pass

        assert metrics.cache_hits == 1


class TestNormalizeScores:
    """Test score normalization functions."""

    def test_minmax_normalization(self):
        """Test min-max normalization."""
        scores = [0.5, 0.7, 0.9, 0.6]
        normalized = normalize_scores(scores, method="minmax")

        assert min(normalized) == 0.0  # Min score maps to 0
        assert max(normalized) == 1.0  # Max score maps to 1
        assert len(normalized) == len(scores)

    def test_minmax_with_range(self):
        """Test min-max with explicit range."""
        scores = [0.6, 0.7, 0.8]
        normalized = normalize_scores(scores, method="minmax", score_range=(0.5, 1.0))

        assert normalized[0] == pytest.approx(0.2, abs=0.01)  # (0.6-0.5)/(1.0-0.5)
        assert normalized[-1] == pytest.approx(0.6, abs=0.01)  # (0.8-0.5)/(1.0-0.5)

    def test_zscore_normalization(self):
        """Test z-score normalization."""
        scores = [0.5, 0.6, 0.7, 0.8, 0.9]
        normalized = normalize_scores(scores, method="zscore")

        # Z-score then sigmoid should give values in (0, 1)
        assert all(0 < s < 1 for s in normalized)

    def test_sigmoid_normalization(self):
        """Test sigmoid normalization."""
        scores = [1.0, 2.0, 3.0, 4.0]
        normalized = normalize_scores(scores, method="sigmoid")

        # Sigmoid maps to (0, 1)
        assert all(0 < s < 1 for s in normalized)
        # Higher scores should map to higher normalized values
        assert normalized[-1] > normalized[0]

    def test_empty_scores(self):
        """Test normalization with empty list."""
        normalized = normalize_scores([], method="minmax")
        assert normalized == []

    def test_single_score(self):
        """Test normalization with single score."""
        normalized = normalize_scores([0.8], method="minmax")
        assert normalized == [1.0]  # Single score is always max

    def test_identical_scores(self):
        """Test normalization with identical scores."""
        scores = [0.5, 0.5, 0.5]
        normalized = normalize_scores(scores, method="minmax")
        assert all(s == 1.0 for s in normalized)  # All identical = all max

    def test_invalid_method(self):
        """Test invalid normalization method."""
        with pytest.raises(ValueError, match="Unknown normalization method"):
            normalize_scores([0.5, 0.6], method="invalid")


class TestCalculateMRR:
    """Test Mean Reciprocal Rank calculation."""

    def test_mrr_perfect(self):
        """Test MRR with perfect ranking."""
        # Relevant doc always at rank 1
        mrr = calculate_mrr([1, 1, 1])
        assert mrr == pytest.approx(1.0)

    def test_mrr_mixed(self):
        """Test MRR with mixed rankings."""
        # Ranks: 1, 3, 2
        # MRR = (1/1 + 1/3 + 1/2) / 3 = (1.0 + 0.333 + 0.5) / 3 = 0.611
        mrr = calculate_mrr([1, 3, 2])
        assert mrr == pytest.approx(0.611, abs=0.01)

    def test_mrr_with_misses(self):
        """Test MRR with some queries having no relevant results."""
        # Ranks: 1, 0 (miss), 2
        mrr = calculate_mrr([1, 0, 2])
        assert mrr == pytest.approx(0.5, abs=0.01)  # (1 + 0 + 0.5) / 3

    def test_mrr_empty(self):
        """Test MRR with empty list."""
        mrr = calculate_mrr([])
        assert mrr == 0.0


class TestRecallAtK:
    """Test Recall@k calculation."""

    def test_recall_perfect(self):
        """Test perfect recall."""
        retrieved = {"doc1", "doc2", "doc3"}
        relevant = {"doc1", "doc2", "doc3"}
        recall = calculate_recall_at_k(retrieved, relevant, k=3)
        assert recall == 1.0

    def test_recall_partial(self):
        """Test partial recall."""
        retrieved = {"doc1", "doc2", "doc5"}
        relevant = {"doc1", "doc2", "doc3", "doc4"}
        # Found 2 out of 4 relevant
        recall = calculate_recall_at_k(retrieved, relevant, k=3)
        assert recall == 0.5

    def test_recall_no_relevant(self):
        """Test recall when no relevant docs exist."""
        retrieved = {"doc1", "doc2"}
        relevant = set()
        recall = calculate_recall_at_k(retrieved, relevant, k=2)
        assert recall == 1.0  # Perfect by definition

    def test_recall_zero(self):
        """Test zero recall."""
        retrieved = {"doc1", "doc2"}
        relevant = {"doc3", "doc4"}
        recall = calculate_recall_at_k(retrieved, relevant, k=2)
        assert recall == 0.0


class TestPrecisionAtK:
    """Test Precision@k calculation."""

    def test_precision_perfect(self):
        """Test perfect precision."""
        retrieved = {"doc1", "doc2", "doc3"}
        relevant = {"doc1", "doc2", "doc3", "doc4"}
        # All 3 retrieved are relevant
        precision = calculate_precision_at_k(retrieved, relevant, k=3)
        assert precision == pytest.approx(1.0)

    def test_precision_partial(self):
        """Test partial precision."""
        retrieved = {"doc1", "doc2", "doc5"}
        relevant = {"doc1", "doc2", "doc3"}
        # 2 out of 3 retrieved are relevant
        precision = calculate_precision_at_k(retrieved, relevant, k=3)
        assert precision == pytest.approx(0.666, abs=0.01)

    def test_precision_zero(self):
        """Test zero precision."""
        retrieved = {"doc1", "doc2"}
        relevant = {"doc3", "doc4"}
        precision = calculate_precision_at_k(retrieved, relevant, k=2)
        assert precision == 0.0

    def test_precision_k_zero(self):
        """Test precision with k=0."""
        retrieved = {"doc1"}
        relevant = {"doc1"}
        precision = calculate_precision_at_k(retrieved, relevant, k=0)
        assert precision == 0.0
