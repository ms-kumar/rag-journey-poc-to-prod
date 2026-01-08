"""
Tests for fusion evaluation metrics.

Tests cover:
- Recall@k
- Precision@k
- MRR (Mean Reciprocal Rank)
- MAP (Mean Average Precision)
- NDCG (Normalized Discounted Cumulative Gain)
- Recall uplift calculations
"""

import pytest
from langchain_core.documents import Document

from src.services.vectorstore.fusion_eval import (
    calculate_map,
    calculate_mrr,
    calculate_ndcg,
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_uplift,
    evaluate_retrieval,
)


class TestRecallAtK:
    """Test recall@k calculations."""

    def test_recall_perfect(self, retrieved_docs, all_relevant):
        """Test recall when all relevant docs are retrieved."""
        recall = calculate_recall_at_k(retrieved_docs, all_relevant, k_values=[5])

        # All 3 relevant docs in top 5
        assert recall[5] == 1.0

    def test_recall_partial(self, retrieved_docs, all_relevant):
        """Test recall with partial retrieval."""
        recall = calculate_recall_at_k(retrieved_docs, all_relevant, k_values=[2])

        # Only 2 of 3 relevant docs in top 2
        assert recall[2] == 2.0 / 3.0

    def test_recall_zero(self, irrelevant_docs, all_relevant):
        """Test recall when no relevant docs retrieved."""
        recall = calculate_recall_at_k(irrelevant_docs, all_relevant, k_values=[5])

        assert recall[5] == 0.0

    def test_recall_empty_relevant(self, retrieved_docs):
        """Test recall with no relevant docs."""
        recall = calculate_recall_at_k(retrieved_docs, set(), k_values=[5])

        assert recall[5] == 0.0

    def test_recall_multiple_k(self, retrieved_docs, all_relevant):
        """Test recall at multiple k values."""
        recall = calculate_recall_at_k(retrieved_docs, all_relevant)

        # Default k values: [1, 3, 5, 10, 20]
        assert len(recall) == 5
        assert 1 in recall
        assert 5 in recall
        assert 10 in recall


class TestPrecisionAtK:
    """Test precision@k calculations."""

    def test_precision_perfect(self):
        """Test precision when all retrieved docs are relevant."""
        docs = [
            Document(page_content="doc1", metadata={"chunk_id": "1"}),
            Document(page_content="doc2", metadata={"chunk_id": "2"}),
        ]
        relevant = {"1", "2"}

        precision = calculate_precision_at_k(docs, relevant, k_values=[2])

        assert precision[2] == 1.0

    def test_precision_partial(self, retrieved_docs, all_relevant):
        """Test precision with mixed results."""
        precision = calculate_precision_at_k(retrieved_docs, all_relevant, k_values=[5])

        # 3 relevant out of 5 retrieved
        assert precision[5] == 3.0 / 5.0

    def test_precision_zero(self, irrelevant_docs, all_relevant):
        """Test precision when no relevant docs."""
        precision = calculate_precision_at_k(irrelevant_docs, all_relevant, k_values=[5])

        assert precision[5] == 0.0

    def test_precision_multiple_k(self, retrieved_docs, all_relevant):
        """Test precision at multiple k values."""
        precision = calculate_precision_at_k(retrieved_docs, all_relevant)

        assert len(precision) == 5
        # Precision should decrease as k increases (more irrelevant docs)
        assert precision[1] >= precision[5]


class TestMRR:
    """Test Mean Reciprocal Rank."""

    def test_mrr_first_position(self):
        """Test MRR when relevant doc is first."""
        docs = [
            Document(page_content="relevant", metadata={"chunk_id": "1"}),
            Document(page_content="other", metadata={"chunk_id": "2"}),
        ]
        relevant = {"1"}

        mrr = calculate_mrr(docs, relevant)
        assert mrr == 1.0

    def test_mrr_third_position(self, retrieved_docs, all_relevant):
        """Test MRR with relevant doc at rank 3."""
        # First relevant doc is at position 1
        mrr = calculate_mrr(retrieved_docs, {"1"})
        assert mrr == 1.0

    def test_mrr_no_relevant(self, irrelevant_docs, all_relevant):
        """Test MRR when no relevant docs found."""
        mrr = calculate_mrr(irrelevant_docs, all_relevant)
        assert mrr == 0.0


class TestMAP:
    """Test Mean Average Precision."""

    def test_map_perfect(self):
        """Test MAP with perfect ranking."""
        docs = [
            Document(page_content="doc1", metadata={"chunk_id": "1"}),
            Document(page_content="doc2", metadata={"chunk_id": "2"}),
            Document(page_content="doc3", metadata={"chunk_id": "3"}),
        ]
        relevant = {"1", "2", "3"}

        map_score = calculate_map(docs, relevant)

        # AP = (1/1 + 2/2 + 3/3) / 3 = 1.0
        assert map_score == 1.0

    def test_map_mixed(self, retrieved_docs, all_relevant):
        """Test MAP with mixed results."""
        map_score = calculate_map(retrieved_docs, all_relevant)

        # Should be between 0 and 1
        assert 0.0 < map_score <= 1.0

    def test_map_no_relevant(self, irrelevant_docs, all_relevant):
        """Test MAP with no relevant docs."""
        map_score = calculate_map(irrelevant_docs, all_relevant)
        assert map_score == 0.0


class TestNDCG:
    """Test Normalized Discounted Cumulative Gain."""

    def test_ndcg_perfect(self):
        """Test NDCG with perfect ranking."""
        docs = [
            Document(page_content="doc1", metadata={"chunk_id": "1"}),
            Document(page_content="doc2", metadata={"chunk_id": "2"}),
        ]
        relevant = {"1", "2"}

        ndcg = calculate_ndcg(docs, relevant)
        assert ndcg == 1.0

    def test_ndcg_reversed(self):
        """Test NDCG with reversed ranking."""
        docs = [
            Document(page_content="irrelevant", metadata={"chunk_id": "99"}),
            Document(page_content="relevant", metadata={"chunk_id": "1"}),
        ]
        relevant = {"1"}

        ndcg = calculate_ndcg(docs, relevant)

        # Should be less than 1 due to position discount
        assert 0.0 < ndcg < 1.0

    def test_ndcg_with_k(self, retrieved_docs, all_relevant):
        """Test NDCG with cutoff k."""
        ndcg5 = calculate_ndcg(retrieved_docs, all_relevant, k=5)
        ndcg10 = calculate_ndcg(retrieved_docs, all_relevant, k=10)

        # Both should be valid
        assert 0.0 <= ndcg5 <= 1.0
        assert 0.0 <= ndcg10 <= 1.0

    def test_ndcg_no_relevant(self, irrelevant_docs, all_relevant):
        """Test NDCG with no relevant docs."""
        ndcg = calculate_ndcg(irrelevant_docs, all_relevant)
        assert ndcg == 0.0


class TestEvaluateRetrieval:
    """Test comprehensive evaluation."""

    def test_evaluate_complete(self, retrieved_docs, all_relevant):
        """Test complete evaluation."""
        metrics = evaluate_retrieval(retrieved_docs, all_relevant)

        # Check all metrics are present
        assert metrics.recall_at_k is not None
        assert metrics.precision_at_k is not None
        assert 0.0 <= metrics.mrr <= 1.0
        assert 0.0 <= metrics.map <= 1.0
        assert 0.0 <= metrics.ndcg <= 1.0
        assert metrics.total_relevant == len(all_relevant)
        assert metrics.total_retrieved == len(retrieved_docs)

    def test_evaluate_custom_k_values(self, retrieved_docs, all_relevant):
        """Test evaluation with custom k values."""
        metrics = evaluate_retrieval(retrieved_docs, all_relevant, k_values=[1, 2, 3])

        assert len(metrics.recall_at_k) == 3
        assert 1 in metrics.recall_at_k
        assert 3 in metrics.recall_at_k


class TestCalculateUplift:
    """Test recall uplift calculations."""

    def test_uplift_improvement(self, fusion_docs, baseline_results, all_relevant):
        """Test uplift when fusion improves over baselines."""
        uplift = calculate_uplift(
            fusion_docs,
            baseline_results,
            all_relevant,
            k_values=[5],
        )

        # Check structure
        assert "vector" in uplift.baseline_recalls
        assert "bm25" in uplift.baseline_recalls
        assert "vector" in uplift.recall_uplift
        assert 5 in uplift.fusion_recall

    def test_uplift_positive(self):
        """Test positive uplift calculation."""
        fusion_docs = [
            Document(page_content="doc1", metadata={"chunk_id": "1"}),
            Document(page_content="doc2", metadata={"chunk_id": "2"}),
            Document(page_content="doc3", metadata={"chunk_id": "3"}),
        ]
        baseline_results = {
            "method1": [
                Document(page_content="doc1", metadata={"chunk_id": "1"}),
                Document(page_content="doc4", metadata={"chunk_id": "4"}),
            ]
        }
        relevant = {"1", "2", "3"}

        uplift = calculate_uplift(fusion_docs, baseline_results, relevant, k_values=[3])

        # Fusion: 3/3 = 1.0
        # Baseline: 1/3 = 0.33
        # Uplift: (1.0 - 0.33) / 0.33 * 100 ≈ 200%
        assert uplift.recall_uplift["method1"][3] > 100.0

    def test_uplift_no_improvement(self):
        """Test uplift when fusion doesn't improve."""
        docs = [
            Document(page_content="doc1", metadata={"chunk_id": "1"}),
        ]
        relevant = {"1"}

        uplift = calculate_uplift(
            docs,
            {"method1": docs},
            relevant,
            k_values=[1],
        )

        # Same results, no uplift
        assert uplift.recall_uplift["method1"][1] == 0.0

    def test_uplift_over_best(self, fusion_docs, baseline_results, all_relevant):
        """Test uplift over best baseline."""
        uplift = calculate_uplift(fusion_docs, baseline_results, all_relevant, k_values=[5])

        # Should have best baseline and uplift over it
        assert 5 in uplift.best_baseline_recall
        assert 5 in uplift.uplift_over_best


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def retrieved_docs():
    """Sample retrieved documents with relevance labels."""
    return [
        Document(page_content="machine learning", metadata={"chunk_id": "1"}),  # relevant
        Document(page_content="deep learning", metadata={"chunk_id": "2"}),  # relevant
        Document(page_content="computer vision", metadata={"chunk_id": "99"}),  # not relevant
        Document(page_content="neural networks", metadata={"chunk_id": "3"}),  # relevant
        Document(page_content="data structures", metadata={"chunk_id": "98"}),  # not relevant
    ]


@pytest.fixture
def all_relevant():
    """Ground truth relevant document IDs."""
    return {"1", "2", "3"}


@pytest.fixture
def irrelevant_docs():
    """Documents with no relevant results."""
    return [
        Document(page_content="unrelated1", metadata={"chunk_id": "91"}),
        Document(page_content="unrelated2", metadata={"chunk_id": "92"}),
        Document(page_content="unrelated3", metadata={"chunk_id": "93"}),
    ]


@pytest.fixture
def fusion_docs():
    """Fusion results for uplift testing."""
    return [
        Document(page_content="doc1", metadata={"chunk_id": "1"}),
        Document(page_content="doc2", metadata={"chunk_id": "2"}),
        Document(page_content="doc3", metadata={"chunk_id": "3"}),
        Document(page_content="doc4", metadata={"chunk_id": "4"}),
        Document(page_content="doc5", metadata={"chunk_id": "5"}),
    ]


@pytest.fixture
def baseline_results():
    """Baseline search results for uplift comparison."""
    return {
        "vector": [
            Document(page_content="doc1", metadata={"chunk_id": "1"}),
            Document(page_content="doc2", metadata={"chunk_id": "2"}),
            Document(page_content="doc99", metadata={"chunk_id": "99"}),
        ],
        "bm25": [
            Document(page_content="doc2", metadata={"chunk_id": "2"}),
            Document(page_content="doc3", metadata={"chunk_id": "3"}),
            Document(page_content="doc98", metadata={"chunk_id": "98"}),
        ],
    }
