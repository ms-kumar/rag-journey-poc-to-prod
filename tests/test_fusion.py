"""
Tests for fusion methods.

Tests cover:
- Reciprocal Rank Fusion (RRF)
- Weighted score fusion
- Score normalization
- Tie-breaking strategies
- Fusion configuration
"""

import pytest
from langchain_core.documents import Document

from src.services.vectorstore.fusion import (
    FusionConfig,
    fuse_results,
    reciprocal_rank_fusion,
    weighted_fusion,
)


class TestReciprocalRankFusion:
    """Test RRF fusion."""

    def test_rrf_basic(self, sample_results):
        """Test basic RRF fusion."""
        result = reciprocal_rank_fusion(sample_results, k=60)

        assert len(result.documents) > 0
        assert result.method == "rrf"
        assert all("fusion_score" in doc.metadata for doc in result.documents)

    def test_rrf_empty_results(self):
        """Test RRF with empty results."""
        result = reciprocal_rank_fusion({}, k=60)
        assert len(result.documents) == 0

    def test_rrf_single_source(self, vector_docs):
        """Test RRF with single search source."""
        results = {"vector": vector_docs}
        result = reciprocal_rank_fusion(results, k=60)

        assert len(result.documents) == len(vector_docs)
        # All docs should have fusion metadata
        for doc in result.documents:
            assert "fusion_score" in doc.metadata
            assert "component_ranks" in doc.metadata

    def test_rrf_scoring(self):
        """Test RRF scoring formula."""
        doc1 = Document(page_content="doc1", metadata={"chunk_id": "1"})
        doc2 = Document(page_content="doc2", metadata={"chunk_id": "2"})

        results = {
            "search1": [doc1, doc2],  # doc1 rank=1, doc2 rank=2
            "search2": [doc2, doc1],  # doc2 rank=1, doc1 rank=2
        }

        result = reciprocal_rank_fusion(results, k=60)

        # doc1: 1/(60+1) + 1/(60+2) ≈ 0.0164 + 0.0161 = 0.0325
        # doc2: 1/(60+2) + 1/(60+1) ≈ 0.0161 + 0.0164 = 0.0325
        # They should have equal scores (will be tie-broken)
        scores = [doc.metadata["fusion_score"] for doc in result.documents]
        assert abs(scores[0] - scores[1]) < 0.0001  # Nearly equal

    def test_rrf_k_parameter(self):
        """Test that k parameter affects scoring."""
        doc1 = Document(page_content="doc1", metadata={"chunk_id": "1"})

        results = {"search1": [doc1]}

        result1 = reciprocal_rank_fusion(results, k=10)
        result2 = reciprocal_rank_fusion(results, k=100)

        score1 = result1.documents[0].metadata["fusion_score"]
        score2 = result2.documents[0].metadata["fusion_score"]

        # Smaller k gives higher score for same rank
        # k=10: 1/(10+1) = 0.0909...
        # k=100: 1/(100+1) = 0.0099...
        assert score1 > score2
        assert abs(score1 - 1 / 11) < 0.0001
        assert abs(score2 - 1 / 101) < 0.0001

    def test_rrf_tie_break_score(self, tie_break_docs):
        """Test tie-breaking by original score."""
        result = reciprocal_rank_fusion(
            tie_break_docs,
            k=60,
            tie_break_strategy="score",
        )

        # Should be sorted by fusion score, then by original score
        assert len(result.documents) == 3

    def test_rrf_tie_break_rank(self, tie_break_docs):
        """Test tie-breaking by average rank."""
        result = reciprocal_rank_fusion(
            tie_break_docs,
            k=60,
            tie_break_strategy="rank",
        )

        assert len(result.documents) == 3

    def test_rrf_tie_break_stable(self, tie_break_docs):
        """Test stable tie-breaking."""
        result = reciprocal_rank_fusion(
            tie_break_docs,
            k=60,
            tie_break_strategy="stable",
        )

        assert len(result.documents) == 3


class TestWeightedFusion:
    """Test weighted score fusion."""

    def test_weighted_basic(self, sample_results):
        """Test basic weighted fusion."""
        weights = {"vector": 0.6, "bm25": 0.4}
        result = weighted_fusion(sample_results, weights=weights)

        assert len(result.documents) > 0
        assert result.method == "weighted"
        assert all("fusion_score" in doc.metadata for doc in result.documents)

    def test_weighted_equal_weights(self, sample_results):
        """Test weighted fusion with equal weights."""
        result = weighted_fusion(sample_results, weights=None)

        # Should use equal weights (0.5, 0.5)
        assert len(result.documents) > 0

    def test_weighted_normalization(self):
        """Test that weights are normalized if they don't sum to 1."""
        doc1 = Document(page_content="doc1", metadata={"chunk_id": "1", "score": 0.9})

        results = {"search1": [doc1]}
        weights = {"search1": 2.0}  # Doesn't sum to 1

        result = weighted_fusion(results, weights=weights, normalize_scores=False)

        # Should still work (weights normalized internally)
        assert len(result.documents) == 1

    def test_weighted_score_combination(self):
        """Test weighted score combination."""
        doc1 = Document(page_content="doc1", metadata={"chunk_id": "1", "score": 0.8})
        doc2 = Document(page_content="doc2", metadata={"chunk_id": "2", "score": 0.6})

        results = {
            "search1": [doc1],  # score 0.8
            "search2": [doc2],  # score 0.6
        }
        weights = {"search1": 0.7, "search2": 0.3}

        result = weighted_fusion(
            results,
            weights=weights,
            normalize_scores=False,
        )

        # doc1: 0.8 * 0.7 = 0.56
        # doc2: 0.6 * 0.3 = 0.18
        assert len(result.documents) == 2
        assert result.documents[0].metadata["chunk_id"] == "1"  # doc1 first

    def test_weighted_with_normalization(self):
        """Test weighted fusion with score normalization."""
        doc1 = Document(page_content="doc1", metadata={"chunk_id": "1", "score": 100})
        doc2 = Document(page_content="doc2", metadata={"chunk_id": "2", "score": 50})

        results = {"search1": [doc1, doc2]}
        weights = {"search1": 1.0}

        result = weighted_fusion(
            results,
            weights=weights,
            normalize_scores=True,
        )

        # Scores should be normalized to [0, 1] before weighting
        assert len(result.documents) == 2

    def test_weighted_multiple_sources(self):
        """Test weighted fusion with multiple search sources."""
        doc1 = Document(page_content="doc1", metadata={"chunk_id": "1", "score": 0.9})
        doc2 = Document(page_content="doc2", metadata={"chunk_id": "2", "score": 0.7})
        doc3 = Document(page_content="doc3", metadata={"chunk_id": "3", "score": 0.8})

        results = {
            "vector": [doc1, doc2],
            "bm25": [doc3, doc1],
            "sparse": [doc2, doc3],
        }
        weights = {"vector": 0.5, "bm25": 0.3, "sparse": 0.2}

        result = weighted_fusion(results, weights=weights, normalize_scores=False)

        # All unique docs should be present
        assert len(result.documents) == 3


class TestFuseResults:
    """Test main fusion interface."""

    def test_fuse_rrf(self, sample_results):
        """Test fusing with RRF method."""
        config = FusionConfig(method="rrf", rrf_k=60)
        result = fuse_results(sample_results, config)

        assert result.method == "rrf"
        assert len(result.documents) > 0

    def test_fuse_weighted(self, sample_results):
        """Test fusing with weighted method."""
        config = FusionConfig(
            method="weighted",
            weights={"vector": 0.6, "bm25": 0.4},
        )
        result = fuse_results(sample_results, config)

        assert result.method == "weighted"
        assert len(result.documents) > 0

    def test_fuse_default_config(self, sample_results):
        """Test fusing with default config."""
        result = fuse_results(sample_results, config=None)

        # Default is RRF
        assert result.method == "rrf"

    def test_fuse_invalid_method(self, sample_results):
        """Test fusing with invalid method."""
        config = FusionConfig(method="invalid")

        with pytest.raises(ValueError, match="Unknown fusion method"):
            fuse_results(sample_results, config)

    def test_fuse_result_metadata(self, sample_results):
        """Test that fusion adds proper metadata."""
        result = fuse_results(sample_results)

        for doc in result.documents:
            assert "fusion_score" in doc.metadata
            assert "fusion_method" in doc.metadata
            assert "component_ranks" in doc.metadata


class TestFusionResult:
    """Test FusionResult class."""

    def test_get_top_k(self, sample_results):
        """Test getting top k results."""
        result = fuse_results(sample_results)

        top_3 = result.get_top_k(3)
        assert len(top_3) <= 3
        assert all(isinstance(doc, Document) for doc in top_3)

    def test_get_top_k_more_than_available(self, sample_results):
        """Test getting more docs than available."""
        result = fuse_results(sample_results)

        top_100 = result.get_top_k(100)
        assert len(top_100) == len(result.documents)


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def vector_docs():
    """Sample vector search results."""
    return [
        Document(
            page_content="machine learning basics",
            metadata={"chunk_id": "1", "score": 0.95, "source": "vector"},
        ),
        Document(
            page_content="deep learning fundamentals",
            metadata={"chunk_id": "2", "score": 0.87, "source": "vector"},
        ),
        Document(
            page_content="neural networks explained",
            metadata={"chunk_id": "3", "score": 0.82, "source": "vector"},
        ),
    ]


@pytest.fixture
def bm25_docs():
    """Sample BM25 search results."""
    return [
        Document(
            page_content="deep learning fundamentals",
            metadata={"chunk_id": "2", "score": 15.3, "source": "bm25"},
        ),
        Document(
            page_content="machine learning basics",
            metadata={"chunk_id": "1", "score": 12.7, "source": "bm25"},
        ),
        Document(
            page_content="reinforcement learning intro",
            metadata={"chunk_id": "4", "score": 9.2, "source": "bm25"},
        ),
    ]


@pytest.fixture
def sample_results(vector_docs, bm25_docs):
    """Sample multi-source results."""
    return {
        "vector": vector_docs,
        "bm25": bm25_docs,
    }


@pytest.fixture
def tie_break_docs():
    """Docs with same RRF scores for tie-breaking tests."""
    doc1 = Document(page_content="doc1", metadata={"chunk_id": "1", "score": 0.9})
    doc2 = Document(page_content="doc2", metadata={"chunk_id": "2", "score": 0.8})
    doc3 = Document(page_content="doc3", metadata={"chunk_id": "3", "score": 0.7})

    return {
        "search1": [doc1, doc2],
        "search2": [doc2, doc1],
        "search3": [doc3],
    }
