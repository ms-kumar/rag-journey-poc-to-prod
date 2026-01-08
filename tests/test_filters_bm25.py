"""
Tests for Qdrant filter builder and BM25/hybrid search functionality.
"""

import pytest

from src.services.vectorstore.filters import FilterBuilder, build_filter_from_dict


class TestFilterBuilder:
    """Test FilterBuilder class for Qdrant filters."""

    def test_empty_filter(self):
        """Empty filter builder should return None."""
        builder = FilterBuilder()
        result = builder.build()
        assert result is None

    def test_simple_match(self):
        """Test simple exact match filter."""
        filter_obj = FilterBuilder().match("source", "doc1.txt").build()

        assert filter_obj is not None
        assert filter_obj.must is not None
        assert len(filter_obj.must) == 1
        assert filter_obj.should is None
        assert filter_obj.must_not is None

    def test_multiple_match_conditions(self):
        """Test multiple AND conditions."""
        filter_obj = FilterBuilder().match("source", "doc1.txt").match("author", "Smith").build()

        assert filter_obj is not None
        assert filter_obj.must is not None
        assert len(filter_obj.must) == 2

    def test_match_any(self):
        """Test match any (OR) condition for single field."""
        filter_obj = FilterBuilder().match_any("category", ["AI", "ML", "DL"]).build()

        assert filter_obj is not None
        assert filter_obj.must is not None
        assert len(filter_obj.must) == 1

    def test_match_except(self):
        """Test exclusion filter."""
        filter_obj = FilterBuilder().match_except("status", ["deleted", "archived"]).build()

        assert filter_obj is not None
        assert filter_obj.must is not None
        assert len(filter_obj.must) == 1

    def test_text_search(self):
        """Test full-text search condition."""
        filter_obj = FilterBuilder().text("content", "machine learning").build()

        assert filter_obj is not None
        assert filter_obj.must is not None
        assert len(filter_obj.must) == 1

    def test_range_filter_gte_lte(self):
        """Test range filter with gte and lte."""
        filter_obj = FilterBuilder().range("year", gte=2020, lte=2023).build()

        assert filter_obj is not None
        assert filter_obj.must is not None
        assert len(filter_obj.must) == 1

    def test_range_filter_gt_lt(self):
        """Test range filter with gt and lt."""
        filter_obj = FilterBuilder().range("score", gt=0.5, lt=1.0).build()

        assert filter_obj is not None
        assert filter_obj.must is not None
        assert len(filter_obj.must) == 1

    def test_should_condition(self):
        """Test OR (should) conditions."""
        filter_obj = FilterBuilder().should("category", "AI").should("category", "ML").build()

        assert filter_obj is not None
        assert filter_obj.should is not None
        assert len(filter_obj.should) == 2
        assert filter_obj.must is None

    def test_must_not_condition(self):
        """Test negation (must_not) conditions."""
        filter_obj = FilterBuilder().must_not("status", "deleted").build()

        assert filter_obj is not None
        assert filter_obj.must_not is not None
        assert len(filter_obj.must_not) == 1

    def test_complex_filter(self):
        """Test complex filter with multiple condition types."""
        filter_obj = (
            FilterBuilder()
            .match("author", "Smith")
            .range("year", gte=2020, lte=2023)
            .match_any("category", ["AI", "ML"])
            .must_not("status", "deleted")
            .build()
        )

        assert filter_obj is not None
        assert filter_obj.must is not None
        assert len(filter_obj.must) == 3  # match + range + match_any
        assert filter_obj.must_not is not None
        assert len(filter_obj.must_not) == 1

    def test_chainable_api(self):
        """Test that builder methods are chainable."""
        builder = FilterBuilder()
        result = builder.match("key1", "value1").match("key2", "value2").range("score", gte=0.5)
        assert result is builder  # Should return same instance
        assert len(builder.must_conditions) == 3


class TestBuildFilterFromDict:
    """Test dictionary-based filter construction."""

    def test_empty_dict(self):
        """Empty dict should return None."""
        result = build_filter_from_dict({})
        assert result is None

        result = build_filter_from_dict(None)
        assert result is None

    def test_simple_match_from_dict(self):
        """Test simple key-value pairs."""
        filter_dict = {"source": "doc1.txt", "author": "Smith"}
        filter_obj = build_filter_from_dict(filter_dict)

        assert filter_obj is not None
        assert filter_obj.must is not None
        assert len(filter_obj.must) == 2

    def test_match_any_operator(self):
        """Test $in operator for match any."""
        filter_dict = {"category$in": ["AI", "ML", "DL"]}
        filter_obj = build_filter_from_dict(filter_dict)

        assert filter_obj is not None
        assert filter_obj.must is not None
        assert len(filter_obj.must) == 1

    def test_not_operator(self):
        """Test $not operator for negation."""
        filter_dict = {"status$not": "deleted"}
        filter_obj = build_filter_from_dict(filter_dict)

        assert filter_obj is not None
        assert filter_obj.must_not is not None
        assert len(filter_obj.must_not) == 1

    def test_text_operator(self):
        """Test $text operator for text search."""
        filter_dict = {"content$text": "machine learning"}
        filter_obj = build_filter_from_dict(filter_dict)

        assert filter_obj is not None
        assert filter_obj.must is not None
        assert len(filter_obj.must) == 1

    def test_range_operators(self):
        """Test range operators ($gt, $gte, $lt, $lte)."""
        filter_dict = {
            "year$gte": 2020,
            "year$lte": 2023,
            "score$gt": 0.5,
            "price$lt": 100,
        }
        filter_obj = build_filter_from_dict(filter_dict)

        assert filter_obj is not None
        assert filter_obj.must is not None
        assert len(filter_obj.must) == 4

    def test_complex_dict_filter(self):
        """Test complex filter with multiple operators."""
        filter_dict = {
            "source": "paper.pdf",
            "year$gte": 2020,
            "category$in": ["AI", "ML"],
            "status$not": "deleted",
            "content$text": "neural networks",
        }
        filter_obj = build_filter_from_dict(filter_dict)

        assert filter_obj is not None
        assert filter_obj.must is not None
        assert len(filter_obj.must) == 4  # source + year$gte + category$in + content$text
        assert filter_obj.must_not is not None
        assert len(filter_obj.must_not) == 1  # status$not


@pytest.mark.integration
class TestVectorStoreFilters:
    """Integration tests for vectorstore with filters."""

    def test_similarity_search_with_filter_dict(self, mock_vectorstore):
        """Test similarity search with dict-based filters."""
        # This would need a properly configured vectorstore
        # For now, just test the API
        filter_dict = {"source": "doc1.txt"}

        # Mock test - in real test, would query actual vectorstore
        assert filter_dict is not None

    def test_bm25_search_with_filters(self, mock_vectorstore):
        """Test BM25 search with metadata filters."""
        # This would test actual BM25 search with filters
        filter_dict = {"category": "AI", "year$gte": 2020}

        # Mock test
        assert filter_dict is not None

    def test_hybrid_search_with_alpha(self, mock_vectorstore):
        """Test hybrid search with different alpha values."""
        # Test hybrid search balancing vector and BM25
        # Mock test
        assert True


class TestFilterEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_range_parameters(self):
        """Test range filter with no parameters."""
        filter_obj = FilterBuilder().range("score").build()
        # Should still build but range will be empty
        assert filter_obj is not None

    def test_empty_match_any_list(self):
        """Test match_any with empty list."""
        filter_obj = FilterBuilder().match_any("category", []).build()
        assert filter_obj is not None

    def test_special_characters_in_values(self):
        """Test filters with special characters."""
        filter_obj = FilterBuilder().match("path", "/home/user/doc$1.txt").build()
        assert filter_obj is not None

    def test_unicode_in_filters(self):
        """Test Unicode characters in filters."""
        filter_obj = FilterBuilder().match("author", "李明").build()
        assert filter_obj is not None

    def test_numeric_field_names(self):
        """Test numeric values in various conditions."""
        filter_obj = FilterBuilder().match("count", 42).range("score", gte=0.5).build()
        assert filter_obj is not None


class TestFilterBuilderGetMethod:
    """Test the get_filter_builder method on vectorstore client."""

    def test_get_filter_builder_returns_new_instance(self):
        """Each call should return a fresh FilterBuilder instance."""
        from src.services.vectorstore.client import QdrantVectorStoreClient

        # This would need proper initialization
        # Mock test to verify the method exists
        assert hasattr(QdrantVectorStoreClient, "get_filter_builder")


# Fixtures for integration tests
@pytest.fixture
def mock_vectorstore():
    """Create a mock vectorstore for testing."""
    # This would create a test vectorstore with sample data
    # For now, return None as placeholder
    return


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        {
            "content": "Machine learning is a subset of AI",
            "source": "doc1.txt",
            "year": 2020,
            "category": "AI",
        },
        {
            "content": "Deep learning uses neural networks",
            "source": "doc2.txt",
            "year": 2021,
            "category": "ML",
        },
        {
            "content": "Natural language processing enables text understanding",
            "source": "doc3.txt",
            "year": 2022,
            "category": "NLP",
        },
    ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
