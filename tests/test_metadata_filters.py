"""
Comprehensive tests for metadata filtering with source/date/tag support.
"""

from datetime import datetime

from src.services.vectorstore.filters import FilterBuilder, build_filter_from_dict


class TestSourceFilters:
    """Test source-specific filtering methods."""

    def test_single_source_filter(self):
        """Test filtering by a single source."""
        filter_obj = FilterBuilder().source("paper.pdf").build()

        assert filter_obj is not None
        assert filter_obj.must is not None
        assert len(filter_obj.must) == 1

    def test_multiple_sources_filter(self):
        """Test filtering by multiple sources."""
        filter_obj = FilterBuilder().sources(["paper1.pdf", "paper2.pdf", "paper3.pdf"]).build()

        assert filter_obj is not None
        assert filter_obj.must is not None
        assert len(filter_obj.must) == 1

    def test_exclude_source_filter(self):
        """Test excluding a specific source."""
        filter_obj = FilterBuilder().exclude_source("draft.txt").build()

        assert filter_obj is not None
        assert filter_obj.must_not is not None
        assert len(filter_obj.must_not) == 1

    def test_source_from_dict(self):
        """Test source filter via dictionary."""
        filter_obj = build_filter_from_dict({"source": "paper.pdf"})

        assert filter_obj is not None
        assert filter_obj.must is not None
        assert len(filter_obj.must) == 1

    def test_sources_from_dict(self):
        """Test multiple sources via dictionary."""
        filter_obj = build_filter_from_dict({"sources": ["doc1.txt", "doc2.txt"]})

        assert filter_obj is not None
        assert filter_obj.must is not None
        assert len(filter_obj.must) == 1


class TestTagFilters:
    """Test tag-specific filtering methods."""

    def test_single_tag_filter(self):
        """Test filtering by a single tag."""
        filter_obj = FilterBuilder().tag("machine-learning").build()

        assert filter_obj is not None
        assert filter_obj.must is not None
        assert len(filter_obj.must) == 1

    def test_multiple_tags_filter(self):
        """Test filtering by multiple tags."""
        filter_obj = FilterBuilder().tags(["ai", "ml", "deep-learning"]).build()

        assert filter_obj is not None
        assert filter_obj.must is not None
        assert len(filter_obj.must) == 1

    def test_tag_from_dict(self):
        """Test tag filter via dictionary."""
        filter_obj = build_filter_from_dict({"tag": "python"})

        assert filter_obj is not None
        assert filter_obj.must is not None
        assert len(filter_obj.must) == 1

    def test_tags_from_dict(self):
        """Test multiple tags via dictionary."""
        filter_obj = build_filter_from_dict({"tags": ["python", "fastapi", "ai"]})

        assert filter_obj is not None
        assert filter_obj.must is not None
        assert len(filter_obj.must) == 1


class TestDateFilters:
    """Test date range filtering methods."""

    def test_date_range_with_datetime_objects(self):
        """Test date range with datetime objects."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)
        filter_obj = FilterBuilder().date_range(after=start, before=end).build()

        assert filter_obj is not None
        assert filter_obj.must is not None
        assert len(filter_obj.must) == 1

    def test_date_range_with_iso_strings(self):
        """Test date range with ISO format strings."""
        filter_obj = FilterBuilder().date_range(after="2024-01-01", before="2024-12-31").build()

        assert filter_obj is not None
        assert filter_obj.must is not None
        assert len(filter_obj.must) == 1

    def test_date_range_after_only(self):
        """Test date range with only after constraint."""
        filter_obj = FilterBuilder().date_range(after="2024-01-01").build()

        assert filter_obj is not None
        assert filter_obj.must is not None
        assert len(filter_obj.must) == 1

    def test_date_range_before_only(self):
        """Test date range with only before constraint."""
        filter_obj = FilterBuilder().date_range(before="2024-12-31").build()

        assert filter_obj is not None
        assert filter_obj.must is not None
        assert len(filter_obj.must) == 1

    def test_date_range_custom_field(self):
        """Test date range with custom field name."""
        filter_obj = FilterBuilder().date_range(after="2024-01-01", field="published_date").build()

        assert filter_obj is not None
        assert filter_obj.must is not None
        assert len(filter_obj.must) == 1

    def test_created_after(self):
        """Test created_after convenience method."""
        filter_obj = FilterBuilder().created_after("2024-01-01").build()

        assert filter_obj is not None
        assert filter_obj.must is not None
        assert len(filter_obj.must) == 1

    def test_created_before(self):
        """Test created_before convenience method."""
        filter_obj = FilterBuilder().created_before("2024-12-31").build()

        assert filter_obj is not None
        assert filter_obj.must is not None
        assert len(filter_obj.must) == 1

    def test_date_filter_from_dict(self):
        """Test date filter via dictionary."""
        filter_obj = build_filter_from_dict(
            {"date_after": "2024-01-01", "date_before": "2024-12-31"}
        )

        assert filter_obj is not None
        assert filter_obj.must is not None
        assert len(filter_obj.must) == 1

    def test_created_filter_from_dict(self):
        """Test created date filter via dictionary."""
        filter_obj = build_filter_from_dict(
            {"created_after": "2024-01-01", "created_before": "2024-12-31"}
        )

        assert filter_obj is not None
        assert filter_obj.must is not None
        assert len(filter_obj.must) == 1  # Combined into single range condition

    def test_date_operator_from_dict(self):
        """Test date operators in dictionary format."""
        filter_obj = build_filter_from_dict({"date$after": "2024-01-01"})

        assert filter_obj is not None
        assert filter_obj.must is not None
        assert len(filter_obj.must) == 1


class TestAuthorFilters:
    """Test author-specific filtering methods."""

    def test_single_author_filter(self):
        """Test filtering by a single author."""
        filter_obj = FilterBuilder().author("John Smith").build()

        assert filter_obj is not None
        assert filter_obj.must is not None
        assert len(filter_obj.must) == 1

    def test_multiple_authors_filter(self):
        """Test filtering by multiple authors."""
        filter_obj = FilterBuilder().authors(["Smith", "Johnson", "Williams"]).build()

        assert filter_obj is not None
        assert filter_obj.must is not None
        assert len(filter_obj.must) == 1

    def test_author_from_dict(self):
        """Test author filter via dictionary."""
        filter_obj = build_filter_from_dict({"author": "Jane Doe"})

        assert filter_obj is not None
        assert filter_obj.must is not None
        assert len(filter_obj.must) == 1

    def test_authors_from_dict(self):
        """Test multiple authors via dictionary."""
        filter_obj = build_filter_from_dict({"authors": ["Smith", "Johnson"]})

        assert filter_obj is not None
        assert filter_obj.must is not None
        assert len(filter_obj.must) == 1


class TestComplexFilters:
    """Test complex combinations of metadata filters."""

    def test_source_and_date_filter(self):
        """Test combining source and date filters."""
        filter_obj = (
            FilterBuilder()
            .source("paper.pdf")
            .date_range(after="2024-01-01", before="2024-12-31")
            .build()
        )

        assert filter_obj is not None
        assert filter_obj.must is not None
        assert len(filter_obj.must) == 2

    def test_tags_and_author_filter(self):
        """Test combining tags and author filters."""
        filter_obj = FilterBuilder().tags(["ai", "ml"]).author("Smith").build()

        assert filter_obj is not None
        assert filter_obj.must is not None
        assert len(filter_obj.must) == 2

    def test_comprehensive_metadata_filter(self):
        """Test complex filter with source, tags, date, and author."""
        filter_obj = (
            FilterBuilder()
            .source("research.pdf")
            .tags(["machine-learning", "nlp"])
            .author("John Smith")
            .date_range(after="2024-01-01")
            .build()
        )

        assert filter_obj is not None
        assert filter_obj.must is not None
        assert len(filter_obj.must) == 4

    def test_filter_with_exclusions(self):
        """Test filter with both inclusions and exclusions."""
        filter_obj = (
            FilterBuilder()
            .sources(["paper1.pdf", "paper2.pdf"])
            .tags(["ai", "ml"])
            .exclude_source("draft.txt")
            .must_not("status", "deleted")
            .build()
        )

        assert filter_obj is not None
        assert filter_obj.must is not None
        assert len(filter_obj.must) == 2
        assert filter_obj.must_not is not None
        assert len(filter_obj.must_not) == 2

    def test_complex_filter_from_dict(self):
        """Test complex filter via dictionary."""
        filter_dict = {
            "source": "paper.pdf",
            "tags": ["ai", "ml"],
            "author": "Smith",
            "date_after": "2024-01-01",
            "date_before": "2024-12-31",
            "year$gte": 2020,
        }
        filter_obj = build_filter_from_dict(filter_dict)

        assert filter_obj is not None
        assert filter_obj.must is not None
        assert len(filter_obj.must) >= 4

    def test_real_world_research_filter(self):
        """Test real-world research paper filtering scenario."""
        filter_obj = (
            FilterBuilder()
            .authors(["Smith", "Johnson", "Lee"])
            .tags(["deep-learning", "computer-vision"])
            .date_range(after="2020-01-01")
            .range("citations", gte=100)
            .must_not("venue", "workshop")
            .build()
        )

        assert filter_obj is not None
        assert filter_obj.must is not None
        assert len(filter_obj.must) >= 3
        assert filter_obj.must_not is not None
        assert len(filter_obj.must_not) == 1

    def test_real_world_document_filter(self):
        """Test real-world document management filtering scenario."""
        filter_obj = (
            FilterBuilder()
            .sources(["report_2024.pdf", "analysis_2024.pdf"])
            .tag("quarterly-report")
            .date_range(after="2024-01-01", before="2024-03-31")
            .author("Finance Team")
            .must_not("status", "draft")
            .build()
        )

        assert filter_obj is not None
        assert filter_obj.must is not None
        assert len(filter_obj.must) == 4
        assert filter_obj.must_not is not None
        assert len(filter_obj.must_not) == 1


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_sources_list(self):
        """Test handling of empty sources list."""
        filter_obj = FilterBuilder().sources([]).build()
        # Empty list should still create a filter
        assert filter_obj is not None

    def test_empty_tags_list(self):
        """Test handling of empty tags list."""
        filter_obj = FilterBuilder().tags([]).build()
        assert filter_obj is not None

    def test_date_range_with_neither_param(self):
        """Test date_range with no parameters."""
        filter_obj = FilterBuilder().date_range().build()
        # Should return None since no conditions added
        assert filter_obj is None

    def test_chaining_order_independence(self):
        """Test that filter building is order-independent."""
        filter1 = FilterBuilder().source("doc.txt").tag("ai").author("Smith").build()
        filter2 = FilterBuilder().author("Smith").tag("ai").source("doc.txt").build()

        # Both should produce valid filters
        assert filter1 is not None
        assert filter2 is not None
        assert len(filter1.must) == len(filter2.must)

    def test_mixed_dict_and_builder_patterns(self):
        """Verify dict and builder produce equivalent results."""
        # Builder approach
        filter1 = (
            FilterBuilder()
            .source("paper.pdf")
            .tags(["ai", "ml"])
            .date_range(after="2024-01-01")
            .build()
        )

        # Dict approach
        filter2 = build_filter_from_dict(
            {"source": "paper.pdf", "tags": ["ai", "ml"], "date_after": "2024-01-01"}
        )

        # Both should produce valid filters with same number of conditions
        assert filter1 is not None
        assert filter2 is not None
        assert len(filter1.must) == len(filter2.must)


class TestBackwardCompatibility:
    """Test backward compatibility with existing filter patterns."""

    def test_legacy_match_still_works(self):
        """Test that legacy match() method still works."""
        filter_obj = FilterBuilder().match("source", "doc.txt").build()
        assert filter_obj is not None

    def test_legacy_range_still_works(self):
        """Test that legacy range() method still works."""
        filter_obj = FilterBuilder().range("year", gte=2020, lte=2024).build()
        assert filter_obj is not None

    def test_legacy_dict_format_still_works(self):
        """Test that legacy dict format still works."""
        filter_obj = build_filter_from_dict({"author": "Smith", "year$gte": 2020})
        assert filter_obj is not None
