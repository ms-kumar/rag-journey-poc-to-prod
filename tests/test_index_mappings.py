"""
Tests for Qdrant index mappings.
"""

import pytest
from qdrant_client.models import (
    IntegerIndexParams,
    KeywordIndexParams,
    TextIndexParams,
    TokenizerType,
)

from src.services.vectorstore.index_mappings import (
    COMMON_PRESETS,
    IndexMapping,
    IndexMappingBuilder,
    get_preset_mappings,
    get_qdrant_field_schema,
)


class TestIndexMapping:
    """Test IndexMapping dataclass."""

    def test_create_keyword_mapping(self):
        """Test creating a keyword index mapping."""
        mapping = IndexMapping(field_name="category", field_type="keyword")
        assert mapping.field_name == "category"
        assert mapping.field_type == "keyword"

    def test_create_integer_mapping(self):
        """Test creating an integer index mapping."""
        mapping = IndexMapping(field_name="year", field_type="integer", range=True)
        assert mapping.field_name == "year"
        assert mapping.field_type == "integer"
        assert mapping.range is True

    def test_create_text_mapping(self):
        """Test creating a text index mapping."""
        mapping = IndexMapping(
            field_name="description",
            field_type="text",
            tokenizer=TokenizerType.WORD,
            min_token_len=3,
            max_token_len=20,
            lowercase=True,
        )
        assert mapping.field_name == "description"
        assert mapping.field_type == "text"
        assert mapping.tokenizer == TokenizerType.WORD
        assert mapping.min_token_len == 3


class TestIndexMappingBuilder:
    """Test IndexMappingBuilder class."""

    def test_empty_builder(self):
        """Test empty builder returns empty list."""
        builder = IndexMappingBuilder()
        mappings = builder.build()
        assert mappings == []

    def test_add_keyword(self):
        """Test adding keyword mapping."""
        mappings = IndexMappingBuilder().add_keyword("category").build()
        assert len(mappings) == 1
        assert mappings[0].field_name == "category"
        assert mappings[0].field_type == "keyword"

    def test_add_integer(self):
        """Test adding integer mapping."""
        mappings = IndexMappingBuilder().add_integer("year", range=True).build()
        assert len(mappings) == 1
        assert mappings[0].field_name == "year"
        assert mappings[0].field_type == "integer"
        assert mappings[0].range is True

    def test_add_float(self):
        """Test adding float mapping."""
        mappings = IndexMappingBuilder().add_float("score", range=True).build()
        assert len(mappings) == 1
        assert mappings[0].field_name == "score"
        assert mappings[0].field_type == "float"
        assert mappings[0].range is True

    def test_add_text(self):
        """Test adding text mapping."""
        mappings = (
            IndexMappingBuilder().add_text("content", min_token_len=3, max_token_len=25).build()
        )
        assert len(mappings) == 1
        assert mappings[0].field_name == "content"
        assert mappings[0].field_type == "text"
        assert mappings[0].min_token_len == 3
        assert mappings[0].max_token_len == 25

    def test_add_datetime(self):
        """Test adding datetime mapping."""
        mappings = IndexMappingBuilder().add_datetime("created_at", range=True).build()
        assert len(mappings) == 1
        assert mappings[0].field_name == "created_at"
        assert mappings[0].field_type == "datetime"
        assert mappings[0].range is True

    def test_add_bool(self):
        """Test adding boolean mapping."""
        mappings = IndexMappingBuilder().add_bool("is_active").build()
        assert len(mappings) == 1
        assert mappings[0].field_name == "is_active"
        assert mappings[0].field_type == "bool"

    def test_add_geo(self):
        """Test adding geo mapping."""
        mappings = IndexMappingBuilder().add_geo("location").build()
        assert len(mappings) == 1
        assert mappings[0].field_name == "location"
        assert mappings[0].field_type == "geo"

    def test_chainable_api(self):
        """Test that builder methods are chainable."""
        builder = IndexMappingBuilder()
        result = (
            builder.add_keyword("category")
            .add_integer("year", range=True)
            .add_float("score", range=True)
            .add_text("content")
        )
        assert result is builder
        mappings = builder.build()
        assert len(mappings) == 4

    def test_complex_mapping(self):
        """Test building complex index configuration."""
        mappings = (
            IndexMappingBuilder()
            .add_keyword("source")
            .add_keyword("category")
            .add_integer("year", range=True)
            .add_integer("chunk_index", range=True)
            .add_float("score", range=True)
            .add_text("abstract", min_token_len=3)
            .add_bool("is_published")
            .add_datetime("created_at", range=True)
            .build()
        )
        assert len(mappings) == 8
        field_names = [m.field_name for m in mappings]
        assert "source" in field_names
        assert "category" in field_names
        assert "year" in field_names
        assert "abstract" in field_names


class TestGetQdrantFieldSchema:
    """Test conversion to Qdrant field schemas."""

    def test_keyword_schema(self):
        """Test keyword mapping to Qdrant schema."""
        mapping = IndexMapping(field_name="category", field_type="keyword")
        schema = get_qdrant_field_schema(mapping)
        assert isinstance(schema, KeywordIndexParams)

    def test_integer_schema(self):
        """Test integer mapping to Qdrant schema."""
        mapping = IndexMapping(field_name="year", field_type="integer", range=True)
        schema = get_qdrant_field_schema(mapping)
        assert isinstance(schema, IntegerIndexParams)
        assert schema.range is True

    def test_float_schema(self):
        """Test float mapping to Qdrant schema."""
        mapping = IndexMapping(field_name="score", field_type="float", range=True)
        schema = get_qdrant_field_schema(mapping)
        # FloatIndexParams doesn't support range/lookup in current Qdrant
        # Returns PayloadSchemaType.FLOAT instead
        from qdrant_client.models import PayloadSchemaType

        assert schema == PayloadSchemaType.FLOAT

    def test_text_schema(self):
        """Test text mapping to Qdrant schema."""
        mapping = IndexMapping(
            field_name="content",
            field_type="text",
            tokenizer=TokenizerType.WORD,
            min_token_len=3,
            max_token_len=20,
            lowercase=True,
        )
        schema = get_qdrant_field_schema(mapping)
        assert isinstance(schema, TextIndexParams)
        assert schema.tokenizer == TokenizerType.WORD
        assert schema.min_token_len == 3
        assert schema.max_token_len == 20
        assert schema.lowercase is True

    def test_text_schema_defaults(self):
        """Test text mapping with default values."""
        mapping = IndexMapping(field_name="content", field_type="text")
        schema = get_qdrant_field_schema(mapping)
        assert isinstance(schema, TextIndexParams)
        assert schema.tokenizer == TokenizerType.WORD
        assert schema.min_token_len == 2
        assert schema.max_token_len == 20
        assert schema.lowercase is True

    def test_unsupported_type_raises_error(self):
        """Test that unsupported field type raises error."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            IndexMapping(field_name="test", field_type="invalid")  # type: ignore


class TestPresetMappings:
    """Test predefined preset mappings."""

    def test_preset_exists(self):
        """Test that presets are defined."""
        assert "document_metadata" in COMMON_PRESETS
        assert "research_paper" in COMMON_PRESETS
        assert "e_commerce" in COMMON_PRESETS
        assert "news_article" in COMMON_PRESETS

    def test_get_document_metadata_preset(self):
        """Test document_metadata preset."""
        mappings = get_preset_mappings("document_metadata")
        assert len(mappings) >= 3
        field_names = [m.field_name for m in mappings]
        assert "source" in field_names
        assert "category" in field_names

    def test_get_research_paper_preset(self):
        """Test research_paper preset."""
        mappings = get_preset_mappings("research_paper")
        assert len(mappings) >= 4
        field_names = [m.field_name for m in mappings]
        assert "year" in field_names
        assert "citations" in field_names

    def test_get_ecommerce_preset(self):
        """Test e_commerce preset."""
        mappings = get_preset_mappings("e_commerce")
        assert len(mappings) >= 4
        field_names = [m.field_name for m in mappings]
        assert "price" in field_names
        assert "rating" in field_names

    def test_get_news_article_preset(self):
        """Test news_article preset."""
        mappings = get_preset_mappings("news_article")
        assert len(mappings) >= 3
        field_names = [m.field_name for m in mappings]
        assert "headline" in field_names
        assert "published_date" in field_names

    def test_invalid_preset_raises_error(self):
        """Test that invalid preset name raises error."""
        with pytest.raises(ValueError, match="Unknown preset"):
            get_preset_mappings("invalid_preset")

    def test_preset_returns_copy(self):
        """Test that preset returns a copy, not reference."""
        mappings1 = get_preset_mappings("document_metadata")
        mappings2 = get_preset_mappings("document_metadata")
        assert mappings1 == mappings2
        assert mappings1 is not mappings2  # Different objects


class TestIndexMappingIntegration:
    """Integration tests for index mappings."""

    def test_build_and_convert_mappings(self):
        """Test building mappings and converting to Qdrant schemas."""
        mappings = (
            IndexMappingBuilder()
            .add_keyword("category")
            .add_integer("year", range=True)
            .add_float("score", range=True)
            .add_text("description")
            .build()
        )

        schemas = [get_qdrant_field_schema(m) for m in mappings]
        assert len(schemas) == 4
        assert isinstance(schemas[0], KeywordIndexParams)
        assert isinstance(schemas[1], IntegerIndexParams)
        # Float returns PayloadSchemaType.FLOAT (not FloatIndexParams)
        from qdrant_client.models import PayloadSchemaType

        assert schemas[2] == PayloadSchemaType.FLOAT
        assert isinstance(schemas[3], TextIndexParams)

    def test_preset_to_schemas(self):
        """Test converting preset mappings to Qdrant schemas."""
        mappings = get_preset_mappings("research_paper")
        schemas = [get_qdrant_field_schema(m) for m in mappings]
        assert len(schemas) == len(mappings)
        # All should convert without error
        for schema in schemas:
            assert schema is not None


class TestIndexMappingEdgeCases:
    """Test edge cases and special scenarios."""

    def test_multiple_indices_same_name(self):
        """Test that builder allows multiple indices on same field."""
        # This might be used to update index config
        mappings = (
            IndexMappingBuilder()
            .add_keyword("field1")
            .add_keyword("field1")  # Duplicate
            .build()
        )
        assert len(mappings) == 2  # Builder doesn't prevent duplicates

    def test_integer_range_and_lookup_options(self):
        """Test integer mapping with specific options."""
        mappings = IndexMappingBuilder().add_integer("count", range=True, lookup=False).build()
        assert mappings[0].range is True
        assert mappings[0].lookup is False

    def test_text_tokenizer_options(self):
        """Test text mapping with different tokenizers."""
        mappings = (
            IndexMappingBuilder()
            .add_text("field1", tokenizer=TokenizerType.WORD)
            .add_text("field2", tokenizer=TokenizerType.WHITESPACE)
            .add_text("field3", tokenizer=TokenizerType.PREFIX)
            .build()
        )
        assert len(mappings) == 3
        assert mappings[0].tokenizer == TokenizerType.WORD
        assert mappings[1].tokenizer == TokenizerType.WHITESPACE
        assert mappings[2].tokenizer == TokenizerType.PREFIX

    def test_float_without_range(self):
        """Test float mapping without range queries."""
        mappings = IndexMappingBuilder().add_float("value", range=False).build()
        assert mappings[0].range is False
        assert mappings[0].lookup is True  # Default


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
