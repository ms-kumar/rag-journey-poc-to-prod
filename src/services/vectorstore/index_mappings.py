"""
Index mapping utilities for Qdrant payload fields.

Provides helpers to create and manage indices on metadata fields for faster filtering.
"""

from qdrant_client.models import (
    DatetimeIndexParams,
    IntegerIndexParams,
    KeywordIndexParams,
    PayloadSchemaType,
    TextIndexParams,
    TokenizerType,
)

from src.schemas.services.vectorstore import IndexMapping


class IndexMappingBuilder:
    """
    Builder for creating index mappings for Qdrant payload fields.

    Example usage:
        mappings = (
            IndexMappingBuilder()
            .add_keyword("category")  # Exact match on category
            .add_integer("year", range=True)  # Range queries on year
            .add_float("score", range=True)  # Range queries on score
            .add_text("description", min_token_len=3)  # Full-text on description
            .build()
        )
    """

    def __init__(self):
        self.mappings: list[IndexMapping] = []

    def add_keyword(self, field_name: str, lookup: bool = True) -> "IndexMappingBuilder":
        """
        Add keyword index for exact string matching.

        Best for: categories, tags, IDs, enum values.

        Args:
            field_name: Field name to index
            lookup: Enable fast exact match lookups

        Returns:
            Self for chaining
        """
        self.mappings.append(
            IndexMapping(field_name=field_name, field_type="keyword", lookup=lookup)
        )
        return self

    def add_integer(
        self, field_name: str, range: bool = True, lookup: bool = True
    ) -> "IndexMappingBuilder":
        """
        Add integer index for numeric fields.

        Best for: counts, IDs, years, rankings.

        Args:
            field_name: Field name to index
            range: Enable range queries (gt, gte, lt, lte)
            lookup: Enable exact match lookups

        Returns:
            Self for chaining
        """
        self.mappings.append(
            IndexMapping(field_name=field_name, field_type="integer", range=range, lookup=lookup)
        )
        return self

    def add_float(
        self, field_name: str, range: bool = True, lookup: bool = True
    ) -> "IndexMappingBuilder":
        """
        Add float index for decimal numeric fields.

        Best for: scores, ratings, prices, probabilities.

        Args:
            field_name: Field name to index
            range: Enable range queries (gt, gte, lt, lte)
            lookup: Enable exact match lookups

        Returns:
            Self for chaining
        """
        self.mappings.append(
            IndexMapping(field_name=field_name, field_type="float", range=range, lookup=lookup)
        )
        return self

    def add_text(
        self,
        field_name: str,
        tokenizer: TokenizerType = TokenizerType.WORD,
        min_token_len: int = 2,
        max_token_len: int = 20,
        lowercase: bool = True,
    ) -> "IndexMappingBuilder":
        """
        Add text index for full-text search (BM25-style).

        Best for: descriptions, abstracts, content snippets.

        Args:
            field_name: Field name to index
            tokenizer: Tokenization strategy (WORD, WHITESPACE, PREFIX)
            min_token_len: Minimum token length to index
            max_token_len: Maximum token length to index
            lowercase: Convert to lowercase before indexing

        Returns:
            Self for chaining
        """
        self.mappings.append(
            IndexMapping(
                field_name=field_name,
                field_type="text",
                tokenizer=tokenizer,
                min_token_len=min_token_len,
                max_token_len=max_token_len,
                lowercase=lowercase,
            )
        )
        return self

    def add_datetime(
        self, field_name: str, range: bool = True, lookup: bool = True
    ) -> "IndexMappingBuilder":
        """
        Add datetime index for timestamp fields.

        Best for: timestamps, dates, creation/update times.

        Args:
            field_name: Field name to index
            range: Enable range queries
            lookup: Enable exact match lookups

        Returns:
            Self for chaining
        """
        self.mappings.append(
            IndexMapping(field_name=field_name, field_type="datetime", range=range, lookup=lookup)
        )
        return self

    def add_bool(self, field_name: str) -> "IndexMappingBuilder":
        """
        Add boolean index for true/false fields.

        Best for: flags, status indicators.

        Args:
            field_name: Field name to index

        Returns:
            Self for chaining
        """
        self.mappings.append(IndexMapping(field_name=field_name, field_type="bool"))
        return self

    def add_geo(self, field_name: str) -> "IndexMappingBuilder":
        """
        Add geo index for geospatial coordinates.

        Best for: locations, GPS coordinates.

        Args:
            field_name: Field name to index (should contain lat/lon)

        Returns:
            Self for chaining
        """
        self.mappings.append(IndexMapping(field_name=field_name, field_type="geo"))
        return self

    def build(self) -> list[IndexMapping]:
        """
        Build and return the list of index mappings.

        Returns:
            List of IndexMapping objects
        """
        return self.mappings


def get_qdrant_field_schema(mapping: IndexMapping):
    """
    Convert an IndexMapping to a Qdrant field schema.

    Args:
        mapping: IndexMapping configuration

    Returns:
        Qdrant field schema (PayloadSchemaType or specific index params)
    """
    if mapping.field_type == "keyword":
        return KeywordIndexParams(type="keyword", is_tenant=False, on_disk=None)  # type: ignore[arg-type]

    if mapping.field_type == "integer":
        return IntegerIndexParams(
            type="integer",  # type: ignore[arg-type]
            range=mapping.range,
            lookup=mapping.lookup,
            on_disk=None,
        )

    if mapping.field_type == "float":
        # FloatIndexParams doesn't support range/lookup in current Qdrant version
        # It only supports 'type' parameter
        return PayloadSchemaType.FLOAT

    if mapping.field_type == "text":
        return TextIndexParams(
            type="text",  # type: ignore[arg-type]
            tokenizer=mapping.tokenizer or TokenizerType.WORD,
            min_token_len=mapping.min_token_len or 2,
            max_token_len=mapping.max_token_len or 20,
            lowercase=mapping.lowercase if mapping.lowercase is not None else True,
            on_disk=None,
        )

    if mapping.field_type == "datetime":
        return DatetimeIndexParams(
            type="datetime",  # type: ignore[arg-type]
            on_disk=None,
        )

    if mapping.field_type == "bool":
        return PayloadSchemaType.BOOL

    if mapping.field_type == "geo":
        return PayloadSchemaType.GEO

    raise ValueError(f"Unsupported field type: {mapping.field_type}")


# Predefined common index mapping presets
COMMON_PRESETS = {
    "document_metadata": [
        IndexMapping(field_name="source", field_type="keyword"),
        IndexMapping(field_name="category", field_type="keyword"),
        IndexMapping(field_name="author", field_type="keyword"),
        IndexMapping(field_name="year", field_type="integer", range=True),
        IndexMapping(field_name="chunk_index", field_type="integer", range=True),
    ],
    "research_paper": [
        IndexMapping(field_name="title", field_type="text"),
        IndexMapping(field_name="authors", field_type="keyword"),
        IndexMapping(field_name="venue", field_type="keyword"),
        IndexMapping(field_name="year", field_type="integer", range=True),
        IndexMapping(field_name="citations", field_type="integer", range=True),
        IndexMapping(field_name="category", field_type="keyword"),
    ],
    "e_commerce": [
        IndexMapping(field_name="product_id", field_type="keyword"),
        IndexMapping(field_name="category", field_type="keyword"),
        IndexMapping(field_name="brand", field_type="keyword"),
        IndexMapping(field_name="price", field_type="float", range=True),
        IndexMapping(field_name="rating", field_type="float", range=True),
        IndexMapping(field_name="in_stock", field_type="bool"),
    ],
    "news_article": [
        IndexMapping(field_name="headline", field_type="text"),
        IndexMapping(field_name="category", field_type="keyword"),
        IndexMapping(field_name="author", field_type="keyword"),
        IndexMapping(field_name="published_date", field_type="datetime", range=True),
        IndexMapping(field_name="tags", field_type="keyword"),
    ],
}


def get_preset_mappings(preset_name: str) -> list[IndexMapping]:
    """
    Get predefined index mappings for common use cases.

    Available presets:
    - "document_metadata": Basic document metadata (source, category, year, etc.)
    - "research_paper": Academic papers (title, authors, citations, etc.)
    - "e_commerce": Product data (price, rating, category, etc.)
    - "news_article": News articles (headline, date, author, etc.)

    Args:
        preset_name: Name of the preset

    Returns:
        List of IndexMapping objects

    Raises:
        ValueError: If preset name is not found
    """
    if preset_name not in COMMON_PRESETS:
        available = ", ".join(COMMON_PRESETS.keys())
        raise ValueError(f"Unknown preset: {preset_name}. Available: {available}")
    return COMMON_PRESETS[preset_name].copy()
