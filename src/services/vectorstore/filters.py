"""
Filter builder utilities for Qdrant queries.

Provides a simple, Pythonic API for building Qdrant filter conditions.
Supports metadata filtering, range queries, and compound conditions.
"""

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from qdrant_client.models import (
    Condition,
    FieldCondition,
    Filter,
    MatchAny,
    MatchExcept,
    MatchText,
    MatchValue,
    Range,
)


class FilterBuilder:
    """
    Builder for Qdrant Filter objects with chainable methods.

    Example usage:
        # Simple metadata filter
        filter = FilterBuilder().match("source", "doc1.txt").build()

        # Range query
        filter = FilterBuilder().range("score", gte=0.5, lte=1.0).build()

        # Multiple conditions (AND)
        filter = (
            FilterBuilder()
            .match("author", "Smith")
            .range("year", gte=2020)
            .build()
        )

        # Text search
        filter = FilterBuilder().text("content", "machine learning").build()

        # Match any of multiple values
        filter = FilterBuilder().match_any("category", ["AI", "ML", "DL"]).build()
    """

    def __init__(self):
        self.must_conditions: list[Condition] = []
        self.should_conditions: list[Condition] = []
        self.must_not_conditions: list[Condition] = []

    def match(self, key: str, value: Any) -> "FilterBuilder":
        """
        Add an exact match condition.

        Args:
            key: Field name to match
            value: Value to match (str, int, bool)

        Returns:
            Self for chaining
        """
        condition = FieldCondition(key=key, match=MatchValue(value=value))
        self.must_conditions.append(condition)
        return self

    def match_any(self, key: str, values: Sequence[Any]) -> "FilterBuilder":
        """
        Match any of the provided values (OR condition for single field).

        Args:
            key: Field name to match
            values: List of values to match

        Returns:
            Self for chaining
        """
        condition = FieldCondition(key=key, match=MatchAny(any=list(values)))
        self.must_conditions.append(condition)
        return self

    def match_except(self, key: str, values: Sequence[Any]) -> "FilterBuilder":
        """
        Exclude specific values (NOT IN).

        Args:
            key: Field name to match
            values: List of values to exclude

        Returns:
            Self for chaining
        """
        # MatchExcept uses 'except' not 'except_'
        condition = FieldCondition(
            key=key,
            match=MatchExcept(
                **{"except": list(values)}
            ),  # Use dict unpacking for reserved keyword
        )
        self.must_conditions.append(condition)
        return self

    def text(self, key: str, text: str) -> "FilterBuilder":
        """
        Add a text search condition (full-text search on a field).

        Args:
            key: Field name to search
            text: Text to search for

        Returns:
            Self for chaining
        """
        condition = FieldCondition(key=key, match=MatchText(text=text))
        self.must_conditions.append(condition)
        return self

    def range(
        self,
        key: str,
        *,
        gt: float | None = None,
        gte: float | None = None,
        lt: float | None = None,
        lte: float | None = None,
    ) -> "FilterBuilder":
        """
        Add a range condition for numeric fields.

        Args:
            key: Field name
            gt: Greater than
            gte: Greater than or equal
            lt: Less than
            lte: Less than or equal

        Returns:
            Self for chaining
        """
        condition = FieldCondition(key=key, range=Range(gt=gt, gte=gte, lt=lt, lte=lte))
        self.must_conditions.append(condition)
        return self

    def should(self, key: str, value: Any) -> "FilterBuilder":
        """
        Add an OR condition (at least one should match).

        Args:
            key: Field name
            value: Value to match

        Returns:
            Self for chaining
        """
        condition = FieldCondition(key=key, match=MatchValue(value=value))
        self.should_conditions.append(condition)
        return self

    def must_not(self, key: str, value: Any) -> "FilterBuilder":
        """
        Add a negation condition (must NOT match).

        Args:
            key: Field name
            value: Value that must not match

        Returns:
            Self for chaining
        """
        condition = FieldCondition(key=key, match=MatchValue(value=value))
        self.must_not_conditions.append(condition)
        return self

    # Convenience methods for common metadata filters

    def source(self, source: str) -> "FilterBuilder":
        """
        Filter by document source (exact match).

        Args:
            source: Source identifier (e.g., filename, URL, document ID)

        Returns:
            Self for chaining

        Example:
            FilterBuilder().source("paper.pdf")
        """
        return self.match("source", source)

    def sources(self, sources: list[str]) -> "FilterBuilder":
        """
        Filter by multiple sources (match any).

        Args:
            sources: List of source identifiers

        Returns:
            Self for chaining

        Example:
            FilterBuilder().sources(["paper1.pdf", "paper2.pdf"])
        """
        return self.match_any("source", sources)

    def exclude_source(self, source: str) -> "FilterBuilder":
        """
        Exclude documents from a specific source.

        Args:
            source: Source identifier to exclude

        Returns:
            Self for chaining

        Example:
            FilterBuilder().exclude_source("draft.txt")
        """
        return self.must_not("source", source)

    def tag(self, tag: str) -> "FilterBuilder":
        """
        Filter by a single tag (exact match).

        Args:
            tag: Tag value

        Returns:
            Self for chaining

        Example:
            FilterBuilder().tag("machine-learning")
        """
        return self.match("tag", tag)

    def tags(self, tags: list[str]) -> "FilterBuilder":
        """
        Filter by multiple tags (match any).

        Args:
            tags: List of tag values

        Returns:
            Self for chaining

        Example:
            FilterBuilder().tags(["ai", "ml", "deep-learning"])
        """
        return self.match_any("tags", tags)

    def date_range(
        self,
        *,
        after: datetime | str | None = None,
        before: datetime | str | None = None,
        field: str = "date",
    ) -> "FilterBuilder":
        """
        Filter by date range. Accepts datetime objects or ISO format strings.

        Args:
            after: Documents created/modified after this date (inclusive)
            before: Documents created/modified before this date (inclusive)
            field: Metadata field name containing the date (default: "date")

        Returns:
            Self for chaining

        Example:
            FilterBuilder().date_range(after="2024-01-01", before="2024-12-31")
            FilterBuilder().date_range(after=datetime(2024, 1, 1))
        """
        gte_value = None
        lte_value = None

        if after is not None:
            if isinstance(after, str):
                after = datetime.fromisoformat(after)
            gte_value = after.timestamp()

        if before is not None:
            if isinstance(before, str):
                before = datetime.fromisoformat(before)
            lte_value = before.timestamp()

        # Only add range if at least one bound is specified
        if gte_value is not None or lte_value is not None:
            self.range(field, gte=gte_value, lte=lte_value)

        return self

    def created_after(self, date: datetime | str, field: str = "created_at") -> "FilterBuilder":
        """
        Filter documents created after a specific date.

        Args:
            date: Cutoff date (datetime or ISO string)
            field: Metadata field name (default: "created_at")

        Returns:
            Self for chaining

        Example:
            FilterBuilder().created_after("2024-01-01")
        """
        return self.date_range(after=date, field=field)

    def created_before(self, date: datetime | str, field: str = "created_at") -> "FilterBuilder":
        """
        Filter documents created before a specific date.

        Args:
            date: Cutoff date (datetime or ISO string)
            field: Metadata field name (default: "created_at")

        Returns:
            Self for chaining

        Example:
            FilterBuilder().created_before("2024-12-31")
        """
        return self.date_range(before=date, field=field)

    def author(self, author: str) -> "FilterBuilder":
        """
        Filter by document author.

        Args:
            author: Author name

        Returns:
            Self for chaining

        Example:
            FilterBuilder().author("John Smith")
        """
        return self.match("author", author)

    def authors(self, authors: list[str]) -> "FilterBuilder":
        """
        Filter by multiple authors (match any).

        Args:
            authors: List of author names

        Returns:
            Self for chaining

        Example:
            FilterBuilder().authors(["Smith", "Johnson"])
        """
        return self.match_any("author", authors)

    def build(self) -> Filter | None:
        """
        Build the final Qdrant Filter object.

        Returns:
            Filter object if any conditions are set, None otherwise
        """
        if not self.must_conditions and not self.should_conditions and not self.must_not_conditions:
            return None

        return Filter(
            must=self.must_conditions if self.must_conditions else None,
            should=self.should_conditions if self.should_conditions else None,
            must_not=self.must_not_conditions if self.must_not_conditions else None,
        )


def build_filter_from_dict(filter_dict: dict[str, Any] | None) -> Filter | None:
    """
    Build a Qdrant Filter from a simple dictionary format.

    Supports simple key-value pairs and special operators:
    - Regular keys: exact match (e.g., {"author": "Smith"})
    - "$in" suffix: match any (e.g., {"category$in": ["AI", "ML"]})
    - "$gt", "$gte", "$lt", "$lte": range queries (e.g., {"year$gte": 2020})
    - "$not": negation (e.g., {"status$not": "deleted"})
    - "$text": text search (e.g., {"content$text": "machine learning"})
    - "$after", "$before": date range (e.g., {"date$after": "2024-01-01"})

    Convenience fields:
    - "source" or "sources": document source filtering
    - "tag" or "tags": tag filtering
    - "author" or "authors": author filtering
    - "date_after", "date_before": date range filtering

    Args:
        filter_dict: Dictionary with field names and values

    Returns:
        Filter object or None if empty

    Example:
        filter_dict = {
            "source": "doc1.txt",
            "year$gte": 2020,
            "category$in": ["AI", "ML"],
            "status$not": "deleted",
            "date_after": "2024-01-01"
        }
        filter = build_filter_from_dict(filter_dict)
    """
    if not filter_dict:
        return None

    builder = FilterBuilder()

    # Handle date range fields specially to combine them
    date_after = filter_dict.get("date_after")
    date_before = filter_dict.get("date_before")
    created_after = filter_dict.get("created_after")
    created_before = filter_dict.get("created_before")

    # Apply combined date ranges
    if date_after or date_before:
        builder.date_range(after=date_after, before=date_before)
    if created_after or created_before:
        builder.date_range(after=created_after, before=created_before, field="created_at")

    # Process other keys
    for key, value in filter_dict.items():
        # Skip already-processed date fields
        if key in ("date_after", "date_before", "created_after", "created_before"):
            continue

        # Convenience fields with special handling
        if key == "sources" and isinstance(value, list):
            builder.sources(value)
        elif key == "source" and isinstance(value, str):
            builder.source(value)
        elif key == "tags" and isinstance(value, list):
            builder.tags(value)
        elif key == "tag" and isinstance(value, str):
            builder.tag(value)
        elif key == "authors" and isinstance(value, list):
            builder.authors(value)
        elif key == "author" and isinstance(value, str):
            builder.author(value)
        # Date range operators
        elif "$after" in key:
            field_name = key.replace("$after", "")
            builder.date_range(after=value, field=field_name or "date")
        elif "$before" in key:
            field_name = key.replace("$before", "")
            builder.date_range(before=value, field=field_name or "date")
        # Other special operators
        elif "$in" in key:
            field_name = key.replace("$in", "")
            if isinstance(value, list):
                builder.match_any(field_name, value)
        elif "$not" in key:
            field_name = key.replace("$not", "")
            builder.must_not(field_name, value)
        elif "$text" in key:
            field_name = key.replace("$text", "")
            builder.text(field_name, value)
        elif any(op in key for op in ["$gt", "$gte", "$lt", "$lte"]):
            # Extract field name and operator
            for op in ["$gte", "$gt", "$lte", "$lt"]:  # Check longer operators first
                if op in key:
                    field_name = key.replace(op, "")
                    kwargs = {op[1:]: value}  # Remove $ from operator
                    builder.range(field_name, **kwargs)
                    break
        else:
            # Simple exact match
            builder.match(key, value)

    return builder.build()
