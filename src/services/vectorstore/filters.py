"""
Filter builder utilities for Qdrant queries.

Provides a simple, Pythonic API for building Qdrant filter conditions.
Supports metadata filtering, range queries, and compound conditions.
"""

from collections.abc import Sequence
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

    Supports simple key-value pairs and some special operators:
    - Regular keys: exact match (e.g., {"author": "Smith"})
    - "$in" suffix: match any (e.g., {"category$in": ["AI", "ML"]})
    - "$gt", "$gte", "$lt", "$lte": range queries (e.g., {"year$gte": 2020})
    - "$not": negation (e.g., {"status$not": "deleted"})
    - "$text": text search (e.g., {"content$text": "machine learning"})

    Args:
        filter_dict: Dictionary with field names and values

    Returns:
        Filter object or None if empty

    Example:
        filter_dict = {
            "source": "doc1.txt",
            "year$gte": 2020,
            "category$in": ["AI", "ML"],
            "status$not": "deleted"
        }
        filter = build_filter_from_dict(filter_dict)
    """
    if not filter_dict:
        return None

    builder = FilterBuilder()

    for key, value in filter_dict.items():
        # Handle special operators
        if "$in" in key:
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
