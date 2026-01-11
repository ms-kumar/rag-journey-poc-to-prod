"""Query understanding service schemas."""

from typing import Any

from pydantic import BaseModel, Field, field_validator


class QueryRewriterConfig(BaseModel):
    """Configuration for query rewriting."""

    expand_acronyms: bool = Field(default=True, description="Expand known acronyms")
    fix_typos: bool = Field(default=True, description="Attempt to fix typos")
    add_context: bool = Field(default=True, description="Add contextual information")
    max_rewrites: int = Field(default=3, ge=1, le=10, description="Maximum number of rewrites")
    min_query_length: int = Field(default=3, ge=1, description="Minimum query length to process")

    @field_validator("max_rewrites")
    @classmethod
    def validate_max_rewrites(cls, v: int) -> int:
        """Validate max_rewrites is reasonable."""
        if v < 1 or v > 10:
            raise ValueError("max_rewrites must be between 1 and 10")
        return v

    @field_validator("min_query_length")
    @classmethod
    def validate_min_query_length(cls, v: int) -> int:
        """Validate min_query_length is positive."""
        if v < 1:
            raise ValueError("min_query_length must be positive")
        return v

    @classmethod
    def from_settings(cls, settings: Any) -> "QueryRewriterConfig":
        """Create config from application settings."""
        query_settings = settings.query_understanding
        return cls(
            expand_acronyms=query_settings.expand_acronyms,
            fix_typos=query_settings.fix_typos,
            add_context=query_settings.add_context,
            max_rewrites=query_settings.max_rewrites,
            min_query_length=query_settings.min_query_length,
        )


class SynonymExpanderConfig(BaseModel):
    """Configuration for synonym expansion."""

    max_synonyms_per_term: int = Field(
        default=3, ge=1, le=10, description="Maximum synonyms per term"
    )
    min_term_length: int = Field(default=3, ge=1, description="Minimum term length to expand")
    expand_all_terms: bool = Field(default=False, description="Expand all terms vs only key terms")

    @field_validator("max_synonyms_per_term")
    @classmethod
    def validate_max_synonyms(cls, v: int) -> int:
        """Validate max_synonyms_per_term is reasonable."""
        if v < 1 or v > 10:
            raise ValueError("max_synonyms_per_term must be between 1 and 10")
        return v

    @field_validator("min_term_length")
    @classmethod
    def validate_min_term_length(cls, v: int) -> int:
        """Validate min_term_length is positive."""
        if v < 1:
            raise ValueError("min_term_length must be positive")
        return v

    @classmethod
    def from_settings(cls, settings: Any) -> "SynonymExpanderConfig":
        """Create config from application settings."""
        query_settings = settings.query_understanding
        return cls(
            max_synonyms_per_term=query_settings.max_synonyms_per_term,
            min_term_length=query_settings.min_term_length,
            expand_all_terms=query_settings.expand_all_terms,
        )
