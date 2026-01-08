from typing import Literal

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Request model for the RAG generate endpoint.

    Fields:
    - `prompt`: the user's prompt to the RAG system.
    - `top_k`: number of retrieved passages to use (defaults to 5).
    - `max_length`: optional generation max tokens override.
    - `metadata_filters`: optional filters applied to retrieval metadata.
    - `search_type`: type of search to perform (vector, bm25, or hybrid).
    - `hybrid_alpha`: weight for hybrid search (0.0=BM25 only, 1.0=vector only).
    """

    prompt: str = Field(..., description="Prompt text to generate from")
    top_k: int = Field(5, description="Number of retrieved passages to return/use")
    max_length: int | None = Field(None, description="Optional generation max tokens")
    metadata_filters: dict | None = Field(
        None, description="Optional metadata filters for retrieval"
    )
    search_type: Literal["vector", "bm25", "hybrid"] = Field(
        "vector", description="Type of search to perform"
    )
    hybrid_alpha: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Weight for hybrid search (0.0=BM25, 1.0=vector, 0.5=balanced)",
    )


class GenerateResponse(BaseModel):
    """Response model for RAG generate endpoint."""

    prompt: str
    answer: str
    context: str | None = Field(None, description="Combined context from retrieved documents")
    sources: list[str] | None = Field(None, description="List of retrieved document texts")
