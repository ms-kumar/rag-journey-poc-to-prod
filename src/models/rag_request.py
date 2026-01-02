from typing import List, Optional
from pydantic import BaseModel, Field

class GenerateRequest(BaseModel):
    """Request model for the RAG generate endpoint.

    Fields:
    - `prompt`: the user's prompt to the RAG system.
    - `top_k`: number of retrieved passages to use (defaults to 5).
    - `max_length`: optional generation max tokens override.
    - `metadata_filters`: optional filters applied to retrieval metadata.
    """

    prompt: str = Field(..., description="Prompt text to generate from")
    top_k: int = Field(5, description="Number of retrieved passages to return/use")
    max_length: Optional[int] = Field(None, description="Optional generation max tokens")
    metadata_filters: Optional[dict] = Field(None, description="Optional metadata filters for retrieval")

class GenerateResponse(BaseModel):
    """Response model for RAG generate endpoint."""
    prompt: str
    answer: str
    context: Optional[str] = Field(None, description="Combined context from retrieved documents")
    sources: Optional[List[str]] = Field(None, description="List of retrieved document texts")