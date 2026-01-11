"""Request and response models for /ask endpoint."""

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Request model for /ask endpoint."""

    question: str = Field(..., description="The question to ask")
    top_k: int | None = Field(5, description="Number of documents to retrieve")


class AskResponse(BaseModel):
    """Response model for /ask endpoint."""

    question: str
    answer: str
    context: str | None = Field(None, description="Combined context from retrieved documents")
    sources: list[str] | None = Field(None, description="List of retrieved document texts")
