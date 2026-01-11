"""Request and response models for data ingestion endpoints."""

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    """Request model for document ingestion."""

    directory: str = Field("./data", description="Directory containing documents to ingest")
    formats: list[str] | None = Field(
        None, description="File formats to ingest (e.g., ['*.txt', '*.md'])"
    )


class IngestResponse(BaseModel):
    """Response model for document ingestion."""

    status: str
    documents_ingested: int
    document_ids: list[str]
    message: str
