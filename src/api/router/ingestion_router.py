"""Data ingestion router for loading and processing documents."""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from src.models.ingestion import IngestRequest, IngestResponse
from src.services.ingestion.factory import get_ingestion_client

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse, tags=["ingestion"])
async def ingest_documents(request: IngestRequest):
    """
    Ingest documents from a directory.

    Supports multiple file formats:
    - .txt: Plain text files
    - .md: Markdown files
    - .html: HTML files
    - .pdf: PDF files (requires PyPDF2)

    Returns the number of documents ingested and their IDs.
    """
    logger.info(f"Ingestion request: directory='{request.directory}', formats={request.formats}")

    try:
        # Validate directory exists
        directory_path = Path(request.directory)
        if not directory_path.exists():
            raise HTTPException(
                status_code=404, detail=f"Directory not found: {request.directory}"
            )

        if not directory_path.is_dir():
            raise HTTPException(
                status_code=400, detail=f"Path is not a directory: {request.directory}"
            )

        # Create ingestion client
        client = get_ingestion_client(
            source_type="local", directory=request.directory, formats=request.formats
        )

        # Ingest documents
        logger.debug("Starting document ingestion...")
        document_ids = client.ingest()
        logger.info(f"Successfully ingested {len(document_ids)} documents")

        return IngestResponse(
            status="success",
            documents_ingested=len(document_ids),
            document_ids=document_ids,
            message=f"Successfully ingested {len(document_ids)} documents from {request.directory}",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.get("/ingest/status", tags=["ingestion"])
async def get_ingestion_status():
    """Get the status of the ingestion service."""
    return {
        "status": "ready",
        "supported_formats": [".txt", ".md", ".html", ".pdf"],
        "message": "Ingestion service is ready",
    }
