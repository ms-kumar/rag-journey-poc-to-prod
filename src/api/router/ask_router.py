import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List

from src.services.pipeline.naive_pipeline.factory import get_naive_pipeline
from src.api.v1.endpoints import rag

logger = logging.getLogger(__name__)

api_router = APIRouter()

# Include the v1 rag router
api_router.include_router(rag.router, prefix="/rag", tags=["rag"])


# --- Direct /ask endpoint using NaivePipeline ---


class AskRequest(BaseModel):
    question: str = Field(..., description="The question to ask")
    top_k: Optional[int] = Field(5, description="Number of documents to retrieve")


class AskResponse(BaseModel):
    question: str
    answer: str
    context: Optional[str] = Field(
        None, description="Combined context from retrieved documents"
    )
    sources: Optional[List[str]] = Field(
        None, description="List of retrieved document texts"
    )


# Lazy singleton pipeline
_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        logger.info("Initializing NaivePipeline for /ask endpoint...")
        _pipeline = get_naive_pipeline()
        logger.info("NaivePipeline ready")
    return _pipeline


def _extract_sources(retrieved) -> List[str]:
    """Extract text content from retrieved documents."""
    sources: List[str] = []
    for doc in retrieved:
        if hasattr(doc, "page_content"):
            sources.append(doc.page_content)
        elif isinstance(doc, str):
            sources.append(doc)
        elif isinstance(doc, dict) and "text" in doc:
            sources.append(doc["text"])
    return sources


@api_router.post("/ask", response_model=AskResponse, tags=["ask"])
async def ask(request: AskRequest):
    """
    Simple /ask endpoint that queries the NaivePipeline.
    Retrieves relevant documents and generates an answer.
    Returns answer + context.
    """
    logger.info(
        f"Received /ask request: question='{request.question[:50]}...', top_k={request.top_k}"
    )

    try:
        pipeline = _get_pipeline()
    except Exception as e:
        logger.error(f"Pipeline initialization error: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    # Retrieve
    logger.debug(f"Retrieving top {request.top_k} documents...")
    retrieved = pipeline.retrieve(request.question, k=request.top_k)
    logger.info(f"Retrieved {len(retrieved)} documents")

    # Extract sources and build context
    sources = _extract_sources(retrieved)
    context = "\n\n".join(sources[:3]) if sources else None
    logger.debug(f"Built context from {len(sources)} sources")

    # Generate
    logger.debug("Generating answer...")
    answer = pipeline.generate(request.question, retrieved_docs=retrieved)
    logger.info(
        f"Generated answer: '{answer[:100]}...'"
        if len(answer) > 100
        else f"Generated answer: '{answer}'"
    )

    return AskResponse(
        question=request.question,
        answer=answer,
        context=context,
        sources=sources,
    )
