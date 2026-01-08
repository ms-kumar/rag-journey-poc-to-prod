import logging

from fastapi import APIRouter, HTTPException

from src.models.rag_request import GenerateRequest, GenerateResponse
from src.services.pipeline.naive_pipeline.factory import get_naive_pipeline

logger = logging.getLogger(__name__)

router = APIRouter()

# Lazy singleton pipeline instance
_pipeline_instance = None


def _get_pipeline():
    global _pipeline_instance
    if _pipeline_instance is None:
        logger.info("Initializing NaivePipeline...")
        _pipeline_instance = get_naive_pipeline()
        logger.info("NaivePipeline initialized successfully")
    return _pipeline_instance


def _extract_sources(retrieved) -> list[str]:
    """Extract text content from retrieved documents."""
    sources: list[str] = []
    for doc in retrieved:
        if hasattr(doc, "page_content"):
            sources.append(doc.page_content)
        elif isinstance(doc, str):
            sources.append(doc)
        elif isinstance(doc, dict) and "text" in doc:
            sources.append(doc["text"])
    return sources


def _build_context(sources: list[str], max_sources: int = 3) -> str:
    """Build context string from sources."""
    return "\n\n".join(sources[:max_sources])


@router.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    RAG Generate endpoint: retrieves relevant documents and generates a response.
    Returns answer + context.

    Supports multiple search types:
    - vector: Semantic similarity search using embeddings (default)
    - bm25: Keyword-based search using BM25 algorithm
    - hybrid: Combined vector + BM25 search for best results
    - sparse: Neural sparse retrieval using SPLADE encoder

    Optional metadata filters can be applied using filter syntax:
    - Exact match: {"source": "doc1.txt"}
    - Range: {"year$gte": 2020, "score$lte": 0.9}
    - Match any: {"category$in": ["AI", "ML"]}
    - Negation: {"status$not": "deleted"}
    """
    logger.info(
        f"Received generate request: prompt='{request.prompt[:50]}...', "
        f"top_k={request.top_k}, search_type={request.search_type}"
    )

    try:
        pipeline = _get_pipeline()
    except Exception as e:
        logger.error(f"Pipeline initialization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline initialization failed: {e}")

    # Retrieve relevant documents with filters and search type
    logger.debug(f"Retrieving top {request.top_k} documents using {request.search_type} search...")
    retrieved = pipeline.retrieve(
        request.prompt,
        k=request.top_k,
        search_type=request.search_type,
        filters=request.metadata_filters,
        hybrid_alpha=request.hybrid_alpha,
        enable_reranking=request.enable_reranking,
    )
    logger.info(f"Retrieved {len(retrieved)} documents")

    # Extract sources and build context
    sources = _extract_sources(retrieved)
    context = _build_context(sources)
    logger.debug(f"Built context from {len(sources)} sources")

    # Generate RAG response
    logger.debug("Generating response...")
    answer = pipeline.generate(request.prompt, retrieved_docs=retrieved)
    logger.info(
        f"Generated answer: '{answer[:100]}...'"
        if len(answer) > 100
        else f"Generated answer: '{answer}'"
    )

    return GenerateResponse(
        prompt=request.prompt,
        answer=answer,
        context=context,
        sources=sources,
    )


@router.post("/ingest")
async def ingest():
    """
    Trigger document ingestion and indexing.
    """
    logger.info("Starting document ingestion...")
    try:
        pipeline = _get_pipeline()
        count = pipeline.ingest_and_index()
        logger.info(f"Ingestion complete: {count} chunks indexed")
        return {"status": "success", "chunks_indexed": count}
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")
