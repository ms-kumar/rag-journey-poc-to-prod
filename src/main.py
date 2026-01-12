import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.router.agent_router import router as agent_router
from src.api.v1.endpoints import health, rag
from src.config import settings
from src.services.cache.factory import make_cache_client
from src.services.embeddings.factory import get_langchain_embeddings_adapter, make_embedding_client
from src.services.generation.factory import make_generation_client
from src.services.guardrails.factory import make_guardrails_client
from src.services.query_understanding.factory import make_query_understanding_client
from src.services.reranker.factory import make_reranker_client
from src.services.vectorstore.factory import make_vectorstore_client

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.app.log_level.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - initializes services once at startup."""
    # Startup
    logger.info(f"Starting {settings.app.name} v{settings.app.version}")
    logger.info(f"Debug mode: {settings.app.debug}")
    logger.info(f"Log level: {settings.app.log_level}")

    # Initialize settings
    app.state.settings = settings
    logger.info("Settings initialized")

    # Initialize core services and store in app.state
    try:
        logger.info("Initializing embedding service...")
        app.state.embedding_service = make_embedding_client(settings.embedding)
        logger.info("✓ Embedding service initialized")

        logger.info("Initializing cache service...")
        app.state.cache_service = make_cache_client(settings.cache)
        logger.info("✓ Cache service initialized")

        logger.info("Initializing vector store service...")
        embeddings = get_langchain_embeddings_adapter(settings.embedding)
        vector_size = settings.embedding.dim
        app.state.vector_store_service = make_vectorstore_client(
            settings.vectorstore,
            embeddings=embeddings,
            vector_size=vector_size,
        )
        logger.info("✓ Vector store service initialized")

        logger.info("Initializing reranker service...")
        app.state.reranker_service = make_reranker_client(settings.reranker)
        logger.info("✓ Reranker service initialized")

        logger.info("Initializing generation service...")
        app.state.generation_service = make_generation_client(settings.generation)
        logger.info("✓ Generation service initialized")

        logger.info("Initializing query understanding service...")
        app.state.query_understanding_service = make_query_understanding_client(settings)
        logger.info("✓ Query understanding service initialized")

        logger.info("Initializing guardrails service...")
        app.state.guardrails_service = make_guardrails_client(settings)
        logger.info("✓ Guardrails service initialized")

        logger.info("All services initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

    yield

    # Shutdown
    logger.info(f"Shutting down {settings.app.name}")


app = FastAPI(
    title=settings.app.name,
    version=settings.app.version,
    debug=settings.app.debug,
    lifespan=lifespan,
)

# Include routers
app.include_router(rag.router, prefix="/api/v1/rag", tags=["rag"])
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(agent_router, prefix="/api/v1", tags=["agent"])


@app.get("/")
def root():
    return {
        "status": "API is running",
        "app": settings.app.name,
        "version": settings.app.version,
        "health_check": "/api/v1/health",
        "detailed_health": "/api/v1/health/detailed",
    }


@app.get("/health")
def legacy_health():
    """Legacy health endpoint - redirects to new health API."""
    return {"status": "healthy", "note": "Use /api/v1/health for detailed checks"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.server.host, port=settings.server.port)

"""
curl -X POST http://localhost:8000/api/v1/rag/ingest

curl -X POST http://localhost:8000/api/v1/rag/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is RAG?", "top_k": 3}'

curl -X POST http://localhost:8000/api/v1/rag/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is machine learning?", "top_k": 3}'
"""
