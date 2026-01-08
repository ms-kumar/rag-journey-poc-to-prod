import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.v1.endpoints import health, rag
from src.config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.app.log_level.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info(f"Starting {settings.app.name} v{settings.app.version}")
    logger.info(f"Debug mode: {settings.app.debug}")
    logger.info(f"Log level: {settings.app.log_level}")
    yield
    # Shutdown (if needed)
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
