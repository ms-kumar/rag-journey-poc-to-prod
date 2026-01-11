"""API schemas for request and response models."""

from src.schemas.api.ask import AskRequest, AskResponse
from src.schemas.api.health import DetailedHealthResponse, HealthCheckResponse, ServiceStatus
from src.schemas.api.ingestion import IngestRequest, IngestResponse
from src.schemas.api.rag_request import GenerateRequest, GenerateResponse

__all__ = [
    # Ask endpoint schemas
    "AskRequest",
    "AskResponse",
    # Health check schemas
    "HealthCheckResponse",
    "DetailedHealthResponse",
    "ServiceStatus",
    # Ingestion schemas
    "IngestRequest",
    "IngestResponse",
    # RAG generation schemas
    "GenerateRequest",
    "GenerateResponse",
]
