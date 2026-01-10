from typing import TYPE_CHECKING

from .client import IngestionClient

if TYPE_CHECKING:
    from src.config import Settings


def get_ingestion_client(source_type: str = "local", **kwargs) -> IngestionClient:
    if source_type == "local":
        return IngestionClient(**kwargs)
    raise ValueError(f"Unknown source_type: {source_type}")


def create_from_settings(settings: "Settings", **overrides) -> IngestionClient:
    """Create ingestion client from application settings."""
    ingestion_settings = settings.ingestion
    directory = overrides.get("directory", ingestion_settings.dir)
    return IngestionClient(directory=directory)
