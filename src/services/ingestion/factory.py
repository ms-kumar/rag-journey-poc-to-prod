"""
Factory for creating ingestion clients from application settings.
"""

import logging
from typing import TYPE_CHECKING

from src.services.ingestion.client import IngestionClient

if TYPE_CHECKING:
    from src.config import IngestionSettings

logger = logging.getLogger(__name__)


def get_ingestion_client(source_type: str = "local", **kwargs) -> IngestionClient:
    """Create ingestion client with specified source type."""
    if source_type == "local":
        logger.info(f"Created IngestionClient with directory={kwargs.get('directory', 'data')}")
        return IngestionClient(**kwargs)
    raise ValueError(f"Unknown source_type: {source_type}")


def make_ingestion_client(settings: "IngestionSettings") -> IngestionClient:
    """
    Create ingestion client from application settings.

    Args:
        settings: Ingestion settings

    Returns:
        Configured ingestion client
    """
    return IngestionClient(directory=settings.dir)


def create_from_settings(settings: "IngestionSettings", **overrides) -> IngestionClient:
    """
    Create ingestion client from application settings with optional overrides.

    Deprecated: Use make_ingestion_client() instead.
    """
    directory = overrides.get("directory", settings.dir)
    return IngestionClient(directory=directory)
