"""
Factory for creating query understanding clients from application settings.
"""

import logging
from typing import TYPE_CHECKING

from src.services.query_understanding.client import QueryUnderstandingClient

if TYPE_CHECKING:
    from src.config import Settings

logger = logging.getLogger(__name__)


def make_query_understanding_client(
    settings: "Settings",
) -> QueryUnderstandingClient:
    """
    Create query understanding client from application settings.

    Args:
        settings: Full application settings object

    Returns:
        Configured QueryUnderstandingClient instance
    """
    query_settings = settings.query_understanding
    logger.info(
        f"Query understanding client created (rewriting={query_settings.enable_rewriting}, synonyms={query_settings.enable_synonyms})"
    )
    return QueryUnderstandingClient(settings)


def create_query_understanding(settings: "Settings") -> QueryUnderstandingClient:
    """
    Create query understanding client from application settings.

    Deprecated: Use make_query_understanding_client() instead.

    Args:
        settings: Full application settings object

    Returns:
        Configured QueryUnderstandingClient instance
    """
    return make_query_understanding_client(settings)


def get_query_understanding_client(
    settings: "Settings",
) -> QueryUnderstandingClient:
    """
    Create a query understanding client (legacy function).

    Args:
        settings: Application settings object

    Returns:
        Configured QueryUnderstandingClient instance

    Note:
        Deprecated: Use make_query_understanding_client(settings) instead.
    """
    return make_query_understanding_client(settings)
