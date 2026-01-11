"""
Factory for creating query understanding clients from application settings.
"""

import logging
from typing import TYPE_CHECKING

from src.services.query_understanding.client import (
    QueryUnderstandingClient,
    QueryUnderstandingConfig,
)

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
    config = QueryUnderstandingConfig.from_settings(settings)
    query_settings = settings.query_understanding
    logger.info(
        f"Query understanding client created (rewriting={query_settings.enable_rewriting}, synonyms={query_settings.enable_synonyms})"
    )
    return QueryUnderstandingClient(config)


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
    enable_rewriting: bool = True,
    enable_synonyms: bool = True,
    enable_intent: bool = False,
) -> QueryUnderstandingClient:
    """
    Create a query understanding client (legacy function).

    Args:
        enable_rewriting: Whether to enable query rewriting
        enable_synonyms: Whether to enable synonym expansion
        enable_intent: Whether to enable intent classification

    Returns:
        Configured QueryUnderstandingClient instance

    Note:
        For new code, prefer using create_query_understanding(settings) instead.
    """
    config = QueryUnderstandingConfig(
        enable_rewriting=enable_rewriting,
        enable_synonyms=enable_synonyms,
        enable_intent_classification=enable_intent,
    )
    return QueryUnderstandingClient(config)
