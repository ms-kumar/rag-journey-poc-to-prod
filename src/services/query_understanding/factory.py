"""
Factory for creating query understanding clients from application settings.
"""

import logging
from typing import TYPE_CHECKING

from src.services.query_understanding.client import (
    QueryUnderstanding,
    QueryUnderstandingConfig,
)

if TYPE_CHECKING:
    from src.config import QueryUnderstandingSettings

logger = logging.getLogger(__name__)


def make_query_understanding_client(
    settings: "QueryUnderstandingSettings",
) -> QueryUnderstanding:
    """
    Create query understanding client from application settings.

    Args:
        settings: Query understanding settings

    Returns:
        Configured QueryUnderstanding instance
    """
    config = QueryUnderstandingConfig.from_settings(settings)
    logger.info(
        f"Query understanding client created (rewriting={settings.enable_rewriting}, synonyms={settings.enable_synonyms})"
    )
    return QueryUnderstanding(config)


def create_query_understanding(settings: "QueryUnderstandingSettings") -> QueryUnderstanding:
    """
    Create query understanding client from application settings.

    Deprecated: Use make_query_understanding_client() instead.

    Args:
        settings: Query understanding settings

    Returns:
        Configured QueryUnderstanding instance
    """
    return make_query_understanding_client(settings)


def get_query_understanding_client(
    enable_rewriting: bool = True,
    enable_synonyms: bool = True,
    enable_intent: bool = False,
) -> QueryUnderstanding:
    """
    Create a query understanding client (legacy function).

    Args:
        enable_rewriting: Whether to enable query rewriting
        enable_synonyms: Whether to enable synonym expansion
        enable_intent: Whether to enable intent classification

    Returns:
        Configured QueryUnderstanding instance

    Note:
        For new code, prefer using create_query_understanding(settings) instead.
    """
    config = QueryUnderstandingConfig(
        enable_rewriting=enable_rewriting,
        enable_synonyms=enable_synonyms,
        enable_intent_classification=enable_intent,
    )
    return QueryUnderstanding(config)
