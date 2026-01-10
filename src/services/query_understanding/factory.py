"""
Factory for creating query understanding clients.
"""

from typing import TYPE_CHECKING

from .client import QueryUnderstanding, QueryUnderstandingConfig

if TYPE_CHECKING:
    from src.config import Settings


def create_query_understanding(settings: "Settings") -> QueryUnderstanding:
    """
    Create query understanding client from application settings.

    Args:
        settings: Application settings

    Returns:
        Configured QueryUnderstanding instance
    """
    config = QueryUnderstandingConfig.from_settings(settings)
    return QueryUnderstanding(config)


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
