"""
Factory for creating reranker clients from application settings.
"""

import logging
from typing import TYPE_CHECKING, Any

from src.services.reranker.client import CrossEncoderReranker, RerankerConfig

if TYPE_CHECKING:
    from src.config import RerankerSettings

logger = logging.getLogger(__name__)


def make_reranker_client(settings: "RerankerSettings") -> CrossEncoderReranker:
    """
    Create reranker client from application settings.

    Args:
        settings: Reranker settings

    Returns:
        Configured CrossEncoderReranker instance
    """
    config = RerankerConfig.from_settings(settings)
    logger.info(f"Reranker client created with model={settings.model_name}")
    return CrossEncoderReranker(config)


def create_reranker(settings: "RerankerSettings") -> CrossEncoderReranker:
    """
    Create reranker from application settings.

    Deprecated: Use make_reranker_client() instead.

    Args:
        settings: Reranker settings

    Returns:
        Configured CrossEncoderReranker instance
    """
    return make_reranker_client(settings)


def get_reranker(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    batch_size: int = 32,
    timeout_seconds: float = 30.0,
    device: str | None = None,
    fallback_enabled: bool = True,
    **kwargs: Any,
) -> CrossEncoderReranker:
    """
    Create a cross-encoder reranker client (legacy function).

    Args:
        model_name: HuggingFace cross-encoder model name
        batch_size: Batch size for scoring
        timeout_seconds: Timeout for scoring operations
        device: Device to run model on (auto-detected if None)
        fallback_enabled: Whether to enable fallback strategies
        **kwargs: Additional configuration options

    Returns:
        Configured CrossEncoderReranker instance

    Note:
        For new code, prefer using create_reranker(settings) instead.
    """
    config = RerankerConfig(
        model_name=model_name,
        batch_size=batch_size,
        timeout_seconds=timeout_seconds,
        device=device,
        fallback_enabled=fallback_enabled,
        **kwargs,
    )

    return CrossEncoderReranker(config)
