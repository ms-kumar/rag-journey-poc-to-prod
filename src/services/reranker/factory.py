"""
Factory for creating reranker clients.
"""

from typing import TYPE_CHECKING, Any

from .client import CrossEncoderReranker, RerankerConfig

if TYPE_CHECKING:
    from src.config import Settings


def create_reranker(settings: "Settings") -> CrossEncoderReranker:
    """
    Create reranker from application settings.

    Args:
        settings: Application settings

    Returns:
        Configured CrossEncoderReranker instance
    """
    config = RerankerConfig.from_settings(settings)
    return CrossEncoderReranker(config)


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
