"""
Factory for creating generation clients from application settings.
"""

import logging
from typing import TYPE_CHECKING, Any

from src.services.generation.client import GenerationConfig, HFGenerator

if TYPE_CHECKING:
    from src.config import GenerationSettings

logger = logging.getLogger(__name__)


def get_generator(
    model_name: str = "gpt2",
    device: int | None = None,
    max_new_tokens: int = 128,
    do_sample: bool = True,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    num_return_sequences: int = 1,
    extra_kwargs: dict[str, Any] | None = None,
):
    """
    Factory: returns an `HFGenerator` when `transformers` is installed, otherwise a `DummyGenerator`.
    """
    config = GenerationConfig(
        model_name=model_name,
        device=device,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=num_return_sequences,
        extra_kwargs=extra_kwargs,
    )

    logger.info(f"Created HFGenerator with model={model_name}")
    return HFGenerator(config=config)


def make_generation_client(settings: "GenerationSettings") -> HFGenerator:
    """
    Create generation client from application settings.

    Args:
        settings: Generation settings

    Returns:
        Configured HFGenerator instance
    """
    config = GenerationConfig(
        model_name=settings.model,
        device=settings.device,
        max_new_tokens=settings.max_length,
        do_sample=True,
        temperature=settings.temperature,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1,
        extra_kwargs=None,
    )
    logger.info(f"Generation client created with model={settings.model}")
    return HFGenerator(config=config)


def create_from_settings(settings: "GenerationSettings", **overrides):
    """
    Create generator from application settings with optional overrides.

    Deprecated: Use make_generation_client() instead.
    """
    config = GenerationConfig.from_settings(settings, **overrides)
    return HFGenerator(config=config)
