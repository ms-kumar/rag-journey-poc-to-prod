from typing import TYPE_CHECKING, Any

from .client import GenerationConfig, HFGenerator

if TYPE_CHECKING:
    from src.config import Settings


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

    return HFGenerator(config=config)


def create_from_settings(settings: "Settings", **overrides):
    """Create generator from application settings."""
    config = GenerationConfig.from_settings(settings, **overrides)
    return HFGenerator(config=config)
