from typing import Any, Dict, List, Optional, Sequence
from .client import HFGenerator, DummyGenerator, GenerationConfig

try:
    from transformers import pipeline
except Exception:
    pipeline = None  # type: ignore


def get_generator(
    model_name: str = "gpt2",
    device: Optional[int] = None,
    max_length: int = 128,
    do_sample: bool = True,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    num_return_sequences: int = 1,
    extra_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    Factory: returns an `HFGenerator` when `transformers` is installed, otherwise a `DummyGenerator`.
    """
    config = GenerationConfig(
        model_name=model_name,
        device=device,
        max_length=max_length,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=num_return_sequences,
        extra_kwargs=extra_kwargs,
    )

    if pipeline is None:
        return DummyGenerator(config=config)
    return HFGenerator(config=config)