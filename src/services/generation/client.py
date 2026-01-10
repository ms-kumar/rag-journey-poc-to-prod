from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from transformers import Pipeline, pipeline

from src.services.retry import RetryConfig, retry_with_backoff
from src.services.truncation import TextTruncator


@dataclass
class GenerationConfig:
    model_name: str = "gpt2"
    device: int | None = None  # -1 for CPU, 0..N for GPU device id
    max_new_tokens: int = 128  # Use max_new_tokens instead of max_length
    do_sample: bool = True
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    num_return_sequences: int = 1
    # Any additional kwargs passed through to pipeline call
    extra_kwargs: dict[str, Any] | None = None
    # Retry configuration
    retry_config: RetryConfig | None = None

    @classmethod
    def from_settings(cls, settings, **kwargs):
        """Create config from application settings."""
        gen_settings = settings.generation
        return cls(
            model_name=gen_settings.model,
            device=gen_settings.device,
            max_new_tokens=gen_settings.max_length,
            temperature=gen_settings.temperature,
            **kwargs,
        )


class DependencyMissingError(RuntimeError):
    pass


class HFGenerator:
    """
    Baseline generator using Hugging Face text-generation pipeline.

    Methods:
    - generate(prompt, **overrides) -> str
    - generate_batch(prompts, **overrides) -> List[str]
    """

    def __init__(self, config: GenerationConfig):
        if pipeline is None:
            raise DependencyMissingError("transformers is not installed")

        self.config = config
        self.retry_config = config.retry_config or RetryConfig(
            max_retries=2,
            initial_delay=0.5,
            max_delay=10.0,
            exponential_base=2.0,
            jitter=True,
        )
        device = config.device if config.device is not None else -1

        # Create pipeline for text-generation. Use return_full_text=False to get only generated continuation.
        self.pipe: Pipeline = pipeline(
            "text-generation",
            model=self.config.model_name,
            device=device,
        )

    def _merge_kwargs(self, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        base = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
            "temperature": self.config.temperature,
            "top_k": self.config.top_k,
            "top_p": self.config.top_p,
            "num_return_sequences": self.config.num_return_sequences,
            "return_full_text": False,
            "truncation": True,
            "pad_token_id": self.pipe.tokenizer.eos_token_id,  # type: ignore[union-attr]
        }
        if self.config.extra_kwargs:
            base.update(self.config.extra_kwargs)
        if overrides:
            base.update(overrides)
        return base

    def generate(self, prompt: str, overrides: dict[str, Any] | None = None) -> str:
        """
        Generate text for a single prompt and return the first generated sequence as a string.
        """  # Apply overflow guard - truncate prompt to model limits (reserve space for output)
        truncator = TextTruncator.from_generation_model(
            self.config.model_name, reserve_output_tokens=self.config.max_new_tokens
        )
        prompt = truncator.truncate(prompt)
        kwargs = self._merge_kwargs(overrides)

        # call pipeline for the single prompt with retry
        @retry_with_backoff(self.retry_config)
        def _generate():
            return self.pipe(prompt, **kwargs)

        out = _generate()

        # pipeline returns a list of generated sequences (dicts) for the prompt
        # handle common response shapes
        if not out:
            return ""
        # if num_return_sequences > 1, pipeline returns list of dicts
        first = out[0]
        if isinstance(first, dict) and "generated_text" in first:
            return first["generated_text"]  # type: ignore[no-any-return]
        # fallback: convert to string
        return str(first)

    def generate_batch(
        self, prompts: Sequence[str], overrides: dict[str, Any] | None = None
    ) -> list[str]:
        """
        Generate texts for a batch of prompts. For baseline simplicity we call the pipeline per prompt.
        """
        results: list[str] = []
        for p in prompts:
            results.append(self.generate(p, overrides=overrides))
        return results
