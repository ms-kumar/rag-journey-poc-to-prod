from collections.abc import Sequence
from typing import Any

from transformers import Pipeline, pipeline

from src.services.retry import RetryConfig, retry_with_backoff
from src.services.truncation import TextTruncator

# GenerationConfig removed - now using GenerationSettings from config.py


class DependencyMissingError(RuntimeError):
    pass


class HFGenerator:
    """
    Baseline generator using Hugging Face text-generation pipeline.

    Methods:
    - generate(prompt, **overrides) -> str
    - generate_batch(prompts, **overrides) -> List[str]
    """

    def __init__(self, config, retry_config=None):
        if pipeline is None:
            raise DependencyMissingError("transformers is not installed")

        self.config = config
        self.retry_config = retry_config or RetryConfig(
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
            model=self.config.model,
            device=device,
        )

    def _merge_kwargs(self, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        # Use max_length as max_new_tokens (GenerationSettings uses max_length)
        max_tokens = getattr(self.config, "max_new_tokens", None) or self.config.max_length
        base = {
            "max_new_tokens": max_tokens,
            "do_sample": self.config.do_sample,
            "temperature": self.config.temperature,
            "top_k": self.config.top_k,
            "top_p": self.config.top_p,
            "num_return_sequences": self.config.num_return_sequences,
            "return_full_text": False,
            "truncation": True,
            "pad_token_id": self.pipe.tokenizer.eos_token_id,  # type: ignore[union-attr]
        }
        if hasattr(self.config, "extra_kwargs") and self.config.extra_kwargs:
            base.update(self.config.extra_kwargs)
        if overrides:
            base.update(overrides)
        return base

    def generate(self, prompt: str, overrides: dict[str, Any] | None = None) -> str:
        """
        Generate text for a single prompt and return the first generated sequence as a string.
        """  # Apply overflow guard - truncate prompt to model limits (reserve space for output)
        model_name = self.config.model
        max_tokens = getattr(self.config, "max_new_tokens", None) or self.config.max_length
        truncator = TextTruncator.from_generation_model(
            model_name, reserve_output_tokens=max_tokens
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
