from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

try:
    from transformers import pipeline, Pipeline
    import transformers
except Exception:
    pipeline = None  # type: ignore
    Pipeline = None  # type: ignore


@dataclass
class GenerationConfig:
    model_name: str = "gpt2"
    device: Optional[int] = None  # -1 for CPU, 0..N for GPU device id
    max_length: int = 128
    do_sample: bool = True
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    num_return_sequences: int = 1
    # Any additional kwargs passed through to pipeline call
    extra_kwargs: Dict[str, Any] = None


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
        device = config.device if config.device is not None else -1

        # Create pipeline for text-generation. Use return_full_text=False to get only generated continuation.
        self.pipe: Pipeline = pipeline(
            "text-generation",
            model=self.config.model_name,
            device=device,
        )

    def _merge_kwargs(self, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        base = {
            "max_length": self.config.max_length,
            "do_sample": self.config.do_sample,
            "temperature": self.config.temperature,
            "top_k": self.config.top_k,
            "top_p": self.config.top_p,
            "num_return_sequences": self.config.num_return_sequences,
            "return_full_text": False,
        }
        if self.config.extra_kwargs:
            base.update(self.config.extra_kwargs)
        if overrides:
            base.update(overrides)
        return base

    def generate(self, prompt: str, overrides: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate text for a single prompt and return the first generated sequence as a string.
        """
        kwargs = self._merge_kwargs(overrides)
        # call pipeline for the single prompt
        out = self.pipe(prompt, **kwargs)
        # pipeline returns a list of generated sequences (dicts) for the prompt
        # handle common response shapes
        if not out:
            return ""
        # if num_return_sequences > 1, pipeline returns list of dicts
        first = out[0]
        if isinstance(first, dict) and "generated_text" in first:
            return first["generated_text"]
        # fallback: convert to string
        return str(first)

    def generate_batch(self, prompts: Sequence[str], overrides: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Generate texts for a batch of prompts. For baseline simplicity we call the pipeline per prompt.
        """
        results: List[str] = []
        for p in prompts:
            results.append(self.generate(p, overrides=overrides))
        return results


class DummyGenerator:
    """
    Simple fallback generator when `transformers` is not available.
    Produces deterministic placeholder outputs for faster dev/test cycles.
    """

    def __init__(self, config: Optional[GenerationConfig] = None):
        self.config = config or GenerationConfig()

    def generate(self, prompt: str, overrides: Optional[Dict[str, Any]] = None) -> str:
        suffix = " ... [generated]"
        return f"{prompt}{suffix}"

    def generate_batch(self, prompts: Sequence[str], overrides: Optional[Dict[str, Any]] = None) -> List[str]:
        return [self.generate(p, overrides=overrides) for p in prompts]
