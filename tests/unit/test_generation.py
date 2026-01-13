"""
Unit tests for generation service.
"""

import pytest

from src.config import GenerationSettings
from src.services.generation.client import DependencyMissingError, HFGenerator


class TestGenerationConfig:
    """Test GenerationSettings dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = GenerationSettings()

        assert config.model == "gpt2"
        assert config.device is None
        assert config.max_length == 128
        assert config.do_sample is True
        assert config.temperature == 1.0
        assert config.top_k == 50
        assert config.top_p == 0.95
        assert config.num_return_sequences == 1

    def test_custom_config(self):
        """Test custom configuration."""
        config = GenerationSettings(
            model="gpt2-medium", device=0, max_length=256, temperature=0.7, top_k=100
        )

        assert config.model == "gpt2-medium"
        assert config.device == 0
        assert config.max_length == 256
        assert config.temperature == 0.7
        assert config.top_k == 100

    def test_cpu_device_config(self):
        """Test CPU device configuration."""
        config = GenerationSettings(device=-1)
        assert config.device == -1

    def test_gpu_device_config(self):
        """Test GPU device configuration."""
        config = GenerationSettings(device=0)
        assert config.device == 0

    def test_zero_temperature(self):
        """Test zero temperature (greedy decoding)."""
        config = GenerationSettings(temperature=0.0, do_sample=False)
        assert config.temperature == 0.0
        assert config.do_sample is False


class TestHFGenerator:
    """Test HFGenerator (integration tests require transformers)."""

    def test_initialization_with_valid_config(self):
        """Test generator initialization with valid config."""
        config = GenerationSettings(model="gpt2", device=-1)

        try:
            generator = HFGenerator(config)
            assert generator.config == config
            assert generator.pipe is not None
        except DependencyMissingError:
            pytest.skip("transformers not installed")

    def test_initialization_with_custom_model(self):
        """Test initialization with custom model name."""
        config = GenerationSettings(model="gpt2")

        try:
            generator = HFGenerator(config)
            assert generator.config.model == "gpt2"
        except DependencyMissingError:
            pytest.skip("transformers not installed")

    @pytest.mark.slow
    def test_generate_simple_prompt(self):
        """Test generating text from simple prompt."""
        config = GenerationSettings(model="gpt2", device=-1, max_length=20, do_sample=True)

        try:
            generator = HFGenerator(config)
            result = generator.generate("Hello world")

            assert isinstance(result, str)
            assert len(result) > 0
        except DependencyMissingError:
            pytest.skip("transformers not installed")

    @pytest.mark.slow
    def test_generate_with_overrides(self):
        """Test generation with parameter overrides."""
        config = GenerationSettings(model="gpt2", device=-1)

        try:
            generator = HFGenerator(config)
            result = generator.generate(
                "Test prompt", overrides={"max_new_tokens": 10, "temperature": 0.5}
            )

            assert isinstance(result, str)
        except DependencyMissingError:
            pytest.skip("transformers not installed")

    @pytest.mark.slow
    def test_generate_batch(self):
        """Test batch generation."""
        config = GenerationSettings(model="gpt2", device=-1, max_length=10)

        try:
            generator = HFGenerator(config)
            prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
            results = generator.generate_batch(prompts)

            assert len(results) == 3
            assert all(isinstance(r, str) for r in results)
        except DependencyMissingError:
            pytest.skip("transformers not installed")

    @pytest.mark.slow
    def test_generate_empty_prompt(self):
        """Test generation with empty prompt."""
        config = GenerationSettings(model="gpt2", device=-1, max_length=10)

        try:
            generator = HFGenerator(config)
            result = generator.generate("")

            assert isinstance(result, str)
        except DependencyMissingError:
            pytest.skip("transformers not installed")

    @pytest.mark.slow
    def test_generate_unicode_prompt(self):
        """Test generation with unicode prompt."""
        config = GenerationSettings(model="gpt2", device=-1, max_length=10)

        try:
            generator = HFGenerator(config)
            result = generator.generate("Hello 世界")

            assert isinstance(result, str)
        except DependencyMissingError:
            pytest.skip("transformers not installed")

    def test_merge_kwargs_default(self):
        """Test merging kwargs with defaults."""
        config = GenerationSettings(max_length=100, temperature=0.8, top_k=40)

        try:
            generator = HFGenerator(config)
            kwargs = generator._merge_kwargs()

            assert kwargs["max_new_tokens"] == 100
            assert kwargs["temperature"] == 0.8
            assert kwargs["top_k"] == 40
            assert kwargs["return_full_text"] is False
        except DependencyMissingError:
            pytest.skip("transformers not installed")

    def test_merge_kwargs_with_overrides(self):
        """Test merging kwargs with overrides."""
        config = GenerationSettings(max_length=100)

        try:
            generator = HFGenerator(config)
            overrides = {"max_new_tokens": 50, "temperature": 0.5}
            kwargs = generator._merge_kwargs(overrides)

            assert kwargs["max_new_tokens"] == 50
            assert kwargs["temperature"] == 0.5
        except DependencyMissingError:
            pytest.skip("transformers not installed")

    @pytest.mark.slow
    def test_generate_deterministic(self):
        """Test deterministic generation with do_sample=False."""
        config = GenerationSettings(model="gpt2", device=-1, max_length=10, do_sample=False)

        try:
            generator = HFGenerator(config)
            prompt = "The quick brown fox"

            result1 = generator.generate(prompt)
            result2 = generator.generate(prompt)

            # Should be identical with do_sample=False
            assert result1 == result2
        except DependencyMissingError:
            pytest.skip("transformers not installed")

    @pytest.mark.slow
    def test_generate_batch_empty_list(self):
        """Test batch generation with empty list."""
        config = GenerationSettings(model="gpt2", device=-1)

        try:
            generator = HFGenerator(config)
            results = generator.generate_batch([])

            assert results == []
        except DependencyMissingError:
            pytest.skip("transformers not installed")

    @pytest.mark.slow
    def test_generate_max_new_tokens_limit(self):
        """Test generation respects max_new_tokens."""
        config = GenerationSettings(model="gpt2", device=-1, max_length=5, do_sample=False)

        try:
            generator = HFGenerator(config)
            result = generator.generate("Once upon a time")

            # Result should be relatively short due to max_length=5
            assert isinstance(result, str)
            # Count tokens roughly (words as proxy)
            assert len(result.split()) <= 10  # Conservative check
        except DependencyMissingError:
            pytest.skip("transformers not installed")
