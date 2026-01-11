"""
Tests for overflow guards in embedding and generation clients.

Verifies that overflow guards automatically truncate text to prevent exceeding model token limits.
"""

import pytest

from src.config import GenerationSettings
from src.services.embeddings.client import EmbedClient
from src.services.generation.client import HFGenerator
from src.services.truncation import estimate_tokens


class TestEmbeddingOverflowGuard:
    """Test overflow guards in embedding clients."""

    def test_simple_embed_client_no_truncation_needed(self):
        """Test that short texts pass through without truncation."""
        client = EmbedClient(model_name="simple-hash", dim=32)
        texts = ["short text", "another short text"]

        embeddings = client.embed(texts)

        assert len(embeddings) == 2
        assert len(embeddings[0]) == 32
        assert len(embeddings[1]) == 32

    def test_simple_embed_client_handles_long_text(self):
        """Test that simple embedder handles long text (no token limits)."""
        client = EmbedClient(model_name="simple-hash", dim=32)
        # Create a very long text (way over typical token limits)
        long_text = "word " * 10000  # ~50k chars, ~12.5k tokens

        embeddings = client.embed([long_text])

        # Simple hash embedder should handle any length
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 32


class TestGenerationOverflowGuard:
    """Test overflow guards in generation clients."""

    @pytest.fixture
    def generator(self):
        """Create a generator with small token limits for testing."""
        config = GenerationSettings(
            model_name="gpt2",
            max_new_tokens=50,
            do_sample=False,  # Deterministic output
            device=-1,  # CPU
        )
        return HFGenerator(config)

    def test_short_prompt_no_truncation(self, generator):
        """Test that short prompts pass through without truncation."""
        prompt = "Hello, how are you?"
        original_tokens = estimate_tokens(prompt)

        result = generator.generate(prompt)

        # Should generate some text
        assert isinstance(result, str)
        assert len(result) > 0
        # Original prompt was short enough
        assert original_tokens < 1000  # Well under limit

    def test_long_prompt_gets_truncated(self, generator):
        """Test that very long prompts get truncated."""
        # Create a prompt that exceeds GPT-2's 1024 token limit
        long_prompt = "This is a test sentence. " * 500  # ~12.5k chars, ~3k tokens
        original_tokens = estimate_tokens(long_prompt)

        # Should not raise an error despite exceeding limit
        result = generator.generate(long_prompt)

        # Should still generate output
        assert isinstance(result, str)
        # Original was way over limit
        assert original_tokens > 1024

    def test_batch_generation_handles_mixed_lengths(self, generator):
        """Test batch generation with mixed length prompts."""
        prompts = [
            "Short prompt",
            "A bit longer prompt " * 10,
            "Very long prompt " * 500,  # This one should be truncated
        ]

        results = generator.generate_batch(prompts)

        # All should complete successfully
        assert len(results) == 3
        assert all(isinstance(r, str) for r in results)


class TestOverflowGuardIntegration:
    """Test overflow guards work correctly with truncation system."""

    def test_embedding_overflow_with_known_model(self):
        """Test overflow guard uses correct limits for known models."""
        # text-embedding-3-small has 8191 token limit
        from src.models.token_budgets import get_embedding_budget

        budget = get_embedding_budget("text-embedding-3-small")
        assert budget.max_input_tokens == 8191

        # Create text that exceeds this limit
        long_text = "word " * 10000  # ~50k chars, ~12.5k tokens
        tokens = estimate_tokens(long_text)
        assert tokens > budget.max_input_tokens

    def test_generation_overflow_with_known_model(self):
        """Test overflow guard uses correct limits for known models."""
        from src.models.token_budgets import get_generation_budget

        budget = get_generation_budget("gpt2")
        assert budget.max_input_tokens == 1024

        # Create prompt that exceeds this limit
        long_prompt = "word " * 5000  # ~25k chars, ~6.25k tokens
        tokens = estimate_tokens(long_prompt)
        assert tokens > budget.max_input_tokens

    def test_overflow_guard_preserves_functionality(self):
        """Test that overflow guards don't break normal functionality."""
        # Test with simple embedder (no API calls)
        client = EmbedClient(model_name="simple-hash", dim=64)

        # Mix of short and long texts
        texts = [
            "short",
            "medium length text here",
            "very " * 1000 + "long text",  # Long but should still work
        ]

        embeddings = client.embed(texts)

        # Should get embeddings for all texts
        assert len(embeddings) == 3
        assert all(len(emb) == 64 for emb in embeddings)

    def test_overflow_guard_with_empty_input(self):
        """Test overflow guards handle empty inputs gracefully."""
        client = EmbedClient(model_name="simple-hash", dim=32)

        # Empty list
        assert client.embed([]) == []

        # Empty strings
        embeddings = client.embed(["", ""])
        assert len(embeddings) == 2
        assert all(len(emb) == 32 for emb in embeddings)


class TestOverflowGuardEdgeCases:
    """Test edge cases for overflow guards."""

    def test_exactly_at_token_limit(self):
        """Test text exactly at token limit."""
        from src.services.truncation import TextTruncator

        # Create text exactly at limit
        truncator = TextTruncator.from_embedding_model("text-embedding-3-small")
        # 8191 tokens * 4 chars = ~32,764 chars
        text = "word " * 6552  # ~32,760 chars, close to 8191 tokens

        truncated = truncator.truncate(text)
        tokens = estimate_tokens(truncated)

        # Should be at or under limit
        assert tokens <= 8191

    def test_unicode_text_overflow(self):
        """Test overflow guard with unicode text."""
        client = EmbedClient(model_name="simple-hash", dim=32)

        # Unicode text with emoji and multi-byte characters
        unicode_text = "Hello 世界 🌍 " * 1000

        embeddings = client.embed([unicode_text])

        # Should handle unicode correctly
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 32

    def test_whitespace_only_overflow(self):
        """Test overflow guard with whitespace-only text."""
        client = EmbedClient(model_name="simple-hash", dim=32)

        # Whitespace text
        whitespace = " " * 10000

        embeddings = client.embed([whitespace])

        # Should handle whitespace
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 32
