"""
Tests for text truncation utilities.
"""

import pytest

from src.models.token_budgets import TokenBudget
from src.services.truncation import (
    TextTruncator,
    TruncationStrategy,
    chars_from_tokens,
    estimate_tokens,
    split_with_overlap,
    truncate_to_budget,
)


class TestTokenEstimation:
    """Test token estimation utilities."""

    def test_estimate_tokens_empty(self):
        assert estimate_tokens("") == 0

    def test_estimate_tokens_short(self):
        # "test" = 4 chars = 1 token (min)
        assert estimate_tokens("test") == 1

    def test_estimate_tokens_typical(self):
        # 39 chars = 9 tokens (39 // 4)
        text = "This is a test sentence with 40 chars!"
        assert estimate_tokens(text) == 9

    def test_estimate_tokens_long(self):
        # 400 chars = 100 tokens
        text = "x" * 400
        assert estimate_tokens(text) == 100

    def test_chars_from_tokens(self):
        assert chars_from_tokens(0) == 0
        assert chars_from_tokens(10) == 40
        assert chars_from_tokens(100) == 400


class TestTextTruncatorInit:
    """Test TextTruncator initialization."""

    def test_basic_init(self):
        truncator = TextTruncator(max_tokens=100)
        assert truncator.max_tokens == 100
        assert truncator.strategy == TruncationStrategy.HEAD
        assert truncator.preserve_words is True

    def test_custom_strategy(self):
        truncator = TextTruncator(max_tokens=100, strategy=TruncationStrategy.TAIL)
        assert truncator.strategy == TruncationStrategy.TAIL

    def test_from_embedding_model(self):
        truncator = TextTruncator.from_embedding_model("text-embedding-3-small")
        assert truncator.max_tokens == 8191

    def test_from_generation_model(self):
        truncator = TextTruncator.from_generation_model("gpt-3.5-turbo")
        assert truncator.max_tokens == 16385

    def test_from_generation_model_with_reserve(self):
        truncator = TextTruncator.from_generation_model("gpt2", reserve_output_tokens=512)
        assert truncator.max_tokens == 1024 - 512


class TestHeadTruncation:
    """Test HEAD truncation strategy."""

    def test_no_truncation_needed(self):
        truncator = TextTruncator(max_tokens=100, strategy=TruncationStrategy.HEAD)
        text = "Short text"
        assert truncator.truncate(text) == text

    def test_simple_truncation(self):
        truncator = TextTruncator(
            max_tokens=10, strategy=TruncationStrategy.HEAD, preserve_words=False
        )
        # 10 tokens = 40 chars
        text = "This is a very long text that needs to be truncated significantly"
        result = truncator.truncate(text)
        assert len(result) <= 43  # 40 chars + "..."
        assert result.endswith("...")

    def test_word_boundary_preservation(self):
        truncator = TextTruncator(
            max_tokens=10, strategy=TruncationStrategy.HEAD, preserve_words=True
        )
        text = "This is a very long text that needs truncation"
        result = truncator.truncate(text)
        assert result.endswith("...")
        # Should not break in middle of word
        assert "..." not in result[:-3]

    def test_empty_text(self):
        truncator = TextTruncator(max_tokens=10, strategy=TruncationStrategy.HEAD)
        assert truncator.truncate("") == ""


class TestTailTruncation:
    """Test TAIL truncation strategy."""

    def test_tail_truncation(self):
        truncator = TextTruncator(
            max_tokens=10, strategy=TruncationStrategy.TAIL, preserve_words=False
        )
        text = "This is a very long text that needs to be truncated significantly"
        result = truncator.truncate(text)
        assert result.startswith("...")
        assert len(result) <= 43  # 40 chars + "..."

    def test_tail_word_boundary(self):
        truncator = TextTruncator(
            max_tokens=10, strategy=TruncationStrategy.TAIL, preserve_words=True
        )
        text = "This is a very long text that needs truncation"
        result = truncator.truncate(text)
        assert result.startswith("...")
        # Check that we keep the end
        assert "truncation" in result


class TestMiddleTruncation:
    """Test MIDDLE truncation strategy."""

    def test_middle_truncation(self):
        truncator = TextTruncator(max_tokens=10, strategy=TruncationStrategy.MIDDLE)
        text = "This is a very long text that needs to be truncated significantly for testing"
        result = truncator.truncate(text)
        assert " ... " in result
        assert len(result) <= 50  # Some buffer for separator

    def test_middle_keeps_ends(self):
        truncator = TextTruncator(max_tokens=15, strategy=TruncationStrategy.MIDDLE)
        text = "Beginning of text with lots of content in the middle and ending here"
        result = truncator.truncate(text)
        assert "Beginning" in result
        assert "ending" in result or "here" in result
        assert " ... " in result


class TestNoneTruncation:
    """Test NONE truncation strategy (error on exceed)."""

    def test_none_no_truncation_needed(self):
        truncator = TextTruncator(max_tokens=100, strategy=TruncationStrategy.NONE)
        text = "Short text"
        assert truncator.truncate(text) == text

    def test_none_raises_on_exceed(self):
        truncator = TextTruncator(max_tokens=10, strategy=TruncationStrategy.NONE)
        text = "This is a very long text that exceeds the limit and should raise an error"
        with pytest.raises(ValueError, match="exceeds token limit"):
            truncator.truncate(text)


class TestBatchTruncation:
    """Test batch truncation."""

    def test_truncate_batch(self):
        truncator = TextTruncator(max_tokens=10, strategy=TruncationStrategy.HEAD)
        texts = [
            "Short",
            "This is a very long text that needs truncation",
            "Another long text that should be truncated as well",
        ]
        results = truncator.truncate_batch(texts)
        assert len(results) == 3
        assert results[0] == "Short"
        assert results[1].endswith("...")
        assert results[2].endswith("...")

    def test_truncate_batch_empty(self):
        truncator = TextTruncator(max_tokens=10)
        assert truncator.truncate_batch([]) == []


class TestTruncateToBudget:
    """Test truncate_to_budget utility."""

    def test_truncate_to_budget(self):
        budget = TokenBudget(max_input_tokens=10, max_output_tokens=0, max_context_window=10)
        text = "This is a very long text that needs to be truncated"
        result = truncate_to_budget(text, budget)
        assert len(result) <= 43  # 40 chars + "..."

    def test_truncate_with_reserve(self):
        budget = TokenBudget(max_input_tokens=20, max_output_tokens=10, max_context_window=30)
        text = "This is a moderately long text that might need truncation"
        result = truncate_to_budget(text, budget, reserve_output_tokens=10)
        # Should use 10 tokens (20 - 10 reserved)
        assert estimate_tokens(result) <= 10

    def test_truncate_different_strategies(self):
        budget = TokenBudget(max_input_tokens=10, max_output_tokens=0, max_context_window=10)
        text = "Beginning text with lots of content in the middle section and more stuff and ending text here"

        head = truncate_to_budget(text, budget, strategy=TruncationStrategy.HEAD)
        assert head.startswith("Beginning")
        assert head.endswith("...")

        tail = truncate_to_budget(text, budget, strategy=TruncationStrategy.TAIL)
        assert tail.startswith("...")
        assert "ending" in tail or "here" in tail

        middle = truncate_to_budget(text, budget, strategy=TruncationStrategy.MIDDLE)
        assert " ... " in middle


class TestSplitWithOverlap:
    """Test split_with_overlap utility."""

    def test_split_no_overlap(self):
        text = "x" * 400  # 100 tokens
        chunks = split_with_overlap(text, max_tokens=25, overlap_tokens=0)
        assert len(chunks) == 4
        for chunk in chunks:
            assert estimate_tokens(chunk) <= 25

    def test_split_with_overlap(self):
        text = "word " * 200  # ~200 tokens
        chunks = split_with_overlap(text, max_tokens=50, overlap_tokens=10)
        assert len(chunks) > 2
        # Chunks should overlap
        for i in range(len(chunks) - 1):
            # Some content from chunk[i] should appear in chunk[i+1]
            assert estimate_tokens(chunks[i]) <= 50

    def test_split_empty_text(self):
        assert split_with_overlap("", max_tokens=10) == []

    def test_split_short_text(self):
        text = "short"
        chunks = split_with_overlap(text, max_tokens=10)
        assert len(chunks) == 1
        assert chunks[0] == "short"

    def test_split_invalid_overlap(self):
        with pytest.raises(ValueError, match="Overlap tokens must be less"):
            split_with_overlap("text", max_tokens=10, overlap_tokens=10)

    def test_split_word_boundaries(self):
        text = "The quick brown fox jumps over the lazy dog multiple times for testing"
        chunks = split_with_overlap(text, max_tokens=10, overlap_tokens=2)
        # Each chunk should end at a word boundary (if possible)
        for chunk in chunks:
            # No broken words (heuristic: shouldn't end with lowercase followed by uppercase)
            if len(chunk) > 1:
                assert not (chunk[-2].islower() and chunk[-1].isupper())


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_character_text(self):
        truncator = TextTruncator(max_tokens=1)
        assert truncator.truncate("a") == "a"

    def test_unicode_text(self):
        truncator = TextTruncator(max_tokens=10)
        text = "Hello 世界 🌍 " * 20
        result = truncator.truncate(text)
        assert len(result) <= 50

    def test_whitespace_only(self):
        truncator = TextTruncator(max_tokens=10)
        text = "   " * 50
        result = truncator.truncate(text)
        # Whitespace gets truncated - either stays same or becomes shorter
        assert len(result) <= len(text)

    def test_no_spaces(self):
        truncator = TextTruncator(max_tokens=10, preserve_words=True)
        text = "abcdefghijklmnopqrstuvwxyz" * 10
        result = truncator.truncate(text)
        # Should still truncate even without spaces
        assert len(result) < len(text)


class TestIntegration:
    """Integration tests with token budgets."""

    def test_with_openai_embedding(self):
        truncator = TextTruncator.from_embedding_model("text-embedding-3-small")
        # Should handle up to 8191 tokens
        text = "word " * 10000  # ~10000 tokens
        result = truncator.truncate(text)
        assert estimate_tokens(result) <= 8191

    def test_with_gpt4(self):
        truncator = TextTruncator.from_generation_model("gpt-4-turbo", reserve_output_tokens=4096)
        # Should handle 128000 - 4096 = 123904 tokens
        text = "x" * 1000000
        result = truncator.truncate(text)
        assert estimate_tokens(result) <= 123904

    def test_with_local_model(self):
        truncator = TextTruncator.from_embedding_model("intfloat/e5-small-v2")
        # Should handle 512 tokens
        text = "word " * 1000
        result = truncator.truncate(text)
        assert estimate_tokens(result) <= 512
