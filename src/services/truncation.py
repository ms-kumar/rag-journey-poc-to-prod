"""
Text truncation utilities for managing token limits.

Provides strategies for truncating text to fit within model token budgets.
"""

from enum import Enum

from src.models.token_budgets import TokenBudget, get_embedding_budget, get_generation_budget


class TruncationStrategy(str, Enum):
    """Truncation strategy for texts exceeding token limits."""

    HEAD = "head"  # Keep beginning, truncate end
    TAIL = "tail"  # Keep end, truncate beginning
    MIDDLE = "middle"  # Keep beginning and end, truncate middle
    NONE = "none"  # No truncation, raise error if exceeded


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.

    Uses a simple heuristic: ~4 characters per token (conservative estimate).
    For production, consider using tiktoken for accurate counts.

    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    if not text:
        return 0
    # Conservative estimate: 4 chars per token
    # Most tokenizers average 3.5-4.5 chars/token
    return max(1, len(text) // 4)


def chars_from_tokens(token_count: int) -> int:
    """
    Estimate character count from token count.

    Args:
        token_count: Number of tokens

    Returns:
        Estimated character count
    """
    return token_count * 4


class TextTruncator:
    """Truncates text to fit within token budgets."""

    def __init__(
        self,
        max_tokens: int,
        strategy: TruncationStrategy = TruncationStrategy.HEAD,
        preserve_words: bool = True,
    ):
        """
        Initialize truncator.

        Args:
            max_tokens: Maximum tokens allowed
            strategy: Truncation strategy to use
            preserve_words: Whether to preserve word boundaries
        """
        self.max_tokens = max_tokens
        self.strategy = strategy
        self.preserve_words = preserve_words

    @classmethod
    def from_embedding_model(
        cls,
        model_name: str,
        strategy: TruncationStrategy = TruncationStrategy.HEAD,
        preserve_words: bool = True,
    ) -> "TextTruncator":
        """
        Create truncator from embedding model budget.

        Args:
            model_name: Name of embedding model
            strategy: Truncation strategy
            preserve_words: Whether to preserve word boundaries

        Returns:
            Configured TextTruncator
        """
        budget = get_embedding_budget(model_name)
        return cls(
            max_tokens=budget.max_input_tokens, strategy=strategy, preserve_words=preserve_words
        )

    @classmethod
    def from_generation_model(
        cls,
        model_name: str,
        strategy: TruncationStrategy = TruncationStrategy.HEAD,
        preserve_words: bool = True,
        reserve_output_tokens: int = 0,
    ) -> "TextTruncator":
        """
        Create truncator from generation model budget.

        Args:
            model_name: Name of generation model
            strategy: Truncation strategy
            preserve_words: Whether to preserve word boundaries
            reserve_output_tokens: Tokens to reserve for output

        Returns:
            Configured TextTruncator
        """
        budget = get_generation_budget(model_name)
        max_input = budget.max_input_tokens - reserve_output_tokens
        return cls(max_tokens=max_input, strategy=strategy, preserve_words=preserve_words)

    def truncate(self, text: str) -> str:
        """
        Truncate text according to strategy.

        Args:
            text: Text to truncate

        Returns:
            Truncated text

        Raises:
            ValueError: If strategy is NONE and text exceeds limit
        """
        if not text:
            return text

        token_count = estimate_tokens(text)

        # No truncation needed
        if token_count <= self.max_tokens:
            return text

        # Error if NONE strategy
        if self.strategy == TruncationStrategy.NONE:
            raise ValueError(
                f"Text exceeds token limit: {token_count} > {self.max_tokens} "
                f"(strategy=NONE, no truncation allowed)"
            )

        # Calculate max characters
        max_chars = chars_from_tokens(self.max_tokens)

        # Apply truncation strategy
        if self.strategy == TruncationStrategy.HEAD:
            return self._truncate_head(text, max_chars)
        if self.strategy == TruncationStrategy.TAIL:
            return self._truncate_tail(text, max_chars)
        if self.strategy == TruncationStrategy.MIDDLE:
            return self._truncate_middle(text, max_chars)

        return text

    def _truncate_head(self, text: str, max_chars: int) -> str:
        """Keep beginning, truncate end."""
        if len(text) <= max_chars:
            return text

        truncated = text[:max_chars]

        if self.preserve_words:
            # Find last complete word
            last_space = truncated.rfind(" ")
            if last_space > 0:
                truncated = truncated[:last_space]

        return truncated.rstrip() + "..."

    def _truncate_tail(self, text: str, max_chars: int) -> str:
        """Keep end, truncate beginning."""
        if len(text) <= max_chars:
            return text

        truncated = text[-max_chars:]

        if self.preserve_words:
            # Find first complete word
            first_space = truncated.find(" ")
            if first_space > 0:
                truncated = truncated[first_space + 1 :]

        return "..." + truncated.lstrip()

    def _truncate_middle(self, text: str, max_chars: int) -> str:
        """Keep beginning and end, truncate middle."""
        if len(text) <= max_chars:
            return text

        # Reserve 5 chars for " ... "
        separator = " ... "
        available = max_chars - len(separator)

        # Split available chars between head and tail
        head_chars = available // 2
        tail_chars = available - head_chars

        head = text[:head_chars]
        tail = text[-tail_chars:] if tail_chars > 0 else ""

        if self.preserve_words:
            # Adjust head to last complete word
            last_space = head.rfind(" ")
            if last_space > 0:
                head = head[:last_space]

            # Adjust tail to first complete word
            if tail:
                first_space = tail.find(" ")
                if first_space > 0:
                    tail = tail[first_space + 1 :]

        return head.rstrip() + separator + tail.lstrip()

    def truncate_batch(self, texts: list[str]) -> list[str]:
        """
        Truncate multiple texts.

        Args:
            texts: List of texts to truncate

        Returns:
            List of truncated texts
        """
        return [self.truncate(text) for text in texts]


def truncate_to_budget(
    text: str,
    budget: TokenBudget,
    strategy: TruncationStrategy = TruncationStrategy.HEAD,
    reserve_output_tokens: int = 0,
) -> str:
    """
    Truncate text to fit within a token budget.

    Args:
        text: Text to truncate
        budget: Token budget to enforce
        strategy: Truncation strategy
        reserve_output_tokens: Tokens to reserve for output (generation models)

    Returns:
        Truncated text
    """
    max_tokens = budget.max_input_tokens - reserve_output_tokens
    truncator = TextTruncator(max_tokens=max_tokens, strategy=strategy)
    return truncator.truncate(text)


def split_with_overlap(text: str, max_tokens: int, overlap_tokens: int = 0) -> list[str]:
    """
    Split text into chunks with optional overlap.

    Args:
        text: Text to split
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Tokens to overlap between chunks

    Returns:
        List of text chunks
    """
    if not text:
        return []

    max_chars = chars_from_tokens(max_tokens)
    overlap_chars = chars_from_tokens(overlap_tokens)
    step = max_chars - overlap_chars

    if step <= 0:
        raise ValueError("Overlap tokens must be less than max tokens")

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk = text[start:end]

        # Try to break at word boundary
        if end < len(text) and " " in chunk:
            last_space = chunk.rfind(" ")
            if last_space > len(chunk) // 2:  # Only if space is in latter half
                chunk = chunk[:last_space]
                end = start + last_space

        chunks.append(chunk.strip())
        start += step

    return chunks
