"""
Token budget definitions for different models.

Defines context windows, max tokens, and cost estimates for various
embedding and generation models.
"""

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class TokenBudget:
    """Token budget configuration for a model."""

    max_input_tokens: int
    max_output_tokens: int
    max_context_window: int
    cost_per_1k_input: float = 0.0  # USD
    cost_per_1k_output: float = 0.0  # USD
    supports_batching: bool = True
    recommended_batch_size: int = 32


class EmbeddingModelBudgets:
    """Token budgets for embedding models."""

    # OpenAI Embeddings
    TEXT_EMBEDDING_3_SMALL: ClassVar[TokenBudget] = TokenBudget(
        max_input_tokens=8191,
        max_output_tokens=0,  # Embeddings don't have output tokens
        max_context_window=8191,
        cost_per_1k_input=0.00002,  # $0.02 per 1M tokens
        supports_batching=True,
        recommended_batch_size=2048,
    )

    TEXT_EMBEDDING_3_LARGE: ClassVar[TokenBudget] = TokenBudget(
        max_input_tokens=8191,
        max_output_tokens=0,
        max_context_window=8191,
        cost_per_1k_input=0.00013,  # $0.13 per 1M tokens
        supports_batching=True,
        recommended_batch_size=2048,
    )

    TEXT_EMBEDDING_ADA_002: ClassVar[TokenBudget] = TokenBudget(
        max_input_tokens=8191,
        max_output_tokens=0,
        max_context_window=8191,
        cost_per_1k_input=0.0001,  # $0.10 per 1M tokens
        supports_batching=True,
        recommended_batch_size=2048,
    )

    # Cohere Embeddings
    EMBED_ENGLISH_V3: ClassVar[TokenBudget] = TokenBudget(
        max_input_tokens=512,
        max_output_tokens=0,
        max_context_window=512,
        cost_per_1k_input=0.0001,  # $0.10 per 1M tokens
        supports_batching=True,
        recommended_batch_size=96,
    )

    EMBED_MULTILINGUAL_V3: ClassVar[TokenBudget] = TokenBudget(
        max_input_tokens=512,
        max_output_tokens=0,
        max_context_window=512,
        cost_per_1k_input=0.0001,
        supports_batching=True,
        recommended_batch_size=96,
    )

    # Local Models (HuggingFace)
    E5_SMALL: ClassVar[TokenBudget] = TokenBudget(
        max_input_tokens=512,
        max_output_tokens=0,
        max_context_window=512,
        cost_per_1k_input=0.0,  # Free (local)
        supports_batching=True,
        recommended_batch_size=32,
    )

    E5_BASE: ClassVar[TokenBudget] = TokenBudget(
        max_input_tokens=512,
        max_output_tokens=0,
        max_context_window=512,
        cost_per_1k_input=0.0,
        supports_batching=True,
        recommended_batch_size=32,
    )

    E5_LARGE: ClassVar[TokenBudget] = TokenBudget(
        max_input_tokens=512,
        max_output_tokens=0,
        max_context_window=512,
        cost_per_1k_input=0.0,
        supports_batching=True,
        recommended_batch_size=16,
    )

    BGE_SMALL: ClassVar[TokenBudget] = TokenBudget(
        max_input_tokens=512,
        max_output_tokens=0,
        max_context_window=512,
        cost_per_1k_input=0.0,
        supports_batching=True,
        recommended_batch_size=32,
    )

    BGE_BASE: ClassVar[TokenBudget] = TokenBudget(
        max_input_tokens=512,
        max_output_tokens=0,
        max_context_window=512,
        cost_per_1k_input=0.0,
        supports_batching=True,
        recommended_batch_size=32,
    )

    BGE_LARGE: ClassVar[TokenBudget] = TokenBudget(
        max_input_tokens=512,
        max_output_tokens=0,
        max_context_window=512,
        cost_per_1k_input=0.0,
        supports_batching=True,
        recommended_batch_size=16,
    )

    # Hash provider (no limits)
    HASH: ClassVar[TokenBudget] = TokenBudget(
        max_input_tokens=100000,
        max_output_tokens=0,
        max_context_window=100000,
        cost_per_1k_input=0.0,
        supports_batching=True,
        recommended_batch_size=1000,
    )


class GenerationModelBudgets:
    """Token budgets for generation models."""

    # OpenAI Models
    GPT_4_TURBO: ClassVar[TokenBudget] = TokenBudget(
        max_input_tokens=128000,
        max_output_tokens=4096,
        max_context_window=128000,
        cost_per_1k_input=0.01,  # $10 per 1M input tokens
        cost_per_1k_output=0.03,  # $30 per 1M output tokens
        supports_batching=False,
        recommended_batch_size=1,
    )

    GPT_4: ClassVar[TokenBudget] = TokenBudget(
        max_input_tokens=8192,
        max_output_tokens=8192,
        max_context_window=8192,
        cost_per_1k_input=0.03,
        cost_per_1k_output=0.06,
        supports_batching=False,
        recommended_batch_size=1,
    )

    GPT_3_5_TURBO: ClassVar[TokenBudget] = TokenBudget(
        max_input_tokens=16385,
        max_output_tokens=4096,
        max_context_window=16385,
        cost_per_1k_input=0.0015,  # $1.50 per 1M tokens
        cost_per_1k_output=0.002,  # $2.00 per 1M tokens
        supports_batching=False,
        recommended_batch_size=1,
    )

    # Anthropic Claude
    CLAUDE_3_OPUS: ClassVar[TokenBudget] = TokenBudget(
        max_input_tokens=200000,
        max_output_tokens=4096,
        max_context_window=200000,
        cost_per_1k_input=0.015,  # $15 per 1M tokens
        cost_per_1k_output=0.075,  # $75 per 1M tokens
        supports_batching=False,
        recommended_batch_size=1,
    )

    CLAUDE_3_SONNET: ClassVar[TokenBudget] = TokenBudget(
        max_input_tokens=200000,
        max_output_tokens=4096,
        max_context_window=200000,
        cost_per_1k_input=0.003,  # $3 per 1M tokens
        cost_per_1k_output=0.015,  # $15 per 1M tokens
        supports_batching=False,
        recommended_batch_size=1,
    )

    CLAUDE_3_HAIKU: ClassVar[TokenBudget] = TokenBudget(
        max_input_tokens=200000,
        max_output_tokens=4096,
        max_context_window=200000,
        cost_per_1k_input=0.00025,  # $0.25 per 1M tokens
        cost_per_1k_output=0.00125,  # $1.25 per 1M tokens
        supports_batching=False,
        recommended_batch_size=1,
    )

    # Local HuggingFace Models
    GPT2: ClassVar[TokenBudget] = TokenBudget(
        max_input_tokens=1024,
        max_output_tokens=1024,
        max_context_window=1024,
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        supports_batching=True,
        recommended_batch_size=8,
    )

    GPT2_MEDIUM: ClassVar[TokenBudget] = TokenBudget(
        max_input_tokens=1024,
        max_output_tokens=1024,
        max_context_window=1024,
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        supports_batching=True,
        recommended_batch_size=4,
    )

    GPT2_LARGE: ClassVar[TokenBudget] = TokenBudget(
        max_input_tokens=1024,
        max_output_tokens=1024,
        max_context_window=1024,
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        supports_batching=True,
        recommended_batch_size=2,
    )

    LLAMA_2_7B: ClassVar[TokenBudget] = TokenBudget(
        max_input_tokens=4096,
        max_output_tokens=4096,
        max_context_window=4096,
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        supports_batching=True,
        recommended_batch_size=4,
    )

    LLAMA_2_13B: ClassVar[TokenBudget] = TokenBudget(
        max_input_tokens=4096,
        max_output_tokens=4096,
        max_context_window=4096,
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        supports_batching=True,
        recommended_batch_size=2,
    )

    MISTRAL_7B: ClassVar[TokenBudget] = TokenBudget(
        max_input_tokens=8192,
        max_output_tokens=8192,
        max_context_window=8192,
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        supports_batching=True,
        recommended_batch_size=4,
    )


# Model name to budget mapping
EMBEDDING_MODEL_BUDGETS = {
    # OpenAI
    "text-embedding-3-small": EmbeddingModelBudgets.TEXT_EMBEDDING_3_SMALL,
    "text-embedding-3-large": EmbeddingModelBudgets.TEXT_EMBEDDING_3_LARGE,
    "text-embedding-ada-002": EmbeddingModelBudgets.TEXT_EMBEDDING_ADA_002,
    # Cohere
    "embed-english-v3.0": EmbeddingModelBudgets.EMBED_ENGLISH_V3,
    "embed-multilingual-v3.0": EmbeddingModelBudgets.EMBED_MULTILINGUAL_V3,
    # HuggingFace E5
    "intfloat/e5-small-v2": EmbeddingModelBudgets.E5_SMALL,
    "intfloat/e5-base-v2": EmbeddingModelBudgets.E5_BASE,
    "intfloat/e5-large-v2": EmbeddingModelBudgets.E5_LARGE,
    # HuggingFace BGE
    "BAAI/bge-small-en-v1.5": EmbeddingModelBudgets.BGE_SMALL,
    "BAAI/bge-base-en-v1.5": EmbeddingModelBudgets.BGE_BASE,
    "BAAI/bge-large-en-v1.5": EmbeddingModelBudgets.BGE_LARGE,
    # Hash
    "simple-hash": EmbeddingModelBudgets.HASH,
    "hash": EmbeddingModelBudgets.HASH,
}

GENERATION_MODEL_BUDGETS = {
    # OpenAI
    "gpt-4-turbo": GenerationModelBudgets.GPT_4_TURBO,
    "gpt-4": GenerationModelBudgets.GPT_4,
    "gpt-3.5-turbo": GenerationModelBudgets.GPT_3_5_TURBO,
    # Anthropic
    "claude-3-opus": GenerationModelBudgets.CLAUDE_3_OPUS,
    "claude-3-sonnet": GenerationModelBudgets.CLAUDE_3_SONNET,
    "claude-3-haiku": GenerationModelBudgets.CLAUDE_3_HAIKU,
    # HuggingFace
    "gpt2": GenerationModelBudgets.GPT2,
    "gpt2-medium": GenerationModelBudgets.GPT2_MEDIUM,
    "gpt2-large": GenerationModelBudgets.GPT2_LARGE,
    "meta-llama/Llama-2-7b-hf": GenerationModelBudgets.LLAMA_2_7B,
    "meta-llama/Llama-2-13b-hf": GenerationModelBudgets.LLAMA_2_13B,
    "mistralai/Mistral-7B-v0.1": GenerationModelBudgets.MISTRAL_7B,
}


def get_embedding_budget(model_name: str) -> TokenBudget:
    """
    Get token budget for an embedding model.

    Args:
        model_name: Model identifier

    Returns:
        TokenBudget for the model, or default if not found
    """
    return EMBEDDING_MODEL_BUDGETS.get(
        model_name,
        TokenBudget(
            max_input_tokens=512,
            max_output_tokens=0,
            max_context_window=512,
            cost_per_1k_input=0.0,
            supports_batching=True,
            recommended_batch_size=32,
        ),
    )


def get_generation_budget(model_name: str) -> TokenBudget:
    """
    Get token budget for a generation model.

    Args:
        model_name: Model identifier

    Returns:
        TokenBudget for the model, or default if not found
    """
    return GENERATION_MODEL_BUDGETS.get(
        model_name,
        TokenBudget(
            max_input_tokens=2048,
            max_output_tokens=2048,
            max_context_window=2048,
            cost_per_1k_input=0.0,
            cost_per_1k_output=0.0,
            supports_batching=True,
            recommended_batch_size=4,
        ),
    )


def estimate_cost(
    model_name: str,
    input_tokens: int,
    output_tokens: int = 0,
    is_embedding: bool = True,
) -> float:
    """
    Estimate cost for a model operation.

    Args:
        model_name: Model identifier
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens (0 for embeddings)
        is_embedding: Whether this is an embedding model

    Returns:
        Estimated cost in USD
    """
    if is_embedding:
        budget = get_embedding_budget(model_name)
    else:
        budget = get_generation_budget(model_name)

    input_cost = (input_tokens / 1000) * budget.cost_per_1k_input
    output_cost = (output_tokens / 1000) * budget.cost_per_1k_output

    return input_cost + output_cost
