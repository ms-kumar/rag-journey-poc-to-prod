"""
Tests for token budget definitions and cost estimation.
"""

import pytest

from src.schemas.services.token_budgets import (
    EMBEDDING_MODEL_BUDGETS,
    GENERATION_MODEL_BUDGETS,
    EmbeddingModelBudgets,
    GenerationModelBudgets,
    TokenBudget,
    estimate_cost,
    get_embedding_budget,
    get_generation_budget,
)


class TestTokenBudget:
    """Test TokenBudget dataclass."""

    def test_token_budget_creation(self):
        budget = TokenBudget(
            max_input_tokens=1000,
            max_output_tokens=500,
            max_context_window=1500,
            cost_per_1k_input=0.001,
            cost_per_1k_output=0.002,
        )
        assert budget.max_input_tokens == 1000
        assert budget.max_output_tokens == 500
        assert budget.cost_per_1k_input == 0.001

    def test_token_budget_frozen(self):
        budget = TokenBudget(max_input_tokens=1000, max_output_tokens=500, max_context_window=1500)
        with pytest.raises(Exception):  # FrozenInstanceError  # noqa: B017
            budget.max_input_tokens = 2000  # type: ignore[misc]


class TestEmbeddingModelBudgets:
    """Test embedding model budgets."""

    def test_openai_budgets(self):
        assert EmbeddingModelBudgets.TEXT_EMBEDDING_3_SMALL.max_input_tokens == 8191
        assert EmbeddingModelBudgets.TEXT_EMBEDDING_3_LARGE.max_input_tokens == 8191
        assert EmbeddingModelBudgets.TEXT_EMBEDDING_ADA_002.max_input_tokens == 8191

    def test_cohere_budgets(self):
        assert EmbeddingModelBudgets.EMBED_ENGLISH_V3.max_input_tokens == 512
        assert EmbeddingModelBudgets.EMBED_MULTILINGUAL_V3.max_input_tokens == 512

    def test_local_model_budgets(self):
        assert EmbeddingModelBudgets.E5_SMALL.cost_per_1k_input == 0.0
        assert EmbeddingModelBudgets.BGE_BASE.cost_per_1k_input == 0.0
        assert EmbeddingModelBudgets.HASH.cost_per_1k_input == 0.0

    def test_embedding_models_have_no_output(self):
        assert EmbeddingModelBudgets.TEXT_EMBEDDING_3_SMALL.max_output_tokens == 0
        assert EmbeddingModelBudgets.E5_BASE.max_output_tokens == 0


class TestGenerationModelBudgets:
    """Test generation model budgets."""

    def test_openai_gpt_budgets(self):
        assert GenerationModelBudgets.GPT_4_TURBO.max_context_window == 128000
        assert GenerationModelBudgets.GPT_4.max_context_window == 8192
        assert GenerationModelBudgets.GPT_3_5_TURBO.max_context_window == 16385

    def test_claude_budgets(self):
        assert GenerationModelBudgets.CLAUDE_3_OPUS.max_context_window == 200000
        assert GenerationModelBudgets.CLAUDE_3_SONNET.max_context_window == 200000
        assert GenerationModelBudgets.CLAUDE_3_HAIKU.max_context_window == 200000

    def test_local_model_budgets(self):
        assert GenerationModelBudgets.GPT2.cost_per_1k_input == 0.0
        assert GenerationModelBudgets.LLAMA_2_7B.cost_per_1k_input == 0.0
        assert GenerationModelBudgets.MISTRAL_7B.cost_per_1k_input == 0.0

    def test_output_token_limits(self):
        assert GenerationModelBudgets.GPT_4_TURBO.max_output_tokens == 4096
        assert GenerationModelBudgets.CLAUDE_3_OPUS.max_output_tokens == 4096
        assert GenerationModelBudgets.GPT2.max_output_tokens == 1024


class TestBudgetLookup:
    """Test budget lookup functions."""

    def test_get_embedding_budget_known_model(self):
        budget = get_embedding_budget("text-embedding-3-small")
        assert budget.max_input_tokens == 8191
        assert budget.cost_per_1k_input == 0.00002

    def test_get_embedding_budget_unknown_model(self):
        budget = get_embedding_budget("unknown-model")
        assert budget.max_input_tokens == 512  # Default
        assert budget.cost_per_1k_input == 0.0

    def test_get_generation_budget_known_model(self):
        budget = get_generation_budget("gpt-4-turbo")
        assert budget.max_context_window == 128000
        assert budget.cost_per_1k_input == 0.01

    def test_get_generation_budget_unknown_model(self):
        budget = get_generation_budget("unknown-llm")
        assert budget.max_input_tokens == 2048  # Default
        assert budget.cost_per_1k_output == 0.0

    def test_all_embedding_models_in_mapping(self):
        # Verify key models are registered
        assert "text-embedding-3-small" in EMBEDDING_MODEL_BUDGETS
        assert "intfloat/e5-small-v2" in EMBEDDING_MODEL_BUDGETS
        assert "BAAI/bge-base-en-v1.5" in EMBEDDING_MODEL_BUDGETS
        assert "simple-hash" in EMBEDDING_MODEL_BUDGETS

    def test_all_generation_models_in_mapping(self):
        # Verify key models are registered
        assert "gpt-4-turbo" in GENERATION_MODEL_BUDGETS
        assert "claude-3-opus" in GENERATION_MODEL_BUDGETS
        assert "gpt2" in GENERATION_MODEL_BUDGETS
        assert "mistralai/Mistral-7B-v0.1" in GENERATION_MODEL_BUDGETS


class TestCostEstimation:
    """Test cost estimation."""

    def test_embedding_cost_openai(self):
        cost = estimate_cost("text-embedding-3-small", input_tokens=1000, is_embedding=True)
        # $0.02 per 1M tokens = $0.00002 per 1K tokens
        assert cost == pytest.approx(0.00002, rel=1e-6)

    def test_embedding_cost_local_model(self):
        cost = estimate_cost("intfloat/e5-small-v2", input_tokens=10000, is_embedding=True)
        assert cost == 0.0  # Local models are free

    def test_generation_cost_gpt4(self):
        cost = estimate_cost(
            "gpt-4-turbo", input_tokens=1000, output_tokens=500, is_embedding=False
        )
        # Input: 1000 * $0.01 = $0.01
        # Output: 500 * $0.03 = $0.015
        # Total: $0.025
        assert cost == pytest.approx(0.025, rel=1e-6)

    def test_generation_cost_claude(self):
        cost = estimate_cost(
            "claude-3-haiku", input_tokens=2000, output_tokens=1000, is_embedding=False
        )
        # Input: 2 * $0.25 = $0.0005
        # Output: 1 * $1.25 = $0.00125
        # Total: $0.00175
        assert cost == pytest.approx(0.00175, rel=1e-6)

    def test_generation_cost_local_model(self):
        cost = estimate_cost("gpt2", input_tokens=5000, output_tokens=2000, is_embedding=False)
        assert cost == 0.0

    def test_cost_large_scale(self):
        # Embedding 1M tokens with GPT-4 embeddings
        cost = estimate_cost("text-embedding-3-small", input_tokens=1_000_000, is_embedding=True)
        assert cost == pytest.approx(0.02, rel=1e-6)  # $0.02

    def test_cost_unknown_model_defaults_to_free(self):
        cost = estimate_cost("random-model", input_tokens=1000, is_embedding=True)
        assert cost == 0.0


class TestBatchRecommendations:
    """Test batch size recommendations."""

    def test_api_models_have_large_batch_sizes(self):
        budget = get_embedding_budget("text-embedding-3-small")
        assert budget.recommended_batch_size == 2048

    def test_local_models_have_smaller_batch_sizes(self):
        budget = get_embedding_budget("intfloat/e5-small-v2")
        assert budget.recommended_batch_size == 32

    def test_large_models_have_smaller_batches(self):
        budget = get_embedding_budget("intfloat/e5-large-v2")
        assert budget.recommended_batch_size == 16

    def test_generation_models_typically_no_batching(self):
        budget = get_generation_budget("gpt-4-turbo")
        assert budget.supports_batching is False
        assert budget.recommended_batch_size == 1
