"""Tests for the ModelSelector class."""

import pytest

from src.services.cost.model_selector import (
    ModelCandidate,
    ModelSelector,
    ModelTier,
    SelectionStrategy,
)


class TestModelCandidate:
    """Tests for ModelCandidate dataclass."""

    def test_model_candidate_creation(self):
        """Test creating a model candidate."""
        candidate = ModelCandidate(
            name="test-model",
            tier=ModelTier.BALANCED,
            cost_per_1k=0.50,
            avg_latency_ms=100.0,
            quality_score=0.80,
        )

        assert candidate.name == "test-model"
        assert candidate.tier == ModelTier.BALANCED
        assert candidate.cost_per_1k == 0.50
        assert candidate.avg_latency_ms == 100.0
        assert candidate.quality_score == 0.80

    def test_efficiency_score(self):
        """Test efficiency score calculation."""
        candidate = ModelCandidate(
            name="test-model",
            tier=ModelTier.BALANCED,
            cost_per_1k=0.50,
            avg_latency_ms=100.0,
            quality_score=0.80,
        )

        # quality / cost = 0.80 / 0.50 = 1.6
        assert candidate.efficiency_score == 1.6

    def test_efficiency_score_zero_cost(self):
        """Test efficiency score with zero cost."""
        candidate = ModelCandidate(
            name="free-model",
            tier=ModelTier.BUDGET,
            cost_per_1k=0.0,
            avg_latency_ms=100.0,
            quality_score=0.50,
        )

        assert candidate.efficiency_score == float("inf")

    def test_speed_score(self):
        """Test speed score calculation."""
        candidate = ModelCandidate(
            name="test-model",
            tier=ModelTier.BALANCED,
            cost_per_1k=0.50,
            avg_latency_ms=100.0,
            quality_score=0.80,
        )

        # 1000 / 100 = 10
        assert candidate.speed_score == 10.0

    def test_speed_score_zero_latency(self):
        """Test speed score with zero latency."""
        candidate = ModelCandidate(
            name="instant-model",
            tier=ModelTier.BUDGET,
            cost_per_1k=0.50,
            avg_latency_ms=0.0,
            quality_score=0.50,
        )

        assert candidate.speed_score == float("inf")


class TestModelTier:
    """Tests for ModelTier enum."""

    def test_tier_values(self):
        """Test tier enum values."""
        assert ModelTier.BUDGET.value == "budget"
        assert ModelTier.BALANCED.value == "balanced"
        assert ModelTier.PREMIUM.value == "premium"
        assert ModelTier.ULTRA.value == "ultra"


class TestSelectionStrategy:
    """Tests for SelectionStrategy enum."""

    def test_strategy_values(self):
        """Test selection strategy values."""
        assert SelectionStrategy.MINIMIZE_COST.value == "minimize_cost"
        assert SelectionStrategy.MINIMIZE_LATENCY.value == "minimize_latency"
        assert SelectionStrategy.MAXIMIZE_QUALITY.value == "maximize_quality"
        assert SelectionStrategy.BALANCED.value == "balanced"
        assert SelectionStrategy.ADAPTIVE.value == "adaptive"


class TestModelSelector:
    """Tests for ModelSelector class."""

    def test_initialization_default_models(self, model_selector: ModelSelector):
        """Test initialization with default models."""
        assert len(model_selector.models) > 0
        assert model_selector.current_budget is None

    def test_initialization_custom_models(self, custom_models: list[ModelCandidate]):
        """Test initialization with custom models."""
        selector = ModelSelector(models=custom_models)

        assert len(selector.models) == 3
        assert selector.models[0].name == "cheap-model"

    def test_select_minimize_cost(self, custom_models: list[ModelCandidate]):
        """Test selecting model with minimize cost strategy."""
        selector = ModelSelector(models=custom_models)

        model = selector.select_model(strategy=SelectionStrategy.MINIMIZE_COST)

        assert model is not None
        assert model.name == "cheap-model"

    def test_select_maximize_quality(self, custom_models: list[ModelCandidate]):
        """Test selecting model with maximize quality strategy."""
        selector = ModelSelector(models=custom_models)

        model = selector.select_model(strategy=SelectionStrategy.MAXIMIZE_QUALITY)

        assert model is not None
        assert model.name == "premium-model"

    def test_select_minimize_latency(self, custom_models: list[ModelCandidate]):
        """Test selecting model with minimize latency strategy."""
        selector = ModelSelector(models=custom_models)

        model = selector.select_model(strategy=SelectionStrategy.MINIMIZE_LATENCY)

        assert model is not None
        assert model.name == "cheap-model"  # Lowest latency

    def test_select_balanced(self, custom_models: list[ModelCandidate]):
        """Test selecting model with balanced strategy."""
        selector = ModelSelector(models=custom_models)

        model = selector.select_model(strategy=SelectionStrategy.BALANCED)

        assert model is not None
        # Should select model with best efficiency score
        # cheap: 0.60 / 0.10 = 6.0
        # balanced: 0.75 / 0.50 = 1.5
        # premium: 0.90 / 2.00 = 0.45
        assert model.name == "cheap-model"

    def test_select_with_min_quality(self, custom_models: list[ModelCandidate]):
        """Test selecting model with minimum quality requirement."""
        selector = ModelSelector(models=custom_models)

        model = selector.select_model(
            strategy=SelectionStrategy.MINIMIZE_COST,
            min_quality=0.70,
        )

        assert model is not None
        # cheap-model has quality 0.60, so should be excluded
        assert model.name == "balanced-model"

    def test_select_with_max_latency(self, custom_models: list[ModelCandidate]):
        """Test selecting model with max latency requirement."""
        selector = ModelSelector(models=custom_models)

        model = selector.select_model(
            strategy=SelectionStrategy.MAXIMIZE_QUALITY,
            max_latency_ms=150,
        )

        assert model is not None
        # premium-model has latency 200ms, should be excluded
        assert model.name == "balanced-model"

    def test_select_with_max_cost(self, custom_models: list[ModelCandidate]):
        """Test selecting model with max cost requirement."""
        selector = ModelSelector(models=custom_models)

        model = selector.select_model(
            strategy=SelectionStrategy.MAXIMIZE_QUALITY,
            max_cost_per_1k=1.0,
        )

        assert model is not None
        # premium-model costs 2.00, should be excluded
        assert model.name == "balanced-model"

    def test_select_by_tier(self, custom_models: list[ModelCandidate]):
        """Test selecting model by tier."""
        selector = ModelSelector(models=custom_models)

        model = selector.select_model(tier=ModelTier.PREMIUM)

        assert model is not None
        assert model.tier == ModelTier.PREMIUM
        assert model.name == "premium-model"

    def test_select_by_tier_not_found(self, custom_models: list[ModelCandidate]):
        """Test selecting by tier when no model in tier."""
        selector = ModelSelector(models=custom_models)

        with pytest.raises(ValueError, match="No models meet"):
            selector.select_model(tier=ModelTier.ULTRA)

    def test_get_tier_models(self, model_selector: ModelSelector):
        """Test getting all models in a tier."""
        budget_models = model_selector.get_tier_models(ModelTier.BUDGET)

        assert len(budget_models) > 0
        for model in budget_models:
            assert model.tier == ModelTier.BUDGET

    def test_set_budget(self, model_selector: ModelSelector):
        """Test setting a budget."""
        model_selector.set_budget(100.0)

        assert model_selector.current_budget == 100.0
        assert model_selector.spent_budget == 0.0

    def test_record_cost(self, model_selector: ModelSelector):
        """Test recording model usage cost."""
        model_selector.set_budget(100.0)
        model_selector.record_cost(10.0)

        assert model_selector.spent_budget == 10.0

    def test_remaining_budget(self, model_selector: ModelSelector):
        """Test getting remaining budget."""
        model_selector.set_budget(100.0)
        model_selector.record_cost(30.0)

        remaining = model_selector.remaining_budget

        assert remaining == 70.0

    def test_remaining_budget_no_budget_set(self, model_selector: ModelSelector):
        """Test remaining budget when no budget is set."""
        remaining = model_selector.remaining_budget
        assert remaining is None

    def test_no_model_meets_requirements(self, custom_models: list[ModelCandidate]):
        """Test when no model meets all requirements."""
        selector = ModelSelector(models=custom_models)

        with pytest.raises(ValueError, match="No models meet"):
            selector.select_model(
                strategy=SelectionStrategy.MINIMIZE_COST,
                min_quality=0.95,  # No model has this quality
            )

    def test_select_no_matching_models(self, custom_models: list[ModelCandidate]):
        """Test selecting when no model meets constraints raises error."""
        selector = ModelSelector(models=custom_models)

        # No model has quality score >= 0.99
        with pytest.raises(ValueError, match="No models meet the specified constraints"):
            selector.select_model(
                strategy=SelectionStrategy.MINIMIZE_COST,
                min_quality=0.99,
            )
