"""Tests for model selection functionality."""

import pytest

from src.services.cost.model_selector import (
    ModelCandidate,
    ModelSelector,
    ModelTier,
    SelectionStrategy,
)


class TestModelCandidate:
    """Test ModelCandidate class."""

    def test_model_candidate_creation(self):
        """Test creating a model candidate."""
        model = ModelCandidate(
            name="test-model",
            tier=ModelTier.BALANCED,
            cost_per_1k=1.0,
            avg_latency_ms=100,
            quality_score=0.8,
        )

        assert model.name == "test-model"
        assert model.tier == ModelTier.BALANCED
        assert model.cost_per_1k == 1.0
        assert model.avg_latency_ms == 100
        assert model.quality_score == 0.8

    def test_efficiency_score(self):
        """Test efficiency score calculation."""
        model = ModelCandidate(
            name="test",
            tier=ModelTier.BALANCED,
            cost_per_1k=2.0,
            avg_latency_ms=100,
            quality_score=0.8,
        )

        # Efficiency = quality / cost
        assert model.efficiency_score == 0.4

    def test_speed_score(self):
        """Test speed score calculation."""
        model = ModelCandidate(
            name="test",
            tier=ModelTier.BALANCED,
            cost_per_1k=1.0,
            avg_latency_ms=100,
            quality_score=0.8,
        )

        # Speed = 1000 / latency
        assert model.speed_score == 10.0


class TestModelSelector:
    """Test ModelSelector class."""

    def test_initialization_with_defaults(self):
        """Test selector initialization with default models."""
        selector = ModelSelector()

        assert len(selector.models) > 0
        assert selector.current_budget is None

    def test_initialization_with_custom_models(self):
        """Test selector initialization with custom models."""
        custom_models = [
            ModelCandidate(
                name="custom1",
                tier=ModelTier.BUDGET,
                cost_per_1k=0.5,
                avg_latency_ms=50,
                quality_score=0.7,
            ),
            ModelCandidate(
                name="custom2",
                tier=ModelTier.PREMIUM,
                cost_per_1k=5.0,
                avg_latency_ms=200,
                quality_score=0.95,
            ),
        ]

        selector = ModelSelector(models=custom_models)

        assert len(selector.models) == 2
        assert selector.models[0].name == "custom1"

    def test_set_budget(self):
        """Test setting budget."""
        selector = ModelSelector()
        selector.set_budget(100.0)

        assert selector.current_budget == 100.0
        assert selector.spent_budget == 0.0
        assert selector.remaining_budget == 100.0

    def test_record_cost(self):
        """Test recording cost."""
        selector = ModelSelector()
        selector.set_budget(100.0)
        selector.record_cost(25.0)

        assert selector.spent_budget == 25.0
        assert selector.remaining_budget == 75.0

    def test_select_minimize_cost(self):
        """Test selecting cheapest model."""
        selector = ModelSelector()

        model = selector.select_model(
            strategy=SelectionStrategy.MINIMIZE_COST,
            min_quality=0.5,
        )

        # Should select one of the cheapest models
        assert model.cost_per_1k <= 0.2
        assert model.quality_score >= 0.5

    def test_select_minimize_latency(self):
        """Test selecting fastest model."""
        selector = ModelSelector()

        model = selector.select_model(
            strategy=SelectionStrategy.MINIMIZE_LATENCY,
            min_quality=0.5,
        )

        # Should select one of the fastest models
        assert model.avg_latency_ms <= 100

    def test_select_maximize_quality(self):
        """Test selecting highest quality model."""
        selector = ModelSelector()

        model = selector.select_model(
            strategy=SelectionStrategy.MAXIMIZE_QUALITY,
            max_cost_per_1k=50.0,
        )

        # Should select high quality model
        assert model.quality_score >= 0.9

    def test_select_balanced(self):
        """Test balanced selection (efficiency)."""
        selector = ModelSelector()

        model = selector.select_model(
            strategy=SelectionStrategy.BALANCED,
            min_quality=0.7,
        )

        # Should select efficient model
        assert model.quality_score >= 0.7
        assert model.efficiency_score > 0

    def test_select_with_quality_constraint(self):
        """Test selection with quality constraint."""
        selector = ModelSelector()

        model = selector.select_model(
            strategy=SelectionStrategy.MINIMIZE_COST,
            min_quality=0.9,
        )

        assert model.quality_score >= 0.9

    def test_select_with_latency_constraint(self):
        """Test selection with latency constraint."""
        selector = ModelSelector()

        model = selector.select_model(
            strategy=SelectionStrategy.MINIMIZE_COST,
            max_latency_ms=100,
        )

        assert model.avg_latency_ms <= 100

    def test_select_with_cost_constraint(self):
        """Test selection with cost constraint."""
        selector = ModelSelector()

        model = selector.select_model(
            strategy=SelectionStrategy.MAXIMIZE_QUALITY,
            max_cost_per_1k=1.0,
        )

        assert model.cost_per_1k <= 1.0

    def test_select_specific_tier(self):
        """Test selecting from specific tier."""
        selector = ModelSelector()

        model = selector.select_model(
            strategy=SelectionStrategy.BALANCED,
            tier=ModelTier.PREMIUM,
        )

        assert model.tier == ModelTier.PREMIUM

    def test_select_with_budget_constraint(self):
        """Test selection respects budget."""
        selector = ModelSelector()
        selector.set_budget(0.001)  # Very low budget

        model = selector.select_model(
            strategy=SelectionStrategy.MAXIMIZE_QUALITY,
        )

        # Should select cheap model due to budget
        assert model.cost_per_1k <= 1.0

    def test_select_no_valid_models(self):
        """Test selection when no models meet constraints."""
        selector = ModelSelector()

        with pytest.raises(ValueError, match="No models meet"):
            selector.select_model(
                strategy=SelectionStrategy.MINIMIZE_COST,
                min_quality=0.99,
                max_cost_per_1k=0.01,
            )

    def test_adaptive_selection_low_budget(self):
        """Test adaptive selection with low budget."""
        selector = ModelSelector()
        selector.set_budget(10.0)
        selector.spent_budget = 9.0  # 90% spent

        model = selector.select_model(
            strategy=SelectionStrategy.ADAPTIVE,
            min_quality=0.5,
        )

        # Should select cheap model
        assert model.cost_per_1k < 1.0

    def test_adaptive_selection_normal_budget(self):
        """Test adaptive selection with normal budget."""
        selector = ModelSelector()
        selector.set_budget(100.0)
        selector.spent_budget = 10.0  # 10% spent

        model = selector.select_model(
            strategy=SelectionStrategy.ADAPTIVE,
            min_quality=0.7,
        )

        # Should optimize for efficiency
        assert model.efficiency_score > 0

    def test_get_tier_models(self):
        """Test getting models from specific tier."""
        selector = ModelSelector()

        budget_models = selector.get_tier_models(ModelTier.BUDGET)

        assert len(budget_models) > 0
        assert all(m.tier == ModelTier.BUDGET for m in budget_models)

    def test_get_comparison_matrix(self):
        """Test getting comparison matrix."""
        selector = ModelSelector()

        matrix = selector.get_comparison_matrix()

        assert "models" in matrix
        assert "tiers" in matrix
        assert len(matrix["models"]) > 0

        # Check first model structure
        model_data = matrix["models"][0]
        assert "name" in model_data
        assert "tier" in model_data
        assert "quality" in model_data
        assert "latency_ms" in model_data
        assert "cost_per_1k" in model_data
        assert "efficiency" in model_data

    def test_comparison_matrix_tiers(self):
        """Test tier summaries in comparison matrix."""
        selector = ModelSelector()

        matrix = selector.get_comparison_matrix()

        # Should have tier summaries
        assert len(matrix["tiers"]) > 0

        for _tier_name, tier_data in matrix["tiers"].items():
            assert "count" in tier_data
            assert "avg_quality" in tier_data
            assert "avg_latency_ms" in tier_data
            assert "avg_cost_per_1k" in tier_data
