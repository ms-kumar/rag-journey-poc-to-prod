"""
Tiered model selection based on quality, latency, and cost requirements.

Implements intelligent model selection strategies to optimize
for different objectives (cost, speed, quality).
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ModelTier(Enum):
    """Model performance tiers."""

    BUDGET = "budget"  # Low cost, acceptable quality
    BALANCED = "balanced"  # Balance of cost and quality
    PREMIUM = "premium"  # High quality, higher cost
    ULTRA = "ultra"  # Highest quality, highest cost


class SelectionStrategy(Enum):
    """Model selection strategy."""

    MINIMIZE_COST = "minimize_cost"  # Choose cheapest model meeting requirements
    MINIMIZE_LATENCY = "minimize_latency"  # Choose fastest model
    MAXIMIZE_QUALITY = "maximize_quality"  # Choose highest quality model
    BALANCED = "balanced"  # Optimize quality/cost ratio
    ADAPTIVE = "adaptive"  # Adapt based on load and budget


@dataclass
class ModelCandidate:
    """Candidate model with its characteristics."""

    name: str
    tier: ModelTier
    cost_per_1k: float  # Cost per 1000 requests
    avg_latency_ms: float  # Average latency in milliseconds
    quality_score: float  # Quality score (0-1)
    max_concurrency: int = 10  # Max concurrent requests

    @property
    def efficiency_score(self) -> float:
        """Efficiency score (quality per dollar)."""
        if self.cost_per_1k == 0:
            return float("inf")
        return self.quality_score / self.cost_per_1k

    @property
    def speed_score(self) -> float:
        """Speed score (inverse of latency)."""
        if self.avg_latency_ms == 0:
            return float("inf")
        return 1000.0 / self.avg_latency_ms


class ModelSelector:
    """
    Select optimal models based on requirements and strategy.

    Features:
    - Quality/latency/cost matrix
    - Tiered model selection
    - Multiple selection strategies
    - Budget-aware selection
    """

    def __init__(self, models: list[ModelCandidate] | None = None):
        """
        Initialize model selector.

        Args:
            models: List of available model candidates
        """
        self.models = models or self._get_default_models()
        self.current_budget: float | None = None
        self.spent_budget = 0.0

    def _get_default_models(self) -> list[ModelCandidate]:
        """Get default model configurations."""
        return [
            # Budget tier - fast, cheap, decent quality
            ModelCandidate(
                name="gpt2",
                tier=ModelTier.BUDGET,
                cost_per_1k=0.10,
                avg_latency_ms=50,
                quality_score=0.65,
                max_concurrency=20,
            ),
            ModelCandidate(
                name="distilgpt2",
                tier=ModelTier.BUDGET,
                cost_per_1k=0.05,
                avg_latency_ms=30,
                quality_score=0.60,
                max_concurrency=30,
            ),
            # Balanced tier - good quality/cost tradeoff
            ModelCandidate(
                name="flan-t5-base",
                tier=ModelTier.BALANCED,
                cost_per_1k=0.50,
                avg_latency_ms=100,
                quality_score=0.75,
                max_concurrency=15,
            ),
            ModelCandidate(
                name="llama-2-7b",
                tier=ModelTier.BALANCED,
                cost_per_1k=0.80,
                avg_latency_ms=150,
                quality_score=0.80,
                max_concurrency=10,
            ),
            # Premium tier - high quality
            ModelCandidate(
                name="gpt-3.5-turbo",
                tier=ModelTier.PREMIUM,
                cost_per_1k=2.00,
                avg_latency_ms=500,
                quality_score=0.90,
                max_concurrency=8,
            ),
            ModelCandidate(
                name="claude-2",
                tier=ModelTier.PREMIUM,
                cost_per_1k=2.50,
                avg_latency_ms=600,
                quality_score=0.92,
                max_concurrency=6,
            ),
            # Ultra tier - best quality
            ModelCandidate(
                name="gpt-4",
                tier=ModelTier.ULTRA,
                cost_per_1k=30.00,
                avg_latency_ms=2000,
                quality_score=0.98,
                max_concurrency=4,
            ),
        ]

    def set_budget(self, budget: float) -> None:
        """Set cost budget for model selection."""
        self.current_budget = budget
        self.spent_budget = 0.0
        logger.info(f"Budget set to ${budget:.2f}")

    def record_cost(self, cost: float) -> None:
        """Record spent cost against budget."""
        self.spent_budget += cost

    @property
    def remaining_budget(self) -> float | None:
        """Get remaining budget."""
        if self.current_budget is None:
            return None
        return self.current_budget - self.spent_budget

    def select_model(
        self,
        strategy: SelectionStrategy = SelectionStrategy.BALANCED,
        min_quality: float = 0.0,
        max_latency_ms: float | None = None,
        max_cost_per_1k: float | None = None,
        tier: ModelTier | None = None,
    ) -> ModelCandidate:
        """
        Select the best model based on strategy and constraints.

        Args:
            strategy: Selection strategy to use
            min_quality: Minimum acceptable quality score
            max_latency_ms: Maximum acceptable latency
            max_cost_per_1k: Maximum cost per 1k requests
            tier: Specific tier to select from

        Returns:
            Selected model candidate

        Raises:
            ValueError: If no models meet the constraints
        """
        # Filter candidates by constraints
        candidates = self._filter_candidates(
            min_quality=min_quality,
            max_latency_ms=max_latency_ms,
            max_cost_per_1k=max_cost_per_1k,
            tier=tier,
        )

        if not candidates:
            raise ValueError("No models meet the specified constraints")

        # Select based on strategy
        if strategy == SelectionStrategy.MINIMIZE_COST:
            selected = min(candidates, key=lambda m: m.cost_per_1k)
        elif strategy == SelectionStrategy.MINIMIZE_LATENCY:
            selected = min(candidates, key=lambda m: m.avg_latency_ms)
        elif strategy == SelectionStrategy.MAXIMIZE_QUALITY:
            selected = max(candidates, key=lambda m: m.quality_score)
        elif strategy == SelectionStrategy.BALANCED:
            selected = max(candidates, key=lambda m: m.efficiency_score)
        elif strategy == SelectionStrategy.ADAPTIVE:
            selected = self._adaptive_selection(candidates)
        else:
            selected = candidates[0]

        logger.info(
            f"Selected {selected.name} ({selected.tier.value}): "
            f"quality={selected.quality_score:.2f}, "
            f"latency={selected.avg_latency_ms:.0f}ms, "
            f"cost=${selected.cost_per_1k:.2f}/1k"
        )

        return selected

    def _filter_candidates(
        self,
        min_quality: float = 0.0,
        max_latency_ms: float | None = None,
        max_cost_per_1k: float | None = None,
        tier: ModelTier | None = None,
    ) -> list[ModelCandidate]:
        """Filter candidates by constraints."""
        candidates = []

        for model in self.models:
            # Quality constraint
            if model.quality_score < min_quality:
                continue

            # Latency constraint
            if max_latency_ms and model.avg_latency_ms > max_latency_ms:
                continue

            # Cost constraint
            if max_cost_per_1k and model.cost_per_1k > max_cost_per_1k:
                continue

            # Tier constraint
            if tier and model.tier != tier:
                continue

            # Budget constraint
            if self.remaining_budget is not None:
                estimated_cost = model.cost_per_1k / 1000  # Cost per single request
                if estimated_cost > self.remaining_budget:
                    continue

            candidates.append(model)

        return candidates

    def _adaptive_selection(self, candidates: list[ModelCandidate]) -> ModelCandidate:
        """
        Adaptive selection based on current conditions.

        Considers budget remaining, current load, and quality requirements.
        """
        # If budget is running low, prefer cheaper models
        if self.remaining_budget is not None and self.current_budget:
            budget_ratio = self.remaining_budget / self.current_budget

            if budget_ratio < 0.2:  # Less than 20% budget remaining
                logger.info("Low budget remaining, selecting cost-efficient model")
                return min(candidates, key=lambda m: m.cost_per_1k)

        # Otherwise optimize for efficiency
        return max(candidates, key=lambda m: m.efficiency_score)

    def get_tier_models(self, tier: ModelTier) -> list[ModelCandidate]:
        """Get all models in a specific tier."""
        return [m for m in self.models if m.tier == tier]

    def get_comparison_matrix(self) -> dict[str, Any]:
        """
        Get quality/latency/cost comparison matrix.

        Returns:
            Dictionary with comparison data for all models
        """
        matrix: dict[str, Any] = {
            "models": [],
            "tiers": {},
        }

        # Per-model data
        for model in sorted(self.models, key=lambda m: m.tier.value):
            matrix["models"].append(
                {
                    "name": model.name,
                    "tier": model.tier.value,
                    "quality": round(model.quality_score, 2),
                    "latency_ms": round(model.avg_latency_ms, 1),
                    "cost_per_1k": round(model.cost_per_1k, 2),
                    "efficiency": round(model.efficiency_score, 2),
                    "max_concurrency": model.max_concurrency,
                }
            )

        # Per-tier summary
        for tier in ModelTier:
            tier_models = self.get_tier_models(tier)
            if tier_models:
                matrix["tiers"][tier.value] = {
                    "count": len(tier_models),
                    "avg_quality": round(
                        sum(m.quality_score for m in tier_models) / len(tier_models), 2
                    ),
                    "avg_latency_ms": round(
                        sum(m.avg_latency_ms for m in tier_models) / len(tier_models), 1
                    ),
                    "avg_cost_per_1k": round(
                        sum(m.cost_per_1k for m in tier_models) / len(tier_models), 2
                    ),
                }

        return matrix

    def print_comparison_matrix(self) -> None:
        """Print quality/latency/cost comparison matrix."""
        print("\n" + "=" * 80)
        print("MODEL COMPARISON MATRIX")
        print("=" * 80)
        print(
            f"{'Model':<20} {'Tier':<10} {'Quality':<10} {'Latency':<12} {'Cost/1k':<10} {'Efficiency':<10}"
        )
        print("-" * 80)

        for model in sorted(self.models, key=lambda m: (m.tier.value, m.cost_per_1k)):
            print(
                f"{model.name:<20} "
                f"{model.tier.value:<10} "
                f"{model.quality_score:<10.2f} "
                f"{model.avg_latency_ms:<12.0f} "
                f"${model.cost_per_1k:<9.2f} "
                f"{model.efficiency_score:<10.2f}"
            )

        print("\nTier Summary:")
        print("-" * 80)

        for tier in ModelTier:
            tier_models = self.get_tier_models(tier)
            if tier_models:
                avg_quality = sum(m.quality_score for m in tier_models) / len(tier_models)
                avg_latency = sum(m.avg_latency_ms for m in tier_models) / len(tier_models)
                avg_cost = sum(m.cost_per_1k for m in tier_models) / len(tier_models)

                print(
                    f"{tier.value.upper():<20} "
                    f"Models: {len(tier_models):<3} "
                    f"Avg Quality: {avg_quality:.2f}  "
                    f"Avg Latency: {avg_latency:.0f}ms  "
                    f"Avg Cost: ${avg_cost:.2f}/1k"
                )
