"""
Experiment definitions and management for A/B testing.

Provides experiment creation, variant assignment, and lifecycle management
for comparing prompts, models, and other RAG configurations.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class ExperimentStatus(str, Enum):
    """Experiment lifecycle status."""

    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ExperimentType(str, Enum):
    """Type of experiment."""

    PROMPT = "prompt"
    MODEL = "model"
    RETRIEVAL = "retrieval"
    RERANKER = "reranker"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    CUSTOM = "custom"


@dataclass
class Variant:
    """A variant in an A/B experiment."""

    name: str
    config: dict[str, Any]
    weight: float = 1.0
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert variant to dictionary."""
        return {
            "name": self.name,
            "config": self.config,
            "weight": self.weight,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Variant:
        """Create variant from dictionary."""
        return cls(
            name=data["name"],
            config=data["config"],
            weight=data.get("weight", 1.0),
            description=data.get("description", ""),
        )


@dataclass
class Assignment:
    """Assignment of a user/request to a variant."""

    experiment_id: str
    variant_name: str
    user_id: str
    assigned_at: datetime = field(default_factory=datetime.utcnow)
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert assignment to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "variant_name": self.variant_name,
            "user_id": self.user_id,
            "assigned_at": self.assigned_at.isoformat(),
            "context": self.context,
        }


@dataclass
class Experiment:
    """An A/B experiment definition."""

    id: str
    name: str
    experiment_type: ExperimentType
    variants: list[Variant]
    status: ExperimentStatus = ExperimentStatus.DRAFT
    description: str = ""
    owner: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    ended_at: datetime | None = None
    target_sample_size: int = 1000
    min_runtime_hours: int = 24
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate experiment after initialization."""
        if not self.variants:
            raise ValueError("Experiment must have at least one variant")
        if len(self.variants) < 2:
            raise ValueError("A/B experiment must have at least 2 variants")

    def get_variant_weights(self) -> dict[str, float]:
        """Get normalized variant weights."""
        total = sum(v.weight for v in self.variants)
        return {v.name: v.weight / total for v in self.variants}

    def get_variant(self, name: str) -> Variant | None:
        """Get variant by name."""
        for variant in self.variants:
            if variant.name == name:
                return variant
        return None

    def start(self) -> None:
        """Start the experiment."""
        if self.status != ExperimentStatus.DRAFT:
            raise ValueError(f"Cannot start experiment in {self.status} status")
        self.status = ExperimentStatus.RUNNING
        self.started_at = datetime.utcnow()

    def pause(self) -> None:
        """Pause the experiment."""
        if self.status != ExperimentStatus.RUNNING:
            raise ValueError(f"Cannot pause experiment in {self.status} status")
        self.status = ExperimentStatus.PAUSED

    def resume(self) -> None:
        """Resume the experiment."""
        if self.status != ExperimentStatus.PAUSED:
            raise ValueError(f"Cannot resume experiment in {self.status} status")
        self.status = ExperimentStatus.RUNNING

    def complete(self) -> None:
        """Complete the experiment."""
        if self.status not in (ExperimentStatus.RUNNING, ExperimentStatus.PAUSED):
            raise ValueError(f"Cannot complete experiment in {self.status} status")
        self.status = ExperimentStatus.COMPLETED
        self.ended_at = datetime.utcnow()

    def cancel(self) -> None:
        """Cancel the experiment."""
        if self.status == ExperimentStatus.COMPLETED:
            raise ValueError("Cannot cancel completed experiment")
        self.status = ExperimentStatus.CANCELLED
        self.ended_at = datetime.utcnow()

    def to_dict(self) -> dict[str, Any]:
        """Convert experiment to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "experiment_type": self.experiment_type.value,
            "variants": [v.to_dict() for v in self.variants],
            "status": self.status.value,
            "description": self.description,
            "owner": self.owner,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "target_sample_size": self.target_sample_size,
            "min_runtime_hours": self.min_runtime_hours,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Experiment:
        """Create experiment from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            experiment_type=ExperimentType(data["experiment_type"]),
            variants=[Variant.from_dict(v) for v in data["variants"]],
            status=ExperimentStatus(data.get("status", "draft")),
            description=data.get("description", ""),
            owner=data.get("owner", ""),
            created_at=datetime.fromisoformat(data["created_at"])
            if "created_at" in data
            else datetime.utcnow(),
            started_at=datetime.fromisoformat(data["started_at"])
            if data.get("started_at")
            else None,
            ended_at=datetime.fromisoformat(data["ended_at"]) if data.get("ended_at") else None,
            target_sample_size=data.get("target_sample_size", 1000),
            min_runtime_hours=data.get("min_runtime_hours", 24),
            metadata=data.get("metadata", {}),
        )


class ExperimentAssigner:
    """Handles deterministic assignment of users to variants."""

    def __init__(self, salt: str = "rag-experiment"):
        """Initialize assigner with salt for hashing."""
        self.salt = salt

    def assign(self, experiment: Experiment, user_id: str) -> Assignment:
        """
        Assign a user to a variant deterministically.

        Uses consistent hashing to ensure the same user always gets
        the same variant for a given experiment.
        """
        if experiment.status != ExperimentStatus.RUNNING:
            raise ValueError(f"Cannot assign to experiment in {experiment.status} status")

        # Create deterministic hash
        hash_input = f"{self.salt}:{experiment.id}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        bucket = (hash_value % 10000) / 10000.0  # 0.0 to 1.0

        # Assign to variant based on weights
        weights = experiment.get_variant_weights()
        cumulative = 0.0
        assigned_variant = experiment.variants[0].name  # Default to first

        for variant_name, weight in weights.items():
            cumulative += weight
            if bucket < cumulative:
                assigned_variant = variant_name
                break

        return Assignment(
            experiment_id=experiment.id,
            variant_name=assigned_variant,
            user_id=user_id,
        )


class ExperimentManager:
    """Manages experiments and assignments."""

    def __init__(
        self,
        storage_path: str | None = None,
        assigner: ExperimentAssigner | None = None,
    ):
        """Initialize experiment manager."""
        self.storage_path = Path(storage_path) if storage_path else None
        self.assigner = assigner or ExperimentAssigner()
        self._experiments: dict[str, Experiment] = {}
        self._assignments: dict[str, list[Assignment]] = {}
        self._outcomes: dict[str, list[dict[str, Any]]] = {}

        if self.storage_path:
            self._load_experiments()

    def _load_experiments(self) -> None:
        """Load experiments from storage."""
        if not self.storage_path or not self.storage_path.exists():
            return

        experiments_file = self.storage_path / "experiments.json"
        if experiments_file.exists():
            with experiments_file.open() as f:
                data = json.load(f)
                for exp_data in data.get("experiments", []):
                    exp = Experiment.from_dict(exp_data)
                    self._experiments[exp.id] = exp

    def _save_experiments(self) -> None:
        """Save experiments to storage."""
        if not self.storage_path:
            return

        self.storage_path.mkdir(parents=True, exist_ok=True)
        experiments_file = self.storage_path / "experiments.json"

        data = {"experiments": [exp.to_dict() for exp in self._experiments.values()]}
        with experiments_file.open("w") as f:
            json.dump(data, f, indent=2)

    def create_experiment(
        self,
        name: str,
        experiment_type: ExperimentType,
        variants: list[Variant],
        description: str = "",
        owner: str = "",
        target_sample_size: int = 1000,
        min_runtime_hours: int = 24,
        metadata: dict[str, Any] | None = None,
    ) -> Experiment:
        """Create a new experiment."""
        experiment = Experiment(
            id=str(uuid.uuid4()),
            name=name,
            experiment_type=experiment_type,
            variants=variants,
            description=description,
            owner=owner,
            target_sample_size=target_sample_size,
            min_runtime_hours=min_runtime_hours,
            metadata=metadata or {},
        )
        self._experiments[experiment.id] = experiment
        self._assignments[experiment.id] = []
        self._outcomes[experiment.id] = []
        self._save_experiments()
        return experiment

    def get_experiment(self, experiment_id: str) -> Experiment | None:
        """Get experiment by ID."""
        return self._experiments.get(experiment_id)

    def get_experiment_by_name(self, name: str) -> Experiment | None:
        """Get experiment by name."""
        for exp in self._experiments.values():
            if exp.name == name:
                return exp
        return None

    def list_experiments(
        self,
        status: ExperimentStatus | None = None,
        experiment_type: ExperimentType | None = None,
    ) -> list[Experiment]:
        """List experiments with optional filters."""
        experiments = list(self._experiments.values())

        if status:
            experiments = [e for e in experiments if e.status == status]
        if experiment_type:
            experiments = [e for e in experiments if e.experiment_type == experiment_type]

        return experiments

    def start_experiment(self, experiment_id: str) -> Experiment:
        """Start an experiment."""
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment.start()
        self._save_experiments()
        return experiment

    def pause_experiment(self, experiment_id: str) -> Experiment:
        """Pause an experiment."""
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment.pause()
        self._save_experiments()
        return experiment

    def complete_experiment(self, experiment_id: str) -> Experiment:
        """Complete an experiment."""
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment.complete()
        self._save_experiments()
        return experiment

    def assign_user(self, experiment_id: str, user_id: str) -> Assignment:
        """Assign a user to an experiment variant."""
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        # Check for existing assignment
        existing = self.get_assignment(experiment_id, user_id)
        if existing:
            return existing

        assignment = self.assigner.assign(experiment, user_id)
        self._assignments.setdefault(experiment_id, []).append(assignment)
        return assignment

    def get_assignment(self, experiment_id: str, user_id: str) -> Assignment | None:
        """Get existing assignment for a user."""
        assignments = self._assignments.get(experiment_id, [])
        for assignment in assignments:
            if assignment.user_id == user_id:
                return assignment
        return None

    def record_outcome(
        self,
        experiment_id: str,
        user_id: str,
        metric_name: str,
        metric_value: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record an outcome for an experiment assignment."""
        assignment = self.get_assignment(experiment_id, user_id)
        if not assignment:
            raise ValueError(f"No assignment found for user {user_id}")

        outcome = {
            "experiment_id": experiment_id,
            "variant_name": assignment.variant_name,
            "user_id": user_id,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "recorded_at": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }
        self._outcomes.setdefault(experiment_id, []).append(outcome)

    def get_outcomes(
        self,
        experiment_id: str,
        variant_name: str | None = None,
        metric_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get outcomes for an experiment."""
        outcomes = self._outcomes.get(experiment_id, [])

        if variant_name:
            outcomes = [o for o in outcomes if o["variant_name"] == variant_name]
        if metric_name:
            outcomes = [o for o in outcomes if o["metric_name"] == metric_name]

        return outcomes

    def get_experiment_stats(self, experiment_id: str) -> dict[str, Any]:
        """Get statistics for an experiment."""
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        assignments = self._assignments.get(experiment_id, [])
        outcomes = self._outcomes.get(experiment_id, [])

        # Count assignments per variant
        variant_counts: dict[str, int] = {}
        for assignment in assignments:
            variant_counts[assignment.variant_name] = (
                variant_counts.get(assignment.variant_name, 0) + 1
            )

        # Calculate metrics per variant
        variant_metrics: dict[str, dict[str, list[float]]] = {}
        for outcome in outcomes:
            variant = outcome["variant_name"]
            metric = outcome["metric_name"]
            value = outcome["metric_value"]

            if variant not in variant_metrics:
                variant_metrics[variant] = {}
            if metric not in variant_metrics[variant]:
                variant_metrics[variant][metric] = []
            variant_metrics[variant][metric].append(value)

        return {
            "experiment_id": experiment_id,
            "status": experiment.status.value,
            "total_assignments": len(assignments),
            "total_outcomes": len(outcomes),
            "variant_counts": variant_counts,
            "variant_metrics": variant_metrics,
        }


# Convenience functions for creating common experiment types
def create_prompt_experiment(
    manager: ExperimentManager,
    name: str,
    control_prompt: str,
    treatment_prompt: str,
    **kwargs: Any,
) -> Experiment:
    """Create an A/B experiment for prompts."""
    variants = [
        Variant(
            name="control",
            config={"prompt": control_prompt},
            description="Control prompt",
        ),
        Variant(
            name="treatment",
            config={"prompt": treatment_prompt},
            description="Treatment prompt",
        ),
    ]
    return manager.create_experiment(
        name=name,
        experiment_type=ExperimentType.PROMPT,
        variants=variants,
        **kwargs,
    )


def create_model_experiment(
    manager: ExperimentManager,
    name: str,
    models: list[dict[str, Any]],
    **kwargs: Any,
) -> Experiment:
    """Create an A/B experiment for models."""
    variants = [
        Variant(
            name=model.get("name", f"model_{i}"),
            config=model,
            weight=model.get("weight", 1.0),
        )
        for i, model in enumerate(models)
    ]
    return manager.create_experiment(
        name=name,
        experiment_type=ExperimentType.MODEL,
        variants=variants,
        **kwargs,
    )


# Global experiment manager
_experiment_manager: ExperimentManager | None = None


def get_experiment_manager() -> ExperimentManager:
    """Get the global experiment manager."""
    global _experiment_manager
    if _experiment_manager is None:
        _experiment_manager = ExperimentManager()
    return _experiment_manager


def set_experiment_manager(manager: ExperimentManager) -> None:
    """Set the global experiment manager."""
    global _experiment_manager
    _experiment_manager = manager
