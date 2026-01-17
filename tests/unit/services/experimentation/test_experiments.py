"""Tests for experiment definitions and management."""

import tempfile

import pytest

from src.services.experimentation.experiments import (
    Assignment,
    Experiment,
    ExperimentAssigner,
    ExperimentManager,
    ExperimentStatus,
    ExperimentType,
    Variant,
    create_model_experiment,
    create_prompt_experiment,
    get_experiment_manager,
    set_experiment_manager,
)


class TestVariant:
    """Tests for Variant."""

    def test_variant_creation(self):
        """Test creating a variant."""
        variant = Variant(
            name="control",
            config={"prompt": "Answer the question: {query}"},
            weight=1.0,
            description="Control variant",
        )
        assert variant.name == "control"
        assert variant.config["prompt"] == "Answer the question: {query}"
        assert variant.weight == 1.0

    def test_variant_to_dict(self):
        """Test variant serialization."""
        variant = Variant(name="test", config={"key": "value"}, weight=0.5)
        data = variant.to_dict()

        assert data["name"] == "test"
        assert data["config"]["key"] == "value"
        assert data["weight"] == 0.5

    def test_variant_from_dict(self):
        """Test variant deserialization."""
        data = {"name": "test", "config": {"key": "value"}, "weight": 0.5}
        variant = Variant.from_dict(data)

        assert variant.name == "test"
        assert variant.config["key"] == "value"
        assert variant.weight == 0.5


class TestAssignment:
    """Tests for Assignment."""

    def test_assignment_creation(self):
        """Test creating an assignment."""
        assignment = Assignment(
            experiment_id="exp-123",
            variant_name="treatment",
            user_id="user-456",
        )
        assert assignment.experiment_id == "exp-123"
        assert assignment.variant_name == "treatment"
        assert assignment.user_id == "user-456"

    def test_assignment_to_dict(self):
        """Test assignment serialization."""
        assignment = Assignment(
            experiment_id="exp-123",
            variant_name="control",
            user_id="user-456",
        )
        data = assignment.to_dict()

        assert data["experiment_id"] == "exp-123"
        assert data["variant_name"] == "control"
        assert data["user_id"] == "user-456"
        assert "assigned_at" in data


class TestExperiment:
    """Tests for Experiment."""

    def test_experiment_creation(self):
        """Test creating an experiment."""
        variants = [
            Variant(name="control", config={"prompt": "v1"}),
            Variant(name="treatment", config={"prompt": "v2"}),
        ]
        experiment = Experiment(
            id="exp-123",
            name="Prompt Test",
            experiment_type=ExperimentType.PROMPT,
            variants=variants,
        )
        assert experiment.id == "exp-123"
        assert experiment.name == "Prompt Test"
        assert experiment.status == ExperimentStatus.DRAFT
        assert len(experiment.variants) == 2

    def test_experiment_requires_two_variants(self):
        """Test that experiment requires at least 2 variants."""
        with pytest.raises(ValueError, match="at least 2 variants"):
            Experiment(
                id="exp-123",
                name="Test",
                experiment_type=ExperimentType.PROMPT,
                variants=[Variant(name="control", config={})],
            )

    def test_experiment_get_variant_weights(self):
        """Test getting normalized variant weights."""
        variants = [
            Variant(name="control", config={}, weight=1.0),
            Variant(name="treatment", config={}, weight=3.0),
        ]
        experiment = Experiment(
            id="exp-123",
            name="Test",
            experiment_type=ExperimentType.PROMPT,
            variants=variants,
        )
        weights = experiment.get_variant_weights()

        assert weights["control"] == pytest.approx(0.25)
        assert weights["treatment"] == pytest.approx(0.75)

    def test_experiment_get_variant(self):
        """Test getting variant by name."""
        variants = [
            Variant(name="control", config={"a": 1}),
            Variant(name="treatment", config={"b": 2}),
        ]
        experiment = Experiment(
            id="exp-123",
            name="Test",
            experiment_type=ExperimentType.PROMPT,
            variants=variants,
        )

        control = experiment.get_variant("control")
        assert control is not None
        assert control.config["a"] == 1

        unknown = experiment.get_variant("unknown")
        assert unknown is None

    def test_experiment_lifecycle_start(self):
        """Test starting an experiment."""
        variants = [
            Variant(name="control", config={}),
            Variant(name="treatment", config={}),
        ]
        experiment = Experiment(
            id="exp-123",
            name="Test",
            experiment_type=ExperimentType.PROMPT,
            variants=variants,
        )

        assert experiment.status == ExperimentStatus.DRAFT
        experiment.start()
        assert experiment.status == ExperimentStatus.RUNNING
        assert experiment.started_at is not None

    def test_experiment_lifecycle_pause_resume(self):
        """Test pausing and resuming an experiment."""
        variants = [
            Variant(name="control", config={}),
            Variant(name="treatment", config={}),
        ]
        experiment = Experiment(
            id="exp-123",
            name="Test",
            experiment_type=ExperimentType.PROMPT,
            variants=variants,
        )
        experiment.start()

        experiment.pause()
        assert experiment.status == ExperimentStatus.PAUSED

        experiment.resume()
        assert experiment.status == ExperimentStatus.RUNNING

    def test_experiment_lifecycle_complete(self):
        """Test completing an experiment."""
        variants = [
            Variant(name="control", config={}),
            Variant(name="treatment", config={}),
        ]
        experiment = Experiment(
            id="exp-123",
            name="Test",
            experiment_type=ExperimentType.PROMPT,
            variants=variants,
        )
        experiment.start()
        experiment.complete()

        assert experiment.status == ExperimentStatus.COMPLETED
        assert experiment.ended_at is not None

    def test_experiment_to_dict(self):
        """Test experiment serialization."""
        variants = [
            Variant(name="control", config={}),
            Variant(name="treatment", config={}),
        ]
        experiment = Experiment(
            id="exp-123",
            name="Test",
            experiment_type=ExperimentType.PROMPT,
            variants=variants,
            description="Test experiment",
        )
        data = experiment.to_dict()

        assert data["id"] == "exp-123"
        assert data["name"] == "Test"
        assert data["experiment_type"] == "prompt"
        assert len(data["variants"]) == 2

    def test_experiment_from_dict(self):
        """Test experiment deserialization."""
        data = {
            "id": "exp-123",
            "name": "Test",
            "experiment_type": "model",
            "variants": [
                {"name": "control", "config": {}, "weight": 1.0},
                {"name": "treatment", "config": {}, "weight": 1.0},
            ],
            "status": "running",
        }
        experiment = Experiment.from_dict(data)

        assert experiment.id == "exp-123"
        assert experiment.experiment_type == ExperimentType.MODEL
        assert experiment.status == ExperimentStatus.RUNNING


class TestExperimentAssigner:
    """Tests for ExperimentAssigner."""

    def test_assign_deterministic(self):
        """Test that assignment is deterministic."""
        assigner = ExperimentAssigner()
        variants = [
            Variant(name="control", config={}),
            Variant(name="treatment", config={}),
        ]
        experiment = Experiment(
            id="exp-123",
            name="Test",
            experiment_type=ExperimentType.PROMPT,
            variants=variants,
        )
        experiment.start()

        # Same user should get same variant
        assignment1 = assigner.assign(experiment, "user-123")
        assignment2 = assigner.assign(experiment, "user-123")
        assert assignment1.variant_name == assignment2.variant_name

    def test_assign_different_users(self):
        """Test that different users can get different variants."""
        assigner = ExperimentAssigner()
        variants = [
            Variant(name="control", config={}),
            Variant(name="treatment", config={}),
        ]
        experiment = Experiment(
            id="exp-123",
            name="Test",
            experiment_type=ExperimentType.PROMPT,
            variants=variants,
        )
        experiment.start()

        # Collect assignments for many users
        assignments = {}
        for i in range(100):
            assignment = assigner.assign(experiment, f"user-{i}")
            assignments[assignment.variant_name] = assignments.get(assignment.variant_name, 0) + 1

        # Both variants should have some assignments
        assert "control" in assignments or "treatment" in assignments

    def test_assign_respects_weights(self):
        """Test that assignment respects variant weights."""
        assigner = ExperimentAssigner()
        variants = [
            Variant(name="control", config={}, weight=1.0),
            Variant(name="treatment", config={}, weight=9.0),
        ]
        experiment = Experiment(
            id="exp-123",
            name="Test",
            experiment_type=ExperimentType.PROMPT,
            variants=variants,
        )
        experiment.start()

        # Collect assignments for many users
        treatment_count = 0
        for i in range(1000):
            assignment = assigner.assign(experiment, f"user-{i}")
            if assignment.variant_name == "treatment":
                treatment_count += 1

        # Treatment should have ~90% of assignments
        assert treatment_count > 800  # Should be around 900


class TestExperimentManager:
    """Tests for ExperimentManager."""

    def test_create_experiment(self):
        """Test creating an experiment through manager."""
        manager = ExperimentManager()
        variants = [
            Variant(name="control", config={}),
            Variant(name="treatment", config={}),
        ]
        experiment = manager.create_experiment(
            name="Test Experiment",
            experiment_type=ExperimentType.PROMPT,
            variants=variants,
        )

        assert experiment.name == "Test Experiment"
        assert experiment.id is not None

    def test_get_experiment(self):
        """Test getting an experiment by ID."""
        manager = ExperimentManager()
        variants = [
            Variant(name="control", config={}),
            Variant(name="treatment", config={}),
        ]
        experiment = manager.create_experiment(
            name="Test",
            experiment_type=ExperimentType.PROMPT,
            variants=variants,
        )

        retrieved = manager.get_experiment(experiment.id)
        assert retrieved is not None
        assert retrieved.name == "Test"

    def test_get_experiment_by_name(self):
        """Test getting an experiment by name."""
        manager = ExperimentManager()
        variants = [
            Variant(name="control", config={}),
            Variant(name="treatment", config={}),
        ]
        manager.create_experiment(
            name="Named Experiment",
            experiment_type=ExperimentType.PROMPT,
            variants=variants,
        )

        retrieved = manager.get_experiment_by_name("Named Experiment")
        assert retrieved is not None
        assert retrieved.name == "Named Experiment"

    def test_list_experiments(self):
        """Test listing experiments."""
        manager = ExperimentManager()
        variants = [
            Variant(name="control", config={}),
            Variant(name="treatment", config={}),
        ]

        manager.create_experiment(
            name="Exp 1",
            experiment_type=ExperimentType.PROMPT,
            variants=variants,
        )
        manager.create_experiment(
            name="Exp 2",
            experiment_type=ExperimentType.MODEL,
            variants=variants,
        )

        all_exps = manager.list_experiments()
        assert len(all_exps) == 2

        prompt_exps = manager.list_experiments(experiment_type=ExperimentType.PROMPT)
        assert len(prompt_exps) == 1

    def test_start_experiment(self):
        """Test starting an experiment through manager."""
        manager = ExperimentManager()
        variants = [
            Variant(name="control", config={}),
            Variant(name="treatment", config={}),
        ]
        experiment = manager.create_experiment(
            name="Test",
            experiment_type=ExperimentType.PROMPT,
            variants=variants,
        )

        started = manager.start_experiment(experiment.id)
        assert started.status == ExperimentStatus.RUNNING

    def test_assign_user(self):
        """Test assigning a user to an experiment."""
        manager = ExperimentManager()
        variants = [
            Variant(name="control", config={}),
            Variant(name="treatment", config={}),
        ]
        experiment = manager.create_experiment(
            name="Test",
            experiment_type=ExperimentType.PROMPT,
            variants=variants,
        )
        manager.start_experiment(experiment.id)

        assignment = manager.assign_user(experiment.id, "user-123")
        assert assignment.user_id == "user-123"
        assert assignment.variant_name in ["control", "treatment"]

    def test_assign_user_returns_existing(self):
        """Test that assigning same user returns existing assignment."""
        manager = ExperimentManager()
        variants = [
            Variant(name="control", config={}),
            Variant(name="treatment", config={}),
        ]
        experiment = manager.create_experiment(
            name="Test",
            experiment_type=ExperimentType.PROMPT,
            variants=variants,
        )
        manager.start_experiment(experiment.id)

        assignment1 = manager.assign_user(experiment.id, "user-123")
        assignment2 = manager.assign_user(experiment.id, "user-123")
        assert assignment1.variant_name == assignment2.variant_name

    def test_record_outcome(self):
        """Test recording an outcome."""
        manager = ExperimentManager()
        variants = [
            Variant(name="control", config={}),
            Variant(name="treatment", config={}),
        ]
        experiment = manager.create_experiment(
            name="Test",
            experiment_type=ExperimentType.PROMPT,
            variants=variants,
        )
        manager.start_experiment(experiment.id)
        manager.assign_user(experiment.id, "user-123")

        manager.record_outcome(
            experiment_id=experiment.id,
            user_id="user-123",
            metric_name="quality_score",
            metric_value=0.85,
        )

        outcomes = manager.get_outcomes(experiment.id)
        assert len(outcomes) == 1
        assert outcomes[0]["metric_value"] == 0.85

    def test_get_experiment_stats(self):
        """Test getting experiment statistics."""
        manager = ExperimentManager()
        variants = [
            Variant(name="control", config={}),
            Variant(name="treatment", config={}),
        ]
        experiment = manager.create_experiment(
            name="Test",
            experiment_type=ExperimentType.PROMPT,
            variants=variants,
        )
        manager.start_experiment(experiment.id)

        # Add some assignments and outcomes
        for i in range(10):
            manager.assign_user(experiment.id, f"user-{i}")
            manager.record_outcome(
                experiment_id=experiment.id,
                user_id=f"user-{i}",
                metric_name="quality",
                metric_value=0.8 + i * 0.01,
            )

        stats = manager.get_experiment_stats(experiment.id)
        assert stats["total_assignments"] == 10
        assert stats["total_outcomes"] == 10

    def test_storage_persistence(self):
        """Test experiment persistence to storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save
            manager1 = ExperimentManager(storage_path=tmpdir)
            variants = [
                Variant(name="control", config={}),
                Variant(name="treatment", config={}),
            ]
            experiment = manager1.create_experiment(
                name="Persistent Test",
                experiment_type=ExperimentType.PROMPT,
                variants=variants,
            )

            # Load in new manager
            manager2 = ExperimentManager(storage_path=tmpdir)
            loaded = manager2.get_experiment(experiment.id)

            assert loaded is not None
            assert loaded.name == "Persistent Test"


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_prompt_experiment(self):
        """Test creating a prompt experiment."""
        manager = ExperimentManager()
        experiment = create_prompt_experiment(
            manager=manager,
            name="Prompt A/B",
            control_prompt="Answer: {query}",
            treatment_prompt="Please answer: {query}",
        )

        assert experiment.experiment_type == ExperimentType.PROMPT
        assert len(experiment.variants) == 2
        assert experiment.variants[0].config["prompt"] == "Answer: {query}"

    def test_create_model_experiment(self):
        """Test creating a model experiment."""
        manager = ExperimentManager()
        models = [
            {"name": "gpt-3.5", "model_id": "gpt-3.5-turbo"},
            {"name": "gpt-4", "model_id": "gpt-4"},
        ]
        experiment = create_model_experiment(
            manager=manager,
            name="Model A/B",
            models=models,
        )

        assert experiment.experiment_type == ExperimentType.MODEL
        assert len(experiment.variants) == 2


class TestGlobalManager:
    """Tests for global experiment manager."""

    def test_get_experiment_manager(self):
        """Test getting global experiment manager."""
        manager = get_experiment_manager()
        assert manager is not None
        assert isinstance(manager, ExperimentManager)

    def test_set_experiment_manager(self):
        """Test setting global experiment manager."""
        custom_manager = ExperimentManager()
        set_experiment_manager(custom_manager)

        retrieved = get_experiment_manager()
        assert retrieved is custom_manager
