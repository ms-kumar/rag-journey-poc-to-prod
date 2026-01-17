"""Tests for feature flags module."""

import tempfile

import pytest

from src.services.experimentation.feature_flags import (
    FeatureFlag,
    FlagEvaluator,
    FlagManager,
    FlagStatus,
    FlagType,
    RolloutConfig,
    create_environment_flag,
    create_gradual_rollout_flag,
    create_user_targeting_flag,
    get_flag_manager,
    set_flag_manager,
)


class TestRolloutConfig:
    """Tests for RolloutConfig."""

    def test_default_config(self):
        """Test default rollout config."""
        config = RolloutConfig()
        assert config.percentage == 0.0
        assert config.enabled_users == []
        assert config.disabled_users == []

    def test_config_to_dict(self):
        """Test config serialization."""
        config = RolloutConfig(
            percentage=50.0,
            enabled_users=["user-1", "user-2"],
        )
        data = config.to_dict()

        assert data["percentage"] == 50.0
        assert "user-1" in data["enabled_users"]

    def test_config_from_dict(self):
        """Test config deserialization."""
        data = {
            "percentage": 75.0,
            "enabled_users": ["user-1"],
            "disabled_users": ["user-2"],
        }
        config = RolloutConfig.from_dict(data)

        assert config.percentage == 75.0
        assert "user-1" in config.enabled_users


class TestFeatureFlag:
    """Tests for FeatureFlag."""

    def test_flag_creation(self):
        """Test creating a feature flag."""
        flag = FeatureFlag(
            name="new_search",
            flag_type=FlagType.BOOLEAN,
            description="New search feature",
        )
        assert flag.name == "new_search"
        assert flag.flag_type == FlagType.BOOLEAN
        assert flag.status == FlagStatus.DISABLED

    def test_flag_enable_disable(self):
        """Test enabling and disabling a flag."""
        flag = FeatureFlag(name="test", flag_type=FlagType.BOOLEAN)

        flag.enable()
        assert flag.status == FlagStatus.ENABLED

        flag.disable()
        assert flag.status == FlagStatus.DISABLED

    def test_flag_archive(self):
        """Test archiving a flag."""
        flag = FeatureFlag(name="test", flag_type=FlagType.BOOLEAN)
        flag.archive()
        assert flag.status == FlagStatus.ARCHIVED

    def test_flag_set_percentage(self):
        """Test setting rollout percentage."""
        flag = FeatureFlag(name="test", flag_type=FlagType.PERCENTAGE)
        flag.set_percentage(50.0)
        assert flag.rollout.percentage == 50.0

    def test_flag_set_percentage_invalid(self):
        """Test setting invalid percentage."""
        flag = FeatureFlag(name="test", flag_type=FlagType.PERCENTAGE)
        with pytest.raises(ValueError):
            flag.set_percentage(150.0)

    def test_flag_add_user_enabled(self):
        """Test adding user to enabled list."""
        flag = FeatureFlag(name="test", flag_type=FlagType.USER_LIST)
        flag.add_user("user-123", enabled=True)

        assert "user-123" in flag.rollout.enabled_users
        assert "user-123" not in flag.rollout.disabled_users

    def test_flag_add_user_disabled(self):
        """Test adding user to disabled list."""
        flag = FeatureFlag(name="test", flag_type=FlagType.USER_LIST)
        flag.add_user("user-123", enabled=False)

        assert "user-123" in flag.rollout.disabled_users
        assert "user-123" not in flag.rollout.enabled_users

    def test_flag_remove_user(self):
        """Test removing user from lists."""
        flag = FeatureFlag(name="test", flag_type=FlagType.USER_LIST)
        flag.add_user("user-123", enabled=True)
        flag.remove_user("user-123")

        assert "user-123" not in flag.rollout.enabled_users
        assert "user-123" not in flag.rollout.disabled_users

    def test_flag_to_dict(self):
        """Test flag serialization."""
        flag = FeatureFlag(
            name="test",
            flag_type=FlagType.PERCENTAGE,
            description="Test flag",
        )
        data = flag.to_dict()

        assert data["name"] == "test"
        assert data["flag_type"] == "percentage"
        assert data["description"] == "Test flag"

    def test_flag_from_dict(self):
        """Test flag deserialization."""
        data = {
            "name": "test",
            "flag_type": "boolean",
            "status": "enabled",
            "description": "Test",
        }
        flag = FeatureFlag.from_dict(data)

        assert flag.name == "test"
        assert flag.flag_type == FlagType.BOOLEAN
        assert flag.status == FlagStatus.ENABLED


class TestFlagEvaluator:
    """Tests for FlagEvaluator."""

    def test_evaluate_disabled_flag(self):
        """Test evaluating a disabled flag."""
        evaluator = FlagEvaluator()
        flag = FeatureFlag(name="test", flag_type=FlagType.BOOLEAN, default_value=False)

        result = evaluator.evaluate(flag, user_id="user-123")
        assert result is False

    def test_evaluate_enabled_boolean_flag(self):
        """Test evaluating an enabled boolean flag."""
        evaluator = FlagEvaluator()
        flag = FeatureFlag(name="test", flag_type=FlagType.BOOLEAN)
        flag.enable()

        result = evaluator.evaluate(flag, user_id="user-123")
        assert result is True

    def test_evaluate_archived_flag(self):
        """Test evaluating an archived flag."""
        evaluator = FlagEvaluator()
        flag = FeatureFlag(name="test", flag_type=FlagType.BOOLEAN)
        flag.archive()

        result = evaluator.evaluate(flag, user_id="user-123")
        assert result is False

    def test_evaluate_user_in_disabled_list(self):
        """Test that disabled users don't get flag."""
        evaluator = FlagEvaluator()
        flag = FeatureFlag(name="test", flag_type=FlagType.BOOLEAN)
        flag.enable()
        flag.add_user("user-123", enabled=False)

        result = evaluator.evaluate(flag, user_id="user-123")
        assert result is False

    def test_evaluate_user_in_enabled_list(self):
        """Test that enabled users get flag."""
        evaluator = FlagEvaluator()
        flag = FeatureFlag(name="test", flag_type=FlagType.PERCENTAGE)
        flag.rollout.percentage = 0  # 0% rollout
        flag.add_user("user-123", enabled=True)
        flag.enable()

        result = evaluator.evaluate(flag, user_id="user-123")
        assert result is True

    def test_evaluate_percentage_deterministic(self):
        """Test that percentage evaluation is deterministic."""
        evaluator = FlagEvaluator()
        flag = FeatureFlag(name="test", flag_type=FlagType.PERCENTAGE)
        flag.set_percentage(50.0)
        flag.enable()

        # Same user should get same result
        result1 = evaluator.evaluate(flag, user_id="user-123")
        result2 = evaluator.evaluate(flag, user_id="user-123")
        assert result1 == result2

    def test_evaluate_percentage_distribution(self):
        """Test that percentage rollout distributes correctly."""
        evaluator = FlagEvaluator()
        flag = FeatureFlag(name="test", flag_type=FlagType.PERCENTAGE)
        flag.set_percentage(50.0)
        flag.enable()

        enabled_count = 0
        for i in range(1000):
            if evaluator.evaluate(flag, user_id=f"user-{i}"):
                enabled_count += 1

        # Should be roughly 50% (allow for variance)
        assert 400 < enabled_count < 600

    def test_evaluate_environment_flag(self):
        """Test environment-based flag."""
        evaluator = FlagEvaluator()
        flag = FeatureFlag(name="test", flag_type=FlagType.ENVIRONMENT)
        flag.rollout.enabled_environments = ["production", "staging"]
        flag.enable()

        assert evaluator.evaluate(flag, environment="production") is True
        assert evaluator.evaluate(flag, environment="development") is False

    def test_evaluate_user_list_flag(self):
        """Test user list flag."""
        evaluator = FlagEvaluator()
        flag = FeatureFlag(name="test", flag_type=FlagType.USER_LIST)
        flag.rollout.enabled_users = ["user-1", "user-2"]
        flag.enable()

        assert evaluator.evaluate(flag, user_id="user-1") is True
        assert evaluator.evaluate(flag, user_id="user-3") is False


class TestFlagManager:
    """Tests for FlagManager."""

    def test_create_flag(self):
        """Test creating a flag through manager."""
        manager = FlagManager()
        flag = manager.create_flag(
            name="new_feature",
            flag_type=FlagType.BOOLEAN,
            description="A new feature",
        )

        assert flag.name == "new_feature"
        assert flag.flag_type == FlagType.BOOLEAN

    def test_create_duplicate_flag(self):
        """Test that creating duplicate flag raises error."""
        manager = FlagManager()
        manager.create_flag(name="test", flag_type=FlagType.BOOLEAN)

        with pytest.raises(ValueError, match="already exists"):
            manager.create_flag(name="test", flag_type=FlagType.BOOLEAN)

    def test_get_flag(self):
        """Test getting a flag by name."""
        manager = FlagManager()
        manager.create_flag(name="test", flag_type=FlagType.BOOLEAN)

        flag = manager.get_flag("test")
        assert flag is not None
        assert flag.name == "test"

    def test_get_nonexistent_flag(self):
        """Test getting a nonexistent flag."""
        manager = FlagManager()
        flag = manager.get_flag("nonexistent")
        assert flag is None

    def test_list_flags(self):
        """Test listing flags."""
        manager = FlagManager()
        manager.create_flag(name="flag1", flag_type=FlagType.BOOLEAN)
        manager.create_flag(name="flag2", flag_type=FlagType.PERCENTAGE)

        all_flags = manager.list_flags()
        assert len(all_flags) == 2

        bool_flags = manager.list_flags(flag_type=FlagType.BOOLEAN)
        assert len(bool_flags) == 1

    def test_delete_flag(self):
        """Test deleting a flag."""
        manager = FlagManager()
        manager.create_flag(name="test", flag_type=FlagType.BOOLEAN)

        result = manager.delete_flag("test")
        assert result is True
        assert manager.get_flag("test") is None

    def test_enable_flag(self):
        """Test enabling a flag through manager."""
        manager = FlagManager()
        manager.create_flag(name="test", flag_type=FlagType.BOOLEAN)

        flag = manager.enable_flag("test")
        assert flag.status == FlagStatus.ENABLED

    def test_disable_flag(self):
        """Test disabling a flag through manager."""
        manager = FlagManager()
        manager.create_flag(name="test", flag_type=FlagType.BOOLEAN)
        manager.enable_flag("test")

        flag = manager.disable_flag("test")
        assert flag.status == FlagStatus.DISABLED

    def test_set_percentage(self):
        """Test setting percentage through manager."""
        manager = FlagManager()
        manager.create_flag(name="test", flag_type=FlagType.PERCENTAGE)

        flag = manager.set_percentage("test", 75.0)
        assert flag.rollout.percentage == 75.0

    def test_is_enabled(self):
        """Test is_enabled convenience method."""
        manager = FlagManager()
        manager.create_flag(name="test", flag_type=FlagType.BOOLEAN)

        assert manager.is_enabled("test") is False

        manager.enable_flag("test")
        assert manager.is_enabled("test") is True

    def test_is_enabled_nonexistent(self):
        """Test is_enabled for nonexistent flag."""
        manager = FlagManager()
        assert manager.is_enabled("nonexistent") is False

    def test_evaluation_log(self):
        """Test that evaluations are logged."""
        manager = FlagManager()
        manager.create_flag(name="test", flag_type=FlagType.BOOLEAN)
        manager.enable_flag("test")

        manager.is_enabled("test", user_id="user-123")

        log = manager.get_evaluation_log()
        assert len(log) == 1
        assert log[0]["flag_name"] == "test"
        assert log[0]["user_id"] == "user-123"

    def test_get_flag_stats(self):
        """Test getting flag statistics."""
        manager = FlagManager()
        manager.create_flag(name="test", flag_type=FlagType.BOOLEAN)
        manager.enable_flag("test")

        for i in range(10):
            manager.is_enabled("test", user_id=f"user-{i}")

        stats = manager.get_flag_stats("test")
        assert stats["total_evaluations"] == 10
        assert stats["unique_users"] == 10

    def test_storage_persistence(self):
        """Test flag persistence to storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save
            manager1 = FlagManager(storage_path=tmpdir)
            manager1.create_flag(
                name="persistent",
                flag_type=FlagType.PERCENTAGE,
                description="Test",
            )
            manager1.set_percentage("persistent", 50.0)

            # Load in new manager
            manager2 = FlagManager(storage_path=tmpdir)
            flag = manager2.get_flag("persistent")

            assert flag is not None
            assert flag.rollout.percentage == 50.0


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_gradual_rollout_flag(self):
        """Test creating gradual rollout flag."""
        manager = FlagManager()
        flag = create_gradual_rollout_flag(
            manager=manager,
            name="rollout_test",
            initial_percentage=10.0,
        )

        assert flag.flag_type == FlagType.PERCENTAGE
        assert flag.rollout.percentage == 10.0
        assert flag.status == FlagStatus.ENABLED

    def test_create_user_targeting_flag(self):
        """Test creating user targeting flag."""
        manager = FlagManager()
        flag = create_user_targeting_flag(
            manager=manager,
            name="beta_users",
            enabled_users=["user-1", "user-2"],
        )

        assert flag.flag_type == FlagType.USER_LIST
        assert "user-1" in flag.rollout.enabled_users

    def test_create_environment_flag(self):
        """Test creating environment flag."""
        manager = FlagManager()
        flag = create_environment_flag(
            manager=manager,
            name="prod_only",
            environments=["production"],
        )

        assert flag.flag_type == FlagType.ENVIRONMENT
        assert "production" in flag.rollout.enabled_environments


class TestGlobalManager:
    """Tests for global flag manager."""

    def test_get_flag_manager(self):
        """Test getting global flag manager."""
        manager = get_flag_manager()
        assert manager is not None
        assert isinstance(manager, FlagManager)

    def test_set_flag_manager(self):
        """Test setting global flag manager."""
        custom_manager = FlagManager()
        set_flag_manager(custom_manager)

        retrieved = get_flag_manager()
        assert retrieved is custom_manager
