"""
Feature flags for gradual rollouts and toggles.

Provides feature flag management with support for:
- Boolean toggles
- Percentage rollouts
- User targeting
- Environment-based flags
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class FlagType(str, Enum):
    """Type of feature flag."""

    BOOLEAN = "boolean"
    PERCENTAGE = "percentage"
    USER_LIST = "user_list"
    ENVIRONMENT = "environment"
    CUSTOM = "custom"


class FlagStatus(str, Enum):
    """Status of a feature flag."""

    ENABLED = "enabled"
    DISABLED = "disabled"
    ARCHIVED = "archived"


@dataclass
class RolloutConfig:
    """Configuration for gradual rollout."""

    percentage: float = 0.0  # 0-100
    enabled_users: list[str] = field(default_factory=list)
    disabled_users: list[str] = field(default_factory=list)
    enabled_environments: list[str] = field(default_factory=list)
    start_time: datetime | None = None
    end_time: datetime | None = None
    ramp_schedule: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "percentage": self.percentage,
            "enabled_users": self.enabled_users,
            "disabled_users": self.disabled_users,
            "enabled_environments": self.enabled_environments,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "ramp_schedule": self.ramp_schedule,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RolloutConfig:
        """Create config from dictionary."""
        return cls(
            percentage=data.get("percentage", 0.0),
            enabled_users=data.get("enabled_users", []),
            disabled_users=data.get("disabled_users", []),
            enabled_environments=data.get("enabled_environments", []),
            start_time=datetime.fromisoformat(data["start_time"])
            if data.get("start_time")
            else None,
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            ramp_schedule=data.get("ramp_schedule", []),
        )


@dataclass
class FeatureFlag:
    """A feature flag definition."""

    name: str
    flag_type: FlagType
    status: FlagStatus = FlagStatus.DISABLED
    description: str = ""
    owner: str = ""
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    default_value: Any = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def enable(self) -> None:
        """Enable the flag."""
        self.status = FlagStatus.ENABLED
        self.updated_at = datetime.utcnow()

    def disable(self) -> None:
        """Disable the flag."""
        self.status = FlagStatus.DISABLED
        self.updated_at = datetime.utcnow()

    def archive(self) -> None:
        """Archive the flag."""
        self.status = FlagStatus.ARCHIVED
        self.updated_at = datetime.utcnow()

    def set_percentage(self, percentage: float) -> None:
        """Set rollout percentage (0-100)."""
        if not 0 <= percentage <= 100:
            raise ValueError("Percentage must be between 0 and 100")
        self.rollout.percentage = percentage
        self.updated_at = datetime.utcnow()

    def add_user(self, user_id: str, enabled: bool = True) -> None:
        """Add user to enabled/disabled list."""
        self.updated_at = datetime.utcnow()
        if enabled:
            if user_id not in self.rollout.enabled_users:
                self.rollout.enabled_users.append(user_id)
            if user_id in self.rollout.disabled_users:
                self.rollout.disabled_users.remove(user_id)
        else:
            if user_id not in self.rollout.disabled_users:
                self.rollout.disabled_users.append(user_id)
            if user_id in self.rollout.enabled_users:
                self.rollout.enabled_users.remove(user_id)

    def remove_user(self, user_id: str) -> None:
        """Remove user from all lists."""
        if user_id in self.rollout.enabled_users:
            self.rollout.enabled_users.remove(user_id)
        if user_id in self.rollout.disabled_users:
            self.rollout.disabled_users.remove(user_id)
        self.updated_at = datetime.utcnow()

    def to_dict(self) -> dict[str, Any]:
        """Convert flag to dictionary."""
        return {
            "name": self.name,
            "flag_type": self.flag_type.value,
            "status": self.status.value,
            "description": self.description,
            "owner": self.owner,
            "rollout": self.rollout.to_dict(),
            "default_value": self.default_value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureFlag:
        """Create flag from dictionary."""
        return cls(
            name=data["name"],
            flag_type=FlagType(data["flag_type"]),
            status=FlagStatus(data.get("status", "disabled")),
            description=data.get("description", ""),
            owner=data.get("owner", ""),
            rollout=RolloutConfig.from_dict(data.get("rollout", {})),
            default_value=data.get("default_value", False),
            created_at=datetime.fromisoformat(data["created_at"])
            if "created_at" in data
            else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"])
            if "updated_at" in data
            else datetime.utcnow(),
            metadata=data.get("metadata", {}),
        )


class FlagEvaluator:
    """Evaluates feature flags for users."""

    def __init__(self, salt: str = "rag-feature-flag"):
        """Initialize evaluator with salt for hashing."""
        self.salt = salt

    def evaluate(
        self,
        flag: FeatureFlag,
        user_id: str | None = None,
        environment: str = "development",
        context: dict[str, Any] | None = None,
    ) -> bool:
        """
        Evaluate a feature flag.

        Returns True if the flag is enabled for the given context.
        """
        # Check if flag is globally disabled or archived
        if flag.status == FlagStatus.DISABLED:
            return bool(flag.default_value)
        if flag.status == FlagStatus.ARCHIVED:
            return False

        # Check time-based constraints
        now = datetime.utcnow()
        if flag.rollout.start_time and now < flag.rollout.start_time:
            return bool(flag.default_value)
        if flag.rollout.end_time and now > flag.rollout.end_time:
            return bool(flag.default_value)

        # Check environment
        if flag.flag_type == FlagType.ENVIRONMENT:
            return environment in flag.rollout.enabled_environments

        # Check user-specific overrides
        if user_id:
            if user_id in flag.rollout.disabled_users:
                return False
            if user_id in flag.rollout.enabled_users:
                return True

        # Boolean flag - just return status
        if flag.flag_type == FlagType.BOOLEAN:
            return flag.status == FlagStatus.ENABLED

        # User list flag - check if user is in enabled list
        if flag.flag_type == FlagType.USER_LIST:
            return user_id in flag.rollout.enabled_users if user_id else False

        # Percentage rollout
        if flag.flag_type == FlagType.PERCENTAGE:
            if not user_id:
                return bool(flag.default_value)

            # Use consistent hashing for deterministic results
            hash_input = f"{self.salt}:{flag.name}:{user_id}"
            hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
            bucket = (hash_value % 10000) / 100.0  # 0-100

            return bucket < flag.rollout.percentage

        # Custom flag - use default
        return bool(flag.default_value)


class FlagManager:
    """Manages feature flags."""

    def __init__(
        self,
        storage_path: str | None = None,
        evaluator: FlagEvaluator | None = None,
        environment: str = "development",
    ):
        """Initialize flag manager."""
        self.storage_path = Path(storage_path) if storage_path else None
        self.evaluator = evaluator or FlagEvaluator()
        self.environment = environment
        self._flags: dict[str, FeatureFlag] = {}
        self._evaluation_log: list[dict[str, Any]] = []

        if self.storage_path:
            self._load_flags()

    def _load_flags(self) -> None:
        """Load flags from storage."""
        if not self.storage_path or not self.storage_path.exists():
            return

        flags_file = self.storage_path / "feature_flags.json"
        if flags_file.exists():
            with flags_file.open() as f:
                data = json.load(f)
                for flag_data in data.get("flags", []):
                    flag = FeatureFlag.from_dict(flag_data)
                    self._flags[flag.name] = flag

    def _save_flags(self) -> None:
        """Save flags to storage."""
        if not self.storage_path:
            return

        self.storage_path.mkdir(parents=True, exist_ok=True)
        flags_file = self.storage_path / "feature_flags.json"

        data = {"flags": [flag.to_dict() for flag in self._flags.values()]}
        with flags_file.open("w") as f:
            json.dump(data, f, indent=2)

    def create_flag(
        self,
        name: str,
        flag_type: FlagType = FlagType.BOOLEAN,
        description: str = "",
        owner: str = "",
        default_value: Any = False,
        metadata: dict[str, Any] | None = None,
    ) -> FeatureFlag:
        """Create a new feature flag."""
        if name in self._flags:
            raise ValueError(f"Flag {name} already exists")

        flag = FeatureFlag(
            name=name,
            flag_type=flag_type,
            description=description,
            owner=owner,
            default_value=default_value,
            metadata=metadata or {},
        )
        self._flags[name] = flag
        self._save_flags()
        return flag

    def get_flag(self, name: str) -> FeatureFlag | None:
        """Get flag by name."""
        return self._flags.get(name)

    def list_flags(
        self,
        status: FlagStatus | None = None,
        flag_type: FlagType | None = None,
    ) -> list[FeatureFlag]:
        """List flags with optional filters."""
        flags = list(self._flags.values())

        if status:
            flags = [f for f in flags if f.status == status]
        if flag_type:
            flags = [f for f in flags if f.flag_type == flag_type]

        return flags

    def update_flag(self, name: str, **updates: Any) -> FeatureFlag:
        """Update a flag's properties."""
        flag = self.get_flag(name)
        if not flag:
            raise ValueError(f"Flag {name} not found")

        for key, value in updates.items():
            if hasattr(flag, key):
                setattr(flag, key, value)

        flag.updated_at = datetime.utcnow()
        self._save_flags()
        return flag

    def delete_flag(self, name: str) -> bool:
        """Delete a flag."""
        if name in self._flags:
            del self._flags[name]
            self._save_flags()
            return True
        return False

    def enable_flag(self, name: str) -> FeatureFlag:
        """Enable a flag."""
        flag = self.get_flag(name)
        if not flag:
            raise ValueError(f"Flag {name} not found")

        flag.enable()
        self._save_flags()
        return flag

    def disable_flag(self, name: str) -> FeatureFlag:
        """Disable a flag."""
        flag = self.get_flag(name)
        if not flag:
            raise ValueError(f"Flag {name} not found")

        flag.disable()
        self._save_flags()
        return flag

    def set_percentage(self, name: str, percentage: float) -> FeatureFlag:
        """Set flag rollout percentage."""
        flag = self.get_flag(name)
        if not flag:
            raise ValueError(f"Flag {name} not found")

        flag.set_percentage(percentage)
        self._save_flags()
        return flag

    def is_enabled(
        self,
        name: str,
        user_id: str | None = None,
        context: dict[str, Any] | None = None,
        log: bool = True,
    ) -> bool:
        """
        Check if a flag is enabled for a user/context.

        This is the main method for checking feature flags in application code.
        """
        flag = self.get_flag(name)
        if not flag:
            return False

        result = self.evaluator.evaluate(
            flag=flag,
            user_id=user_id,
            environment=self.environment,
            context=context,
        )

        if log:
            self._evaluation_log.append(
                {
                    "flag_name": name,
                    "user_id": user_id,
                    "result": result,
                    "environment": self.environment,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

        return result

    def get_evaluation_log(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent flag evaluations."""
        return self._evaluation_log[-limit:]

    def get_flag_stats(self, name: str) -> dict[str, Any]:
        """Get statistics for a flag."""
        flag = self.get_flag(name)
        if not flag:
            raise ValueError(f"Flag {name} not found")

        evaluations = [e for e in self._evaluation_log if e["flag_name"] == name]
        enabled_count = sum(1 for e in evaluations if e["result"])
        total_count = len(evaluations)

        return {
            "flag_name": name,
            "status": flag.status.value,
            "total_evaluations": total_count,
            "enabled_evaluations": enabled_count,
            "enabled_percentage": (enabled_count / total_count * 100) if total_count > 0 else 0,
            "unique_users": len({e["user_id"] for e in evaluations if e["user_id"]}),
        }


# Convenience functions for common flag patterns
def create_gradual_rollout_flag(
    manager: FlagManager,
    name: str,
    initial_percentage: float = 0.0,
    **kwargs: Any,
) -> FeatureFlag:
    """Create a flag for gradual percentage rollout."""
    flag = manager.create_flag(
        name=name,
        flag_type=FlagType.PERCENTAGE,
        **kwargs,
    )
    flag.rollout.percentage = initial_percentage
    flag.enable()
    manager._save_flags()
    return flag


def create_user_targeting_flag(
    manager: FlagManager,
    name: str,
    enabled_users: list[str],
    **kwargs: Any,
) -> FeatureFlag:
    """Create a flag for user targeting."""
    flag = manager.create_flag(
        name=name,
        flag_type=FlagType.USER_LIST,
        **kwargs,
    )
    flag.rollout.enabled_users = enabled_users
    flag.enable()
    manager._save_flags()
    return flag


def create_environment_flag(
    manager: FlagManager,
    name: str,
    environments: list[str],
    **kwargs: Any,
) -> FeatureFlag:
    """Create a flag for environment targeting."""
    flag = manager.create_flag(
        name=name,
        flag_type=FlagType.ENVIRONMENT,
        **kwargs,
    )
    flag.rollout.enabled_environments = environments
    flag.enable()
    manager._save_flags()
    return flag


# Global flag manager
_flag_manager: FlagManager | None = None


def get_flag_manager() -> FlagManager:
    """Get the global flag manager."""
    global _flag_manager
    if _flag_manager is None:
        _flag_manager = FlagManager()
    return _flag_manager


def set_flag_manager(manager: FlagManager) -> None:
    """Set the global flag manager."""
    global _flag_manager
    _flag_manager = manager
