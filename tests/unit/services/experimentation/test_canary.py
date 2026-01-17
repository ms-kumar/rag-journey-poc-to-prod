"""Tests for canary deployment module."""

from src.services.experimentation.canary import (
    CanaryConfig,
    CanaryDeployment,
    CanaryManager,
    CanaryMetrics,
    CanaryStatus,
    RollbackReason,
)


class TestCanaryConfig:
    """Tests for CanaryConfig."""

    def test_default_config(self):
        """Test default canary configuration."""
        config = CanaryConfig()

        assert config.initial_percentage == 1.0
        assert config.max_percentage == 100.0
        assert config.increment_percentage == 10.0

    def test_custom_config(self):
        """Test custom canary configuration."""
        config = CanaryConfig(
            initial_percentage=5.0,
            max_percentage=50.0,
            increment_percentage=5.0,
            max_error_rate=0.02,
        )

        assert config.initial_percentage == 5.0
        assert config.max_percentage == 50.0
        assert config.max_error_rate == 0.02

    def test_config_to_dict(self):
        """Test config serialization."""
        config = CanaryConfig(
            max_error_rate=0.02,
            max_latency_p99_ms=150.0,
        )
        data = config.to_dict()

        assert data["max_error_rate"] == 0.02
        assert data["max_latency_p99_ms"] == 150.0


class TestCanaryMetrics:
    """Tests for CanaryMetrics."""

    def test_metrics_creation(self):
        """Test metrics creation."""
        metrics = CanaryMetrics()
        metrics.request_count = 1000
        metrics.error_count = 5
        metrics.latencies_ms = [50.0] * 1000

        assert metrics.request_count == 1000
        assert metrics.error_rate == 0.005

    def test_metrics_empty(self):
        """Test metrics with no requests."""
        metrics = CanaryMetrics()

        assert metrics.error_rate == 0.0
        assert metrics.latency_p99 == 0.0

    def test_metrics_record_request(self):
        """Test recording a request."""
        metrics = CanaryMetrics()
        metrics.record_request(latency_ms=50.0, is_error=False)

        assert metrics.request_count == 1
        assert metrics.error_count == 0
        assert len(metrics.latencies_ms) == 1

    def test_metrics_record_request_with_error(self):
        """Test recording a request with error."""
        metrics = CanaryMetrics()
        metrics.record_request(latency_ms=100.0, is_error=True)

        assert metrics.request_count == 1
        assert metrics.error_count == 1

    def test_metrics_latency_p99(self):
        """Test P99 latency calculation."""
        metrics = CanaryMetrics()
        # Add latencies: 1-100
        for i in range(1, 101):
            metrics.latencies_ms.append(float(i))

        # P99 should be around 99
        assert metrics.latency_p99 == 100.0

    def test_metrics_to_dict(self):
        """Test metrics serialization."""
        metrics = CanaryMetrics()
        metrics.request_count = 500
        metrics.error_count = 10
        metrics.latencies_ms = [50.0] * 500

        data = metrics.to_dict()

        assert data["request_count"] == 500
        assert data["error_rate"] == 0.02


class TestCanaryDeployment:
    """Tests for CanaryDeployment."""

    def test_deployment_creation(self):
        """Test creating a deployment."""
        config = CanaryConfig()
        deployment = CanaryDeployment(
            id="deploy-123",
            name="API v2",
            baseline_version="v1.0",
            canary_version="v2.0",
            config=config,
        )

        assert deployment.id == "deploy-123"
        assert deployment.status == CanaryStatus.PENDING
        assert deployment.current_percentage == 0.0

    def test_deployment_start(self):
        """Test starting a deployment."""
        config = CanaryConfig(initial_percentage=5.0)
        deployment = CanaryDeployment(
            id="deploy-123",
            name="API v2",
            baseline_version="v1.0",
            canary_version="v2.0",
            config=config,
        )

        deployment.start()

        assert deployment.status == CanaryStatus.RUNNING
        assert deployment.current_percentage == 5.0
        assert deployment.started_at is not None

    def test_deployment_increment_traffic(self):
        """Test incrementing traffic."""
        config = CanaryConfig(
            initial_percentage=5.0,
            increment_percentage=10.0,
            max_percentage=50.0,
        )
        deployment = CanaryDeployment(
            id="deploy-123",
            name="API v2",
            baseline_version="v1.0",
            canary_version="v2.0",
            config=config,
        )

        deployment.start()
        result = deployment.increment_traffic()

        assert result is True
        assert deployment.current_percentage == 15.0

    def test_deployment_increment_capped(self):
        """Test incrementing traffic is capped at max."""
        config = CanaryConfig(
            initial_percentage=45.0,
            increment_percentage=10.0,
            max_percentage=50.0,
        )
        deployment = CanaryDeployment(
            id="deploy-123",
            name="API v2",
            baseline_version="v1.0",
            canary_version="v2.0",
            config=config,
        )

        deployment.start()
        result = deployment.increment_traffic()

        assert result is True
        assert deployment.current_percentage == 50.0

    def test_deployment_promote(self):
        """Test promoting deployment."""
        config = CanaryConfig()
        deployment = CanaryDeployment(
            id="deploy-123",
            name="API v2",
            baseline_version="v1.0",
            canary_version="v2.0",
            config=config,
        )

        deployment.start()
        deployment.promote()

        assert deployment.status == CanaryStatus.PROMOTED
        assert deployment.ended_at is not None

    def test_deployment_rollback(self):
        """Test rolling back deployment."""
        config = CanaryConfig()
        deployment = CanaryDeployment(
            id="deploy-123",
            name="API v2",
            baseline_version="v1.0",
            canary_version="v2.0",
            config=config,
        )

        deployment.start()
        deployment.rollback(reason=RollbackReason.ERROR_RATE)

        assert deployment.status == CanaryStatus.ROLLED_BACK
        assert deployment.rollback_reason == RollbackReason.ERROR_RATE
        assert deployment.current_percentage == 0.0

    def test_deployment_pause_resume(self):
        """Test pausing and resuming deployment."""
        config = CanaryConfig()
        deployment = CanaryDeployment(
            id="deploy-123",
            name="API v2",
            baseline_version="v1.0",
            canary_version="v2.0",
            config=config,
        )

        deployment.start()
        deployment.pause()
        assert deployment.status == CanaryStatus.PAUSED

        deployment.resume()
        assert deployment.status == CanaryStatus.RUNNING

    def test_deployment_should_rollback_high_error_rate(self):
        """Test should_rollback with high error rate."""
        config = CanaryConfig(
            max_error_rate=0.05,
            min_samples_before_decision=10,
        )
        deployment = CanaryDeployment(
            id="deploy-123",
            name="API v2",
            baseline_version="v1.0",
            canary_version="v2.0",
            config=config,
        )

        deployment.start()

        # Record high error rate
        for i in range(100):
            deployment.canary_metrics.record_request(
                latency_ms=50.0,
                is_error=(i % 5 == 0),  # 20% error rate
            )

        should_rollback, reason = deployment.should_rollback()
        assert should_rollback is True
        assert reason == RollbackReason.ERROR_RATE

    def test_deployment_to_dict(self):
        """Test deployment serialization."""
        config = CanaryConfig()
        deployment = CanaryDeployment(
            id="deploy-123",
            name="API v2",
            baseline_version="v1.0",
            canary_version="v2.0",
            config=config,
        )

        data = deployment.to_dict()

        assert data["id"] == "deploy-123"
        assert data["name"] == "API v2"
        assert data["status"] == "pending"


class TestCanaryManager:
    """Tests for CanaryManager."""

    def test_create_deployment(self):
        """Test creating a deployment."""
        manager = CanaryManager()
        config = CanaryConfig(initial_percentage=5.0)

        deployment = manager.create_deployment(
            deployment_id="deploy-123",
            name="API v2",
            baseline_version="v1.0",
            canary_version="v2.0",
            config=config,
        )

        assert deployment.name == "API v2"
        assert deployment.status == CanaryStatus.PENDING

    def test_start_deployment(self):
        """Test starting a deployment."""
        manager = CanaryManager()
        deployment = manager.create_deployment(
            deployment_id="deploy-123",
            name="API v2",
            baseline_version="v1.0",
            canary_version="v2.0",
        )

        manager.start_deployment(deployment.id)

        updated = manager.get_deployment(deployment.id)
        assert updated is not None
        assert updated.status == CanaryStatus.RUNNING

    def test_get_nonexistent_deployment(self):
        """Test getting nonexistent deployment returns None."""
        manager = CanaryManager()

        result = manager.get_deployment("nonexistent")
        assert result is None

    def test_list_deployments(self):
        """Test listing deployments."""
        manager = CanaryManager()

        manager.create_deployment("deploy-1", "Deploy 1", "v1", "v2")
        manager.create_deployment("deploy-2", "Deploy 2", "v1", "v2")

        all_deployments = manager.list_deployments()
        assert len(all_deployments) == 2

    def test_list_active_deployments(self):
        """Test listing active deployments."""
        manager = CanaryManager()

        d1 = manager.create_deployment("deploy-1", "Deploy 1", "v1", "v2")
        manager.create_deployment("deploy-2", "Deploy 2", "v1", "v2")

        manager.start_deployment(d1.id)

        active = manager.list_deployments(status=CanaryStatus.RUNNING)
        assert len(active) == 1
        assert active[0].id == d1.id

    def test_record_canary_metrics(self):
        """Test recording canary metrics."""
        manager = CanaryManager()
        deployment = manager.create_deployment("deploy-123", "API v2", "v1", "v2")
        manager.start_deployment(deployment.id)

        manager.record_request(
            deployment_id=deployment.id,
            is_canary=True,
            latency_ms=50.0,
            is_error=False,
        )

        updated = manager.get_deployment(deployment.id)
        assert updated is not None
        assert updated.canary_metrics.request_count == 1

    def test_record_baseline_metrics(self):
        """Test recording baseline metrics."""
        manager = CanaryManager()
        deployment = manager.create_deployment("deploy-123", "API v2", "v1", "v2")
        manager.start_deployment(deployment.id)

        manager.record_request(
            deployment_id=deployment.id,
            is_canary=False,
            latency_ms=50.0,
            is_error=False,
        )

        updated = manager.get_deployment(deployment.id)
        assert updated is not None
        assert updated.baseline_metrics.request_count == 1

    def test_promote_deployment(self):
        """Test promoting deployment."""
        manager = CanaryManager()
        deployment = manager.create_deployment("deploy-123", "API v2", "v1", "v2")
        manager.start_deployment(deployment.id)

        manager.promote_deployment(deployment.id)

        updated = manager.get_deployment(deployment.id)
        assert updated is not None
        assert updated.status == CanaryStatus.PROMOTED

    def test_rollback_deployment(self):
        """Test rolling back deployment."""
        manager = CanaryManager()
        deployment = manager.create_deployment("deploy-123", "API v2", "v1", "v2")
        manager.start_deployment(deployment.id)

        manager.rollback_deployment(deployment.id, reason=RollbackReason.MANUAL)

        updated = manager.get_deployment(deployment.id)
        assert updated is not None
        assert updated.status == CanaryStatus.ROLLED_BACK

    def test_route_request_baseline(self):
        """Test routing request to baseline when not running."""
        manager = CanaryManager()
        deployment = manager.create_deployment("deploy-123", "API v2", "v1", "v2")

        # Not started yet
        is_canary = manager.route_request(deployment.id)
        assert is_canary is False

    def test_route_request_random(self):
        """Test routing requests with percentage."""
        manager = CanaryManager()
        config = CanaryConfig(initial_percentage=50.0)
        deployment = manager.create_deployment(
            "deploy-123",
            "API v2",
            "v1",
            "v2",
            config=config,
        )
        manager.start_deployment(deployment.id)

        # Route many requests
        canary_count = 0
        for i in range(100):
            if manager.route_request(deployment.id, user_id=f"user-{i}"):
                canary_count += 1

        # Should be roughly 50% (with some variance)
        assert 30 < canary_count < 70

    def test_callbacks_on_rollback(self):
        """Test callback is called on rollback."""
        rollback_called = {"count": 0}

        def on_rollback(deployment, reason):
            rollback_called["count"] += 1

        manager = CanaryManager(on_rollback=on_rollback)
        deployment = manager.create_deployment("deploy-123", "API v2", "v1", "v2")
        manager.start_deployment(deployment.id)
        manager.rollback_deployment(deployment.id)

        assert rollback_called["count"] == 1

    def test_callbacks_on_promote(self):
        """Test callback is called on promote."""
        promote_called = {"count": 0}

        def on_promote(deployment):
            promote_called["count"] += 1

        manager = CanaryManager(on_promote=on_promote)
        deployment = manager.create_deployment("deploy-123", "API v2", "v1", "v2")
        manager.start_deployment(deployment.id)
        manager.promote_deployment(deployment.id)

        assert promote_called["count"] == 1


class TestRollbackReason:
    """Tests for RollbackReason enum."""

    def test_reason_values(self):
        """Test rollback reason values."""
        assert RollbackReason.MANUAL.value == "manual"
        assert RollbackReason.ERROR_RATE.value == "error_rate"
        assert RollbackReason.LATENCY.value == "latency"
        assert RollbackReason.QUALITY.value == "quality"


class TestCanaryStatus:
    """Tests for CanaryStatus enum."""

    def test_status_values(self):
        """Test canary status values."""
        assert CanaryStatus.PENDING.value == "pending"
        assert CanaryStatus.RUNNING.value == "running"
        assert CanaryStatus.PROMOTED.value == "promoted"
        assert CanaryStatus.ROLLED_BACK.value == "rolled_back"
