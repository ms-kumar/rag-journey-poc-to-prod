"""Tests for the Autoscaler class."""

from src.services.cost.autoscaler import (
    Autoscaler,
    AutoscalingPolicy,
    LoadMetrics,
    ScalingDecision,
)


class TestAutoscalingPolicy:
    """Tests for AutoscalingPolicy dataclass."""

    def test_default_values(self):
        """Test default policy values."""
        policy = AutoscalingPolicy()

        assert policy.scale_up_cpu_threshold == 70.0
        assert policy.scale_down_cpu_threshold == 30.0
        assert policy.min_instances == 1
        assert policy.max_instances == 10
        assert policy.cooldown_period == 60

    def test_custom_values(self):
        """Test custom policy values."""
        policy = AutoscalingPolicy(
            min_instances=2,
            max_instances=20,
            scale_up_cpu_threshold=80.0,
        )

        assert policy.min_instances == 2
        assert policy.max_instances == 20
        assert policy.scale_up_cpu_threshold == 80.0


class TestLoadMetrics:
    """Tests for LoadMetrics dataclass."""

    def test_load_metrics_creation(self):
        """Test creating load metrics."""
        metrics = LoadMetrics(
            cpu_usage=50.0,
            queue_size=10,
            active_requests=5,
            avg_latency_ms=100.0,
            p95_latency_ms=200.0,
            error_rate=1.0,
            requests_per_second=50.0,
        )

        assert metrics.cpu_usage == 50.0
        assert metrics.queue_size == 10
        assert metrics.active_requests == 5


class TestAutoscaler:
    """Tests for Autoscaler class."""

    def test_initialization_default(self, autoscaler: Autoscaler):
        """Test autoscaler initialization with defaults."""
        assert autoscaler.current_instances == 1
        assert autoscaler.current_concurrency == 1
        assert autoscaler.policy is not None

    def test_initialization_custom_policy(self, autoscaler_with_policy: Autoscaler):
        """Test autoscaler with custom policy."""
        assert autoscaler_with_policy.policy.max_instances == 5
        assert autoscaler_with_policy.policy.cooldown_period == 0

    def test_should_scale_no_change_normal_load(
        self,
        autoscaler_with_policy: Autoscaler,
        normal_load_metrics: LoadMetrics,
    ):
        """Test no scaling needed with normal load."""
        decision = autoscaler_with_policy.should_scale(normal_load_metrics)
        assert decision == ScalingDecision.NO_CHANGE

    def test_should_scale_up_high_cpu(
        self,
        autoscaler_with_policy: Autoscaler,
        high_load_metrics: LoadMetrics,
    ):
        """Test scale up when CPU is high."""
        decision = autoscaler_with_policy.should_scale(high_load_metrics)
        assert decision == ScalingDecision.SCALE_UP

    def test_should_scale_down_low_cpu(
        self,
        autoscaler_with_policy: Autoscaler,
        low_load_metrics: LoadMetrics,
    ):
        """Test scale down when CPU is low."""
        # First scale up to have room to scale down
        autoscaler_with_policy.current_instances = 3
        decision = autoscaler_with_policy.should_scale(low_load_metrics)
        assert decision == ScalingDecision.SCALE_DOWN

    def test_should_scale_up_high_latency(self, autoscaler_with_policy: Autoscaler):
        """Test scale up when latency is high."""
        metrics = LoadMetrics(
            cpu_usage=50.0,
            queue_size=10,
            active_requests=5,
            avg_latency_ms=500.0,
            p95_latency_ms=1500.0,  # Above threshold
            error_rate=1.0,
            requests_per_second=50.0,
        )

        decision = autoscaler_with_policy.should_scale(metrics)
        assert decision == ScalingDecision.SCALE_UP

    def test_should_scale_up_high_queue(self, autoscaler_with_policy: Autoscaler):
        """Test scale up when queue is large."""
        metrics = LoadMetrics(
            cpu_usage=50.0,
            queue_size=100,  # Above threshold
            active_requests=5,
            avg_latency_ms=100.0,
            p95_latency_ms=200.0,
            error_rate=1.0,
            requests_per_second=50.0,
        )

        decision = autoscaler_with_policy.should_scale(metrics)
        assert decision == ScalingDecision.SCALE_UP

    def test_cannot_scale_up_at_max(
        self,
        autoscaler_with_policy: Autoscaler,
        high_load_metrics: LoadMetrics,
    ):
        """Test cannot scale up when at max instances."""
        autoscaler_with_policy.current_instances = 5  # At max

        decision = autoscaler_with_policy.should_scale(high_load_metrics)
        assert decision == ScalingDecision.NO_CHANGE

    def test_cannot_scale_down_at_min(
        self,
        autoscaler_with_policy: Autoscaler,
        low_load_metrics: LoadMetrics,
    ):
        """Test cannot scale down when at min instances."""
        autoscaler_with_policy.current_instances = 1  # At min

        decision = autoscaler_with_policy.should_scale(low_load_metrics)
        assert decision == ScalingDecision.NO_CHANGE

    def test_scale_up_execution(self, autoscaler_with_policy: Autoscaler):
        """Test scaling up execution."""
        new_instances = autoscaler_with_policy.scale(ScalingDecision.SCALE_UP)

        assert new_instances == 2
        assert autoscaler_with_policy.current_instances == 2
        assert autoscaler_with_policy.last_scale_time > 0

    def test_scale_down_execution(self, autoscaler_with_policy: Autoscaler):
        """Test scaling down execution."""
        autoscaler_with_policy.current_instances = 4

        new_instances = autoscaler_with_policy.scale(ScalingDecision.SCALE_DOWN)

        assert new_instances == 2
        assert autoscaler_with_policy.current_instances == 2

    def test_scale_no_change(self, autoscaler_with_policy: Autoscaler):
        """Test no change in scaling."""
        original = autoscaler_with_policy.current_instances

        new_instances = autoscaler_with_policy.scale(ScalingDecision.NO_CHANGE)

        assert new_instances == original

    def test_scale_up_respects_max(self, autoscaler_with_policy: Autoscaler):
        """Test scale up respects max instances."""
        autoscaler_with_policy.current_instances = 4

        new_instances = autoscaler_with_policy.scale(ScalingDecision.SCALE_UP)

        assert new_instances == 5  # Max is 5

    def test_scale_down_respects_min(self, autoscaler_with_policy: Autoscaler):
        """Test scale down respects min instances."""
        autoscaler_with_policy.current_instances = 2
        autoscaler_with_policy.policy.scale_down_factor = 0.3  # Would go below 1

        new_instances = autoscaler_with_policy.scale(ScalingDecision.SCALE_DOWN)

        assert new_instances == 1  # Min is 1

    def test_concurrency_increases_on_scale_up(self, autoscaler_with_policy: Autoscaler):
        """Test concurrency increases when scaling up."""
        # Start with concurrency > 1 so int(n * 1.5) actually increases
        autoscaler_with_policy.current_concurrency = 2
        original_concurrency = autoscaler_with_policy.current_concurrency

        autoscaler_with_policy.scale(ScalingDecision.SCALE_UP)

        assert autoscaler_with_policy.current_concurrency > original_concurrency

    def test_concurrency_decreases_on_scale_down(self, autoscaler_with_policy: Autoscaler):
        """Test concurrency decreases when scaling down."""
        autoscaler_with_policy.current_instances = 4
        autoscaler_with_policy.current_concurrency = 10

        autoscaler_with_policy.scale(ScalingDecision.SCALE_DOWN)

        assert autoscaler_with_policy.current_concurrency < 10

    def test_history_recorded(self, autoscaler_with_policy: Autoscaler):
        """Test scaling history is recorded."""
        autoscaler_with_policy.scale(ScalingDecision.SCALE_UP)

        assert len(autoscaler_with_policy.history) == 1
        assert autoscaler_with_policy.history[0]["decision"] == ScalingDecision.SCALE_UP.value

    def test_cooldown_period_enforcement(self, autoscaler: Autoscaler):
        """Test cooldown period is enforced."""
        # Use default autoscaler with 60s cooldown
        # Simulate recent scale action
        import time

        autoscaler.last_scale_time = time.time()

        metrics = LoadMetrics(
            cpu_usage=85.0,
            queue_size=100,
            active_requests=20,
            avg_latency_ms=500.0,
            p95_latency_ms=1500.0,
            error_rate=5.0,
            requests_per_second=10.0,
        )

        decision = autoscaler.should_scale(metrics)
        assert decision == ScalingDecision.NO_CHANGE  # Should be blocked by cooldown

    def test_budget_limit_enforcement(self, autoscaler_with_policy: Autoscaler):
        """Test budget limit is enforced."""
        autoscaler_with_policy.policy.budget_limit = 100.0
        autoscaler_with_policy.total_cost = 100.0

        # Should not be able to scale up
        assert not autoscaler_with_policy._can_scale_up()
