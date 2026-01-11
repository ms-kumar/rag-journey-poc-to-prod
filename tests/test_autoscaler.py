"""Tests for autoscaling functionality."""

import time

from src.services.cost.autoscaler import (
    Autoscaler,
    AutoscalingPolicy,
    LoadMetrics,
    ScalingDecision,
)


class TestAutoscalingPolicy:
    """Test AutoscalingPolicy class."""

    def test_default_policy(self):
        """Test default policy values."""
        policy = AutoscalingPolicy()

        assert policy.min_instances == 1
        assert policy.max_instances == 10
        assert policy.scale_up_cpu_threshold == 70.0
        assert policy.scale_down_cpu_threshold == 30.0
        assert policy.cooldown_period == 60

    def test_custom_policy(self):
        """Test custom policy values."""
        policy = AutoscalingPolicy(
            min_instances=2,
            max_instances=20,
            scale_up_cpu_threshold=80.0,
            cooldown_period=30,
        )

        assert policy.min_instances == 2
        assert policy.max_instances == 20
        assert policy.scale_up_cpu_threshold == 80.0
        assert policy.cooldown_period == 30


class TestLoadMetrics:
    """Test LoadMetrics class."""

    def test_load_metrics_creation(self):
        """Test creating load metrics."""
        metrics = LoadMetrics(
            cpu_usage=75.0,
            queue_size=30,
            active_requests=15,
            avg_latency_ms=200.0,
            p95_latency_ms=350.0,
            error_rate=2.0,
            requests_per_second=50.0,
        )

        assert metrics.cpu_usage == 75.0
        assert metrics.queue_size == 30
        assert metrics.active_requests == 15


class TestAutoscaler:
    """Test Autoscaler class."""

    def test_initialization(self):
        """Test autoscaler initialization."""
        autoscaler = Autoscaler()

        assert autoscaler.current_instances == 1
        assert autoscaler.current_concurrency >= 1
        assert autoscaler.total_cost == 0.0

    def test_initialization_with_policy(self):
        """Test initialization with custom policy."""
        policy = AutoscalingPolicy(min_instances=2, max_instances=5)
        autoscaler = Autoscaler(policy)

        assert autoscaler.current_instances == 2
        assert autoscaler.policy.max_instances == 5

    def test_should_scale_up_cpu(self):
        """Test scale up decision based on CPU."""
        autoscaler = Autoscaler()

        metrics = LoadMetrics(
            cpu_usage=85.0,  # Above threshold
            queue_size=10,
            active_requests=5,
            avg_latency_ms=200,
            p95_latency_ms=300,
            error_rate=1.0,
            requests_per_second=50,
        )

        decision = autoscaler.should_scale(metrics)
        assert decision == ScalingDecision.SCALE_UP

    def test_should_scale_down_cpu(self):
        """Test scale down decision based on CPU."""
        policy = AutoscalingPolicy(min_instances=1, cooldown_period=0)
        autoscaler = Autoscaler(policy)
        autoscaler.current_instances = 3

        metrics = LoadMetrics(
            cpu_usage=20.0,  # Below threshold
            queue_size=2,
            active_requests=1,
            avg_latency_ms=100,
            p95_latency_ms=150,
            error_rate=0.5,
            requests_per_second=10,
        )

        decision = autoscaler.should_scale(metrics)
        assert decision == ScalingDecision.SCALE_DOWN

    def test_should_scale_up_queue(self):
        """Test scale up decision based on queue size."""
        policy = AutoscalingPolicy(cooldown_period=0)
        autoscaler = Autoscaler(policy)

        metrics = LoadMetrics(
            cpu_usage=50.0,
            queue_size=60,  # Above threshold
            active_requests=30,
            avg_latency_ms=200,
            p95_latency_ms=300,
            error_rate=1.0,
            requests_per_second=50,
        )

        decision = autoscaler.should_scale(metrics)
        assert decision == ScalingDecision.SCALE_UP

    def test_should_scale_up_latency(self):
        """Test scale up decision based on latency."""
        policy = AutoscalingPolicy(
            cooldown_period=0,
            scale_up_latency_threshold=500.0,
        )
        autoscaler = Autoscaler(policy)

        metrics = LoadMetrics(
            cpu_usage=50.0,
            queue_size=20,
            active_requests=10,
            avg_latency_ms=800,
            p95_latency_ms=1200,  # Above threshold
            error_rate=1.0,
            requests_per_second=50,
        )

        decision = autoscaler.should_scale(metrics)
        assert decision == ScalingDecision.SCALE_UP

    def test_cooldown_period(self):
        """Test cooldown period prevents scaling."""
        policy = AutoscalingPolicy(cooldown_period=60)
        autoscaler = Autoscaler(policy)
        autoscaler.last_scale_time = time.time()  # Just scaled

        metrics = LoadMetrics(
            cpu_usage=90.0,  # Should trigger scale up
            queue_size=100,
            active_requests=50,
            avg_latency_ms=1000,
            p95_latency_ms=1500,
            error_rate=5.0,
            requests_per_second=100,
        )

        decision = autoscaler.should_scale(metrics)
        assert decision == ScalingDecision.NO_CHANGE

    def test_scale_up_execution(self):
        """Test executing scale up."""
        autoscaler = Autoscaler()
        initial_instances = autoscaler.current_instances

        new_instances = autoscaler.scale(ScalingDecision.SCALE_UP)

        assert new_instances > initial_instances
        assert autoscaler.current_instances == new_instances

    def test_scale_down_execution(self):
        """Test executing scale down."""
        autoscaler = Autoscaler()
        autoscaler.current_instances = 4

        new_instances = autoscaler.scale(ScalingDecision.SCALE_DOWN)

        assert new_instances < 4
        assert autoscaler.current_instances == new_instances

    def test_scale_respects_max_instances(self):
        """Test scaling respects max instances limit."""
        policy = AutoscalingPolicy(max_instances=3)
        autoscaler = Autoscaler(policy)
        autoscaler.current_instances = 3

        new_instances = autoscaler.scale(ScalingDecision.SCALE_UP)

        assert new_instances == 3  # Should not exceed max

    def test_scale_respects_min_instances(self):
        """Test scaling respects min instances limit."""
        policy = AutoscalingPolicy(min_instances=2)
        autoscaler = Autoscaler(policy)
        autoscaler.current_instances = 2

        new_instances = autoscaler.scale(ScalingDecision.SCALE_DOWN)

        assert new_instances == 2  # Should not go below min

    def test_auto_scale(self):
        """Test automatic scaling."""
        policy = AutoscalingPolicy(cooldown_period=0)
        autoscaler = Autoscaler(policy)

        high_load_metrics = LoadMetrics(
            cpu_usage=90.0,
            queue_size=80,
            active_requests=40,
            avg_latency_ms=800,
            p95_latency_ms=1200,
            error_rate=2.0,
            requests_per_second=100,
        )

        decision, new_instances = autoscaler.auto_scale(high_load_metrics)

        assert decision == ScalingDecision.SCALE_UP
        assert new_instances > 1

    def test_budget_prevents_scale_up(self):
        """Test budget limit prevents scaling up."""
        policy = AutoscalingPolicy(
            budget_limit=10.0,
            cooldown_period=0,
        )
        autoscaler = Autoscaler(policy)
        autoscaler.total_cost = 10.0  # At budget limit

        metrics = LoadMetrics(
            cpu_usage=90.0,  # Should trigger scale up
            queue_size=100,
            active_requests=50,
            avg_latency_ms=1000,
            p95_latency_ms=1500,
            error_rate=5.0,
            requests_per_second=100,
        )

        decision = autoscaler.should_scale(metrics)
        assert decision == ScalingDecision.NO_CHANGE

    def test_record_cost(self):
        """Test recording operational cost."""
        autoscaler = Autoscaler()

        autoscaler.record_cost(10.0)
        autoscaler.record_cost(5.0)

        assert autoscaler.total_cost == 15.0

    def test_get_current_capacity(self):
        """Test getting current capacity."""
        autoscaler = Autoscaler()
        autoscaler.current_instances = 3
        autoscaler.current_concurrency = 10

        capacity = autoscaler.get_current_capacity()

        assert capacity["instances"] == 3
        assert capacity["concurrency"] == 10
        assert capacity["total_capacity"] == 30

    def test_get_scaling_history(self):
        """Test getting scaling history."""
        policy = AutoscalingPolicy(cooldown_period=0)
        autoscaler = Autoscaler(policy)

        # Trigger some scaling events
        autoscaler.scale(ScalingDecision.SCALE_UP)
        time.sleep(0.1)
        autoscaler.scale(ScalingDecision.SCALE_UP)

        history = autoscaler.get_scaling_history(limit=5)

        assert len(history) == 2
        assert all("decision" in event for event in history)
        assert all("timestamp" in event for event in history)

    def test_reset(self):
        """Test resetting autoscaler."""
        autoscaler = Autoscaler()
        autoscaler.current_instances = 5
        autoscaler.total_cost = 100.0
        autoscaler.scale(ScalingDecision.SCALE_UP)

        autoscaler.reset()

        assert autoscaler.current_instances == 1
        assert autoscaler.total_cost == 0.0
        assert len(autoscaler.history) == 0

    def test_concurrency_increases_with_scale_up(self):
        """Test concurrency increases when scaling up."""
        autoscaler = Autoscaler()
        autoscaler.current_concurrency = 10  # Start with higher concurrency
        initial_concurrency = autoscaler.current_concurrency

        autoscaler.scale(ScalingDecision.SCALE_UP)

        assert autoscaler.current_concurrency > initial_concurrency

    def test_concurrency_decreases_with_scale_down(self):
        """Test concurrency decreases when scaling down."""
        autoscaler = Autoscaler()
        autoscaler.current_instances = 4
        autoscaler.current_concurrency = 50

        autoscaler.scale(ScalingDecision.SCALE_DOWN)

        assert autoscaler.current_concurrency < 50
