"""
Tests for cache metrics and monitoring.
"""

import time

import pytest

from src.config import CacheSettings
from src.services.cache.metrics import CacheMetrics, CacheStats, CacheTimer


class TestCacheStats:
    """Tests for CacheStats dataclass."""

    def test_defaults(self):
        """Test default values."""
        stats = CacheStats()

        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.sets == 0
        assert stats.deletes == 0
        assert stats.total_latency_ms == 0.0

    def test_total_requests(self):
        """Test total requests calculation."""
        stats = CacheStats(hits=10, misses=5)

        assert stats.total_requests == 15

    def test_hit_rate_zero_requests(self):
        """Test hit rate with zero requests."""
        stats = CacheStats()

        assert stats.hit_rate == 0.0

    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        stats = CacheStats(hits=8, misses=2)

        assert stats.hit_rate == 0.8

    def test_avg_latency_zero_requests(self):
        """Test average latency with zero requests."""
        stats = CacheStats()

        assert stats.avg_latency_ms == 0.0

    def test_avg_latency_calculation(self):
        """Test average latency calculation."""
        stats = CacheStats(hits=5, misses=5, total_latency_ms=100.0)

        assert stats.avg_latency_ms == 10.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = CacheStats(hits=10, misses=2, sets=12)
        data = stats.to_dict()

        assert data["hits"] == 10
        assert data["misses"] == 2
        assert data["sets"] == 12
        assert data["total_requests"] == 12
        assert "hit_rate" in data
        assert "avg_latency_ms" in data


class TestStalenessConfig:
    """Tests for staleness configuration."""

    def test_defaults(self):
        """Test default configuration."""
        config = CacheSettings()

        assert config.staleness_check_interval == 300
        assert config.staleness_threshold == 3600
        assert config.staleness_auto_invalidate is False

    def test_custom_config(self):
        """Test custom configuration."""
        config = CacheSettings(
            staleness_check_interval=600,
            staleness_threshold=7200,
            staleness_auto_invalidate=True,
        )

        assert config.staleness_check_interval == 600
        assert config.staleness_threshold == 7200
        assert config.staleness_auto_invalidate is True


class TestCacheMetrics:
    """Tests for cache metrics."""

    @pytest.fixture
    def metrics(self):
        """Create metrics instance."""
        config = CacheSettings(staleness_check_interval=1, staleness_threshold=2)
        return CacheMetrics(config)

    def test_initialization(self, metrics):
        """Test metrics initialization."""
        assert metrics._global_stats.hits == 0
        assert metrics._global_stats.misses == 0
        assert len(metrics._key_stats) == 0

    def test_record_hit(self, metrics):
        """Test recording cache hit."""
        metrics.record_hit("key1", latency_ms=5.0)

        assert metrics._global_stats.hits == 1
        assert metrics._global_stats.total_latency_ms == 5.0
        assert metrics._key_stats["key1"].hits == 1

    def test_record_miss(self, metrics):
        """Test recording cache miss."""
        metrics.record_miss("key1", latency_ms=3.0)

        assert metrics._global_stats.misses == 1
        assert metrics._global_stats.total_latency_ms == 3.0
        assert metrics._key_stats["key1"].misses == 1

    def test_record_set(self, metrics):
        """Test recording cache set."""
        metrics.record_set("key1")

        assert metrics._global_stats.sets == 1
        assert metrics._key_stats["key1"].sets == 1

    def test_record_delete(self, metrics):
        """Test recording cache delete."""
        metrics.record_set("key1")
        metrics.record_delete("key1")

        assert metrics._global_stats.deletes == 1
        assert "key1" not in metrics._access_times

    def test_record_invalidation(self, metrics):
        """Test recording invalidation."""
        metrics.record_invalidation(5)

        assert metrics._global_stats.invalidations == 5

    def test_multiple_operations(self, metrics):
        """Test recording multiple operations."""
        metrics.record_hit("key1", 5.0)
        metrics.record_miss("key2", 3.0)
        metrics.record_hit("key1", 4.0)
        metrics.record_set("key3")

        assert metrics._global_stats.hits == 2
        assert metrics._global_stats.misses == 1
        assert metrics._global_stats.sets == 1
        assert metrics._global_stats.total_requests == 3

    def test_hit_rate_calculation(self, metrics):
        """Test hit rate calculation."""
        metrics.record_hit("key1")
        metrics.record_hit("key2")
        metrics.record_miss("key3")

        assert metrics._global_stats.hit_rate == pytest.approx(2 / 3)

    def test_check_staleness_interval_not_elapsed(self, metrics):
        """Test staleness check when interval not elapsed."""
        result = metrics.check_staleness()

        assert result["checked"] is False
        assert "next_check_in" in result

    def test_check_staleness_forced(self, metrics):
        """Test forced staleness check."""
        metrics.record_set("key1")

        result = metrics.check_staleness(force=True)

        assert result["checked"] is True
        assert "stale_count" in result
        assert result["stale_count"] == 0

    def test_check_staleness_with_stale_entries(self, metrics):
        """Test staleness detection."""
        # Record old access
        metrics.record_set("old_key")

        # Wait for entry to become stale
        time.sleep(2.5)  # Threshold is 2 seconds

        result = metrics.check_staleness(force=True)

        assert result["checked"] is True
        assert result["stale_count"] >= 1

    def test_get_global_stats(self, metrics):
        """Test getting global statistics."""
        metrics.record_hit("key1")
        metrics.record_miss("key2")

        stats = metrics.get_global_stats()

        assert isinstance(stats, CacheStats)
        assert stats.hits == 1
        assert stats.misses == 1

    def test_get_key_stats(self, metrics):
        """Test getting key-specific statistics."""
        metrics.record_hit("key1")
        metrics.record_hit("key1")
        metrics.record_miss("key1")

        stats = metrics.get_key_stats("key1")

        assert stats.hits == 2
        assert stats.misses == 1

    def test_get_summary(self, metrics):
        """Test getting comprehensive summary."""
        metrics.record_hit("key1")
        metrics.record_miss("key2")

        summary = metrics.get_summary()

        assert "global_stats" in summary
        assert "staleness" in summary
        assert "top_keys_by_hits" in summary
        assert "time_series" in summary

    def test_reset(self, metrics):
        """Test resetting metrics."""
        metrics.record_hit("key1")
        metrics.record_miss("key2")

        metrics.reset()

        assert metrics._global_stats.hits == 0
        assert metrics._global_stats.misses == 0
        assert len(metrics._key_stats) == 0

    def test_meets_target_hit_rate_success(self, metrics):
        """Test meeting target hit rate."""
        # 7 hits, 3 misses = 70% hit rate
        for _ in range(7):
            metrics.record_hit("key1")
        for _ in range(3):
            metrics.record_miss("key2")

        assert metrics.meets_target_hit_rate(0.6)
        assert not metrics.meets_target_hit_rate(0.8)

    def test_meets_target_hit_rate_zero_requests(self, metrics):
        """Test target hit rate with no requests."""
        assert not metrics.meets_target_hit_rate(0.6)

    def test_repr(self, metrics):
        """Test string representation."""
        metrics.record_hit("key1")
        metrics.record_miss("key2")

        repr_str = repr(metrics)

        assert "CacheMetrics" in repr_str
        assert "hit_rate" in repr_str
        assert "requests" in repr_str


class TestCacheTimer:
    """Tests for cache timer context manager."""

    @pytest.fixture
    def metrics(self):
        """Create metrics instance."""
        return CacheMetrics()

    def test_timer_for_hit(self, metrics):
        """Test timing cache hit."""
        with CacheTimer(metrics, "key1", "get") as timer:
            time.sleep(0.01)  # Simulate some work
            timer.mark_hit()

        assert metrics._global_stats.hits == 1
        assert metrics._global_stats.total_latency_ms > 0

    def test_timer_for_miss(self, metrics):
        """Test timing cache miss."""
        with CacheTimer(metrics, "key1", "get"):
            time.sleep(0.01)

        assert metrics._global_stats.misses == 1
        assert metrics._global_stats.total_latency_ms > 0

    def test_timer_for_set(self, metrics):
        """Test timing cache set."""
        with CacheTimer(metrics, "key1", "set"):
            time.sleep(0.01)

        assert metrics._global_stats.sets == 1

    def test_timer_for_delete(self, metrics):
        """Test timing cache delete."""
        metrics.record_set("key1")

        with CacheTimer(metrics, "key1", "delete"):
            time.sleep(0.01)

        assert metrics._global_stats.deletes == 1

    def test_timer_latency_tracking(self, metrics):
        """Test latency tracking accuracy."""
        with CacheTimer(metrics, "key1", "get") as timer:
            time.sleep(0.05)  # 50ms
            timer.mark_hit()

        # Latency should be approximately 50ms (with some tolerance)
        assert 40 < metrics._global_stats.avg_latency_ms < 100


class TestCacheMetricsIntegration:
    """Integration tests for cache metrics."""

    def test_realistic_workflow(self):
        """Test realistic cache workflow."""
        metrics = CacheMetrics()

        # Simulate cache operations
        for i in range(100):
            key = f"key_{i % 10}"

            if i % 3 == 0:  # 33% misses
                metrics.record_miss(key, latency_ms=5.0)
            else:  # 67% hits
                metrics.record_hit(key, latency_ms=2.0)

        stats = metrics.get_global_stats()

        assert stats.total_requests == 100
        assert 0.6 < stats.hit_rate < 0.7
        assert stats.avg_latency_ms > 0

        # Should meet 60% target
        assert metrics.meets_target_hit_rate(0.6)

    def test_per_key_tracking(self):
        """Test per-key statistics tracking."""
        metrics = CacheMetrics()

        # Different patterns for different keys
        for _ in range(10):
            metrics.record_hit("hot_key")  # Very hot
        for _ in range(5):
            metrics.record_miss("cold_key")  # Always miss

        hot_stats = metrics.get_key_stats("hot_key")
        cold_stats = metrics.get_key_stats("cold_key")

        assert hot_stats.hits == 10
        assert hot_stats.hit_rate == 1.0
        assert cold_stats.misses == 5
        assert cold_stats.hit_rate == 0.0
