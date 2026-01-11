"""
Cache metrics and monitoring for tracking performance and staleness.
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from src.config import CacheSettings

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Cache statistics."""

    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    invalidations: int = 0
    total_latency_ms: float = 0.0
    stale_entries: int = 0

    @property
    def total_requests(self) -> int:
        """Total cache requests (hits + misses)."""
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0.0-1.0)."""
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests

    @property
    def avg_latency_ms(self) -> float:
        """Average latency in milliseconds."""
        if self.total_requests == 0:
            return 0.0
        return self.total_latency_ms / self.total_requests

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "deletes": self.deletes,
            "invalidations": self.invalidations,
            "total_requests": self.total_requests,
            "hit_rate": round(self.hit_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "stale_entries": self.stale_entries,
        }


class CacheMetrics:
    """
    Cache metrics tracker with staleness monitoring.

    Tracks:
    - Hit/miss rates
    - Latency
    - Cache operations
    - Staleness detection
    """

    def __init__(
        self,
        staleness_config: CacheSettings | None = None,
    ):
        """
        Initialize cache metrics.

        Args:
            staleness_config: Configuration for staleness monitoring
        """
        self.staleness_config = staleness_config

        # Global stats
        self._global_stats = CacheStats()

        # Per-key stats
        self._key_stats: dict[str, CacheStats] = defaultdict(CacheStats)

        # Staleness tracking
        self._access_times: dict[str, datetime] = {}
        self._last_staleness_check = datetime.now()

        # Time-series data (last hour, grouped by minute)
        self._time_series: dict[str, list[tuple[datetime, float]]] = {
            "hit_rate": [],
            "latency": [],
        }

    def record_hit(self, key: str, latency_ms: float = 0.0) -> None:
        """
        Record cache hit.

        Args:
            key: Cache key
            latency_ms: Operation latency in milliseconds
        """
        self._global_stats.hits += 1
        self._global_stats.total_latency_ms += latency_ms

        self._key_stats[key].hits += 1
        self._key_stats[key].total_latency_ms += latency_ms

        # Update access time
        self._access_times[key] = datetime.now()

        # Record time-series data
        self._record_time_series("hit_rate", self._global_stats.hit_rate)

    def record_miss(self, key: str, latency_ms: float = 0.0) -> None:
        """
        Record cache miss.

        Args:
            key: Cache key
            latency_ms: Operation latency in milliseconds
        """
        self._global_stats.misses += 1
        self._global_stats.total_latency_ms += latency_ms

        self._key_stats[key].misses += 1
        self._key_stats[key].total_latency_ms += latency_ms

        # Record time-series data
        self._record_time_series("hit_rate", self._global_stats.hit_rate)

    def record_set(self, key: str) -> None:
        """
        Record cache set operation.

        Args:
            key: Cache key
        """
        self._global_stats.sets += 1
        self._key_stats[key].sets += 1

        # Update access time
        self._access_times[key] = datetime.now()

    def record_delete(self, key: str) -> None:
        """
        Record cache delete operation.

        Args:
            key: Cache key
        """
        self._global_stats.deletes += 1
        self._key_stats[key].deletes += 1

        # Remove from access times
        self._access_times.pop(key, None)

    def record_invalidation(self, count: int = 1) -> None:
        """
        Record cache invalidation.

        Args:
            count: Number of entries invalidated
        """
        self._global_stats.invalidations += count

    def _record_time_series(self, metric: str, value: float) -> None:
        """Record time-series data point."""
        now = datetime.now()
        self._time_series[metric].append((now, value))

        # Keep only last hour of data
        cutoff = now - timedelta(hours=1)
        self._time_series[metric] = [
            (ts, val) for ts, val in self._time_series[metric] if ts > cutoff
        ]

    def check_staleness(self, force: bool = False) -> dict[str, Any]:
        """
        Check for stale cache entries.

        Args:
            force: Force staleness check even if interval hasn't elapsed

        Returns:
            Dictionary with staleness information
        """
        now = datetime.now()

        # Check if interval elapsed
        elapsed = (now - self._last_staleness_check).total_seconds()
        if self.staleness_config is None:
            return {"checked": False, "error": "No staleness config"}

        if not force and elapsed < self.staleness_config.staleness_check_interval:
            return {
                "checked": False,
                "next_check_in": int(self.staleness_config.staleness_check_interval - elapsed),
            }

        # Check for stale entries
        stale_keys = []
        threshold = timedelta(seconds=self.staleness_config.staleness_threshold)

        for key, last_access in self._access_times.items():
            if now - last_access > threshold:
                stale_keys.append(key)

        # Update global stats
        self._global_stats.stale_entries = len(stale_keys)
        self._last_staleness_check = now

        logger.info(f"Staleness check: {len(stale_keys)} stale entries found")

        return {
            "checked": True,
            "stale_count": len(stale_keys),
            "stale_keys": stale_keys[:10],  # Sample of stale keys
            "total_entries": len(self._access_times),
            "staleness_threshold_seconds": self.staleness_config.staleness_threshold,
        }

    def get_global_stats(self) -> CacheStats:
        """Get global cache statistics."""
        return self._global_stats

    def get_key_stats(self, key: str) -> CacheStats:
        """
        Get statistics for specific key.

        Args:
            key: Cache key

        Returns:
            Statistics for the key
        """
        return self._key_stats[key]

    def get_summary(self) -> dict[str, Any]:
        """
        Get comprehensive metrics summary.

        Returns:
            Dictionary with all metrics
        """
        staleness_info = self.check_staleness()

        return {
            "global_stats": self._global_stats.to_dict(),
            "staleness": staleness_info,
            "top_keys_by_hits": self._get_top_keys("hits", limit=10),
            "top_keys_by_misses": self._get_top_keys("misses", limit=10),
            "time_series": {
                metric: [(ts.isoformat(), float(val)) for ts, val in series[-60:]]
                for metric, series in self._time_series.items()
            },
        }

    def _get_top_keys(self, metric: str, limit: int = 10) -> list[tuple[str, int]]:
        """Get top keys by metric."""
        sorted_keys = sorted(
            self._key_stats.items(),
            key=lambda x: getattr(x[1], metric),
            reverse=True,
        )
        return [(key, getattr(stats, metric)) for key, stats in sorted_keys[:limit]]

    def reset(self) -> None:
        """Reset all metrics."""
        self._global_stats = CacheStats()
        self._key_stats.clear()
        self._access_times.clear()
        self._time_series = {
            "hit_rate": [],
            "latency": [],
        }
        logger.info("Cache metrics reset")

    def meets_target_hit_rate(self, target: float = 0.6) -> bool:
        """
        Check if cache hit rate meets target.

        Args:
            target: Target hit rate (default 60%)

        Returns:
            True if hit rate >= target
        """
        return self._global_stats.hit_rate >= target

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CacheMetrics(hit_rate={self._global_stats.hit_rate:.2%}, "
            f"requests={self._global_stats.total_requests})"
        )


class CacheTimer:
    """Context manager for timing cache operations."""

    def __init__(self, metrics: CacheMetrics, key: str, operation: str = "get"):
        """
        Initialize cache timer.

        Args:
            metrics: Cache metrics instance
            key: Cache key
            operation: Operation type ('get', 'set', etc.)
        """
        self.metrics = metrics
        self.key = key
        self.operation = operation
        self.start_time: float | None = None
        self.hit: bool = False

    def __enter__(self):
        """Start timing."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and record metrics."""
        if self.start_time is None:
            return

        latency_ms = (time.perf_counter() - self.start_time) * 1000

        if self.operation == "get":
            if self.hit:
                self.metrics.record_hit(self.key, latency_ms)
            else:
                self.metrics.record_miss(self.key, latency_ms)
        elif self.operation == "set":
            self.metrics.record_set(self.key)
        elif self.operation == "delete":
            self.metrics.record_delete(self.key)

    def mark_hit(self) -> None:
        """Mark operation as cache hit."""
        self.hit = True
