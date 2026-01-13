"""
Tests for retry and backoff utilities.
"""

import asyncio
import time

import pytest

from src.services.retry import (
    RetryableClient,
    RetryConfig,
    async_retry_with_backoff,
    calculate_delay,
    retry_with_backoff,
    should_retry,
)


class TestCalculateDelay:
    """Tests for exponential backoff delay calculation."""

    def test_basic_delay_calculation(self):
        """Test basic exponential backoff calculation."""
        # attempt 0: 1.0 * 2^0 = 1.0
        delay = calculate_delay(0, initial_delay=1.0, exponential_base=2.0, jitter=False)
        assert delay == 1.0

        # attempt 1: 1.0 * 2^1 = 2.0
        delay = calculate_delay(1, initial_delay=1.0, exponential_base=2.0, jitter=False)
        assert delay == 2.0

        # attempt 2: 1.0 * 2^2 = 4.0
        delay = calculate_delay(2, initial_delay=1.0, exponential_base=2.0, jitter=False)
        assert delay == 4.0

    def test_max_delay_cap(self):
        """Test that delay is capped at max_delay."""
        # attempt 10 would be 1024, but should be capped at 10
        delay = calculate_delay(
            10, initial_delay=1.0, exponential_base=2.0, max_delay=10.0, jitter=False
        )
        assert delay == 10.0

    def test_jitter_adds_randomness(self):
        """Test that jitter creates random delays."""
        delays = [
            calculate_delay(3, initial_delay=1.0, exponential_base=2.0, jitter=True)
            for _ in range(10)
        ]
        # With jitter, delays should vary
        assert len(set(delays)) > 1
        # All delays should be between 0 and 8 (1.0 * 2^3)
        assert all(0 <= d <= 8.0 for d in delays)

    def test_custom_parameters(self):
        """Test with custom initial delay and base."""
        delay = calculate_delay(
            2, initial_delay=2.0, exponential_base=3.0, max_delay=100.0, jitter=False
        )
        # 2.0 * 3^2 = 18.0
        assert delay == 18.0


class TestShouldRetry:
    """Tests for retry decision logic."""

    def test_retry_on_retryable_exceptions(self):
        """Test that retryable exceptions trigger retry."""
        config = RetryConfig()
        assert should_retry(
            ConnectionError("timeout"), config.retryable_exceptions, config.fatal_exceptions
        )
        assert should_retry(
            TimeoutError("timeout"), config.retryable_exceptions, config.fatal_exceptions
        )
        assert should_retry(
            OSError("network error"), config.retryable_exceptions, config.fatal_exceptions
        )

    def test_no_retry_on_fatal_exceptions(self):
        """Test that fatal exceptions don't trigger retry."""
        config = RetryConfig()
        assert not should_retry(
            ValueError("bad value"), config.retryable_exceptions, config.fatal_exceptions
        )
        assert not should_retry(
            TypeError("bad type"), config.retryable_exceptions, config.fatal_exceptions
        )
        assert not should_retry(
            KeyError("missing key"), config.retryable_exceptions, config.fatal_exceptions
        )

    def test_no_retry_on_unknown_exceptions(self):
        """Test that unknown exceptions don't trigger retry by default."""
        config = RetryConfig()
        assert not should_retry(
            RuntimeError("unknown error"),
            config.retryable_exceptions,
            config.fatal_exceptions,
        )


class TestRetryWithBackoff:
    """Tests for synchronous retry decorator."""

    def test_success_on_first_try(self):
        """Test that successful calls don't retry."""
        call_count = 0

        @retry_with_backoff(RetryConfig(max_retries=3))
        def succeed():
            nonlocal call_count
            call_count += 1
            return "success"

        result = succeed()
        assert result == "success"
        assert call_count == 1

    def test_success_after_retries(self):
        """Test that function succeeds after transient failures."""
        call_count = 0

        @retry_with_backoff(RetryConfig(max_retries=3, initial_delay=0.01))
        def succeed_on_third_try():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("temporary failure")
            return "success"

        result = succeed_on_third_try()
        assert result == "success"
        assert call_count == 3

    def test_max_retries_exceeded(self):
        """Test that function fails after max retries."""
        call_count = 0

        @retry_with_backoff(RetryConfig(max_retries=2, initial_delay=0.01))
        def always_fail():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("persistent failure")

        with pytest.raises(ConnectionError, match="persistent failure"):
            always_fail()
        assert call_count == 3  # 1 initial + 2 retries

    def test_fatal_exception_no_retry(self):
        """Test that fatal exceptions don't trigger retries."""
        call_count = 0

        @retry_with_backoff(RetryConfig(max_retries=3))
        def raise_fatal():
            nonlocal call_count
            call_count += 1
            raise ValueError("fatal error")

        with pytest.raises(ValueError, match="fatal error"):
            raise_fatal()
        assert call_count == 1  # No retries

    def test_retry_delay_timing(self):
        """Test that retry delays are applied correctly."""
        call_times = []

        @retry_with_backoff(
            RetryConfig(max_retries=2, initial_delay=0.1, exponential_base=2.0, jitter=False)
        )
        def fail_with_timing():
            call_times.append(time.time())
            if len(call_times) < 3:
                raise ConnectionError("temporary")
            return "success"

        fail_with_timing()

        # Check delays between calls
        # First delay: ~0.1s, Second delay: ~0.2s
        if len(call_times) >= 2:
            delay1 = call_times[1] - call_times[0]
            assert 0.08 <= delay1 <= 0.15  # Allow some tolerance
        if len(call_times) >= 3:
            delay2 = call_times[2] - call_times[1]
            assert 0.18 <= delay2 <= 0.25


class TestAsyncRetryWithBackoff:
    """Tests for asynchronous retry decorator."""

    @pytest.mark.asyncio
    async def test_async_success_on_first_try(self):
        """Test that successful async calls don't retry."""
        call_count = 0

        @async_retry_with_backoff(RetryConfig(max_retries=3))
        async def async_succeed():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.001)
            return "success"

        result = await async_succeed()
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_async_success_after_retries(self):
        """Test that async function succeeds after transient failures."""
        call_count = 0

        @async_retry_with_backoff(RetryConfig(max_retries=3, initial_delay=0.01))
        async def async_succeed_on_third_try():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("temporary failure")
            return "success"

        result = await async_succeed_on_third_try()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_async_max_retries_exceeded(self):
        """Test that async function fails after max retries."""
        call_count = 0

        @async_retry_with_backoff(RetryConfig(max_retries=2, initial_delay=0.01))
        async def async_always_fail():
            nonlocal call_count
            call_count += 1
            raise TimeoutError("persistent timeout")

        with pytest.raises(TimeoutError, match="persistent timeout"):
            await async_always_fail()
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_async_fatal_exception_no_retry(self):
        """Test that fatal exceptions in async functions don't trigger retries."""
        call_count = 0

        @async_retry_with_backoff(RetryConfig(max_retries=3))
        async def async_raise_fatal():
            nonlocal call_count
            call_count += 1
            raise TypeError("fatal type error")

        with pytest.raises(TypeError, match="fatal type error"):
            await async_raise_fatal()
        assert call_count == 1


class TestRetryableClient:
    """Tests for RetryableClient base class."""

    def test_retryable_client_initialization(self):
        """Test RetryableClient initialization."""
        config = RetryConfig(max_retries=5)
        client = RetryableClient(config)
        assert client.retry_config.max_retries == 5

    def test_retryable_client_default_config(self):
        """Test RetryableClient with default config."""
        client = RetryableClient()
        assert client.retry_config.max_retries == 3

    def test_with_retry_wrapper(self):
        """Test with_retry method wraps functions correctly."""
        client = RetryableClient(RetryConfig(max_retries=2, initial_delay=0.01))

        call_count = 0

        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("temporary")
            return "success"

        wrapped = client.with_retry(flaky_function)
        result = wrapped()
        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_with_async_retry_wrapper(self):
        """Test with_async_retry method wraps async functions correctly."""
        client = RetryableClient(RetryConfig(max_retries=2, initial_delay=0.01))

        call_count = 0

        async def async_flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("temporary")
            return "success"

        wrapped = client.with_async_retry(async_flaky_function)
        result = await wrapped()
        assert result == "success"
        assert call_count == 2


class TestRetryIntegration:
    """Integration tests for retry functionality."""

    def test_custom_retryable_exceptions(self):
        """Test retry with custom retryable exceptions."""
        call_count = 0

        class CustomError(Exception):
            pass

        @retry_with_backoff(
            RetryConfig(
                max_retries=2,
                initial_delay=0.01,
                retryable_exceptions=(CustomError,),
                fatal_exceptions=(ValueError,),
            )
        )
        def custom_retry():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise CustomError("should retry")
            return "success"

        result = custom_retry()
        assert result == "success"
        assert call_count == 2

    def test_custom_fatal_exceptions(self):
        """Test that custom fatal exceptions prevent retry."""
        call_count = 0

        class CustomFatalError(Exception):
            pass

        @retry_with_backoff(
            RetryConfig(
                max_retries=3,
                retryable_exceptions=(Exception,),
                fatal_exceptions=(CustomFatalError,),
            )
        )
        def raise_custom_fatal():
            nonlocal call_count
            call_count += 1
            raise CustomFatalError("should not retry")

        with pytest.raises(CustomFatalError):
            raise_custom_fatal()
        assert call_count == 1

    def test_exponential_backoff_progression(self):
        """Test that delays follow exponential progression."""
        call_times = []

        @retry_with_backoff(
            RetryConfig(
                max_retries=3,
                initial_delay=0.05,
                exponential_base=2.0,
                jitter=False,
            )
        )
        def track_timing():
            call_times.append(time.time())
            if len(call_times) < 4:
                raise ConnectionError("retry")
            return "done"

        track_timing()

        # Verify exponential progression in delays
        delays = [call_times[i + 1] - call_times[i] for i in range(len(call_times) - 1)]
        # First delay ~0.05, second ~0.1, third ~0.2
        assert len(delays) == 3
        # Each delay should be approximately double the previous
        # (with some tolerance for timing precision)
        assert delays[1] > delays[0] * 1.5
        assert delays[2] > delays[1] * 1.5
