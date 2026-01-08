"""
Retry and backoff utilities for resilient external service calls.

Provides decorators and utilities for implementing exponential backoff
with jitter for handling transient failures.
"""

import asyncio
import functools
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

# Type variable for generic function return type
T = TypeVar("T")


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    initial_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True
    # Exceptions to retry on (tuple of exception types)
    retryable_exceptions: tuple[type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
        OSError,
    )
    # Exceptions to never retry (tuple of exception types)
    fatal_exceptions: tuple[type[Exception], ...] = (
        ValueError,
        TypeError,
        KeyError,
    )


def calculate_delay(
    attempt: int,
    initial_delay: float = 1.0,
    exponential_base: float = 2.0,
    max_delay: float = 60.0,
    jitter: bool = True,
) -> float:
    """
    Calculate exponential backoff delay with optional jitter.

    Args:
        attempt: Current attempt number (0-indexed)
        initial_delay: Base delay in seconds
        exponential_base: Exponential base for backoff
        max_delay: Maximum delay in seconds
        jitter: Whether to add random jitter

    Returns:
        Delay in seconds
    """
    # Calculate exponential backoff: initial_delay * base^attempt
    delay = min(initial_delay * (exponential_base**attempt), max_delay)

    # Add jitter: random value between 0 and calculated delay
    if jitter:
        delay = random.uniform(0, delay)

    return delay


def should_retry(
    exception: Exception,
    retryable_exceptions: tuple[type[Exception], ...],
    fatal_exceptions: tuple[type[Exception], ...],
) -> bool:
    """
    Determine if an exception should trigger a retry.

    Args:
        exception: The raised exception
        retryable_exceptions: Tuple of exception types that should be retried
        fatal_exceptions: Tuple of exception types that should never be retried

    Returns:
        True if the exception should trigger a retry, False otherwise
    """
    # Never retry fatal exceptions
    if isinstance(exception, fatal_exceptions):
        return False

    # Retry if it's a retryable exception
    return isinstance(exception, retryable_exceptions)


def retry_with_backoff(config: RetryConfig | None = None):
    """
    Decorator for synchronous functions with exponential backoff retry.

    Args:
        config: RetryConfig instance. If None, uses default configuration.

    Example:
        @retry_with_backoff(RetryConfig(max_retries=5, initial_delay=2.0))
        def fetch_data():
            response = requests.get("https://api.example.com/data")
            response.raise_for_status()
            return response.json()
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None

            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    # Check if we should retry
                    if not should_retry(e, config.retryable_exceptions, config.fatal_exceptions):
                        logger.error(f"{func.__name__}: Fatal exception, not retrying: {e}")
                        raise

                    # Check if we've exhausted retries
                    if attempt >= config.max_retries:
                        logger.error(
                            f"{func.__name__}: Max retries ({config.max_retries}) exceeded"
                        )
                        break

                    # Calculate delay and wait
                    delay = calculate_delay(
                        attempt,
                        config.initial_delay,
                        config.exponential_base,
                        config.max_delay,
                        config.jitter,
                    )

                    logger.warning(
                        f"{func.__name__}: Attempt {attempt + 1}/{config.max_retries + 1} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )

                    time.sleep(delay)

            # If we get here, all retries failed
            logger.error(
                f"{func.__name__}: All retry attempts failed. Last exception: {last_exception}"
            )
            raise last_exception

        return wrapper

    return decorator


def async_retry_with_backoff(config: RetryConfig | None = None):
    """
    Decorator for asynchronous functions with exponential backoff retry.

    Args:
        config: RetryConfig instance. If None, uses default configuration.

    Example:
        @async_retry_with_backoff(RetryConfig(max_retries=5, initial_delay=2.0))
        async def fetch_data():
            async with httpx.AsyncClient() as client:
                response = await client.get("https://api.example.com/data")
                response.raise_for_status()
                return response.json()
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None

            for attempt in range(config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    # Check if we should retry
                    if not should_retry(e, config.retryable_exceptions, config.fatal_exceptions):
                        logger.error(f"{func.__name__}: Fatal exception, not retrying: {e}")
                        raise

                    # Check if we've exhausted retries
                    if attempt >= config.max_retries:
                        logger.error(
                            f"{func.__name__}: Max retries ({config.max_retries}) exceeded"
                        )
                        break

                    # Calculate delay and wait
                    delay = calculate_delay(
                        attempt,
                        config.initial_delay,
                        config.exponential_base,
                        config.max_delay,
                        config.jitter,
                    )

                    logger.warning(
                        f"{func.__name__}: Attempt {attempt + 1}/{config.max_retries + 1} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )

                    await asyncio.sleep(delay)

            # If we get here, all retries failed
            logger.error(
                f"{func.__name__}: All retry attempts failed. Last exception: {last_exception}"
            )
            raise last_exception

        return wrapper

    return decorator


class RetryableClient:
    """
    Base class for clients that need retry logic.

    Provides a convenient way to configure retry behavior for all methods
    of a client class.
    """

    def __init__(self, retry_config: RetryConfig | None = None):
        """
        Initialize retryable client.

        Args:
            retry_config: RetryConfig instance. If None, uses default configuration.
        """
        self.retry_config = retry_config or RetryConfig()

    def with_retry(self, func: Callable[..., T]) -> Callable[..., T]:
        """
        Wrap a function with retry logic.

        Args:
            func: Function to wrap

        Returns:
            Wrapped function with retry logic
        """
        return retry_with_backoff(self.retry_config)(func)

    def with_async_retry(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """
        Wrap an async function with retry logic.

        Args:
            func: Async function to wrap

        Returns:
            Wrapped async function with retry logic
        """
        return async_retry_with_backoff(self.retry_config)(func)
