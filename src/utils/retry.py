import asyncio
import time
from typing import TypeVar, Callable, Optional, Type, Tuple, Any
from functools import wraps
import structlog


logger = structlog.get_logger(__name__)

T = TypeVar('T')


class RetryConfig:
    """Configuration for retry behavior"""

    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None
    ):
        """
        Args:
            max_attempts: Maximum number of retry attempts (including first try)
            initial_delay: Initial delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            exponential_base: Base for exponential backoff (delay = initial * base^attempt)
            jitter: Whether to add random jitter to delay
            retryable_exceptions: Tuple of exception types that should trigger retry.
                                 If None, all exceptions are retryable.
        """
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions or (Exception,)

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number (0-indexed)"""
        import random

        # Exponential backoff
        delay = min(
            self.initial_delay * (self.exponential_base ** attempt),
            self.max_delay
        )

        # Add jitter if enabled (randomize between 50% and 100% of calculated delay)
        if self.jitter:
            delay = delay * (0.5 + random.random() * 0.5)

        return delay

    def is_retryable(self, exception: Exception) -> bool:
        """Check if exception is retryable"""
        return isinstance(exception, self.retryable_exceptions)


async def retry_async(
    func: Callable[..., T],
    *args,
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[Exception, int, float], None]] = None,
    **kwargs
) -> T:
    """
    Retry an async function with exponential backoff

    Args:
        func: Async function to retry
        *args: Positional arguments for func
        config: Retry configuration (uses default if None)
        on_retry: Optional callback called before each retry (exception, attempt, delay)
        **kwargs: Keyword arguments for func

    Returns:
        Result of func

    Raises:
        Last exception if all retries exhausted
    """
    if config is None:
        config = RetryConfig()

    last_exception = None

    for attempt in range(config.max_attempts):
        try:
            return await func(*args, **kwargs)

        except Exception as e:
            last_exception = e

            # Check if this exception is retryable
            if not config.is_retryable(e):
                logger.warning(
                    "Non-retryable exception encountered",
                    exception_type=type(e).__name__,
                    error=str(e)
                )
                raise

            # Check if we have retries left
            if attempt >= config.max_attempts - 1:
                logger.error(
                    "All retry attempts exhausted",
                    max_attempts=config.max_attempts,
                    exception_type=type(e).__name__,
                    error=str(e)
                )
                raise

            # Calculate delay for next retry
            delay = config.calculate_delay(attempt)

            logger.warning(
                "Operation failed, retrying",
                attempt=attempt + 1,
                max_attempts=config.max_attempts,
                delay_seconds=f"{delay:.2f}",
                exception_type=type(e).__name__,
                error=str(e)
            )

            # Call on_retry callback if provided
            if on_retry:
                try:
                    on_retry(e, attempt + 1, delay)
                except Exception as callback_error:
                    logger.error(
                        "Error in retry callback",
                        error=str(callback_error)
                    )

            # Wait before retrying
            await asyncio.sleep(delay)

    # This should never be reached, but for type safety
    if last_exception:
        raise last_exception
    raise RuntimeError("Retry logic error: no exception but all attempts used")


def retry_sync(
    func: Callable[..., T],
    *args,
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[Exception, int, float], None]] = None,
    **kwargs
) -> T:
    """
    Retry a sync function with exponential backoff

    Args:
        func: Sync function to retry
        *args: Positional arguments for func
        config: Retry configuration (uses default if None)
        on_retry: Optional callback called before each retry (exception, attempt, delay)
        **kwargs: Keyword arguments for func

    Returns:
        Result of func

    Raises:
        Last exception if all retries exhausted
    """
    if config is None:
        config = RetryConfig()

    last_exception = None

    for attempt in range(config.max_attempts):
        try:
            return func(*args, **kwargs)

        except Exception as e:
            last_exception = e

            # Check if this exception is retryable
            if not config.is_retryable(e):
                logger.warning(
                    "Non-retryable exception encountered",
                    exception_type=type(e).__name__,
                    error=str(e)
                )
                raise

            # Check if we have retries left
            if attempt >= config.max_attempts - 1:
                logger.error(
                    "All retry attempts exhausted",
                    max_attempts=config.max_attempts,
                    exception_type=type(e).__name__,
                    error=str(e)
                )
                raise

            # Calculate delay for next retry
            delay = config.calculate_delay(attempt)

            logger.warning(
                "Operation failed, retrying",
                attempt=attempt + 1,
                max_attempts=config.max_attempts,
                delay_seconds=f"{delay:.2f}",
                exception_type=type(e).__name__,
                error=str(e)
            )

            # Call on_retry callback if provided
            if on_retry:
                try:
                    on_retry(e, attempt + 1, delay)
                except Exception as callback_error:
                    logger.error(
                        "Error in retry callback",
                        error=str(callback_error)
                    )

            # Wait before retrying
            time.sleep(delay)

    # This should never be reached, but for type safety
    if last_exception:
        raise last_exception
    raise RuntimeError("Retry logic error: no exception but all attempts used")


def with_retry(config: Optional[RetryConfig] = None):
    """
    Decorator for adding retry logic to async functions

    Usage:
        @with_retry(RetryConfig(max_attempts=5, initial_delay=2.0))
        async def my_function():
            # ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await retry_async(func, *args, config=config, **kwargs)
        return wrapper
    return decorator


def with_retry_sync(config: Optional[RetryConfig] = None):
    """
    Decorator for adding retry logic to sync functions

    Usage:
        @with_retry_sync(RetryConfig(max_attempts=5, initial_delay=2.0))
        def my_function():
            # ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return retry_sync(func, *args, config=config, **kwargs)
        return wrapper
    return decorator


# Predefined retry configurations for common scenarios

# Network operations: more retries, longer delays
NETWORK_RETRY_CONFIG = RetryConfig(
    max_attempts=5,
    initial_delay=1.0,
    max_delay=30.0,
    exponential_base=2.0,
    jitter=True
)

# Database operations: moderate retries
DATABASE_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    initial_delay=0.5,
    max_delay=10.0,
    exponential_base=2.0,
    jitter=True
)

# API calls: quick retries
API_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    initial_delay=1.0,
    max_delay=15.0,
    exponential_base=2.0,
    jitter=True
)

# Fast operations: minimal retries
FAST_RETRY_CONFIG = RetryConfig(
    max_attempts=2,
    initial_delay=0.1,
    max_delay=1.0,
    exponential_base=2.0,
    jitter=False
)
