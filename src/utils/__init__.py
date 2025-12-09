"""
Utility modules for the Code Embedding AI system

This package provides utility functions and classes for:
- Retry logic with exponential backoff
- Error tracking and monitoring
- Other shared utilities
"""

from .retry import (
    retry_async,
    retry_sync,
    with_retry,
    with_retry_sync,
    RetryConfig,
    NETWORK_RETRY_CONFIG,
    DATABASE_RETRY_CONFIG,
    API_RETRY_CONFIG,
    FAST_RETRY_CONFIG
)

from .error_tracker import (
    get_error_tracker,
    reset_error_tracker,
    ErrorTracker,
    ErrorCategory,
    ErrorSeverity,
    ErrorRecord,
    ErrorStatistics
)

__all__ = [
    # Retry utilities
    "retry_async",
    "retry_sync",
    "with_retry",
    "with_retry_sync",
    "RetryConfig",
    "NETWORK_RETRY_CONFIG",
    "DATABASE_RETRY_CONFIG",
    "API_RETRY_CONFIG",
    "FAST_RETRY_CONFIG",

    # Error tracking
    "get_error_tracker",
    "reset_error_tracker",
    "ErrorTracker",
    "ErrorCategory",
    "ErrorSeverity",
    "ErrorRecord",
    "ErrorStatistics",
]
