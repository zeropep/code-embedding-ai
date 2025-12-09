import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from threading import Lock
from enum import Enum
import structlog


logger = structlog.get_logger(__name__)


class ErrorSeverity(str, Enum):
    """Error severity levels"""
    LOW = "low"          # Minor issues, no impact on functionality
    MEDIUM = "medium"    # Moderate issues, degraded performance
    HIGH = "high"        # Significant issues, feature unavailable
    CRITICAL = "critical"  # System-wide failures


class ErrorCategory(str, Enum):
    """Error categories for classification"""
    NETWORK = "network"              # Network/connection errors
    DATABASE = "database"            # Database operation errors
    EMBEDDING = "embedding"          # Embedding generation errors
    VALIDATION = "validation"        # Input validation errors
    AUTHENTICATION = "authentication"  # Auth/permission errors
    INTERNAL = "internal"            # Internal system errors
    EXTERNAL_API = "external_api"    # External API errors
    TIMEOUT = "timeout"              # Timeout errors
    UNKNOWN = "unknown"              # Uncategorized errors


@dataclass
class ErrorRecord:
    """Individual error record"""
    timestamp: float
    category: ErrorCategory
    severity: ErrorSeverity
    error_type: str
    error_message: str
    context: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    resolved: bool = False


@dataclass
class ErrorStatistics:
    """Aggregated error statistics"""
    total_errors: int = 0
    errors_by_category: Dict[ErrorCategory, int] = field(default_factory=lambda: defaultdict(int))
    errors_by_severity: Dict[ErrorSeverity, int] = field(default_factory=lambda: defaultdict(int))
    errors_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    recent_errors: List[ErrorRecord] = field(default_factory=list)
    error_rate_per_minute: float = 0.0
    first_error_timestamp: Optional[float] = None
    last_error_timestamp: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "total_errors": self.total_errors,
            "errors_by_category": {k.value: v for k, v in self.errors_by_category.items()},
            "errors_by_severity": {k.value: v for k, v in self.errors_by_severity.items()},
            "errors_by_type": dict(self.errors_by_type),
            "error_rate_per_minute": round(self.error_rate_per_minute, 2),
            "first_error_timestamp": self.first_error_timestamp,
            "last_error_timestamp": self.last_error_timestamp,
            "recent_errors_count": len(self.recent_errors),
            "recent_errors": [
                {
                    "timestamp": err.timestamp,
                    "category": err.category.value,
                    "severity": err.severity.value,
                    "error_type": err.error_type,
                    "error_message": err.error_message,
                    "context": err.context,
                    "resolved": err.resolved
                }
                for err in self.recent_errors[:10]  # Return last 10 errors
            ]
        }


class ErrorTracker:
    """Thread-safe error tracker for monitoring system errors"""

    def __init__(self, max_recent_errors: int = 100):
        """
        Args:
            max_recent_errors: Maximum number of recent errors to keep in memory
        """
        self.max_recent_errors = max_recent_errors
        self._errors: List[ErrorRecord] = []
        self._stats = ErrorStatistics()
        self._lock = Lock()
        self._start_time = time.time()

        logger.info("ErrorTracker initialized", max_recent_errors=max_recent_errors)

    def record_error(
        self,
        error: Exception,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        stack_trace: Optional[str] = None
    ) -> ErrorRecord:
        """
        Record an error occurrence

        Args:
            error: The exception that occurred
            category: Error category
            severity: Error severity level
            context: Additional context information
            stack_trace: Optional stack trace string

        Returns:
            ErrorRecord for the recorded error
        """
        error_record = ErrorRecord(
            timestamp=time.time(),
            category=category,
            severity=severity,
            error_type=type(error).__name__,
            error_message=str(error),
            context=context or {},
            stack_trace=stack_trace,
            resolved=False
        )

        with self._lock:
            # Add to recent errors list
            self._errors.append(error_record)

            # Trim if exceeds max size
            if len(self._errors) > self.max_recent_errors:
                self._errors = self._errors[-self.max_recent_errors:]

            # Update statistics
            self._stats.total_errors += 1
            self._stats.errors_by_category[category] += 1
            self._stats.errors_by_severity[severity] += 1
            self._stats.errors_by_type[error_record.error_type] += 1
            self._stats.recent_errors = self._errors.copy()

            # Update timestamps
            if self._stats.first_error_timestamp is None:
                self._stats.first_error_timestamp = error_record.timestamp
            self._stats.last_error_timestamp = error_record.timestamp

            # Calculate error rate
            self._update_error_rate()

        logger.warning(
            "Error recorded",
            category=category.value,
            severity=severity.value,
            error_type=error_record.error_type,
            error_message=error_record.error_message,
            total_errors=self._stats.total_errors
        )

        return error_record

    def _update_error_rate(self):
        """Update the error rate calculation (errors per minute)"""
        if not self._errors:
            self._stats.error_rate_per_minute = 0.0
            return

        # Calculate error rate over last minute
        current_time = time.time()
        one_minute_ago = current_time - 60

        recent_errors = [
            err for err in self._errors
            if err.timestamp >= one_minute_ago
        ]

        self._stats.error_rate_per_minute = len(recent_errors)

    def get_statistics(
        self,
        category: Optional[ErrorCategory] = None,
        severity: Optional[ErrorSeverity] = None,
        time_window_seconds: Optional[float] = None
    ) -> ErrorStatistics:
        """
        Get error statistics with optional filtering

        Args:
            category: Filter by error category
            severity: Filter by severity level
            time_window_seconds: Only include errors from last N seconds

        Returns:
            ErrorStatistics object
        """
        with self._lock:
            # Apply filters if specified
            filtered_errors = self._errors

            if time_window_seconds:
                cutoff_time = time.time() - time_window_seconds
                filtered_errors = [
                    err for err in filtered_errors
                    if err.timestamp >= cutoff_time
                ]

            if category:
                filtered_errors = [
                    err for err in filtered_errors
                    if err.category == category
                ]

            if severity:
                filtered_errors = [
                    err for err in filtered_errors
                    if err.severity == severity
                ]

            # Build filtered statistics
            filtered_stats = ErrorStatistics(
                total_errors=len(filtered_errors),
                recent_errors=filtered_errors.copy()
            )

            for err in filtered_errors:
                filtered_stats.errors_by_category[err.category] += 1
                filtered_stats.errors_by_severity[err.severity] += 1
                filtered_stats.errors_by_type[err.error_type] += 1

            if filtered_errors:
                filtered_stats.first_error_timestamp = filtered_errors[0].timestamp
                filtered_stats.last_error_timestamp = filtered_errors[-1].timestamp

            # Calculate error rate for filtered results
            if time_window_seconds and filtered_errors:
                minutes = time_window_seconds / 60.0
                filtered_stats.error_rate_per_minute = len(filtered_errors) / minutes if minutes > 0 else 0

            return filtered_stats

    def get_errors_by_category(self, category: ErrorCategory, limit: int = 10) -> List[ErrorRecord]:
        """Get recent errors for a specific category"""
        with self._lock:
            category_errors = [
                err for err in reversed(self._errors)
                if err.category == category
            ]
            return category_errors[:limit]

    def get_errors_by_severity(self, severity: ErrorSeverity, limit: int = 10) -> List[ErrorRecord]:
        """Get recent errors for a specific severity level"""
        with self._lock:
            severity_errors = [
                err for err in reversed(self._errors)
                if err.severity == severity
            ]
            return severity_errors[:limit]

    def get_critical_errors(self, limit: int = 10) -> List[ErrorRecord]:
        """Get recent critical errors"""
        return self.get_errors_by_severity(ErrorSeverity.CRITICAL, limit)

    def mark_resolved(self, error_record: ErrorRecord):
        """Mark an error as resolved"""
        with self._lock:
            if error_record in self._errors:
                error_record.resolved = True
                logger.info(
                    "Error marked as resolved",
                    error_type=error_record.error_type,
                    timestamp=error_record.timestamp
                )

    def clear_errors(self):
        """Clear all error records"""
        with self._lock:
            self._errors.clear()
            self._stats = ErrorStatistics()
            self._start_time = time.time()
            logger.info("Error tracker cleared")

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get overall health status based on error patterns

        Returns:
            Dictionary with health status information
        """
        with self._lock:
            # Get errors in last 5 minutes
            recent_stats = self.get_statistics(time_window_seconds=300)

            # Determine health status
            status = "healthy"
            issues = []

            # Check critical errors
            critical_count = recent_stats.errors_by_severity.get(ErrorSeverity.CRITICAL, 0)
            if critical_count > 0:
                status = "critical"
                issues.append(f"{critical_count} critical errors in last 5 minutes")

            # Check high severity errors
            high_count = recent_stats.errors_by_severity.get(ErrorSeverity.HIGH, 0)
            if high_count > 5:
                if status != "critical":
                    status = "degraded"
                issues.append(f"{high_count} high-severity errors in last 5 minutes")

            # Check error rate
            if recent_stats.error_rate_per_minute > 10:
                if status == "healthy":
                    status = "degraded"
                issues.append(f"High error rate: {recent_stats.error_rate_per_minute:.1f} errors/minute")

            return {
                "status": status,
                "total_errors_lifetime": self._stats.total_errors,
                "errors_last_5_min": recent_stats.total_errors,
                "error_rate_per_minute": recent_stats.error_rate_per_minute,
                "critical_errors": critical_count,
                "high_severity_errors": high_count,
                "issues": issues,
                "uptime_seconds": time.time() - self._start_time
            }


# Global error tracker instance
_global_error_tracker: Optional[ErrorTracker] = None
_tracker_lock = Lock()


def get_error_tracker() -> ErrorTracker:
    """Get the global error tracker instance (singleton)"""
    global _global_error_tracker

    if _global_error_tracker is None:
        with _tracker_lock:
            if _global_error_tracker is None:
                _global_error_tracker = ErrorTracker(max_recent_errors=100)

    return _global_error_tracker


def reset_error_tracker():
    """Reset the global error tracker (useful for testing)"""
    global _global_error_tracker

    with _tracker_lock:
        _global_error_tracker = ErrorTracker(max_recent_errors=100)
