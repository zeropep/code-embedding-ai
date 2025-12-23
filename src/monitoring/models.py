from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum
import time
import threading


class LogLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(Enum):
    PERFORMANCE = "performance"
    ERROR_RATE = "error_rate"
    RESOURCE_USAGE = "resource_usage"
    SERVICE_HEALTH = "service_health"
    SECURITY = "security"


class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class LogEntry:
    """Structured log entry"""
    timestamp: float
    level: LogLevel
    component: str
    message: str
    context: Dict[str, Any] = field(default_factory=dict)
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "level": self.level.value,
            "component": self.component,
            "message": self.message,
            "context": self.context,
            "request_id": self.request_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "error_details": self.error_details
        }


@dataclass
class MetricValue:
    """Individual metric measurement"""
    name: str
    value: float
    metric_type: MetricType
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "type": self.metric_type.value,
            "timestamp": self.timestamp,
            "labels": self.labels,
            "description": self.description
        }


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot"""
    timestamp: float = field(default_factory=time.time)

    # Request metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0

    # Processing metrics
    embeddings_generated: int = 0
    chunks_processed: int = 0
    files_parsed: int = 0
    security_scans_completed: int = 0

    # Error metrics
    parsing_errors: int = 0
    embedding_errors: int = 0
    database_errors: int = 0
    security_errors: int = 0

    # Resource metrics
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    disk_usage_mb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "requests": {
                "total": self.total_requests,
                "successful": self.successful_requests,
                "failed": self.failed_requests,
                "success_rate": self.successful_requests / max(self.total_requests, 1),
                "avg_response_time_ms": self.avg_response_time_ms,
                "p95_response_time_ms": self.p95_response_time_ms,
                "p99_response_time_ms": self.p99_response_time_ms
            },
            "processing": {
                "embeddings_generated": self.embeddings_generated,
                "chunks_processed": self.chunks_processed,
                "files_parsed": self.files_parsed,
                "security_scans_completed": self.security_scans_completed
            },
            "errors": {
                "parsing_errors": self.parsing_errors,
                "embedding_errors": self.embedding_errors,
                "database_errors": self.database_errors,
                "security_errors": self.security_errors,
                "total_errors": (self.parsing_errors + self.embedding_errors +
                                 self.database_errors + self.security_errors)
            },
            "resources": {
                "memory_usage_mb": self.memory_usage_mb,
                "cpu_usage_percent": self.cpu_usage_percent,
                "disk_usage_mb": self.disk_usage_mb
            }
        }


@dataclass
class Alert:
    """Alert/notification"""
    id: str
    alert_type: AlertType
    severity: LogLevel
    title: str
    message: str
    triggered_at: float = field(default_factory=time.time)
    resolved_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_resolved(self) -> bool:
        return self.resolved_at is not None

    @property
    def duration_seconds(self) -> float:
        end_time = self.resolved_at or time.time()
        return end_time - self.triggered_at

    def resolve(self):
        """Mark alert as resolved"""
        self.resolved_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "triggered_at": self.triggered_at,
            "resolved_at": self.resolved_at,
            "is_resolved": self.is_resolved,
            "duration_seconds": self.duration_seconds,
            "metadata": self.metadata
        }


@dataclass
class HealthCheck:
    """Health check result"""
    component: str
    status: str  # "healthy", "degraded", "unhealthy"
    message: str
    checked_at: float = field(default_factory=time.time)
    response_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "status": self.status,
            "message": self.message,
            "checked_at": self.checked_at,
            "response_time_ms": self.response_time_ms,
            "metadata": self.metadata
        }


@dataclass
class MonitoringConfig:
    """Configuration for monitoring system"""
    # Logging configuration
    log_level: LogLevel = LogLevel.INFO
    log_file_path: Optional[str] = "logs/pipeline.log"
    log_rotation_size_mb: int = 10
    log_retention_days: int = 30
    enable_structured_logging: bool = True
    enable_console_logging: bool = True

    # Metrics configuration
    enable_metrics: bool = True
    metrics_collection_interval_seconds: int = 60
    metrics_retention_hours: int = 24

    # Alerting configuration
    enable_alerting: bool = True
    error_rate_threshold: float = 0.1  # 10% error rate
    response_time_threshold_ms: float = 5000  # 5 seconds
    memory_usage_threshold_mb: float = 1000  # 1GB
    cpu_usage_threshold_percent: float = 80.0  # 80%

    # Health check configuration
    health_check_interval_seconds: int = 30
    health_check_timeout_seconds: int = 10

    # Performance monitoring
    enable_performance_monitoring: bool = True
    track_request_metrics: bool = True
    track_processing_metrics: bool = True
    track_resource_metrics: bool = True


class MetricsCollector:
    """Thread-safe metrics collector"""

    def __init__(self):
        self._metrics: Dict[str, List[MetricValue]] = {}
        self._lock = threading.Lock()

    def record_metric(self, metric: MetricValue):
        """Record a metric value"""
        with self._lock:
            if metric.name not in self._metrics:
                self._metrics[metric.name] = []
            self._metrics[metric.name].append(metric)

    def get_metrics(self, metric_name: Optional[str] = None) -> Dict[str, List[MetricValue]]:
        """Get metrics by name or all metrics"""
        with self._lock:
            if metric_name:
                return {metric_name: self._metrics.get(metric_name, [])}
            return self._metrics.copy()

    def clear_metrics(self, before_timestamp: Optional[float] = None):
        """Clear old metrics"""
        with self._lock:
            if before_timestamp:
                for metric_name in self._metrics:
                    self._metrics[metric_name] = [
                        m for m in self._metrics[metric_name]
                        if m.timestamp >= before_timestamp
                    ]
            else:
                self._metrics.clear()

    def get_metric_summary(self, metric_name: str, time_window_seconds: Optional[float] = None) -> Dict[str, Any]:
        """Get summary statistics for a metric"""
        with self._lock:
            metrics = self._metrics.get(metric_name, [])

            if time_window_seconds:
                cutoff_time = time.time() - time_window_seconds
                metrics = [m for m in metrics if m.timestamp >= cutoff_time]

            if not metrics:
                return {"count": 0, "min": 0, "max": 0, "avg": 0, "sum": 0}

            values = [m.value for m in metrics]
            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "sum": sum(values),
                "latest": values[-1] if values else 0
            }


@dataclass
class ErrorContext:
    """Context information for error tracking"""
    component: str
    operation: str
    request_id: Optional[str] = None
    user_context: Optional[Dict[str, Any]] = None
    system_context: Optional[Dict[str, Any]] = None
    stack_trace: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "operation": self.operation,
            "request_id": self.request_id,
            "user_context": self.user_context,
            "system_context": self.system_context,
            "stack_trace": self.stack_trace
        }


@dataclass
class SystemStatus:
    """Overall system status"""
    overall_status: str  # "healthy", "degraded", "unhealthy"
    component_statuses: Dict[str, HealthCheck] = field(default_factory=dict)
    active_alerts: List[Alert] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)
    uptime_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_status": self.overall_status,
            "component_statuses": {name: check.to_dict() for name, check in self.component_statuses.items()},
            "active_alerts": [alert.to_dict() for alert in self.active_alerts],
            "alert_count": len(self.active_alerts),
            "last_updated": self.last_updated,
            "uptime_seconds": self.uptime_seconds
        }
