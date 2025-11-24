"""
Monitoring module for Code Embedding AI Pipeline

Provides comprehensive monitoring capabilities including:
- Structured logging with rotation and JSON output
- Metrics collection and aggregation
- Health monitoring and status checks
- Alerting and notification system
- Performance tracking and analysis
"""

from .models import (
    LogLevel, AlertType, MetricType,
    LogEntry, MetricValue, PerformanceMetrics, Alert, HealthCheck,
    SystemStatus, MonitoringConfig
)

from .logger import StructuredLogger, ComponentLogger, LoggerFactory
from .metrics_collector import AdvancedMetricsCollector
from .alert_manager import AlertManager, AlertRule
from .health_monitor import HealthMonitor
from .monitoring_service import MonitoringService, MonitoredOperation

__all__ = [
    # Models
    'LogLevel', 'AlertType', 'MetricType',
    'LogEntry', 'MetricValue', 'PerformanceMetrics', 'Alert', 'HealthCheck',
    'SystemStatus', 'MonitoringConfig',

    # Logging
    'StructuredLogger', 'ComponentLogger', 'LoggerFactory',

    # Metrics
    'AdvancedMetricsCollector',

    # Alerting
    'AlertManager', 'AlertRule',

    # Health monitoring
    'HealthMonitor',

    # Main service
    'MonitoringService', 'MonitoredOperation'
]
