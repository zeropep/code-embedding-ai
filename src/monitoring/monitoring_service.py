import time
from typing import Dict, Any, Callable, List

from .models import MonitoringConfig, PerformanceMetrics, SystemStatus
from .logger import LoggerFactory, ComponentLogger
from .metrics_collector import AdvancedMetricsCollector
from .alert_manager import AlertManager
from .health_monitor import HealthMonitor


class MonitoringService:
    """Centralized monitoring service that orchestrates all monitoring components"""

    def __init__(self, config: MonitoringConfig = None):
        self.config = config or MonitoringConfig()

        # Initialize components
        self.logger_factory = LoggerFactory(self.config)
        self.main_logger = self.logger_factory.get_main_logger()
        self.metrics_collector = AdvancedMetricsCollector(self.config)
        self.alert_manager = AlertManager(self.config, self.metrics_collector)
        self.health_monitor = HealthMonitor(self.config, self.alert_manager)

        # Service state
        self._is_running = False
        self._start_time = time.time()

        # Setup default notification handlers
        self._setup_default_notifications()

        self.main_logger.info("Monitoring service initialized",
                              config_enabled=self.config.enable_metrics)

    def _setup_default_notifications(self):
        """Setup default alert notification handlers"""

        def log_alert_notification(alert):
            """Log alert notifications"""
            if alert.is_resolved:
                self.main_logger.info(
                    f"Alert resolved: {alert.title}",
                    alert_id=alert.id,
                    alert_type=alert.alert_type.value,
                    duration_seconds=alert.duration_seconds
                )
            else:
                self.main_logger.error(
                    f"Alert triggered: {alert.title}",
                    alert_id=alert.id,
                    alert_type=alert.alert_type.value,
                    severity=alert.severity.value,
                    message=alert.message
                )

        self.alert_manager.add_notification_callback(log_alert_notification)

    async def start(self):
        """Start all monitoring components"""
        if self._is_running:
            return

        self.main_logger.info("Starting monitoring service")

        try:
            # Start metrics collection
            await self.metrics_collector.start_collection()

            # Start alert monitoring
            await self.alert_manager.start_monitoring()

            # Start health monitoring
            await self.health_monitor.start_monitoring()

            self._is_running = True
            self._start_time = time.time()

            self.main_logger.info("Monitoring service started successfully")

        except Exception as e:
            self.main_logger.error("Failed to start monitoring service", error=e)
            await self.stop()
            raise

    async def stop(self):
        """Stop all monitoring components"""
        if not self._is_running:
            return

        self.main_logger.info("Stopping monitoring service")

        try:
            # Stop components in reverse order
            await self.health_monitor.stop_monitoring()
            await self.alert_manager.stop_monitoring()
            await self.metrics_collector.stop_collection()

            self._is_running = False

            self.main_logger.info("Monitoring service stopped successfully")

        except Exception as e:
            self.main_logger.error("Error stopping monitoring service", error=e)

    def get_logger(self, component: str) -> 'ComponentLogger':
        """Get a component-specific logger"""
        return self.logger_factory.get_logger(component)

    def record_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a custom metric"""
        from .models import MetricValue, MetricType

        metric = MetricValue(
            name=name,
            value=value,
            metric_type=MetricType.GAUGE,
            labels=labels or {}
        )
        self.metrics_collector.record_metric(metric)

    def record_request(self, duration_ms: float, status_code: int = 200, endpoint: str = "unknown"):
        """Record HTTP request metrics"""
        self.metrics_collector.record_request_time(duration_ms, status_code, endpoint)

    def record_error(self, component: str, error_type: str, operation: str = "unknown"):
        """Record error occurrence"""
        self.metrics_collector.record_error(component, error_type, operation)

    def record_operation(self, operation: str, duration_seconds: float, component: str = "unknown"):
        """Record operation timing"""
        self.metrics_collector.record_operation_time(operation, duration_seconds, component)

    def record_business_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record business/application specific metrics"""
        self.metrics_collector.record_business_metric(name, value, labels)

    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics"""
        return self.metrics_collector.get_performance_snapshot()

    def get_system_status(self) -> SystemStatus:
        """Get overall system status"""
        return self.health_monitor.get_system_status()

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        performance = self.get_performance_metrics()
        system_status = self.get_system_status()
        alert_summary = self.alert_manager.get_alert_summary()
        health_summary = self.health_monitor.get_health_summary()

        return {
            "monitoring_service": {
                "is_running": self._is_running,
                "uptime_seconds": time.time() - self._start_time,
                "config_enabled": self.config.enable_metrics
            },
            "performance": performance.to_dict(),
            "system_status": system_status.to_dict(),
            "alerts": alert_summary,
            "health": health_summary
        }

    def register_health_check(self, name: str, check_function: Callable,
                              timeout_seconds: float = 10.0, critical: bool = True,
                              description: str = ""):
        """Register a custom health check"""
        self.health_monitor.register_health_check(
            name=name,
            check_function=check_function,
            timeout_seconds=timeout_seconds,
            critical=critical,
            description=description
        )

    def setup_pipeline_monitoring(self, pipeline_components: Dict[str, Any]):
        """Setup monitoring for pipeline components"""
        self.main_logger.info("Setting up pipeline monitoring")

        # Setup default health checks
        self.health_monitor.setup_default_health_checks(pipeline_components)

        # Setup component-specific monitoring
        for component_name, component in pipeline_components.items():
            self._setup_component_monitoring(component_name, component)

    def _setup_component_monitoring(self, name: str, component: Any):
        """Setup monitoring for a specific component"""
        logger = self.get_logger(name)

        # Add component-specific health checks based on component type
        if hasattr(component, 'health_check'):
            self.register_health_check(
                name=f"{name}_health",
                check_function=component.health_check,
                description=f"Health check for {name} component"
            )

        logger.info(f"Monitoring setup completed for {name}")

    def create_alert(self, title: str, message: str, severity: str = "warning",
                     alert_type: str = "performance", metadata: Dict[str, Any] = None):
        """Create a manual alert"""
        from .models import AlertType, LogLevel

        # Convert string types to enums
        severity_map = {
            "debug": LogLevel.DEBUG,
            "info": LogLevel.INFO,
            "warning": LogLevel.WARNING,
            "error": LogLevel.ERROR,
            "critical": LogLevel.CRITICAL
        }

        type_map = {
            "performance": AlertType.PERFORMANCE,
            "error_rate": AlertType.ERROR_RATE,
            "resource_usage": AlertType.RESOURCE_USAGE,
            "service_health": AlertType.SERVICE_HEALTH,
            "security": AlertType.SECURITY
        }

        alert_severity = severity_map.get(severity, LogLevel.WARNING)
        alert_type_enum = type_map.get(alert_type, AlertType.PERFORMANCE)

        return self.alert_manager.create_manual_alert(
            title=title,
            message=message,
            alert_type=alert_type_enum,
            severity=alert_severity,
            metadata=metadata or {}
        )

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        return self.alert_manager.resolve_alert(alert_id)

    def export_metrics(self, format_type: str = "prometheus") -> str:
        """Export metrics in specified format"""
        if format_type.lower() == "prometheus":
            return self.metrics_collector.export_prometheus_format()
        else:
            # JSON format
            import json
            metrics = self.metrics_collector.get_metrics()
            return json.dumps({
                name: [metric.to_dict() for metric in metric_list]
                for name, metric_list in metrics.items()
            }, indent=2)

    def export_logs(self, file_path: str, hours: int = 24, level_filter: str = None):
        """Export logs to file"""
        from .models import LogLevel

        level_map = {
            "debug": LogLevel.DEBUG,
            "info": LogLevel.INFO,
            "warning": LogLevel.WARNING,
            "error": LogLevel.ERROR,
            "critical": LogLevel.CRITICAL
        }

        filter_level = level_map.get(level_filter) if level_filter else None
        self.main_logger.export_logs(file_path, hours, filter_level)

    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get error summary"""
        return self.main_logger.get_error_summary(hours)

    def get_recent_logs(self, limit: int = 100, level_filter: str = None) -> List[Dict[str, Any]]:
        """Get recent log entries"""
        from .models import LogLevel

        level_map = {
            "debug": LogLevel.DEBUG,
            "info": LogLevel.INFO,
            "warning": LogLevel.WARNING,
            "error": LogLevel.ERROR,
            "critical": LogLevel.CRITICAL
        }

        filter_level = level_map.get(level_filter) if level_filter else None
        logs = self.main_logger.get_recent_logs(limit, filter_level)
        return [log.to_dict() for log in logs]

    def reset_metrics(self):
        """Reset all metrics"""
        self.metrics_collector.reset_metrics()
        self.main_logger.info("Metrics reset")

    def update_config(self, **kwargs):
        """Update monitoring configuration"""
        updated = False
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                updated = True
                self.main_logger.info(f"Config updated: {key} = {value}")

        if updated:
            # Apply configuration changes
            self._apply_config_changes()

    def _apply_config_changes(self):
        """Apply configuration changes to components"""
        # This would restart components if necessary
        # For now, just log the change
        self.main_logger.info("Configuration changes applied")


# Context manager for operation monitoring
class MonitoredOperation:
    """Context manager for monitoring operations"""

    def __init__(self, monitoring_service: MonitoringService, operation_name: str, component: str = "unknown"):
        self.monitoring_service = monitoring_service
        self.operation_name = operation_name
        self.component = component
        self.start_time = None
        self.logger = monitoring_service.get_logger(component)

    def __enter__(self):
        self.start_time = time.time()
        self.logger.log_operation_start(self.operation_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time

        if exc_type is None:
            # Success
            self.logger.log_operation_complete(self.operation_name, duration)
            self.monitoring_service.record_operation(self.operation_name, duration, self.component)
        else:
            # Error
            self.logger.log_operation_error(self.operation_name, exc_val)
            self.monitoring_service.record_error(self.component, exc_type.__name__, self.operation_name)

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return self.__exit__(exc_type, exc_val, exc_tb)
