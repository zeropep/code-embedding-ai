"""
Tests for monitoring, logging, and alerting functionality
Based on actual implementation in src/monitoring/
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import asyncio
import time

from src.monitoring.models import (
    MonitoringConfig, AlertType, MetricType, LogLevel,
    MetricValue, PerformanceMetrics, Alert, HealthCheck,
    LogEntry, MetricsCollector, SystemStatus
)
from src.monitoring.metrics_collector import AdvancedMetricsCollector
from src.monitoring.alert_manager import AlertManager, AlertRule
from src.monitoring.health_monitor import HealthMonitor


class TestMonitoringModels:
    """Test monitoring model classes"""

    def test_log_level_enum(self):
        """Test LogLevel enum values"""
        assert LogLevel.DEBUG.value == "debug"
        assert LogLevel.INFO.value == "info"
        assert LogLevel.WARNING.value == "warning"
        assert LogLevel.ERROR.value == "error"
        assert LogLevel.CRITICAL.value == "critical"

    def test_alert_type_enum(self):
        """Test AlertType enum values"""
        assert AlertType.PERFORMANCE.value == "performance"
        assert AlertType.ERROR_RATE.value == "error_rate"
        assert AlertType.RESOURCE_USAGE.value == "resource_usage"
        assert AlertType.SERVICE_HEALTH.value == "service_health"
        assert AlertType.SECURITY.value == "security"

    def test_metric_type_enum(self):
        """Test MetricType enum values"""
        assert MetricType.COUNTER.value == "counter"
        assert MetricType.GAUGE.value == "gauge"
        assert MetricType.HISTOGRAM.value == "histogram"
        assert MetricType.TIMER.value == "timer"

    def test_monitoring_config_defaults(self):
        """Test MonitoringConfig default values"""
        config = MonitoringConfig()

        assert config.enable_metrics is True
        assert config.enable_alerting is True
        assert config.log_level == LogLevel.INFO
        assert config.metrics_retention_hours == 24
        assert config.health_check_interval_seconds == 30

    def test_monitoring_config_custom_values(self):
        """Test MonitoringConfig with custom values"""
        config = MonitoringConfig(
            enable_metrics=False,
            enable_alerting=False,
            log_level=LogLevel.DEBUG,
            error_rate_threshold=0.2,
            response_time_threshold_ms=10000
        )

        assert config.enable_metrics is False
        assert config.enable_alerting is False
        assert config.log_level == LogLevel.DEBUG
        assert config.error_rate_threshold == 0.2
        assert config.response_time_threshold_ms == 10000


class TestMetricValue:
    """Test MetricValue dataclass"""

    def test_metric_value_creation(self):
        """Test creating MetricValue"""
        metric = MetricValue(
            name="test_metric",
            value=42.5,
            metric_type=MetricType.GAUGE,
            labels={"component": "api"}
        )

        assert metric.name == "test_metric"
        assert metric.value == 42.5
        assert metric.metric_type == MetricType.GAUGE
        assert metric.labels == {"component": "api"}
        assert metric.timestamp > 0

    def test_metric_value_to_dict(self):
        """Test MetricValue to_dict method"""
        metric = MetricValue(
            name="test_metric",
            value=100,
            metric_type=MetricType.COUNTER,
            description="Test counter"
        )

        data = metric.to_dict()
        assert data["name"] == "test_metric"
        assert data["value"] == 100
        assert data["type"] == "counter"
        assert data["description"] == "Test counter"


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass"""

    def test_performance_metrics_defaults(self):
        """Test PerformanceMetrics default values"""
        metrics = PerformanceMetrics()

        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 0
        assert metrics.avg_response_time_ms == 0.0
        assert metrics.embeddings_generated == 0

    def test_performance_metrics_to_dict(self):
        """Test PerformanceMetrics to_dict method"""
        metrics = PerformanceMetrics(
            total_requests=100,
            successful_requests=95,
            failed_requests=5,
            avg_response_time_ms=150.5
        )

        data = metrics.to_dict()
        assert data["requests"]["total"] == 100
        assert data["requests"]["successful"] == 95
        assert data["requests"]["failed"] == 5
        assert data["requests"]["success_rate"] == 0.95


class TestAlert:
    """Test Alert dataclass"""

    def test_alert_creation(self):
        """Test creating Alert"""
        alert = Alert(
            id="alert-123",
            alert_type=AlertType.ERROR_RATE,
            severity=LogLevel.ERROR,
            title="High Error Rate",
            message="Error rate exceeded threshold"
        )

        assert alert.id == "alert-123"
        assert alert.alert_type == AlertType.ERROR_RATE
        assert alert.severity == LogLevel.ERROR
        assert alert.is_resolved is False

    def test_alert_resolve(self):
        """Test resolving an alert"""
        alert = Alert(
            id="alert-456",
            alert_type=AlertType.PERFORMANCE,
            severity=LogLevel.WARNING,
            title="Test Alert",
            message="Test message"
        )

        assert alert.is_resolved is False
        alert.resolve()
        assert alert.is_resolved is True
        assert alert.resolved_at is not None

    def test_alert_duration(self):
        """Test alert duration calculation"""
        alert = Alert(
            id="alert-789",
            alert_type=AlertType.SERVICE_HEALTH,
            severity=LogLevel.CRITICAL,
            title="Service Down",
            message="Service not responding"
        )

        time.sleep(0.1)  # Small delay
        duration = alert.duration_seconds
        assert duration >= 0.1


class TestHealthCheck:
    """Test HealthCheck dataclass"""

    def test_health_check_creation(self):
        """Test creating HealthCheck"""
        check = HealthCheck(
            component="database",
            status="healthy",
            message="Connection OK",
            response_time_ms=15.5
        )

        assert check.component == "database"
        assert check.status == "healthy"
        assert check.message == "Connection OK"
        assert check.response_time_ms == 15.5

    def test_health_check_to_dict(self):
        """Test HealthCheck to_dict method"""
        check = HealthCheck(
            component="api",
            status="degraded",
            message="High latency",
            response_time_ms=500.0,
            metadata={"latency_ms": 500}
        )

        data = check.to_dict()
        assert data["component"] == "api"
        assert data["status"] == "degraded"
        assert data["metadata"] == {"latency_ms": 500}


class TestMetricsCollector:
    """Test MetricsCollector class"""

    def test_record_metric(self):
        """Test recording a metric"""
        collector = MetricsCollector()

        metric = MetricValue(
            name="test.counter",
            value=1,
            metric_type=MetricType.COUNTER
        )
        collector.record_metric(metric)

        metrics = collector.get_metrics("test.counter")
        assert "test.counter" in metrics
        assert len(metrics["test.counter"]) == 1
        assert metrics["test.counter"][0].value == 1

    def test_get_all_metrics(self):
        """Test getting all metrics"""
        collector = MetricsCollector()

        collector.record_metric(MetricValue("metric1", 10, MetricType.GAUGE))
        collector.record_metric(MetricValue("metric2", 20, MetricType.GAUGE))

        all_metrics = collector.get_metrics()
        assert "metric1" in all_metrics
        assert "metric2" in all_metrics

    def test_clear_metrics(self):
        """Test clearing metrics"""
        collector = MetricsCollector()

        collector.record_metric(MetricValue("test", 100, MetricType.COUNTER))
        assert len(collector.get_metrics()) > 0

        collector.clear_metrics()
        assert len(collector.get_metrics()) == 0

    def test_metric_summary(self):
        """Test getting metric summary"""
        collector = MetricsCollector()

        for i in range(5):
            collector.record_metric(MetricValue("summary_test", i * 10, MetricType.GAUGE))

        summary = collector.get_metric_summary("summary_test")
        assert summary["count"] == 5
        assert summary["min"] == 0
        assert summary["max"] == 40
        assert summary["avg"] == 20


class TestAdvancedMetricsCollector:
    """Test AdvancedMetricsCollector class"""

    @pytest.fixture
    def config(self):
        return MonitoringConfig(enable_metrics=True)

    @pytest.fixture
    def collector(self, config):
        return AdvancedMetricsCollector(config)

    def test_record_request_time(self, collector):
        """Test recording request time"""
        collector.record_request_time(150.5, status_code=200, endpoint="/api/search")

        metrics = collector.get_metrics("http.request_duration_ms")
        assert "http.request_duration_ms" in metrics

    def test_record_error(self, collector):
        """Test recording error"""
        collector.record_error("parser", "validation_error", "parse_file")

        metrics = collector.get_metrics("errors_total")
        assert "errors_total" in metrics

    def test_record_operation_time(self, collector):
        """Test recording operation time"""
        collector.record_operation_time("embedding_generation", 2.5, "embedding_service")

        metrics = collector.get_metrics("operation.duration_seconds")
        assert "operation.duration_seconds" in metrics

    def test_get_performance_snapshot(self, collector):
        """Test getting performance snapshot"""
        # Record some metrics
        collector.record_request_time(100, status_code=200)
        collector.record_request_time(200, status_code=200)
        collector.record_request_time(150, status_code=500)

        snapshot = collector.get_performance_snapshot()
        assert isinstance(snapshot, PerformanceMetrics)
        assert snapshot.total_requests >= 0

    def test_get_error_rate(self, collector):
        """Test getting error rate"""
        collector.record_request_time(100, status_code=200)
        collector.record_request_time(100, status_code=200)
        collector.record_request_time(100, status_code=500)
        collector.record_request_time(100, status_code=500)

        error_rate = collector.get_error_rate()
        assert 0 <= error_rate <= 1


class TestAlertRule:
    """Test AlertRule dataclass"""

    def test_alert_rule_creation(self):
        """Test creating AlertRule"""
        rule = AlertRule(
            name="high_error_rate",
            alert_type=AlertType.ERROR_RATE,
            severity=LogLevel.ERROR,
            condition=lambda ctx: ctx.get("error_rate", 0) > 0.1,
            threshold=0.1,
            description="Error rate exceeds 10%"
        )

        assert rule.name == "high_error_rate"
        assert rule.alert_type == AlertType.ERROR_RATE
        assert rule.threshold == 0.1
        assert rule.cooldown_seconds == 900  # Default


class TestAlertManager:
    """Test AlertManager class"""

    @pytest.fixture
    def config(self):
        return MonitoringConfig(
            enable_alerting=True,
            error_rate_threshold=0.1
        )

    @pytest.fixture
    def metrics_collector(self, config):
        return AdvancedMetricsCollector(config)

    @pytest.fixture
    def alert_manager(self, config, metrics_collector):
        return AlertManager(config, metrics_collector)

    def test_alert_manager_initialization(self, alert_manager):
        """Test AlertManager initialization"""
        assert alert_manager is not None
        assert len(alert_manager.rules) > 0  # Default rules should be set up

    @pytest.mark.asyncio
    async def test_create_manual_alert(self, alert_manager):
        """Test creating a manual alert"""
        alert = alert_manager.create_manual_alert(
            title="Test Alert",
            message="This is a test alert",
            alert_type=AlertType.PERFORMANCE,
            severity=LogLevel.WARNING
        )

        assert alert is not None
        assert alert.title == "Test Alert"
        assert alert.alert_type == AlertType.PERFORMANCE

        # Alert should be in active alerts
        active = alert_manager.get_active_alerts()
        assert len(active) >= 1

    @pytest.mark.asyncio
    async def test_resolve_alert(self, alert_manager):
        """Test resolving an alert"""
        alert = alert_manager.create_manual_alert(
            title="To Resolve",
            message="Will be resolved",
            alert_type=AlertType.SERVICE_HEALTH
        )

        # Resolve it
        result = alert_manager.resolve_alert(alert.id)
        assert result is True

        # Should no longer be in active alerts
        active = alert_manager.get_active_alerts()
        resolved_ids = [a.id for a in active]
        assert alert.id not in resolved_ids

    @pytest.mark.asyncio
    async def test_get_active_alerts_filtered(self, alert_manager):
        """Test getting active alerts with filter"""
        alert_manager.create_manual_alert(
            title="Error Alert",
            message="Error",
            alert_type=AlertType.ERROR_RATE
        )
        alert_manager.create_manual_alert(
            title="Performance Alert",
            message="Performance",
            alert_type=AlertType.PERFORMANCE
        )

        error_alerts = alert_manager.get_active_alerts(alert_type=AlertType.ERROR_RATE)
        assert all(a.alert_type == AlertType.ERROR_RATE for a in error_alerts)

    @pytest.mark.asyncio
    async def test_alert_summary(self, alert_manager):
        """Test getting alert summary"""
        alert_manager.create_manual_alert(
            title="Alert 1",
            message="Message 1",
            alert_type=AlertType.ERROR_RATE
        )

        summary = alert_manager.get_alert_summary(hours=24)
        assert "total_alerts" in summary
        assert "active_alerts" in summary
        assert "alerts_by_type" in summary

    def test_suppress_rule(self, alert_manager):
        """Test rule suppression"""
        alert_manager.suppress_rule("high_error_rate", 3600)  # 1 hour

        assert alert_manager._is_rule_suppressed("high_error_rate") is True

        alert_manager.unsuppress_rule("high_error_rate")
        assert alert_manager._is_rule_suppressed("high_error_rate") is False

    def test_add_custom_rule(self, alert_manager):
        """Test adding custom rule"""
        custom_rule = AlertRule(
            name="custom_rule",
            alert_type=AlertType.SECURITY,
            severity=LogLevel.CRITICAL,
            condition=lambda ctx: False,
            threshold=1.0
        )

        initial_count = len(alert_manager.rules)
        alert_manager.add_custom_rule(custom_rule)
        assert len(alert_manager.rules) == initial_count + 1

    def test_remove_rule(self, alert_manager):
        """Test removing a rule"""
        initial_count = len(alert_manager.rules)
        alert_manager.remove_rule("high_error_rate")
        assert len(alert_manager.rules) == initial_count - 1


class TestHealthMonitor:
    """Test HealthMonitor class"""

    @pytest.fixture
    def config(self):
        return MonitoringConfig(
            health_check_interval_seconds=10,
            health_check_timeout_seconds=5
        )

    @pytest.fixture
    def health_monitor(self, config):
        return HealthMonitor(config)

    def test_health_monitor_initialization(self, health_monitor):
        """Test HealthMonitor initialization"""
        assert health_monitor is not None
        assert health_monitor.config is not None
        assert len(health_monitor.health_checks) == 0

    def test_register_health_check(self, health_monitor):
        """Test registering a health check"""
        health_monitor.register_health_check(
            name="test_component",
            check_function=lambda: True,
            timeout_seconds=5.0,
            critical=True
        )

        assert "test_component" in health_monitor.health_checks

    def test_unregister_health_check(self, health_monitor):
        """Test unregistering a health check"""
        health_monitor.register_health_check(
            name="to_remove",
            check_function=lambda: True
        )
        assert "to_remove" in health_monitor.health_checks

        health_monitor.unregister_health_check("to_remove")
        assert "to_remove" not in health_monitor.health_checks

    @pytest.mark.asyncio
    async def test_run_health_checks(self, health_monitor):
        """Test running health checks"""
        health_monitor.register_health_check(
            name="healthy_component",
            check_function=lambda: {"status": "healthy", "message": "OK"}
        )

        status = await health_monitor.run_health_checks()
        assert isinstance(status, SystemStatus)
        assert "healthy_component" in status.component_statuses

    @pytest.mark.asyncio
    async def test_health_check_timeout(self, health_monitor):
        """Test health check timeout handling"""
        async def slow_check():
            await asyncio.sleep(10)
            return True

        health_monitor.register_health_check(
            name="slow_component",
            check_function=slow_check,
            timeout_seconds=0.1  # Very short timeout
        )

        status = await health_monitor.run_health_checks()
        result = status.component_statuses.get("slow_component")
        assert result is not None
        assert result.status == "unhealthy"
        assert "timed out" in result.message.lower()

    @pytest.mark.asyncio
    async def test_health_check_exception_handling(self, health_monitor):
        """Test health check exception handling"""
        def failing_check():
            raise ValueError("Check failed!")

        health_monitor.register_health_check(
            name="failing_component",
            check_function=failing_check
        )

        status = await health_monitor.run_health_checks()
        result = status.component_statuses.get("failing_component")
        assert result is not None
        assert result.status == "unhealthy"

    def test_get_uptime(self, health_monitor):
        """Test getting system uptime"""
        uptime = health_monitor.get_uptime()
        assert uptime >= 0

    def test_get_health_summary(self, health_monitor):
        """Test getting health summary"""
        summary = health_monitor.get_health_summary()

        assert "total_health_checks" in summary
        assert "healthy_components" in summary
        assert "unhealthy_components" in summary
        assert "uptime_seconds" in summary


class TestLogEntry:
    """Test LogEntry dataclass"""

    def test_log_entry_creation(self):
        """Test creating LogEntry"""
        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            component="api",
            message="Request received",
            context={"endpoint": "/search"},
            request_id="req-123"
        )

        assert entry.level == LogLevel.INFO
        assert entry.component == "api"
        assert entry.request_id == "req-123"

    def test_log_entry_to_dict(self):
        """Test LogEntry to_dict method"""
        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.ERROR,
            component="parser",
            message="Parse failed",
            error_details={"exception": "ValueError"}
        )

        data = entry.to_dict()
        assert data["level"] == "error"
        assert data["component"] == "parser"
        assert data["error_details"] == {"exception": "ValueError"}


class TestSystemStatus:
    """Test SystemStatus dataclass"""

    def test_system_status_creation(self):
        """Test creating SystemStatus"""
        status = SystemStatus(
            overall_status="healthy",
            uptime_seconds=3600
        )

        assert status.overall_status == "healthy"
        assert status.uptime_seconds == 3600
        assert len(status.component_statuses) == 0
        assert len(status.active_alerts) == 0

    def test_system_status_to_dict(self):
        """Test SystemStatus to_dict method"""
        check = HealthCheck(
            component="db",
            status="healthy",
            message="OK"
        )
        status = SystemStatus(
            overall_status="healthy",
            component_statuses={"db": check},
            uptime_seconds=7200
        )

        data = status.to_dict()
        assert data["overall_status"] == "healthy"
        assert "db" in data["component_statuses"]
        assert data["alert_count"] == 0
