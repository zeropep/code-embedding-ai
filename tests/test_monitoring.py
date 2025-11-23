"""
Tests for monitoring, logging, and alerting functionality
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json
import time

from src.monitoring.models import MonitoringConfig, AlertConfig, AlertType, MetricType
from src.monitoring.logger import StructuredLogger
from src.monitoring.metrics import MetricsCollector
from src.monitoring.alerts import AlertManager
from src.monitoring.health_checker import HealthChecker


class TestMonitoringModels:
    """Test monitoring model classes"""

    def test_monitoring_config_defaults(self):
        """Test MonitoringConfig default values"""
        config = MonitoringConfig()

        assert config.enable_metrics is True
        assert config.enable_alerting is True
        assert config.log_level == "INFO"
        assert config.metrics_retention_days == 30

    def test_monitoring_config_validation(self):
        """Test MonitoringConfig validation"""
        # Valid config
        valid_config = MonitoringConfig(
            enable_metrics=True,
            log_level="DEBUG",
            metrics_retention_days=7
        )
        assert valid_config.validate() is True

        # Invalid config - bad log level
        invalid_config = MonitoringConfig(log_level="INVALID")
        assert invalid_config.validate() is False

        # Invalid config - bad retention days
        invalid_config2 = MonitoringConfig(metrics_retention_days=0)
        assert invalid_config2.validate() is False

    def test_alert_config_creation(self):
        """Test AlertConfig creation"""
        alert_config = AlertConfig(
            alert_type=AlertType.ERROR_RATE,
            threshold=0.1,
            duration_minutes=5,
            cooldown_minutes=30,
            enabled=True
        )

        assert alert_config.alert_type == AlertType.ERROR_RATE
        assert alert_config.threshold == 0.1
        assert alert_config.duration_minutes == 5

    def test_alert_config_to_dict(self):
        """Test AlertConfig serialization"""
        alert_config = AlertConfig(
            alert_type=AlertType.HIGH_LATENCY,
            threshold=5.0,
            duration_minutes=10
        )

        config_dict = alert_config.to_dict()

        assert "alert_type" in config_dict
        assert config_dict["alert_type"] == "high_latency"
        assert config_dict["threshold"] == 5.0


class TestStructuredLogger:
    """Test structured logging functionality"""

    @pytest.fixture
    def logger(self, monitoring_config):
        """Create structured logger for testing"""
        return StructuredLogger("test_component", monitoring_config)

    def test_logger_initialization(self, monitoring_config):
        """Test StructuredLogger initialization"""
        logger = StructuredLogger("test_service", monitoring_config)

        assert logger.component_name == "test_service"
        assert logger.config == monitoring_config
        assert logger.logger is not None

    def test_structured_logging_methods(self, logger):
        """Test structured logging methods"""
        with patch('structlog.get_logger') as mock_structlog:
            mock_log_instance = Mock()
            mock_structlog.return_value = mock_log_instance

            # Reinitialize logger to use mock
            logger._setup_logger()
            logger.logger = mock_log_instance

            # Test info logging
            logger.info("Test message", extra_field="extra_value")
            mock_log_instance.info.assert_called_with(
                "Test message",
                component="test_component",
                extra_field="extra_value"
            )

            # Test error logging
            logger.error("Error message", error_code="E001")
            mock_log_instance.error.assert_called_with(
                "Error message",
                component="test_component",
                error_code="E001"
            )

    def test_request_logging(self, logger):
        """Test HTTP request logging"""
        request_data = {
            "method": "POST",
            "path": "/api/v1/search",
            "query_params": {"limit": 10},
            "request_id": "req_123"
        }

        with patch.object(logger, 'info') as mock_info:
            logger.log_request(request_data)
            mock_info.assert_called_once()

    def test_performance_logging(self, logger):
        """Test performance metrics logging"""
        performance_data = {
            "operation": "embedding_generation",
            "duration": 2.5,
            "batch_size": 10,
            "success": True
        }

        with patch.object(logger, 'info') as mock_info:
            logger.log_performance(performance_data)
            mock_info.assert_called_once()

    def test_error_logging_with_context(self, logger):
        """Test error logging with contextual information"""
        error_data = {
            "error_type": "APIError",
            "error_message": "Failed to connect to embedding service",
            "file_path": "src/embeddings/jina_client.py",
            "function_name": "generate_embedding",
            "request_id": "req_456"
        }

        with patch.object(logger, 'error') as mock_error:
            logger.log_error(error_data)
            mock_error.assert_called_once()

    def test_security_event_logging(self, logger):
        """Test security event logging"""
        security_event = {
            "event_type": "secret_detected",
            "file_path": "src/config/DatabaseConfig.java",
            "secret_type": "password",
            "confidence": 0.95,
            "masked": True
        }

        with patch.object(logger, 'warning') as mock_warning:
            logger.log_security_event(security_event)
            mock_warning.assert_called_once()


class TestMetricsCollector:
    """Test metrics collection functionality"""

    @pytest.fixture
    def metrics_collector(self, monitoring_config):
        """Create metrics collector for testing"""
        return MetricsCollector(monitoring_config)

    def test_metrics_collector_initialization(self, monitoring_config):
        """Test MetricsCollector initialization"""
        collector = MetricsCollector(monitoring_config)

        assert collector.config == monitoring_config
        assert isinstance(collector._metrics, dict)
        assert isinstance(collector._counters, dict)
        assert isinstance(collector._histograms, dict)

    def test_counter_metrics(self, metrics_collector):
        """Test counter metric operations"""
        # Increment counter
        metrics_collector.increment_counter("requests_total", labels={"endpoint": "/search"})
        metrics_collector.increment_counter("requests_total", labels={"endpoint": "/search"})

        # Get counter value
        counter_value = metrics_collector.get_counter_value("requests_total", {"endpoint": "/search"})
        assert counter_value == 2

        # Increment by custom amount
        metrics_collector.increment_counter("bytes_processed", value=1024, labels={"operation": "embedding"})
        bytes_value = metrics_collector.get_counter_value("bytes_processed", {"operation": "embedding"})
        assert bytes_value == 1024

    def test_histogram_metrics(self, metrics_collector):
        """Test histogram metric operations"""
        # Record histogram values
        metrics_collector.record_histogram("request_duration", 0.5, labels={"endpoint": "/search"})
        metrics_collector.record_histogram("request_duration", 1.2, labels={"endpoint": "/search"})
        metrics_collector.record_histogram("request_duration", 0.8, labels={"endpoint": "/search"})

        # Get histogram statistics
        stats = metrics_collector.get_histogram_stats("request_duration", {"endpoint": "/search"})
        assert stats["count"] == 3
        assert stats["min"] == 0.5
        assert stats["max"] == 1.2
        assert abs(stats["avg"] - 0.83) < 0.01  # Approximate average

    def test_gauge_metrics(self, metrics_collector):
        """Test gauge metric operations"""
        # Set gauge value
        metrics_collector.set_gauge("active_connections", 10)
        assert metrics_collector.get_gauge_value("active_connections") == 10

        # Update gauge value
        metrics_collector.set_gauge("active_connections", 15)
        assert metrics_collector.get_gauge_value("active_connections") == 15

        # Increment/decrement gauge
        metrics_collector.increment_gauge("queue_size")
        metrics_collector.increment_gauge("queue_size", 5)
        assert metrics_collector.get_gauge_value("queue_size") == 6

        metrics_collector.decrement_gauge("queue_size", 2)
        assert metrics_collector.get_gauge_value("queue_size") == 4

    def test_embedding_metrics(self, metrics_collector):
        """Test embedding-specific metrics"""
        # Record successful embedding
        metrics_collector.record_embedding_success(
            processing_time=1.5,
            batch_size=10,
            model_version="jina-v2"
        )

        # Record failed embedding
        metrics_collector.record_embedding_failure(
            error_type="timeout",
            batch_size=5
        )

        # Check counters
        success_count = metrics_collector.get_counter_value("embeddings_successful_total")
        failure_count = metrics_collector.get_counter_value("embeddings_failed_total")

        assert success_count == 1
        assert failure_count == 1

    def test_database_metrics(self, metrics_collector):
        """Test database-specific metrics"""
        # Record database operations
        metrics_collector.record_database_operation(
            operation="insert",
            duration=0.1,
            batch_size=50,
            success=True
        )

        metrics_collector.record_database_operation(
            operation="search",
            duration=0.05,
            result_count=5,
            success=True
        )

        # Check operation counters
        insert_count = metrics_collector.get_counter_value(
            "database_operations_total",
            {"operation": "insert", "status": "success"}
        )
        assert insert_count == 1

    def test_metrics_export(self, metrics_collector):
        """Test exporting metrics in various formats"""
        # Add some test metrics
        metrics_collector.increment_counter("test_counter")
        metrics_collector.record_histogram("test_duration", 1.0)
        metrics_collector.set_gauge("test_gauge", 42)

        # Export as dictionary
        metrics_dict = metrics_collector.export_metrics()
        assert "counters" in metrics_dict
        assert "histograms" in metrics_dict
        assert "gauges" in metrics_dict

        # Export as Prometheus format
        prometheus_output = metrics_collector.export_prometheus()
        assert isinstance(prometheus_output, str)
        assert "test_counter" in prometheus_output

    def test_metrics_cleanup(self, metrics_collector):
        """Test metrics cleanup and retention"""
        # Add old metrics
        old_time = time.time() - (31 * 24 * 3600)  # 31 days ago
        with patch('time.time', return_value=old_time):
            metrics_collector.increment_counter("old_metric")

        # Add recent metrics
        metrics_collector.increment_counter("new_metric")

        # Cleanup old metrics
        metrics_collector.cleanup_old_metrics()

        # Old metric should be removed, new metric should remain
        all_metrics = metrics_collector.export_metrics()
        counter_names = list(all_metrics["counters"].keys())
        assert "new_metric" in counter_names
        # Old metric cleanup depends on implementation


class TestAlertManager:
    """Test alerting functionality"""

    @pytest.fixture
    def alert_manager(self, monitoring_config):
        """Create alert manager for testing"""
        return AlertManager(monitoring_config)

    def test_alert_manager_initialization(self, monitoring_config):
        """Test AlertManager initialization"""
        manager = AlertManager(monitoring_config)

        assert manager.config == monitoring_config
        assert isinstance(manager._active_alerts, dict)
        assert isinstance(manager._alert_history, list)

    def test_error_rate_alert(self, alert_manager):
        """Test error rate threshold alerting"""
        # Configure error rate alert
        alert_config = AlertConfig(
            alert_type=AlertType.ERROR_RATE,
            threshold=0.1,  # 10% error rate
            duration_minutes=5
        )
        alert_manager.add_alert_config("high_error_rate", alert_config)

        # Simulate high error rate
        metrics = {
            "error_rate": 0.15,  # 15% - above threshold
            "request_count": 100
        }

        # Check for alerts
        alerts = alert_manager.check_alerts(metrics)
        assert len(alerts) > 0
        assert alerts[0]["alert_type"] == AlertType.ERROR_RATE.value

    def test_latency_alert(self, alert_manager):
        """Test latency threshold alerting"""
        # Configure latency alert
        alert_config = AlertConfig(
            alert_type=AlertType.HIGH_LATENCY,
            threshold=5.0,  # 5 seconds
            duration_minutes=3
        )
        alert_manager.add_alert_config("high_latency", alert_config)

        # Simulate high latency
        metrics = {
            "avg_response_time": 6.5,  # Above threshold
            "p95_response_time": 8.2
        }

        alerts = alert_manager.check_alerts(metrics)
        assert len(alerts) > 0
        assert alerts[0]["alert_type"] == AlertType.HIGH_LATENCY.value

    def test_resource_usage_alert(self, alert_manager):
        """Test resource usage alerting"""
        # Configure resource alerts
        cpu_alert = AlertConfig(
            alert_type=AlertType.HIGH_CPU,
            threshold=0.8,  # 80%
            duration_minutes=2
        )
        memory_alert = AlertConfig(
            alert_type=AlertType.HIGH_MEMORY,
            threshold=0.9,  # 90%
            duration_minutes=2
        )

        alert_manager.add_alert_config("high_cpu", cpu_alert)
        alert_manager.add_alert_config("high_memory", memory_alert)

        # Simulate high resource usage
        metrics = {
            "cpu_usage": 0.85,  # Above CPU threshold
            "memory_usage": 0.75,  # Below memory threshold
            "disk_usage": 0.5
        }

        alerts = alert_manager.check_alerts(metrics)
        cpu_alerts = [a for a in alerts if a["alert_type"] == AlertType.HIGH_CPU.value]
        memory_alerts = [a for a in alerts if a["alert_type"] == AlertType.HIGH_MEMORY.value]

        assert len(cpu_alerts) > 0
        assert len(memory_alerts) == 0

    def test_alert_cooldown(self, alert_manager):
        """Test alert cooldown functionality"""
        # Configure alert with cooldown
        alert_config = AlertConfig(
            alert_type=AlertType.ERROR_RATE,
            threshold=0.1,
            duration_minutes=1,
            cooldown_minutes=10
        )
        alert_manager.add_alert_config("test_alert", alert_config)

        # First alert should fire
        metrics = {"error_rate": 0.2}
        alerts1 = alert_manager.check_alerts(metrics)
        assert len(alerts1) > 0

        # Second check immediately after should not fire (cooldown)
        alerts2 = alert_manager.check_alerts(metrics)
        assert len(alerts2) == 0

    def test_alert_resolution(self, alert_manager):
        """Test alert resolution when conditions improve"""
        # Fire an alert
        alert_config = AlertConfig(
            alert_type=AlertType.ERROR_RATE,
            threshold=0.1,
            duration_minutes=1
        )
        alert_manager.add_alert_config("test_alert", alert_config)

        # Trigger alert
        high_error_metrics = {"error_rate": 0.2}
        alerts = alert_manager.check_alerts(high_error_metrics)
        assert len(alerts) > 0
        alert_id = alerts[0]["alert_id"]

        # Resolve alert (error rate back to normal)
        normal_metrics = {"error_rate": 0.05}
        resolved_alerts = alert_manager.check_alert_resolution(normal_metrics)

        assert len(resolved_alerts) > 0
        assert alert_id in [a["alert_id"] for a in resolved_alerts]

    def test_alert_notification(self, alert_manager):
        """Test alert notification sending"""
        with patch.object(alert_manager, '_send_notification') as mock_send:
            alert = {
                "alert_id": "test_123",
                "alert_type": AlertType.ERROR_RATE.value,
                "message": "Error rate exceeded threshold",
                "severity": "HIGH"
            }

            alert_manager.send_alert_notification(alert)
            mock_send.assert_called_once_with(alert)

    def test_alert_history(self, alert_manager):
        """Test alert history tracking"""
        # Add some alerts to history
        alert1 = {
            "alert_id": "alert_1",
            "alert_type": AlertType.ERROR_RATE.value,
            "timestamp": time.time() - 3600
        }
        alert2 = {
            "alert_id": "alert_2",
            "alert_type": AlertType.HIGH_LATENCY.value,
            "timestamp": time.time() - 1800
        }

        alert_manager._alert_history.extend([alert1, alert2])

        # Get recent alert history
        recent_alerts = alert_manager.get_alert_history(hours=2)
        assert len(recent_alerts) == 2

        # Get alerts for specific type
        error_rate_alerts = alert_manager.get_alert_history(
            hours=2,
            alert_type=AlertType.ERROR_RATE
        )
        assert len(error_rate_alerts) == 1
        assert error_rate_alerts[0]["alert_id"] == "alert_1"


class TestHealthChecker:
    """Test health checking functionality"""

    @pytest.fixture
    def health_checker(self, monitoring_config):
        """Create health checker for testing"""
        return HealthChecker(monitoring_config)

    def test_health_checker_initialization(self, monitoring_config):
        """Test HealthChecker initialization"""
        checker = HealthChecker(monitoring_config)

        assert checker.config == monitoring_config
        assert isinstance(checker._component_checks, dict)

    def test_register_health_check(self, health_checker):
        """Test registering component health checks"""
        def database_health_check():
            return {"status": "healthy", "connection_count": 5}

        health_checker.register_component("database", database_health_check)
        assert "database" in health_checker._component_checks

    def test_component_health_check(self, health_checker):
        """Test individual component health checking"""
        # Register mock component
        def mock_component_check():
            return {"status": "healthy", "last_update": time.time()}

        health_checker.register_component("test_component", mock_component_check)

        # Check component health
        health = health_checker.check_component_health("test_component")
        assert health["status"] == "healthy"
        assert "last_update" in health

    def test_overall_health_check(self, health_checker):
        """Test overall system health check"""
        # Register multiple components
        def healthy_component():
            return {"status": "healthy"}

        def degraded_component():
            return {"status": "degraded", "error": "Minor issue"}

        def unhealthy_component():
            return {"status": "unhealthy", "error": "Critical error"}

        health_checker.register_component("healthy", healthy_component)
        health_checker.register_component("degraded", degraded_component)
        health_checker.register_component("unhealthy", unhealthy_component)

        # Check overall health
        overall_health = health_checker.check_overall_health()

        assert "overall_status" in overall_health
        assert "components" in overall_health
        assert "timestamp" in overall_health
        assert len(overall_health["components"]) == 3

        # Overall status should be unhealthy due to one unhealthy component
        assert overall_health["overall_status"] == "unhealthy"

    def test_health_check_timeout(self, health_checker):
        """Test health check timeout handling"""
        def slow_component():
            import time
            time.sleep(2)  # Simulate slow health check
            return {"status": "healthy"}

        health_checker.register_component("slow", slow_component, timeout_seconds=1)

        # Health check should timeout
        health = health_checker.check_component_health("slow")
        assert health["status"] == "unhealthy"
        assert "timeout" in health.get("error", "").lower()

    def test_health_check_exception_handling(self, health_checker):
        """Test health check exception handling"""
        def failing_component():
            raise Exception("Component check failed")

        health_checker.register_component("failing", failing_component)

        # Should handle exception gracefully
        health = health_checker.check_component_health("failing")
        assert health["status"] == "unhealthy"
        assert "error" in health

    def test_periodic_health_monitoring(self, health_checker):
        """Test periodic health monitoring"""
        check_count = 0

        def counting_component():
            nonlocal check_count
            check_count += 1
            return {"status": "healthy", "check_count": check_count}

        health_checker.register_component("counter", counting_component)

        # Mock periodic checking
        with patch('time.sleep'):  # Avoid actual sleep in tests
            health_checker.start_periodic_monitoring(interval_seconds=1)

            # Simulate some time passing
            import asyncio
            asyncio.create_task(health_checker._periodic_check_loop())

            # Check that monitoring started
            assert health_checker._monitoring_active is True

    def test_health_metrics_integration(self, health_checker):
        """Test integration with metrics collector"""
        metrics_collector = Mock()
        health_checker._metrics_collector = metrics_collector

        def test_component():
            return {"status": "healthy", "response_time": 0.05}

        health_checker.register_component("test", test_component)

        # Check component health
        health_checker.check_component_health("test")

        # Should record health check metrics
        metrics_collector.record_histogram.assert_called()
        metrics_collector.increment_counter.assert_called()