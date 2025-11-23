import asyncio
import time
import uuid
from typing import Dict, List, Optional, Callable, Any
from collections import defaultdict, deque
from dataclasses import dataclass

from .models import Alert, AlertType, LogLevel, MonitoringConfig, PerformanceMetrics
from .metrics_collector import AdvancedMetricsCollector


@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    alert_type: AlertType
    severity: LogLevel
    condition: Callable[[Dict[str, Any]], bool]
    threshold: float
    window_seconds: int = 300
    cooldown_seconds: int = 900  # 15 minutes
    description: str = ""


class AlertManager:
    """Manages alerting and notifications"""

    def __init__(self, config: MonitoringConfig, metrics_collector: AdvancedMetricsCollector):
        self.config = config
        self.metrics_collector = metrics_collector

        # Alert storage
        self.active_alerts: Dict[str, Alert] = {}
        self.resolved_alerts: deque = deque(maxlen=1000)
        self.alert_history: List[Alert] = []

        # Alert rules
        self.rules: List[AlertRule] = []
        self._setup_default_rules()

        # Alert suppression
        self.suppressed_rules: Dict[str, float] = {}  # rule_name -> suppress_until_timestamp

        # Background monitoring
        self._monitoring_task: Optional[asyncio.Task] = None
        self._is_monitoring = False

        # Notification callbacks
        self.notification_callbacks: List[Callable[[Alert], None]] = []

    def _setup_default_rules(self):
        """Setup default alert rules"""
        if not self.config.enable_alerting:
            return

        # Error rate alert
        self.rules.append(AlertRule(
            name="high_error_rate",
            alert_type=AlertType.ERROR_RATE,
            severity=LogLevel.ERROR,
            condition=lambda ctx: ctx.get("error_rate", 0) > self.config.error_rate_threshold,
            threshold=self.config.error_rate_threshold,
            description=f"Error rate exceeds {self.config.error_rate_threshold * 100}%"
        ))

        # Response time alert
        self.rules.append(AlertRule(
            name="high_response_time",
            alert_type=AlertType.PERFORMANCE,
            severity=LogLevel.WARNING,
            condition=lambda ctx: ctx.get("avg_response_time_ms", 0) > self.config.response_time_threshold_ms,
            threshold=self.config.response_time_threshold_ms,
            description=f"Average response time exceeds {self.config.response_time_threshold_ms}ms"
        ))

        # Memory usage alert
        self.rules.append(AlertRule(
            name="high_memory_usage",
            alert_type=AlertType.RESOURCE_USAGE,
            severity=LogLevel.WARNING,
            condition=lambda ctx: ctx.get("memory_usage_mb", 0) > self.config.memory_usage_threshold_mb,
            threshold=self.config.memory_usage_threshold_mb,
            description=f"Memory usage exceeds {self.config.memory_usage_threshold_mb}MB"
        ))

        # CPU usage alert
        self.rules.append(AlertRule(
            name="high_cpu_usage",
            alert_type=AlertType.RESOURCE_USAGE,
            severity=LogLevel.WARNING,
            condition=lambda ctx: ctx.get("cpu_usage_percent", 0) > self.config.cpu_usage_threshold_percent,
            threshold=self.config.cpu_usage_threshold_percent,
            description=f"CPU usage exceeds {self.config.cpu_usage_threshold_percent}%"
        ))

        # Service health alerts
        self.rules.append(AlertRule(
            name="embedding_service_unhealthy",
            alert_type=AlertType.SERVICE_HEALTH,
            severity=LogLevel.CRITICAL,
            condition=lambda ctx: not ctx.get("embedding_service_healthy", True),
            threshold=1.0,
            description="Embedding service is not responding"
        ))

        self.rules.append(AlertRule(
            name="database_connection_failed",
            alert_type=AlertType.SERVICE_HEALTH,
            severity=LogLevel.CRITICAL,
            condition=lambda ctx: not ctx.get("database_connected", True),
            threshold=1.0,
            description="Database connection has failed"
        ))

        # Security alerts
        self.rules.append(AlertRule(
            name="high_secrets_detected",
            alert_type=AlertType.SECURITY,
            severity=LogLevel.WARNING,
            condition=lambda ctx: ctx.get("secrets_detected_count", 0) > 10,
            threshold=10.0,
            window_seconds=3600,  # 1 hour window
            description="High number of secrets detected in processed files"
        ))

    def add_notification_callback(self, callback: Callable[[Alert], None]):
        """Add a notification callback function"""
        self.notification_callbacks.append(callback)

    def remove_notification_callback(self, callback: Callable[[Alert], None]):
        """Remove a notification callback function"""
        if callback in self.notification_callbacks:
            self.notification_callbacks.remove(callback)

    async def start_monitoring(self):
        """Start alert monitoring"""
        if self._is_monitoring:
            return

        self._is_monitoring = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())

    async def stop_monitoring(self):
        """Stop alert monitoring"""
        self._is_monitoring = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while self._is_monitoring:
            try:
                await asyncio.sleep(60)  # Check every minute

                if not self._is_monitoring:
                    break

                await self._evaluate_rules()

            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log error but continue monitoring
                pass

    async def _evaluate_rules(self):
        """Evaluate all alert rules"""
        if not self.config.enable_alerting:
            return

        # Get current metrics context
        context = await self._gather_metrics_context()

        for rule in self.rules:
            # Check if rule is suppressed
            if self._is_rule_suppressed(rule.name):
                continue

            try:
                # Evaluate rule condition
                if rule.condition(context):
                    await self._trigger_alert(rule, context)
                else:
                    # Check if we should resolve an existing alert
                    await self._resolve_alert_if_exists(rule.name)

            except Exception as e:
                # Log error evaluating rule
                pass

    async def _gather_metrics_context(self) -> Dict[str, Any]:
        """Gather current metrics for rule evaluation"""
        context = {}

        # Performance metrics
        performance = self.metrics_collector.get_performance_snapshot()
        context.update({
            "error_rate": (performance.failed_requests / max(performance.total_requests, 1)),
            "avg_response_time_ms": performance.avg_response_time_ms,
            "memory_usage_mb": performance.memory_usage_mb,
            "cpu_usage_percent": performance.cpu_usage_percent,
            "total_requests": performance.total_requests,
            "failed_requests": performance.failed_requests
        })

        # Service health (would need to be implemented with actual health checks)
        context.update({
            "embedding_service_healthy": True,  # Placeholder
            "database_connected": True,  # Placeholder
            "parser_service_healthy": True,  # Placeholder
        })

        # Security metrics
        context.update({
            "secrets_detected_count": 0,  # Placeholder - would come from security scanner
        })

        return context

    async def _trigger_alert(self, rule: AlertRule, context: Dict[str, Any]):
        """Trigger an alert"""
        alert_id = f"{rule.name}_{int(time.time())}"

        # Check if alert already exists
        existing_alert_id = None
        for aid, alert in self.active_alerts.items():
            if alert.title.startswith(rule.name):
                existing_alert_id = aid
                break

        if existing_alert_id:
            # Alert already active, don't create duplicate
            return

        # Create new alert
        alert = Alert(
            id=alert_id,
            alert_type=rule.alert_type,
            severity=rule.severity,
            title=rule.name.replace("_", " ").title(),
            message=rule.description,
            metadata={
                "rule": rule.name,
                "threshold": rule.threshold,
                "current_value": context.get(rule.name.split("_")[1], 0),
                "context": context
            }
        )

        # Store active alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)

        # Suppress rule for cooldown period
        self.suppressed_rules[rule.name] = time.time() + rule.cooldown_seconds

        # Send notifications
        await self._send_notifications(alert)

    async def _resolve_alert_if_exists(self, rule_name: str):
        """Resolve alert if it exists and condition is no longer met"""
        alert_to_resolve = None
        alert_id_to_remove = None

        for alert_id, alert in self.active_alerts.items():
            if alert.metadata.get("rule") == rule_name:
                alert_to_resolve = alert
                alert_id_to_remove = alert_id
                break

        if alert_to_resolve:
            # Resolve alert
            alert_to_resolve.resolve()

            # Move to resolved alerts
            self.resolved_alerts.append(alert_to_resolve)
            del self.active_alerts[alert_id_to_remove]

            # Send resolution notification
            await self._send_notifications(alert_to_resolve)

    def _is_rule_suppressed(self, rule_name: str) -> bool:
        """Check if rule is currently suppressed"""
        suppress_until = self.suppressed_rules.get(rule_name, 0)
        return time.time() < suppress_until

    async def _send_notifications(self, alert: Alert):
        """Send alert notifications"""
        for callback in self.notification_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                # Log notification error
                pass

    def create_manual_alert(self, title: str, message: str, alert_type: AlertType,
                          severity: LogLevel = LogLevel.WARNING, metadata: Dict[str, Any] = None) -> Alert:
        """Manually create an alert"""
        alert_id = str(uuid.uuid4())
        alert = Alert(
            id=alert_id,
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            metadata=metadata or {}
        )

        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)

        # Send notifications asynchronously
        asyncio.create_task(self._send_notifications(alert))

        return alert

    def resolve_alert(self, alert_id: str) -> bool:
        """Manually resolve an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolve()

            # Move to resolved alerts
            self.resolved_alerts.append(alert)
            del self.active_alerts[alert_id]

            # Send resolution notification
            asyncio.create_task(self._send_notifications(alert))
            return True

        return False

    def get_active_alerts(self, alert_type: Optional[AlertType] = None,
                         severity: Optional[LogLevel] = None) -> List[Alert]:
        """Get active alerts with optional filtering"""
        alerts = list(self.active_alerts.values())

        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return sorted(alerts, key=lambda a: a.triggered_at, reverse=True)

    def get_alert_history(self, hours: int = 24, limit: int = 100) -> List[Alert]:
        """Get alert history"""
        cutoff_time = time.time() - (hours * 3600)

        filtered_alerts = [
            alert for alert in self.alert_history
            if alert.triggered_at >= cutoff_time
        ]

        return sorted(filtered_alerts, key=lambda a: a.triggered_at, reverse=True)[:limit]

    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get alert summary statistics"""
        cutoff_time = time.time() - (hours * 3600)

        recent_alerts = [
            alert for alert in self.alert_history
            if alert.triggered_at >= cutoff_time
        ]

        # Count by type
        type_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        for alert in recent_alerts:
            type_counts[alert.alert_type.value] += 1
            severity_counts[alert.severity.value] += 1

        # Calculate resolution statistics
        resolved_alerts = [a for a in recent_alerts if a.is_resolved]
        avg_resolution_time = 0
        if resolved_alerts:
            avg_resolution_time = sum(a.duration_seconds for a in resolved_alerts) / len(resolved_alerts)

        return {
            "total_alerts": len(recent_alerts),
            "active_alerts": len(self.active_alerts),
            "resolved_alerts": len(resolved_alerts),
            "alerts_by_type": dict(type_counts),
            "alerts_by_severity": dict(severity_counts),
            "avg_resolution_time_seconds": avg_resolution_time,
            "time_period_hours": hours
        }

    def suppress_rule(self, rule_name: str, duration_seconds: int):
        """Manually suppress a rule"""
        self.suppressed_rules[rule_name] = time.time() + duration_seconds

    def unsuppress_rule(self, rule_name: str):
        """Remove rule suppression"""
        if rule_name in self.suppressed_rules:
            del self.suppressed_rules[rule_name]

    def add_custom_rule(self, rule: AlertRule):
        """Add a custom alert rule"""
        self.rules.append(rule)

    def remove_rule(self, rule_name: str):
        """Remove an alert rule"""
        self.rules = [rule for rule in self.rules if rule.name != rule_name]

    def export_alert_config(self) -> Dict[str, Any]:
        """Export alert configuration"""
        return {
            "rules": [
                {
                    "name": rule.name,
                    "alert_type": rule.alert_type.value,
                    "severity": rule.severity.value,
                    "threshold": rule.threshold,
                    "window_seconds": rule.window_seconds,
                    "cooldown_seconds": rule.cooldown_seconds,
                    "description": rule.description
                }
                for rule in self.rules
            ],
            "suppressed_rules": {
                rule_name: {"suppress_until": timestamp}
                for rule_name, timestamp in self.suppressed_rules.items()
            },
            "active_alerts_count": len(self.active_alerts),
            "monitoring_enabled": self._is_monitoring
        }