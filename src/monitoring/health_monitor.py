import asyncio
import time
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass

from .models import HealthCheck, SystemStatus, AlertType, LogLevel, MonitoringConfig
from .alert_manager import AlertManager


@dataclass
class HealthCheckConfig:
    """Configuration for a health check"""
    name: str
    check_function: Callable[[], Any]
    timeout_seconds: float = 10.0
    critical: bool = True
    description: str = ""


class HealthMonitor:
    """System health monitoring"""

    def __init__(self, config: MonitoringConfig, alert_manager: Optional[AlertManager] = None):
        self.config = config
        self.alert_manager = alert_manager

        # Health checks
        self.health_checks: Dict[str, HealthCheckConfig] = {}
        self.last_results: Dict[str, HealthCheck] = {}

        # System status
        self.system_start_time = time.time()
        self.last_health_check_time = 0

        # Background monitoring
        self._monitoring_task: Optional[asyncio.Task] = None
        self._is_monitoring = False

    def register_health_check(self, name: str, check_function: Callable[[], Any],
                              timeout_seconds: float = 10.0, critical: bool = True,
                              description: str = ""):
        """Register a new health check"""
        self.health_checks[name] = HealthCheckConfig(
            name=name,
            check_function=check_function,
            timeout_seconds=timeout_seconds,
            critical=critical,
            description=description
        )

    def unregister_health_check(self, name: str):
        """Unregister a health check"""
        if name in self.health_checks:
            del self.health_checks[name]
        if name in self.last_results:
            del self.last_results[name]

    async def start_monitoring(self):
        """Start health monitoring"""
        if self._is_monitoring:
            return

        self._is_monitoring = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())

    async def stop_monitoring(self):
        """Stop health monitoring"""
        self._is_monitoring = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

    async def _monitoring_loop(self):
        """Background health monitoring loop"""
        while self._is_monitoring:
            try:
                await asyncio.sleep(self.config.health_check_interval_seconds)

                if not self._is_monitoring:
                    break

                await self.run_health_checks()

            except asyncio.CancelledError:
                break
            except Exception:
                # Log error but continue monitoring
                pass

    async def run_health_checks(self) -> SystemStatus:
        """Run all health checks and return system status"""
        start_time = time.time()
        component_statuses = {}

        for name, check_config in self.health_checks.items():
            try:
                result = await self._run_single_health_check(check_config)
                component_statuses[name] = result
                self.last_results[name] = result

                # Generate alerts for unhealthy components
                if result.status == "unhealthy" and check_config.critical and self.alert_manager:
                    await self._create_health_alert(name, result, check_config)

            except Exception as e:
                # Create failed health check result
                result = HealthCheck(
                    component=name,
                    status="unhealthy",
                    message=f"Health check failed: {str(e)}",
                    response_time_ms=(time.time() - start_time) * 1000
                )
                component_statuses[name] = result
                self.last_results[name] = result

        # Determine overall system status
        overall_status = self._calculate_overall_status(component_statuses)

        # Get active alerts
        active_alerts = self.alert_manager.get_active_alerts() if self.alert_manager else []

        system_status = SystemStatus(
            overall_status=overall_status,
            component_statuses=component_statuses,
            active_alerts=active_alerts,
            uptime_seconds=time.time() - self.system_start_time
        )

        self.last_health_check_time = time.time()
        return system_status

    async def _run_single_health_check(self, check_config: HealthCheckConfig) -> HealthCheck:
        """Run a single health check with timeout"""
        start_time = time.time()

        try:
            # Run check with timeout
            result = await asyncio.wait_for(
                self._execute_check_function(check_config.check_function),
                timeout=check_config.timeout_seconds
            )

            response_time_ms = (time.time() - start_time) * 1000

            # Interpret result
            if isinstance(result, dict):
                status = result.get("status", "healthy")
                message = result.get("message", "OK")
                metadata = result.get("metadata", {})
            elif isinstance(result, bool):
                status = "healthy" if result else "unhealthy"
                message = "OK" if result else "Check failed"
                metadata = {}
            elif isinstance(result, str):
                status = "healthy" if "ok" in result.lower() else "unhealthy"
                message = result
                metadata = {}
            else:
                status = "healthy" if result else "unhealthy"
                message = str(result) if result else "Check failed"
                metadata = {}

            return HealthCheck(
                component=check_config.name,
                status=status,
                message=message,
                response_time_ms=response_time_ms,
                metadata=metadata
            )

        except asyncio.TimeoutError:
            return HealthCheck(
                component=check_config.name,
                status="unhealthy",
                message=f"Health check timed out after {check_config.timeout_seconds} seconds",
                response_time_ms=check_config.timeout_seconds * 1000
            )
        except Exception as e:
            return HealthCheck(
                component=check_config.name,
                status="unhealthy",
                message=f"Health check error: {str(e)}",
                response_time_ms=(time.time() - start_time) * 1000
            )

    async def _execute_check_function(self, check_function: Callable):
        """Execute health check function (sync or async)"""
        if asyncio.iscoroutinefunction(check_function):
            return await check_function()
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, check_function)

    def _calculate_overall_status(self, component_statuses: Dict[str, HealthCheck]) -> str:
        """Calculate overall system status from component statuses"""
        if not component_statuses:
            return "unknown"

        # Count statuses
        healthy_count = 0
        degraded_count = 0
        unhealthy_count = 0

        for result in component_statuses.values():
            if result.status == "healthy":
                healthy_count += 1
            elif result.status == "degraded":
                degraded_count += 1
            else:
                unhealthy_count += 1

        # Determine overall status
        total_checks = len(component_statuses)

        if unhealthy_count == 0 and degraded_count == 0:
            return "healthy"
        elif unhealthy_count == 0 and degraded_count > 0:
            return "degraded"
        elif unhealthy_count < total_checks / 2:  # Less than half unhealthy
            return "degraded"
        else:
            return "unhealthy"

    async def _create_health_alert(self, component_name: str, result: HealthCheck,
                                   check_config: HealthCheckConfig):
        """Create alert for unhealthy component"""
        if not self.alert_manager:
            return

        alert_title = f"Component {component_name} is unhealthy"
        alert_message = f"{component_name}: {result.message}"

        # Check if alert already exists
        existing_alerts = self.alert_manager.get_active_alerts(AlertType.SERVICE_HEALTH)
        for alert in existing_alerts:
            if component_name in alert.title:
                return  # Alert already exists

        self.alert_manager.create_manual_alert(
            title=alert_title,
            message=alert_message,
            alert_type=AlertType.SERVICE_HEALTH,
            severity=LogLevel.CRITICAL if check_config.critical else LogLevel.WARNING,
            metadata={
                "component": component_name,
                "status": result.status,
                "response_time_ms": result.response_time_ms,
                "check_config": {
                    "timeout_seconds": check_config.timeout_seconds,
                    "critical": check_config.critical,
                    "description": check_config.description
                }
            }
        )

    def get_system_status(self) -> SystemStatus:
        """Get current system status"""
        if not self.last_results:
            # Run health checks synchronously if no recent results
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If event loop is running, return basic status
                return SystemStatus(
                    overall_status="unknown",
                    uptime_seconds=time.time() - self.system_start_time
                )
            else:
                # If no event loop, run sync
                return asyncio.run(self.run_health_checks())

        # Use last results
        active_alerts = self.alert_manager.get_active_alerts() if self.alert_manager else []

        return SystemStatus(
            overall_status=self._calculate_overall_status(self.last_results),
            component_statuses=self.last_results.copy(),
            active_alerts=active_alerts,
            uptime_seconds=time.time() - self.system_start_time
        )

    def get_component_health(self, component_name: str) -> Optional[HealthCheck]:
        """Get health status for a specific component"""
        return self.last_results.get(component_name)

    def get_uptime(self) -> float:
        """Get system uptime in seconds"""
        return time.time() - self.system_start_time

    def get_health_summary(self) -> Dict[str, Any]:
        """Get health monitoring summary"""
        total_checks = len(self.health_checks)
        healthy_checks = sum(1 for r in self.last_results.values() if r.status == "healthy")
        unhealthy_checks = sum(1 for r in self.last_results.values() if r.status == "unhealthy")
        degraded_checks = sum(1 for r in self.last_results.values() if r.status == "degraded")

        return {
            "total_health_checks": total_checks,
            "healthy_components": healthy_checks,
            "degraded_components": degraded_checks,
            "unhealthy_components": unhealthy_checks,
            "overall_status": self._calculate_overall_status(self.last_results),
            "last_check_time": self.last_health_check_time,
            "uptime_seconds": self.get_uptime(),
            "monitoring_active": self._is_monitoring
        }

    def setup_default_health_checks(self, pipeline_components: Dict[str, Any]):
        """Setup default health checks for pipeline components"""

        # Database health check
        if "vector_store" in pipeline_components:
            vector_store = pipeline_components["vector_store"]
            self.register_health_check(
                name="database",
                check_function=lambda: self._check_database_health(vector_store),
                critical=True,
                description="Vector database connectivity and basic operations"
            )

        # Embedding service health check
        if "embedding_service" in pipeline_components:
            embedding_service = pipeline_components["embedding_service"]
            self.register_health_check(
                name="embedding_service",
                check_function=lambda: self._check_embedding_service_health(embedding_service),
                critical=True,
                description="Embedding generation service availability"
            )

        # Git monitor health check
        if "git_monitor" in pipeline_components:
            git_monitor = pipeline_components["git_monitor"]
            self.register_health_check(
                name="git_monitor",
                check_function=lambda: self._check_git_monitor_health(git_monitor),
                critical=False,
                description="Git repository monitoring and change detection"
            )

        # System resources health check
        self.register_health_check(
            name="system_resources",
            check_function=self._check_system_resources,
            critical=False,
            description="System resource availability (memory, disk, CPU)"
        )

    def _check_database_health(self, vector_store) -> Dict[str, Any]:
        """Check vector database health"""
        try:
            health = vector_store.health_check()
            if health.get("vector_store_status") == "healthy":
                return {
                    "status": "healthy",
                    "message": "Database is accessible and responsive",
                    "metadata": health
                }
            else:
                return {
                    "status": "unhealthy",
                    "message": f"Database health check failed: {health.get('error', 'Unknown error')}",
                    "metadata": health
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Database health check error: {str(e)}"
            }

    def _check_embedding_service_health(self, embedding_service) -> Dict[str, Any]:
        """Check embedding service health"""
        try:
            if not embedding_service._is_running:
                return {
                    "status": "unhealthy",
                    "message": "Embedding service is not running"
                }

            metrics = embedding_service.get_metrics()
            return {
                "status": "healthy",
                "message": "Embedding service is running",
                "metadata": {
                    "is_running": embedding_service._is_running,
                    "total_requests": metrics.get("total_requests", 0),
                    "success_rate": metrics.get("success_rate", 0)
                }
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Embedding service check error: {str(e)}"
            }

    def _check_git_monitor_health(self, git_monitor) -> Dict[str, Any]:
        """Check Git monitor health"""
        try:
            if not git_monitor.repo:
                return {
                    "status": "unhealthy",
                    "message": "Git repository not connected"
                }

            return {
                "status": "healthy",
                "message": "Git monitor is connected",
                "metadata": {
                    "connected": True,
                    "repo_path": str(git_monitor.repo_path),
                    "current_branch": git_monitor.repo.active_branch.name
                }
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Git monitor check error: {str(e)}"
            }

    def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource health"""
        try:
            import psutil

            # Check memory usage
            memory = psutil.virtual_memory()
            memory_usage_percent = memory.percent

            # Check disk usage
            disk = psutil.disk_usage('/')
            disk_usage_percent = (disk.used / disk.total) * 100

            # Check CPU usage
            cpu_usage_percent = psutil.cpu_percent(interval=1)

            # Determine status
            if memory_usage_percent > 90 or disk_usage_percent > 90 or cpu_usage_percent > 90:
                status = "unhealthy"
                message = "Critical resource usage detected"
            elif memory_usage_percent > 80 or disk_usage_percent > 80 or cpu_usage_percent > 80:
                status = "degraded"
                message = "High resource usage detected"
            else:
                status = "healthy"
                message = "Resource usage is normal"

            return {
                "status": status,
                "message": message,
                "metadata": {
                    "memory_usage_percent": memory_usage_percent,
                    "disk_usage_percent": disk_usage_percent,
                    "cpu_usage_percent": cpu_usage_percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_free_gb": disk.free / (1024**3)
                }
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"System resources check error: {str(e)}"
            }
