import time
import psutil
from typing import Dict, Any, Optional, List
from collections import defaultdict, deque
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .models import (MetricValue, MetricType, PerformanceMetrics, MonitoringConfig,
                     MetricsCollector as BaseMetricsCollector)


class AdvancedMetricsCollector(BaseMetricsCollector):
    """Advanced metrics collector with automatic resource monitoring and aggregation"""

    def __init__(self, config: MonitoringConfig):
        super().__init__()
        self.config = config
        self.start_time = time.time()

        # Performance tracking
        self.request_times: deque = deque(maxlen=1000)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.operation_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # Resource monitoring
        self.process = psutil.Process()
        self.system_metrics: Dict[str, float] = {}

        # Background collection
        self._collection_task: Optional[asyncio.Task] = None
        self._is_collecting = False

        # Thread pool for resource monitoring
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="metrics")

    async def start_collection(self):
        """Start automatic metrics collection"""
        if self._is_collecting:
            return

        self._is_collecting = True
        self._collection_task = asyncio.create_task(self._collection_loop())

    async def stop_collection(self):
        """Stop automatic metrics collection"""
        self._is_collecting = False
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass

        self._executor.shutdown(wait=True)

    async def _collection_loop(self):
        """Background metrics collection loop"""
        while self._is_collecting:
            try:
                await asyncio.sleep(self.config.metrics_collection_interval_seconds)

                if not self._is_collecting:
                    break

                # Collect system metrics in background thread
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(self._executor, self._collect_system_metrics)

            except asyncio.CancelledError:
                break
            except Exception:
                # Log error but continue collection
                pass

    def _collect_system_metrics(self):
        """Collect system resource metrics"""
        try:
            # CPU usage
            cpu_percent = self.process.cpu_percent()
            self.record_metric(MetricValue(
                name="system.cpu_usage_percent",
                value=cpu_percent,
                metric_type=MetricType.GAUGE,
                labels={"component": "system"}
            ))

            # Memory usage
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            self.record_metric(MetricValue(
                name="system.memory_usage_mb",
                value=memory_mb,
                metric_type=MetricType.GAUGE,
                labels={"component": "system"}
            ))

            # Disk I/O
            disk_io = self.process.io_counters()
            self.record_metric(MetricValue(
                name="system.disk_read_bytes",
                value=disk_io.read_bytes,
                metric_type=MetricType.COUNTER,
                labels={"component": "system"}
            ))
            self.record_metric(MetricValue(
                name="system.disk_write_bytes",
                value=disk_io.write_bytes,
                metric_type=MetricType.COUNTER,
                labels={"component": "system"}
            ))

            # Thread count
            thread_count = self.process.num_threads()
            self.record_metric(MetricValue(
                name="system.thread_count",
                value=thread_count,
                metric_type=MetricType.GAUGE,
                labels={"component": "system"}
            ))

            # File descriptors (Unix-like systems)
            try:
                if hasattr(self.process, 'num_fds'):
                    fd_count = self.process.num_fds()
                    self.record_metric(MetricValue(
                        name="system.file_descriptors",
                        value=fd_count,
                        metric_type=MetricType.GAUGE,
                        labels={"component": "system"}
                    ))
            except Exception:
                pass

        except Exception:
            # Log error but don't fail
            pass

    def record_request_time(self, duration_ms: float, status_code: int = 200, endpoint: str = "unknown"):
        """Record HTTP request timing"""
        self.request_times.append((time.time(), duration_ms, status_code))

        # Record as metric
        self.record_metric(MetricValue(
            name="http.request_duration_ms",
            value=duration_ms,
            metric_type=MetricType.HISTOGRAM,
            labels={
                "endpoint": endpoint,
                "status_code": str(status_code),
                "status_class": f"{status_code // 100}xx"
            }
        ))

        # Count requests
        self.record_metric(MetricValue(
            name="http.requests_total",
            value=1,
            metric_type=MetricType.COUNTER,
            labels={
                "endpoint": endpoint,
                "status_code": str(status_code),
                "status_class": f"{status_code // 100}xx"
            }
        ))

    def record_error(self, component: str, error_type: str, operation: str = "unknown"):
        """Record error occurrence"""
        error_key = f"{component}.{error_type}"
        self.error_counts[error_key] += 1

        self.record_metric(MetricValue(
            name="errors_total",
            value=1,
            metric_type=MetricType.COUNTER,
            labels={
                "component": component,
                "error_type": error_type,
                "operation": operation
            }
        ))

    def record_operation_time(self, operation: str, duration_seconds: float, component: str = "unknown"):
        """Record operation timing"""
        self.operation_times[operation].append((time.time(), duration_seconds))

        self.record_metric(MetricValue(
            name="operation.duration_seconds",
            value=duration_seconds,
            metric_type=MetricType.HISTOGRAM,
            labels={
                "operation": operation,
                "component": component
            }
        ))

    def record_business_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record business/application specific metrics"""
        self.record_metric(MetricValue(
            name=f"business.{name}",
            value=value,
            metric_type=MetricType.GAUGE,
            labels=labels or {}
        ))

    def get_performance_snapshot(self) -> PerformanceMetrics:
        """Get current performance metrics snapshot"""
        now = time.time()
        metrics = PerformanceMetrics(timestamp=now)

        # Request metrics
        recent_requests = [
            (timestamp, duration, status) for timestamp, duration, status in self.request_times
            if now - timestamp < 300  # Last 5 minutes
        ]

        if recent_requests:
            metrics.total_requests = len(recent_requests)
            metrics.successful_requests = sum(1 for _, _, status in recent_requests if status < 400)
            metrics.failed_requests = metrics.total_requests - metrics.successful_requests

            durations = [duration for _, duration, _ in recent_requests]
            metrics.avg_response_time_ms = sum(durations) / len(durations)

            # Calculate percentiles
            sorted_durations = sorted(durations)
            if sorted_durations:
                p95_idx = int(0.95 * len(sorted_durations))
                p99_idx = int(0.99 * len(sorted_durations))
                metrics.p95_response_time_ms = sorted_durations[min(p95_idx, len(sorted_durations) - 1)]
                metrics.p99_response_time_ms = sorted_durations[min(p99_idx, len(sorted_durations) - 1)]

        # Error metrics
        metrics.parsing_errors = self.error_counts.get("parser.error", 0)
        metrics.embedding_errors = self.error_counts.get("embedding.error", 0)
        metrics.database_errors = self.error_counts.get("database.error", 0)
        metrics.security_errors = self.error_counts.get("security.error", 0)

        # Business metrics from recorded metrics
        embedding_metrics = self.get_metric_summary("business.embeddings_generated")
        chunks_metrics = self.get_metric_summary("business.chunks_processed")
        files_metrics = self.get_metric_summary("business.files_parsed")

        metrics.embeddings_generated = int(embedding_metrics.get("sum", 0))
        metrics.chunks_processed = int(chunks_metrics.get("sum", 0))
        metrics.files_parsed = int(files_metrics.get("sum", 0))

        # Resource metrics
        cpu_metrics = self.get_metric_summary("system.cpu_usage_percent", time_window_seconds=60)
        memory_metrics = self.get_metric_summary("system.memory_usage_mb", time_window_seconds=60)

        metrics.cpu_usage_percent = cpu_metrics.get("latest", 0)
        metrics.memory_usage_mb = memory_metrics.get("latest", 0)

        return metrics

    def get_error_rate(self, time_window_seconds: int = 300) -> float:
        """Calculate error rate over time window"""
        now = time.time()
        cutoff = now - time_window_seconds

        recent_requests = [
            (timestamp, duration, status) for timestamp, duration, status in self.request_times
            if timestamp >= cutoff
        ]

        if not recent_requests:
            return 0.0

        error_requests = sum(1 for _, _, status in recent_requests if status >= 400)
        return error_requests / len(recent_requests)

    def get_avg_response_time(self, time_window_seconds: int = 300) -> float:
        """Calculate average response time over time window"""
        now = time.time()
        cutoff = now - time_window_seconds

        recent_requests = [
            duration for timestamp, duration, status in self.request_times
            if timestamp >= cutoff
        ]

        if not recent_requests:
            return 0.0

        return sum(recent_requests) / len(recent_requests)

    def get_top_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most frequent errors"""
        sorted_errors = sorted(self.error_counts.items(), key=lambda x: x[1], reverse=True)
        return [
            {"error_type": error_type, "count": count}
            for error_type, count in sorted_errors[:limit]
        ]

    def get_operation_stats(self, operation: str) -> Dict[str, Any]:
        """Get statistics for a specific operation"""
        operation_data = self.operation_times.get(operation, deque())

        if not operation_data:
            return {"count": 0, "avg_duration": 0, "min_duration": 0, "max_duration": 0}

        durations = [duration for _, duration in operation_data]
        return {
            "count": len(durations),
            "avg_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "total_duration": sum(durations)
        }

    def reset_metrics(self):
        """Reset all metrics"""
        super().clear_metrics()
        self.request_times.clear()
        self.error_counts.clear()
        self.operation_times.clear()
        self.system_metrics.clear()

    def export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format"""
        output = []

        # Get all metrics
        all_metrics = self.get_metrics()

        for metric_name, metric_list in all_metrics.items():
            if not metric_list:
                continue

            # Group by labels
            grouped_metrics = defaultdict(list)
            for metric in metric_list:
                label_str = ",".join([f'{k}="{v}"' for k, v in sorted(metric.labels.items())])
                grouped_metrics[label_str].append(metric)

            # Output each group
            for label_str, metrics in grouped_metrics.items():
                latest_metric = max(metrics, key=lambda m: m.timestamp)

                if latest_metric.metric_type == MetricType.COUNTER:
                    metric_line = f"{metric_name}_total"
                elif latest_metric.metric_type == MetricType.HISTOGRAM:
                    metric_line = f"{metric_name}_bucket"
                else:
                    metric_line = metric_name

                if label_str:
                    metric_line += f"{{{label_str}}}"

                metric_line += f" {latest_metric.value} {int(latest_metric.timestamp * 1000)}"
                output.append(metric_line)

        return "\n".join(output)
