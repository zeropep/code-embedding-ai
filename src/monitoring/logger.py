import logging
import logging.handlers
import json
import sys
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import structlog
from pythonjsonlogger import jsonlogger

from .models import LogLevel, LogEntry, MonitoringConfig


class StructuredLogger:
    """Enhanced structured logger with JSON output and rotation"""

    def __init__(self, config: MonitoringConfig, component: str = "pipeline"):
        self.config = config
        self.component = component
        self.log_entries: List[LogEntry] = []

        # Setup structured logging
        self._setup_structlog()
        self.logger = structlog.get_logger(component)

    def _setup_structlog(self):
        """Configure structlog with proper processors and formatters"""

        processors = [
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
        ]

        if self.config.enable_structured_logging:
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(structlog.dev.ConsoleRenderer())

        # Configure logging level
        level_map = {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.CRITICAL: logging.CRITICAL
        }
        log_level = level_map.get(self.config.log_level, logging.INFO)

        # Setup file handler if configured
        handlers = []

        if self.config.log_file_path:
            # Ensure log directory exists
            log_file = Path(self.config.log_file_path)
            log_file.parent.mkdir(parents=True, exist_ok=True)

            # Rotating file handler
            file_handler = logging.handlers.RotatingFileHandler(
                filename=self.config.log_file_path,
                maxBytes=self.config.log_rotation_size_mb * 1024 * 1024,
                backupCount=self.config.log_retention_days
            )

            if self.config.enable_structured_logging:
                formatter = jsonlogger.JsonFormatter(
                    fmt='%(asctime)s %(name)s %(levelname)s %(message)s'
                )
                file_handler.setFormatter(formatter)

            file_handler.setLevel(log_level)
            handlers.append(file_handler)

        # Console handler
        if self.config.enable_console_logging:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)
            handlers.append(console_handler)

        # Configure structlog
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.make_filtering_bound_logger(log_level),
            logger_factory=structlog.stdlib.LoggerFactory(),
            context_class=dict,
            cache_logger_on_first_use=True,
        )

        # Configure standard library logging
        logging.basicConfig(
            level=log_level,
            handlers=handlers
        )

    def _create_log_entry(self, level: LogLevel, message: str, **kwargs) -> LogEntry:
        """Create structured log entry"""
        context = kwargs.copy()

        # Extract special fields
        request_id = context.pop('request_id', None)
        user_id = context.pop('user_id', None)
        session_id = context.pop('session_id', None)
        error_details = context.pop('error_details', None)

        entry = LogEntry(
            timestamp=datetime.now().timestamp(),
            level=level,
            component=self.component,
            message=message,
            context=context,
            request_id=request_id,
            user_id=user_id,
            session_id=session_id,
            error_details=error_details
        )

        # Store entry for retrieval
        self.log_entries.append(entry)

        # Keep only recent entries
        if len(self.log_entries) > 10000:
            self.log_entries = self.log_entries[-5000:]

        return entry

    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self._create_log_entry(LogLevel.DEBUG, message, **kwargs)
        self.logger.debug(message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message"""
        self._create_log_entry(LogLevel.INFO, message, **kwargs)
        self.logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self._create_log_entry(LogLevel.WARNING, message, **kwargs)
        self.logger.warning(message, **kwargs)

    def error(self, message: str, error: Optional[Exception] = None, **kwargs):
        """Log error message with optional exception details"""
        if error:
            kwargs['error_details'] = {
                'exception_type': type(error).__name__,
                'exception_message': str(error),
                'stack_trace': traceback.format_exc()
            }

        self._create_log_entry(LogLevel.ERROR, message, **kwargs)
        self.logger.error(message, exc_info=error, **kwargs)

    def critical(self, message: str, error: Optional[Exception] = None, **kwargs):
        """Log critical message"""
        if error:
            kwargs['error_details'] = {
                'exception_type': type(error).__name__,
                'exception_message': str(error),
                'stack_trace': traceback.format_exc()
            }

        self._create_log_entry(LogLevel.CRITICAL, message, **kwargs)
        self.logger.critical(message, exc_info=error, **kwargs)

    def log_operation_start(self, operation: str, **kwargs):
        """Log the start of an operation"""
        self.info(f"Operation started: {operation}",
                  operation=operation,
                  operation_phase="start",
                  **kwargs)

    def log_operation_complete(self, operation: str, duration_seconds: float, **kwargs):
        """Log successful operation completion"""
        self.info(f"Operation completed: {operation}",
                  operation=operation,
                  operation_phase="complete",
                  duration_seconds=duration_seconds,
                  **kwargs)

    def log_operation_error(self, operation: str, error: Exception, **kwargs):
        """Log operation error"""
        self.error(f"Operation failed: {operation}",
                   error=error,
                   operation=operation,
                   operation_phase="error",
                   **kwargs)

    def log_request_start(self, request_id: str, method: str, path: str, **kwargs):
        """Log HTTP request start"""
        self.info("Request started",
                  request_id=request_id,
                  http_method=method,
                  http_path=path,
                  request_phase="start",
                  **kwargs)

    def log_request_complete(self, request_id: str, status_code: int, duration_ms: float, **kwargs):
        """Log HTTP request completion"""
        log_level = LogLevel.ERROR if status_code >= 400 else LogLevel.INFO

        if log_level == LogLevel.ERROR:
            self.error("Request completed with error",
                       request_id=request_id,
                       http_status_code=status_code,
                       response_time_ms=duration_ms,
                       request_phase="complete",
                       **kwargs)
        else:
            self.info("Request completed",
                      request_id=request_id,
                      http_status_code=status_code,
                      response_time_ms=duration_ms,
                      request_phase="complete",
                      **kwargs)

    def log_performance_metrics(self, metrics: Dict[str, Any], **kwargs):
        """Log performance metrics"""
        self.info("Performance metrics",
                  metrics=metrics,
                  log_type="performance",
                  **kwargs)

    def log_security_event(self, event_type: str, details: Dict[str, Any], **kwargs):
        """Log security-related events"""
        self.warning(f"Security event: {event_type}",
                     security_event_type=event_type,
                     security_details=details,
                     log_type="security",
                     **kwargs)

    def log_business_event(self, event: str, **kwargs):
        """Log business logic events"""
        self.info(f"Business event: {event}",
                  business_event=event,
                  log_type="business",
                  **kwargs)

    def get_recent_logs(self, limit: int = 100, level_filter: Optional[LogLevel] = None) -> List[LogEntry]:
        """Get recent log entries"""
        logs = self.log_entries

        if level_filter:
            logs = [entry for entry in logs if entry.level == level_filter]

        return logs[-limit:] if limit > 0 else logs

    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get error summary for the specified time period"""
        cutoff_time = datetime.now().timestamp() - (hours * 3600)

        error_logs = [
            entry for entry in self.log_entries
            if entry.timestamp >= cutoff_time and entry.level in [LogLevel.ERROR, LogLevel.CRITICAL]
        ]

        error_types = {}
        for entry in error_logs:
            error_type = entry.error_details.get('exception_type', 'Unknown') if entry.error_details else 'Unknown'
            error_types[error_type] = error_types.get(error_type, 0) + 1

        return {
            "total_errors": len(error_logs),
            "error_types": error_types,
            "time_period_hours": hours,
            "error_rate_per_hour": len(error_logs) / max(hours, 1)
        }

    def export_logs(self, file_path: str, hours: Optional[int] = None, level_filter: Optional[LogLevel] = None):
        """Export logs to a file"""
        logs = self.log_entries

        if hours:
            cutoff_time = datetime.now().timestamp() - (hours * 3600)
            logs = [entry for entry in logs if entry.timestamp >= cutoff_time]

        if level_filter:
            logs = [entry for entry in logs if entry.level == level_filter]

        with open(file_path, 'w', encoding='utf-8') as f:
            for entry in logs:
                f.write(json.dumps(entry.to_dict()) + '\n')


class ComponentLogger:
    """Component-specific logger wrapper"""

    def __init__(self, structured_logger: StructuredLogger, component: str):
        self.structured_logger = structured_logger
        self.component = component

    def debug(self, message: str, **kwargs):
        kwargs['component'] = self.component
        self.structured_logger.debug(message, **kwargs)

    def info(self, message: str, **kwargs):
        kwargs['component'] = self.component
        self.structured_logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs):
        kwargs['component'] = self.component
        self.structured_logger.warning(message, **kwargs)

    def error(self, message: str, error: Optional[Exception] = None, **kwargs):
        kwargs['component'] = self.component
        self.structured_logger.error(message, error=error, **kwargs)

    def critical(self, message: str, error: Optional[Exception] = None, **kwargs):
        kwargs['component'] = self.component
        self.structured_logger.critical(message, error=error, **kwargs)


class LoggerFactory:
    """Factory for creating component-specific loggers"""

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.main_logger = StructuredLogger(config, "main")
        self._component_loggers: Dict[str, ComponentLogger] = {}

    def get_logger(self, component: str) -> ComponentLogger:
        """Get or create a component-specific logger"""
        if component not in self._component_loggers:
            self._component_loggers[component] = ComponentLogger(self.main_logger, component)
        return self._component_loggers[component]

    def get_main_logger(self) -> StructuredLogger:
        """Get the main structured logger"""
        return self.main_logger
