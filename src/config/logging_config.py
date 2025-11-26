"""
Logging configuration for the application
"""
import os
import logging
import logging.handlers
import structlog
from pathlib import Path


def setup_file_logging(log_dir: str = None, log_file: str = None, log_level: str = "INFO"):
    """
    Setup file-based logging for the entire application

    Args:
        log_dir: Directory for log files (default: ./logs)
        log_file: Log file name (default: app.log)
        log_level: Logging level (default: INFO)
    """
    # Get log directory from environment or use default
    if log_dir is None:
        log_dir = os.getenv("LOG_DIR", "./logs")

    # Get log file name from environment or use default
    if log_file is None:
        log_file = os.getenv("LOG_FILE", "app.log")

    # Get log level from environment or use default
    log_level = os.getenv("LOG_LEVEL", log_level).upper()

    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Full path to log file
    log_file_path = log_path / log_file

    # Configure log level
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    numeric_level = level_map.get(log_level, logging.INFO)

    # Setup file handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        filename=str(log_file_path),
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=7,  # Keep 7 backup files
        encoding='utf-8'
    )
    file_handler.setLevel(numeric_level)

    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Setup formatters
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.dev.ConsoleRenderer(colors=False),
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Add handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Log the initialization
    logger = structlog.get_logger(__name__)
    logger.info("Logging configured",
                log_dir=str(log_dir),
                log_file=log_file,
                log_level=log_level,
                log_file_path=str(log_file_path))

    return str(log_file_path)


def get_logger(name: str):
    """
    Get a logger instance

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)
