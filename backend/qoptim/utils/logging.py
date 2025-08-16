"""
Comprehensive logging utilities for Q-Optim framework.
"""

import logging
import sys
import os
import json
from typing import Optional, Dict, Any, Union
from pathlib import Path
from datetime import datetime
import traceback
import functools

try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False

try:
    from colorlog import ColoredFormatter
    COLORLOG_AVAILABLE = True
except ImportError:
    COLORLOG_AVAILABLE = False


class QOptimLogger:
    """
    Enhanced logger for Q-Optim framework with structured logging capabilities.
    """
    
    _loggers = {}
    _configured = False
    
    @classmethod
    def configure(
        cls,
        level: str = "INFO",
        log_file: Optional[str] = None,
        enable_structlog: bool = True,
        enable_colors: bool = True,
        log_format: Optional[str] = None,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5
    ) -> None:
        """
        Configure global logging settings for Q-Optim.
        
        Args:
            level: Logging level
            log_file: Optional file path for logging
            enable_structlog: Use structured logging if available
            enable_colors: Use colored output if available
            log_format: Custom log format string
            max_file_size: Maximum log file size before rotation
            backup_count: Number of backup files to keep
        """
        if cls._configured:
            return
        
        # Set up root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, level.upper()))
        
        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Configure console handler
        console_handler = logging.StreamHandler(sys.stdout)
        
        if enable_colors and COLORLOG_AVAILABLE:
            console_format = log_format or (
                "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s%(reset)s"
            )
            console_formatter = ColoredFormatter(
                console_format,
                datefmt='%Y-%m-%d %H:%M:%S',
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green', 
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }
            )
        else:
            console_format = log_format or (
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            console_formatter = logging.Formatter(
                console_format,
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(getattr(logging, level.upper()))
        root_logger.addHandler(console_handler)
        
        # Configure file handler if specified
        if log_file:
            try:
                from logging.handlers import RotatingFileHandler
                
                log_path = Path(log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                
                file_handler = RotatingFileHandler(
                    log_file,
                    maxBytes=max_file_size,
                    backupCount=backup_count
                )
                
                file_format = (
                    "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
                )
                file_formatter = logging.Formatter(
                    file_format,
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                
                file_handler.setFormatter(file_formatter)
                file_handler.setLevel(logging.DEBUG)  # File logs everything
                root_logger.addHandler(file_handler)
                
            except Exception as e:
                console_handler.emit(
                    logging.LogRecord(
                        name="qoptim.logging",
                        level=logging.WARNING,
                        pathname="",
                        lineno=0,
                        msg=f"Failed to set up file logging: {e}",
                        args=(),
                        exc_info=None
                    )
                )
        
        # Configure structured logging if available and enabled
        if enable_structlog and STRUCTLOG_AVAILABLE:
            try:
                structlog.configure(
                    processors=[
                        structlog.stdlib.filter_by_level,
                        structlog.stdlib.add_logger_name,
                        structlog.stdlib.add_log_level,
                        structlog.stdlib.PositionalArgumentsFormatter(),
                        structlog.processors.TimeStamper(fmt="ISO"),
                        structlog.processors.StackInfoRenderer(),
                        structlog.processors.format_exc_info,
                        structlog.processors.UnicodeDecoder(),
                        structlog.processors.JSONRenderer()
                    ],
                    context_class=dict,
                    logger_factory=structlog.stdlib.LoggerFactory(),
                    wrapper_class=structlog.stdlib.BoundLogger,
                    cache_logger_on_first_use=True,
                )
            except Exception:
                pass  # Fall back to standard logging
        
        cls._configured = True
    
    @classmethod
    def get_logger(cls, name: str, level: Optional[str] = None) -> logging.Logger:
        """
        Get a logger instance with consistent configuration.
        
        Args:
            name: Logger name (typically __name__)
            level: Optional logging level override
            
        Returns:
            Configured logger instance
        """
        if not cls._configured:
            cls.configure()
        
        if name in cls._loggers:
            logger = cls._loggers[name]
        else:
            logger = logging.getLogger(name)
            cls._loggers[name] = logger
        
        if level:
            logger.setLevel(getattr(logging, level.upper()))
        
        return logger


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance with consistent formatting.
    
    Args:
        name: Logger name (typically __name__)
        level: Logging level override
        
    Returns:
        Configured logger instance
    """
    return QOptimLogger.get_logger(name, level)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    **kwargs
) -> None:
    """
    Setup logging configuration for Q-Optim.
    
    Args:
        level: Logging level
        log_file: Optional file path for logging
        **kwargs: Additional configuration options
    """
    QOptimLogger.configure(level=level, log_file=log_file, **kwargs)


class LoggingMixin:
    """
    Mixin class to add logging capabilities to any class.
    """
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        if not hasattr(self, '_logger'):
            self._logger = get_logger(self.__class__.__module__ + '.' + self.__class__.__name__)
        return self._logger


def log_execution_time(func):
    """
    Decorator to log function execution time.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = datetime.now()
        
        try:
            result = func(*args, **kwargs)
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.debug(f"{func.__name__} executed in {execution_time:.4f} seconds")
            return result
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(
                f"{func.__name__} failed after {execution_time:.4f} seconds: {str(e)}"
            )
            raise
    
    return wrapper


def log_method_calls(cls):
    """
    Class decorator to log all method calls.
    
    Args:
        cls: Class to decorate
        
    Returns:
        Decorated class
    """
    for attr_name in dir(cls):
        attr = getattr(cls, attr_name)
        if callable(attr) and not attr_name.startswith('_'):
            setattr(cls, attr_name, log_execution_time(attr))
    
    return cls


class StructuredLogger:
    """
    Structured logger for complex data logging.
    """
    
    def __init__(self, name: str):
        self.logger = get_logger(name)
        self.context = {}
    
    def bind(self, **context) -> 'StructuredLogger':
        """
        Bind context to logger.
        
        Args:
            **context: Context key-value pairs
            
        Returns:
            New logger instance with bound context
        """
        new_logger = StructuredLogger(self.logger.name)
        new_logger.context = {**self.context, **context}
        return new_logger
    
    def _log(self, level: str, message: str, **extra) -> None:
        """
        Log with structured context.
        
        Args:
            level: Log level
            message: Log message
            **extra: Additional context
        """
        log_data = {
            'message': message,
            'timestamp': datetime.now().isoformat(),
            **self.context,
            **extra
        }
        
        log_level = getattr(logging, level.upper())
        
        if STRUCTLOG_AVAILABLE:
            # Use structured logging if available
            struct_logger = structlog.get_logger(self.logger.name)
            getattr(struct_logger, level.lower())(**log_data)
        else:
            # Fall back to JSON logging
            self.logger.log(log_level, json.dumps(log_data))
    
    def debug(self, message: str, **extra) -> None:
        """Log debug message with context."""
        self._log('DEBUG', message, **extra)
    
    def info(self, message: str, **extra) -> None:
        """Log info message with context."""
        self._log('INFO', message, **extra)
    
    def warning(self, message: str, **extra) -> None:
        """Log warning message with context."""
        self._log('WARNING', message, **extra)
    
    def error(self, message: str, **extra) -> None:
        """Log error message with context."""
        self._log('ERROR', message, **extra)
    
    def critical(self, message: str, **extra) -> None:
        """Log critical message with context."""
        self._log('CRITICAL', message, **extra)


def get_structured_logger(name: str) -> StructuredLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(name)


class ExceptionLogger:
    """
    Context manager for exception logging.
    """
    
    def __init__(self, logger: logging.Logger, context: str = ""):
        self.logger = logger
        self.context = context
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type is not None:
            error_msg = f"Exception in {self.context}: {exc_value}" if self.context else f"Exception: {exc_value}"
            self.logger.error(error_msg, exc_info=True)
        return False  # Don't suppress the exception


# Performance logging utilities
class PerformanceLogger:
    """
    Performance and profiling logger.
    """
    
    def __init__(self, name: str):
        self.logger = get_logger(name)
        self.metrics = {}
    
    def log_metric(self, name: str, value: float, unit: str = "") -> None:
        """
        Log a performance metric.
        
        Args:
            name: Metric name
            value: Metric value
            unit: Unit of measurement
        """
        self.metrics[name] = {'value': value, 'unit': unit, 'timestamp': datetime.now()}
        self.logger.info(f"METRIC: {name}={value}{unit}")
    
    def log_memory_usage(self) -> None:
        """
        Log current memory usage.
        """
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            self.log_metric("memory_rss", memory_info.rss / 1024 / 1024, "MB")
            self.log_metric("memory_vms", memory_info.vms / 1024 / 1024, "MB")
        except ImportError:
            self.logger.debug("psutil not available for memory monitoring")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of logged metrics.
        
        Returns:
            Dictionary of metrics
        """
        return self.metrics.copy()


# Initialize default logging on import
setup_logging()
