"""
Logging configuration for VNStock Hub backend.

Provides two loggers:
- main_logger: General logging to console (INFO level)
- background_logger: Background task logging to file only (DEBUG level)
"""
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Log files (relative to backend folder)
LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_FILE = LOG_DIR / "background_tasks.log"
MAIN_LOG_FILE = LOG_DIR / "backend.log"

# Log formats
CONSOLE_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
FILE_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Module-level logger references
_main_logger = None
_background_logger = None


def setup_logging():
    """Initialize logging configuration. Should be called once at startup."""
    global _main_logger, _background_logger

    # Ensure logs directory exists
    LOG_DIR.mkdir(exist_ok=True)

    # Truncate log files on startup to ensure fresh logs
    for log_path in [LOG_FILE, MAIN_LOG_FILE]:
        try:
            with open(log_path, 'w') as f:
                f.truncate(0)
        except Exception:
            pass # Ignore errors if file doesn't exist or is locked

    # === Main Logger (console output + file) ===
    _main_logger = logging.getLogger("vnstock_hub")
    _main_logger.setLevel(logging.DEBUG) # Allow file to capture DEBUG
    _main_logger.propagate = False

    # Clear existing handlers to avoid duplicates on reload
    _main_logger.handlers.clear()

    # Console handler for main logger (INFO level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(CONSOLE_FORMAT, DATE_FORMAT))
    _main_logger.addHandler(console_handler)

    # File handler for main logger (DEBUG level)
    main_file_handler = RotatingFileHandler(
        MAIN_LOG_FILE,
        maxBytes=10 * 1024 * 1024,  # 10 MB per file
        backupCount=5,
        encoding="utf-8"
    )
    main_file_handler.setLevel(logging.DEBUG)
    main_file_handler.setFormatter(logging.Formatter(FILE_FORMAT, DATE_FORMAT))
    _main_logger.addHandler(main_file_handler)

    # === Background Logger (file only + critical errors to console) ===
    _background_logger = logging.getLogger("vnstock_hub.background")
    _background_logger.setLevel(logging.DEBUG)
    _background_logger.propagate = False

    # Clear existing handlers
    _background_logger.handlers.clear()

    # Rotating file handler for background logger
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=10 * 1024 * 1024,  # 10 MB per file
        backupCount=5,               # Keep 5 backup files
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(FILE_FORMAT, DATE_FORMAT))
    _background_logger.addHandler(file_handler)

    # Also add console handler for WARNING and above (critical errors)
    console_error_handler = logging.StreamHandler()
    console_error_handler.setLevel(logging.WARNING)
    console_error_handler.setFormatter(logging.Formatter(CONSOLE_FORMAT, DATE_FORMAT))
    _background_logger.addHandler(console_error_handler)

    return _main_logger, _background_logger


def get_main_logger() -> logging.Logger:
    """Get the main logger for general operations."""
    global _main_logger
    if _main_logger is None:
        setup_logging()
    return _main_logger


def get_background_logger() -> logging.Logger:
    """Get the background logger for background tasks."""
    global _background_logger
    if _background_logger is None:
        setup_logging()
    return _background_logger


def log_background_start(task_name: str, details: str = ""):
    """
    Log background task start.
    Shows brief message on console + detailed entry in log file.
    """
    main = get_main_logger()
    bg = get_background_logger()

    console_msg = f"[BACKGROUND] {task_name} started"
    if details:
        console_msg += f" ({details})"

    main.info(console_msg)
    bg.info(f"=== {task_name} STARTED === {details}")


def log_background_complete(task_name: str, summary: str = ""):
    """
    Log background task completion.
    Shows brief message on console + detailed entry in log file.
    """
    main = get_main_logger()
    bg = get_background_logger()

    console_msg = f"[BACKGROUND] {task_name} completed"
    if summary:
        console_msg += f" - {summary}"

    main.info(console_msg)
    bg.info(f"=== {task_name} COMPLETED === {summary}")


def log_background_error(task_name: str, error: str):
    """
    Log background task error.
    Shows warning on console + error entry in log file.
    """
    main = get_main_logger()
    bg = get_background_logger()

    main.warning(f"[BACKGROUND] {task_name} failed: {error}")
    bg.error(f"=== {task_name} FAILED === {error}")
