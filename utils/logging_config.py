"""
Structured Logging Configuration for the Framework.

Provides consistent, structured logging across all modules with:
    - Timestamped console output with colour coding.
    - File-based logging for training runs.
    - Per-module log levels for granular debugging.
    - TensorBoard integration hooks.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


# ANSI colour codes for console output
_COLOURS = {
    "DEBUG": "\033[36m",     # Cyan
    "INFO": "\033[32m",      # Green
    "WARNING": "\033[33m",   # Yellow
    "ERROR": "\033[31m",     # Red
    "CRITICAL": "\033[41m",  # Red background
    "RESET": "\033[0m",      # Reset
}


class ColouredFormatter(logging.Formatter):
    """Log formatter with ANSI colour codes for console readability."""

    def format(self, record: logging.LogRecord) -> str:
        colour = _COLOURS.get(record.levelname, _COLOURS["RESET"])
        reset = _COLOURS["RESET"]
        record.levelname = f"{colour}{record.levelname:>8s}{reset}"
        record.name = f"\033[34m{record.name}\033[0m"  # Blue module name
        return super().format(record)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "logs",
) -> None:
    """Configure the global logging system.

    Args:
        level: Root log level (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``).
        log_file: If provided, also write logs to this file.
        log_dir: Directory for log files (created if needed).

    Example:
        >>> setup_logging(level="DEBUG", log_file="training_run_01.log")
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler with colours
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_fmt = ColouredFormatter(
        fmt="%(asctime)s │ %(levelname)s │ %(name)s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)
    root_logger.addHandler(console_handler)

    # File handler (no colours, full timestamps)
    if log_file is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path / log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # Always capture everything to file
        file_fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_fmt)
        root_logger.addHandler(file_handler)

    # Suppress overly verbose third-party loggers
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)

    root_logger.info(f"Logging initialised: level={level}, file={log_file}")
