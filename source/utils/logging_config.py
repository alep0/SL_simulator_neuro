"""
logging_config.py
=================
Centralised logging configuration for the neuro_sl_simulator project.

Import this module early in any script or application entry-point:

    import source.utils.logging_config  # noqa: F401

This configures the root logger to write:
    * INFO and above to stdout (console).
    * DEBUG and above to ``logs/simulation.log`` (rotating, 5 MB × 3 files).

Environment variables
---------------------
LOG_LEVEL  Override the console log level (e.g. ``LOG_LEVEL=DEBUG``).
LOG_DIR    Override the log-file directory   (default: ``logs/``).
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path


def setup_logging(
    log_dir: str | Path | None = None,
    log_filename: str = "simulation.log",
    console_level: int | None = None,
) -> logging.Logger:
    """
    Configure the root logger and return it.

    Parameters
    ----------
    log_dir      : Path or str  Directory for the rotating log file.
                                Defaults to the ``LOG_DIR`` env-var or ``logs/``.
    log_filename : str          Name of the rotating log file.
    console_level: int          ``logging.INFO`` by default; overridden by
                                the ``LOG_LEVEL`` env-var.

    Returns
    -------
    root_logger : logging.Logger
    """
    # ---- Resolve parameters ------------------------------------------------
    if log_dir is None:
        log_dir = Path(os.environ.get("LOG_DIR", "logs"))
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    env_level = os.environ.get("LOG_LEVEL", "").upper()
    _level_map = {
        "DEBUG":    logging.DEBUG,
        "INFO":     logging.INFO,
        "WARNING":  logging.WARNING,
        "ERROR":    logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    if console_level is None:
        console_level = _level_map.get(env_level, logging.INFO)

    # ---- Formatter ---------------------------------------------------------
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ---- Handlers ----------------------------------------------------------
    # Console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(fmt)

    # Rotating file (5 MB × 3 backups)
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / log_filename,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    # ---- Root logger -------------------------------------------------------
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)          # capture everything; handlers filter
    root.handlers.clear()                 # avoid duplicate handlers on re-import
    root.addHandler(console_handler)
    root.addHandler(file_handler)

    root.info(
        "Logging initialised – console=%s, file=%s",
        logging.getLevelName(console_level),
        log_dir / log_filename,
    )
    return root


# Auto-configure when the module is imported
_logger = setup_logging()
