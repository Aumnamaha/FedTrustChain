"""Modular logging helper for FedTrustChain components."""

from __future__ import annotations

import logging
from datetime import datetime


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger instance.

    Format: ``[YYYY-MM-DD HH:MM:SS] [LEVEL] [COMPONENT]: message``

    Parameters
    ----------
    name : str
        The component or module name (e.g. ``"FederatedClient-0"``,
        ``"FederatedServer"``, ``"DatasetLoader"``).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if called multiple times
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="[%s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Custom format method to inject the component name with brackets
        class _BracketFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                # Prepend [COMPONENT] before the message
                msg = super().format(record)
                return f"[{name}] {msg}"

        handler.setFormatter(_BracketFormatter(
            fmt="[%(asctime)s] [%(levelname)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

    return logger
