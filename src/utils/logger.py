#!/usr/bin/env python3
"""
Simple logger utility for consistent project-wide logging.

Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("This is a message")
"""

import logging


def get_logger(name: str = __name__) -> logging.Logger:
    """
    Create or retrieve a logger with a console handler.

    - Sets log level to INFO by default.
    - Avoids duplicate handlers.

    Args:
        name: Name of the logger.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
