"""
Utilities Module

Common utility functions and helpers for the Neural SDK.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional


def setup_logging(level: str = "INFO") -> None:
    """Set up standardized logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def safe_get(data: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Safely get a value from a dictionary with nested key support."""
    try:
        keys = key.split(".")
        value = data
        for k in keys:
            value = value[k]
        return value
    except (KeyError, TypeError):
        return default


def format_price(price: float, precision: int = 2) -> str:
    """Format price with proper precision."""
    return f"${price:.{precision}f}"


def timestamp_to_string(
    timestamp: datetime, format_str: str = "%Y-%m-%d %H:%M:%S"
) -> str:
    """Convert timestamp to formatted string."""
    return timestamp.strftime(format_str)


def validate_config(config: Dict[str, Any], required_keys: list) -> bool:
    """Validate that all required keys are present in config."""
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        logging.error(f"Missing required configuration keys: {missing_keys}")
        return False
    return True
