"""
Kalshi API Infrastructure

A Python library for accessing market data from Kalshi prediction markets.
"""

# Public API re-exports (aligned with current package layout)
from .config.settings import KalshiConfig, get_config  # type: ignore
from .data_sources.kalshi.auth import KalshiAuth  # type: ignore
from .data_sources.kalshi.client import KalshiClient  # type: ignore
from .utils import setup_logging  # type: ignore

__version__ = "1.0.1"

__all__ = [
    'KalshiConfig',
    'get_config',
    'KalshiAuth',
    'KalshiClient',
    'setup_logging'
]
