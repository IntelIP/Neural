"""
Neural SDK - Professional-grade SDK for algorithmic trading on prediction markets.

This package provides tools for:
- Authentication with Kalshi API
- Historical and real-time market data collection
- Trading strategy development and backtesting
- Risk management and position sizing
- Order execution via REST and FIX protocols
"""

__version__ = "0.1.0"
__author__ = "Neural Contributors"
__license__ = "MIT"

from neural import auth
from neural import data_collection
from neural import analysis
from neural import trading

__all__ = [
    "__version__",
    "auth",
    "data_collection",
    "analysis",
    "trading",
]
