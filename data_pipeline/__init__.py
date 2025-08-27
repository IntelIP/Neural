"""
Kalshi WebSocket Infrastructure

A Python library for streaming real-time market data from Kalshi prediction markets.
"""

from .config import KalshiConfig, get_config
from .core import KalshiAuth, KalshiClient
from .streaming import KalshiWebSocket, MessageHandler, DefaultMessageHandler
from .utils import setup_logging

__version__ = "1.0.0"

__all__ = [
    'KalshiConfig',
    'get_config',
    'KalshiAuth',
    'KalshiClient',
    'KalshiWebSocket',
    'MessageHandler',
    'DefaultMessageHandler',
    'setup_logging'
]