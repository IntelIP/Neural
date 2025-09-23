"""
Neural SDK - Institutional-Grade Sports Trading Infrastructure for Kalshi

Neural is a comprehensive SDK designed for institutional trading of sports event
contracts on Kalshi's prediction market platform. It provides a robust, scalable
infrastructure for data collection, analysis, trading, and deployment.

Key Features:
    - Multi-source data collection (WebSocket & REST APIs)
    - Real-time market data streaming
    - Advanced trading strategies
    - Risk management tools
    - Production-ready deployment infrastructure

Version: 0.1.0
License: MIT
"""

__version__ = "0.1.0"
__author__ = "Neural SDK Team"

# Core module imports - only import what exists
from neural.data_collection import (
    WebSocketDataSource,
    RestDataSource,
    ConfigManager,
    BaseDataSource,
    ConnectionState
)

__all__ = [
    "WebSocketDataSource",
    "RestDataSource",
    "BaseDataSource",
    "ConnectionState",
    "ConfigManager",
    "__version__"
]