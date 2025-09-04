"""
Neural SDK Streaming Module

Real-time market data streaming and WebSocket functionality.
This module provides user-friendly wrappers around the data pipeline
WebSocket infrastructure.
"""

from .websocket import NeuralWebSocket
from .market_stream import MarketStream, NFLMarketStream
from .handlers import StreamEventHandler

__all__ = [
    "NeuralWebSocket",
    "MarketStream", 
    "NFLMarketStream",
    "StreamEventHandler",
]
