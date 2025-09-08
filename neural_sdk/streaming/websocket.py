"""
Neural WebSocket Implementation

Provides a unified WebSocket interface for streaming market data.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime

from neural_sdk.data_sources.kalshi.websocket_adapter import KalshiWebSocketAdapter, KalshiChannel
from neural_sdk.data_sources.base.websocket_source import ConnectionConfig

logger = logging.getLogger(__name__)


class NeuralWebSocket:
    """
    Unified WebSocket interface for Neural SDK.
    
    Provides a simplified API for streaming real-time market data
    from various sources.
    """
    
    def __init__(self, sdk_instance=None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize NeuralWebSocket.
        
        Args:
            sdk_instance: Parent NeuralSDK instance
            config: Configuration dictionary
        """
        self.sdk = sdk_instance
        self.config = config or {}
        self.adapter = None
        self.subscriptions = {}
        self.message_handlers = []
        self._connected = False
        
    async def connect(self):
        """Connect to WebSocket server."""
        try:
            # Initialize Kalshi adapter by default
            self.adapter = KalshiWebSocketAdapter(self.config)
            await self.adapter.connect()
            self._connected = True
            logger.info("WebSocket connected successfully")
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            raise
            
    async def disconnect(self):
        """Disconnect from WebSocket server."""
        if self.adapter:
            await self.adapter.disconnect()
            self._connected = False
            logger.info("WebSocket disconnected")
            
    async def subscribe(self, market_ticker: str, channels: Optional[List[str]] = None):
        """
        Subscribe to market data channels.
        
        Args:
            market_ticker: Market ticker to subscribe to
            channels: List of channel names (defaults to all channels)
        """
        if not self._connected:
            await self.connect()
            
        if channels is None:
            channels = ['ticker', 'orderbook_delta', 'trade']
            
        # Convert string channels to KalshiChannel enum
        kalshi_channels = []
        for channel in channels:
            try:
                if hasattr(KalshiChannel, channel.upper()):
                    kalshi_channels.append(getattr(KalshiChannel, channel.upper()))
                else:
                    # Try the exact name
                    for kc in KalshiChannel:
                        if kc.value == channel:
                            kalshi_channels.append(kc)
                            break
            except:
                logger.warning(f"Unknown channel: {channel}")
                
        if kalshi_channels:
            await self.adapter.subscribe_market(market_ticker, kalshi_channels)
            self.subscriptions[market_ticker] = kalshi_channels
            logger.info(f"Subscribed to {market_ticker} channels: {channels}")
            
    async def unsubscribe(self, market_ticker: str):
        """
        Unsubscribe from market data.
        
        Args:
            market_ticker: Market ticker to unsubscribe from
        """
        if market_ticker in self.subscriptions:
            await self.adapter.unsubscribe_market(market_ticker)
            del self.subscriptions[market_ticker]
            logger.info(f"Unsubscribed from {market_ticker}")
            
    def add_message_handler(self, handler: Callable):
        """
        Add a message handler callback.
        
        Args:
            handler: Callback function to handle messages
        """
        self.message_handlers.append(handler)
        if self.adapter:
            self.adapter.add_message_handler(handler)
            
    async def stream(self):
        """
        Start streaming data.
        
        Yields messages as they arrive.
        """
        if not self._connected:
            await self.connect()
            
        async for message in self.adapter.stream():
            yield message
            
    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._connected
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()