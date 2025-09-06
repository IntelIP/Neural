"""
Kalshi WebSocket Adapter

Real-time market data streaming from Kalshi via WebSocket.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Set
from datetime import datetime
import json
from enum import Enum

from ..base.websocket_source import WebSocketDataSource, ConnectionConfig
from neural_sdk.data_pipeline.data_sources.kalshi.client import KalshiClient

logger = logging.getLogger(__name__)


class KalshiChannel(Enum):
    """Kalshi WebSocket channels."""
    ORDERBOOK_DELTA = "orderbook_delta"
    TICKER = "ticker"
    TRADE = "trade"
    FILL = "fill"
    MARKET_POSITIONS = "market_positions"
    MARKET_LIFECYCLE = "market_lifecycle_v2"
    MULTIVARIATE = "multivariate"


class KalshiWebSocketAdapter(WebSocketDataSource):
    """
    WebSocket adapter for Kalshi real-time market data.
    
    Provides streaming access to:
    - Orderbook updates
    - Price tickers
    - Executed trades
    - Order fills
    - Market lifecycle events
    """
    
    def __init__(self, config: Optional[Any] = None):
        """
        Initialize Kalshi WebSocket adapter.
        
        Args:
            config: KalshiConfig object or None to use environment
        """
        # Use existing Kalshi client for config
        self.kalshi_client = KalshiClient(config)
        self.config_obj = self.kalshi_client.config
        
        # Determine WebSocket URL based on environment
        if self.config_obj.environment == "production":
            ws_url = "wss://api.elections.kalshi.com"
        else:
            ws_url = "wss://demo-api.elections.kalshi.com"
        
        # Create WebSocket config
        ws_config = ConnectionConfig(
            url=ws_url,
            api_key=self.config_obj.api_key_id,
            heartbeat_interval=30,
            reconnect_interval=5,
            max_reconnect_attempts=10,
            connection_timeout=30,
            message_queue_size=10000,
            max_subscriptions=100
        )
        
        super().__init__(ws_config, name="KalshiWebSocket")
        
        # Track subscriptions by channel and market
        self._channel_subscriptions: Dict[str, Set[str]] = {}
        self._market_subscriptions: Set[str] = set()
        
        # Message ID counter
        self._message_id = 0
        
        # Pending requests
        self._pending_requests: Dict[int, asyncio.Future] = {}
        
        logger.info("Kalshi WebSocket adapter initialized")
    
    # Authentication
    
    async def _authenticate(self) -> bool:
        """
        Authenticate with Kalshi WebSocket.
        
        Kalshi uses API key authentication during connection.
        Authentication is handled via headers in the base class.
        """
        # Authentication is handled via headers in connection
        # No additional authentication message needed
        return True
    
    # Subscription management
    
    async def subscribe_market(
        self,
        market_ticker: str,
        channels: Optional[List[KalshiChannel]] = None
    ) -> bool:
        """
        Subscribe to market data.
        
        Args:
            market_ticker: Market ticker symbol
            channels: List of channels to subscribe (defaults to ticker & orderbook)
            
        Returns:
            True if subscribed successfully
        """
        if channels is None:
            channels = [KalshiChannel.TICKER, KalshiChannel.ORDERBOOK_DELTA]
        
        success = True
        for channel in channels:
            if await self._subscribe_to_channel(channel.value, market_ticker):
                self._market_subscriptions.add(market_ticker)
            else:
                success = False
        
        return success
    
    async def subscribe_markets(
        self,
        market_tickers: List[str],
        channels: Optional[List[KalshiChannel]] = None
    ) -> bool:
        """
        Subscribe to multiple markets.
        
        Args:
            market_tickers: List of market tickers
            channels: Channels to subscribe
            
        Returns:
            True if all subscribed successfully
        """
        success = True
        for ticker in market_tickers:
            if not await self.subscribe_market(ticker, channels):
                success = False
        return success
    
    async def unsubscribe_market(
        self,
        market_ticker: str,
        channels: Optional[List[KalshiChannel]] = None
    ) -> bool:
        """
        Unsubscribe from market data.
        
        Args:
            market_ticker: Market ticker symbol
            channels: Channels to unsubscribe (defaults to all)
            
        Returns:
            True if unsubscribed successfully
        """
        if channels is None:
            # Unsubscribe from all channels for this market
            channels = list(KalshiChannel)
        
        success = True
        for channel in channels:
            if not await self._unsubscribe_from_channel(channel.value, market_ticker):
                success = False
        
        if market_ticker in self._market_subscriptions:
            self._market_subscriptions.remove(market_ticker)
        
        return success
    
    async def _subscribe_to_channel(
        self,
        channel: str,
        market_ticker: Optional[str] = None
    ) -> bool:
        """
        Subscribe to specific channel.
        
        Args:
            channel: Channel name
            market_ticker: Optional market ticker
            
        Returns:
            True if subscribed successfully
        """
        self._message_id += 1
        
        params = {"channels": [channel]}
        if market_ticker:
            params["market_ticker"] = market_ticker
        
        message = {
            "id": self._message_id,
            "cmd": "subscribe",
            "params": params
        }
        
        # Create future for response
        future = asyncio.Future()
        self._pending_requests[self._message_id] = future
        
        if await self.send(message):
            try:
                # Wait for response with timeout
                result = await asyncio.wait_for(future, timeout=5.0)
                
                if result.get("success"):
                    # Track subscription
                    if channel not in self._channel_subscriptions:
                        self._channel_subscriptions[channel] = set()
                    if market_ticker:
                        self._channel_subscriptions[channel].add(market_ticker)
                    
                    logger.info(f"Subscribed to {channel} for {market_ticker or 'all'}")
                    return True
                else:
                    logger.error(f"Subscription failed: {result.get('error')}")
                    
            except asyncio.TimeoutError:
                logger.error(f"Subscription timeout for {channel}")
            finally:
                self._pending_requests.pop(self._message_id, None)
        
        return False
    
    async def _unsubscribe_from_channel(
        self,
        channel: str,
        market_ticker: Optional[str] = None
    ) -> bool:
        """
        Unsubscribe from specific channel.
        
        Args:
            channel: Channel name
            market_ticker: Optional market ticker
            
        Returns:
            True if unsubscribed successfully
        """
        self._message_id += 1
        
        params = {"channels": [channel]}
        if market_ticker:
            params["market_ticker"] = market_ticker
        
        message = {
            "id": self._message_id,
            "cmd": "unsubscribe",
            "params": params
        }
        
        if await self.send(message):
            # Remove from tracking
            if channel in self._channel_subscriptions and market_ticker:
                self._channel_subscriptions[channel].discard(market_ticker)
            
            logger.info(f"Unsubscribed from {channel} for {market_ticker or 'all'}")
            return True
        
        return False
    
    # Internal subscription methods for base class
    
    async def _subscribe_internal(self, subscription: str) -> bool:
        """
        Internal subscription implementation.
        
        Args:
            subscription: Market ticker or channel:ticker format
            
        Returns:
            True if subscribed successfully
        """
        # Parse subscription format
        if ":" in subscription:
            channel, ticker = subscription.split(":", 1)
            return await self._subscribe_to_channel(channel, ticker)
        else:
            # Default to ticker channel for market
            return await self.subscribe_market(subscription)
    
    async def _unsubscribe_internal(self, subscription: str) -> bool:
        """
        Internal unsubscription implementation.
        
        Args:
            subscription: Market ticker or channel:ticker format
            
        Returns:
            True if unsubscribed successfully
        """
        if ":" in subscription:
            channel, ticker = subscription.split(":", 1)
            return await self._unsubscribe_from_channel(channel, ticker)
        else:
            return await self.unsubscribe_market(subscription)
    
    # Message processing
    
    async def _process_message(self, message: Dict[str, Any]):
        """
        Process incoming Kalshi WebSocket message.
        
        Args:
            message: Parsed message data
        """
        # Check if this is a response to a request
        msg_id = message.get("id")
        if msg_id and msg_id in self._pending_requests:
            future = self._pending_requests.get(msg_id)
            if future and not future.done():
                future.set_result(message)
            return
        
        # Process data messages
        msg_type = message.get("type")
        channel = message.get("channel")
        
        if not msg_type and not channel:
            # Could be subscription response
            return
        
        # Parse message based on channel
        if channel == KalshiChannel.TICKER.value:
            await self._process_ticker(message)
        elif channel == KalshiChannel.ORDERBOOK_DELTA.value:
            await self._process_orderbook(message)
        elif channel == KalshiChannel.TRADE.value:
            await self._process_trade(message)
        elif channel == KalshiChannel.FILL.value:
            await self._process_fill(message)
        elif channel == KalshiChannel.MARKET_POSITIONS.value:
            await self._process_positions(message)
        elif channel == KalshiChannel.MARKET_LIFECYCLE.value:
            await self._process_lifecycle(message)
        else:
            logger.debug(f"Unknown message type: {channel}")
    
    async def _process_ticker(self, message: Dict[str, Any]):
        """Process ticker update."""
        data = message.get("data", {})
        market_ticker = data.get("market_ticker")
        
        ticker_data = {
            "type": "ticker",
            "market_ticker": market_ticker,
            "yes_price": data.get("yes_price"),
            "no_price": data.get("no_price"),
            "yes_bid": data.get("yes_bid"),
            "yes_ask": data.get("yes_ask"),
            "no_bid": data.get("no_bid"),
            "no_ask": data.get("no_ask"),
            "volume": data.get("volume"),
            "open_interest": data.get("open_interest"),
            "timestamp": datetime.utcnow()
        }
        
        await self._emit_event("ticker", ticker_data)
        await self._emit_event(f"ticker:{market_ticker}", ticker_data)
    
    async def _process_orderbook(self, message: Dict[str, Any]):
        """Process orderbook delta."""
        data = message.get("data", {})
        market_ticker = data.get("market_ticker")
        
        orderbook_data = {
            "type": "orderbook_delta",
            "market_ticker": market_ticker,
            "yes_bids": data.get("yes_bids", []),
            "yes_asks": data.get("yes_asks", []),
            "no_bids": data.get("no_bids", []),
            "no_asks": data.get("no_asks", []),
            "timestamp": datetime.utcnow()
        }
        
        await self._emit_event("orderbook", orderbook_data)
        await self._emit_event(f"orderbook:{market_ticker}", orderbook_data)
    
    async def _process_trade(self, message: Dict[str, Any]):
        """Process executed trade."""
        data = message.get("data", {})
        market_ticker = data.get("market_ticker")
        
        trade_data = {
            "type": "trade",
            "market_ticker": market_ticker,
            "trade_id": data.get("trade_id"),
            "price": data.get("price"),
            "size": data.get("size"),
            "side": data.get("side"),  # yes/no
            "timestamp": data.get("created_time", datetime.utcnow())
        }
        
        await self._emit_event("trade", trade_data)
        await self._emit_event(f"trade:{market_ticker}", trade_data)
    
    async def _process_fill(self, message: Dict[str, Any]):
        """Process order fill."""
        data = message.get("data", {})
        
        fill_data = {
            "type": "fill",
            "order_id": data.get("order_id"),
            "market_ticker": data.get("market_ticker"),
            "price": data.get("price"),
            "size": data.get("size"),
            "side": data.get("side"),
            "timestamp": data.get("created_time", datetime.utcnow())
        }
        
        await self._emit_event("fill", fill_data)
    
    async def _process_positions(self, message: Dict[str, Any]):
        """Process position update."""
        data = message.get("data", {})
        
        position_data = {
            "type": "position",
            "market_ticker": data.get("market_ticker"),
            "position": data.get("position"),
            "average_price": data.get("average_price"),
            "realized_pnl": data.get("realized_pnl"),
            "unrealized_pnl": data.get("unrealized_pnl"),
            "timestamp": datetime.utcnow()
        }
        
        await self._emit_event("position", position_data)
    
    async def _process_lifecycle(self, message: Dict[str, Any]):
        """Process market lifecycle event."""
        data = message.get("data", {})
        market_ticker = data.get("market_ticker")
        
        lifecycle_data = {
            "type": "lifecycle",
            "market_ticker": market_ticker,
            "status": data.get("status"),
            "event": data.get("event"),
            "timestamp": datetime.utcnow()
        }
        
        await self._emit_event("lifecycle", lifecycle_data)
        await self._emit_event(f"lifecycle:{market_ticker}", lifecycle_data)
    
    # Heartbeat
    
    async def _send_heartbeat(self) -> bool:
        """
        Send heartbeat to keep connection alive.
        
        Kalshi WebSocket may not require explicit heartbeat,
        but we can send a ping frame.
        """
        try:
            if self._ws:
                await self._ws.ping()
                return True
        except Exception as e:
            logger.error(f"Heartbeat failed: {e}")
        return False
    
    # Utility methods
    
    def get_subscribed_markets(self) -> List[str]:
        """Get list of subscribed markets."""
        return list(self._market_subscriptions)
    
    def get_channel_subscriptions(self) -> Dict[str, List[str]]:
        """Get subscriptions by channel."""
        return {
            channel: list(markets)
            for channel, markets in self._channel_subscriptions.items()
        }
    
    async def get_market_snapshot(self, market_ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get current market snapshot.
        
        Note: This would typically query REST API for full snapshot.
        WebSocket provides updates only.
        """
        # This would integrate with REST adapter for snapshot
        logger.warning("Market snapshot requires REST API integration")
        return None