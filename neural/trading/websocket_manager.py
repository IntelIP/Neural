"""
Kalshi WebSocket Manager

This module provides real-time streaming capabilities for Kalshi market data
and trading operations using WebSocket connections.

WebSocket URL: wss://api.elections.kalshi.com/trade-api/ws/v2
Demo URL: wss://demo-api.kalshi.co/trade-api/ws/v2

Supported message types:
- Orderbook updates
- Trade executions
- Market status changes
- Fill notifications
- Order status updates
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass
from enum import Enum
import websockets
from websockets.exceptions import ConnectionClosed, InvalidHandshake

from .kalshi_client import KalshiClient, Environment

logger = logging.getLogger(__name__)


class SubscriptionType(Enum):
    """Types of WebSocket subscriptions."""
    ORDERBOOK = "orderbook_delta"
    TRADES = "trade"
    MARKET_STATUS = "market_status"
    FILLS = "fill"
    ORDERS = "order"


@dataclass
class WebSocketMessage:
    """WebSocket message structure."""
    id: str
    type: str
    seq: int
    msg: Dict[str, Any]
    timestamp: datetime
    
    @classmethod
    def from_raw(cls, data: Dict[str, Any]) -> 'WebSocketMessage':
        """Create from raw WebSocket data."""
        return cls(
            id=data.get("id", ""),
            type=data.get("type", ""),
            seq=data.get("seq", 0),
            msg=data.get("msg", {}),
            timestamp=datetime.now()
        )


@dataclass
class Subscription:
    """WebSocket subscription."""
    id: str
    type: SubscriptionType
    params: Dict[str, Any]
    handler: Optional[Callable[[WebSocketMessage], None]] = None


class WebSocketManager:
    """
    Kalshi WebSocket connection manager.
    
    Handles:
    - WebSocket connection lifecycle
    - Subscription management
    - Message routing and handling
    - Automatic reconnection
    - Authentication for trading operations
    """
    
    def __init__(
        self,
        kalshi_client: KalshiClient,
        max_reconnect_attempts: int = 10,
        reconnect_delay: float = 5.0,
        ping_interval: int = 30,
        ping_timeout: int = 10
    ):
        """
        Initialize WebSocket manager.
        
        Args:
            kalshi_client: Authenticated Kalshi client
            max_reconnect_attempts: Maximum reconnection attempts
            reconnect_delay: Delay between reconnection attempts
            ping_interval: Ping interval in seconds
            ping_timeout: Ping timeout in seconds
        """
        self.client = kalshi_client
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        
        # WebSocket connection
        self.websocket = None
        self.connected = False
        self.reconnect_count = 0
        
        # Subscription management
        self.subscriptions: Dict[str, Subscription] = {}
        self.message_handlers: Dict[SubscriptionType, List[Callable]] = {
            subscription_type: [] for subscription_type in SubscriptionType
        }
        
        # Message tracking
        self.last_seq_num = 0
        self.message_queue = asyncio.Queue()
        
        # Control flags
        self.running = False
        self.should_reconnect = True
        
        # WebSocket URL based on environment
        if self.client.config.environment == Environment.PRODUCTION:
            self.ws_url = "wss://api.elections.kalshi.com/trade-api/ws/v2"
        else:
            self.ws_url = "wss://demo-api.kalshi.co/trade-api/ws/v2"
        
        logger.info(f"Initialized WebSocket manager for {self.client.config.environment.value}")
    
    async def connect(self):
        """Establish WebSocket connection."""
        try:
            # Prepare headers for authentication if available
            headers = {}
            if self.client.authenticated and self.client.access_token:
                headers["Authorization"] = f"Bearer {self.client.access_token}"
            
            logger.info(f"Connecting to WebSocket: {self.ws_url}")
            
            # WebSocket connection with headers
            additional_headers = None
            if headers:
                additional_headers = list(headers.items())
            
            self.websocket = await websockets.connect(
                self.ws_url,
                additional_headers=additional_headers,
                ping_interval=self.ping_interval,
                ping_timeout=self.ping_timeout
            )
            
            self.connected = True
            self.reconnect_count = 0
            logger.info("WebSocket connection established")
            
            # Start message processing
            if not self.running:
                self.running = True
                asyncio.create_task(self._message_loop())
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            self.connected = False
            raise
    
    async def disconnect(self):
        """Close WebSocket connection."""
        self.should_reconnect = False
        self.running = False
        
        if self.websocket:
            try:
                await self.websocket.close()
                logger.info("WebSocket connection closed")
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}")
            finally:
                self.websocket = None
                self.connected = False
    
    async def _reconnect(self):
        """Attempt to reconnect WebSocket."""
        if not self.should_reconnect:
            return
        
        while self.reconnect_count < self.max_reconnect_attempts:
            self.reconnect_count += 1
            logger.info(f"Reconnection attempt {self.reconnect_count}/{self.max_reconnect_attempts}")
            
            try:
                await asyncio.sleep(self.reconnect_delay * self.reconnect_count)  # Exponential backoff
                await self.connect()
                
                # Re-subscribe to all previous subscriptions
                await self._resubscribe_all()
                return
                
            except Exception as e:
                logger.error(f"Reconnection attempt {self.reconnect_count} failed: {e}")
        
        logger.error("Max reconnection attempts reached, giving up")
        self.should_reconnect = False
    
    async def _resubscribe_all(self):
        """Re-subscribe to all previous subscriptions after reconnection."""
        for subscription in self.subscriptions.values():
            try:
                await self._send_subscription(subscription)
                logger.info(f"Re-subscribed to {subscription.type.value}: {subscription.id}")
            except Exception as e:
                logger.error(f"Failed to re-subscribe to {subscription.id}: {e}")
    
    async def _message_loop(self):
        """Main message processing loop."""
        while self.running:
            try:
                if not self.connected or not self.websocket:
                    await asyncio.sleep(1)
                    continue
                
                # Receive message
                raw_message = await self.websocket.recv()
                message_data = json.loads(raw_message)
                
                # Create message object
                message = WebSocketMessage.from_raw(message_data)
                
                # Track sequence numbers for message ordering
                if message.seq > self.last_seq_num:
                    self.last_seq_num = message.seq
                elif message.seq < self.last_seq_num:
                    logger.warning(f"Out-of-order message: seq={message.seq}, last={self.last_seq_num}")
                
                # Route message to handlers
                await self._route_message(message)
                
            except ConnectionClosed:
                logger.warning("WebSocket connection closed")
                self.connected = False
                if self.should_reconnect:
                    asyncio.create_task(self._reconnect())
                break
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode WebSocket message: {e}")
                
            except Exception as e:
                logger.error(f"Error in message loop: {e}")
                await asyncio.sleep(1)
    
    async def _route_message(self, message: WebSocketMessage):
        """Route message to appropriate handlers."""
        try:
            # Determine subscription type from message
            subscription_type = self._get_subscription_type(message)
            
            if subscription_type:
                # Call registered handlers
                handlers = self.message_handlers.get(subscription_type, [])
                for handler in handlers:
                    try:
                        await self._call_handler(handler, message)
                    except Exception as e:
                        logger.error(f"Handler error for {subscription_type.value}: {e}")
            else:
                logger.debug(f"Unhandled message type: {message.type}")
                
        except Exception as e:
            logger.error(f"Error routing message: {e}")
    
    def _get_subscription_type(self, message: WebSocketMessage) -> Optional[SubscriptionType]:
        """Determine subscription type from message."""
        message_type = message.type.lower()
        
        if "orderbook" in message_type:
            return SubscriptionType.ORDERBOOK
        elif "trade" in message_type:
            return SubscriptionType.TRADES
        elif "market" in message_type:
            return SubscriptionType.MARKET_STATUS
        elif "fill" in message_type:
            return SubscriptionType.FILLS
        elif "order" in message_type:
            return SubscriptionType.ORDERS
        
        return None
    
    async def _call_handler(self, handler: Callable, message: WebSocketMessage):
        """Call message handler safely."""
        if asyncio.iscoroutinefunction(handler):
            await handler(message)
        else:
            handler(message)
    
    async def _send_subscription(self, subscription: Subscription):
        """Send subscription message to WebSocket."""
        if not self.connected or not self.websocket:
            raise RuntimeError("WebSocket not connected")
        
        subscribe_message = {
            "id": subscription.id,
            "cmd": "subscribe",
            "params": {
                "channel": subscription.type.value,
                **subscription.params
            }
        }
        
        await self.websocket.send(json.dumps(subscribe_message))
        logger.debug(f"Sent subscription: {subscription.id}")
    
    async def _send_unsubscription(self, subscription_id: str):
        """Send unsubscription message to WebSocket."""
        if not self.connected or not self.websocket:
            raise RuntimeError("WebSocket not connected")
        
        unsubscribe_message = {
            "id": subscription_id,
            "cmd": "unsubscribe"
        }
        
        await self.websocket.send(json.dumps(unsubscribe_message))
        logger.debug(f"Sent unsubscription: {subscription_id}")
    
    # Public API Methods
    
    async def subscribe_orderbook(
        self, 
        ticker: str, 
        handler: Optional[Callable[[WebSocketMessage], None]] = None
    ) -> str:
        """
        Subscribe to orderbook updates for a market.
        
        Args:
            ticker: Market ticker
            handler: Optional message handler
            
        Returns:
            Subscription ID
        """
        subscription_id = f"orderbook_{ticker}_{datetime.now().timestamp()}"
        
        subscription = Subscription(
            id=subscription_id,
            type=SubscriptionType.ORDERBOOK,
            params={"ticker": ticker},
            handler=handler
        )
        
        self.subscriptions[subscription_id] = subscription
        
        if handler:
            self.message_handlers[SubscriptionType.ORDERBOOK].append(handler)
        
        if self.connected:
            await self._send_subscription(subscription)
        
        logger.info(f"Subscribed to orderbook: {ticker}")
        return subscription_id
    
    async def subscribe_trades(
        self, 
        ticker: str, 
        handler: Optional[Callable[[WebSocketMessage], None]] = None
    ) -> str:
        """Subscribe to trade updates for a market."""
        subscription_id = f"trades_{ticker}_{datetime.now().timestamp()}"
        
        subscription = Subscription(
            id=subscription_id,
            type=SubscriptionType.TRADES,
            params={"ticker": ticker},
            handler=handler
        )
        
        self.subscriptions[subscription_id] = subscription
        
        if handler:
            self.message_handlers[SubscriptionType.TRADES].append(handler)
        
        if self.connected:
            await self._send_subscription(subscription)
        
        logger.info(f"Subscribed to trades: {ticker}")
        return subscription_id
    
    async def subscribe_market_status(
        self, 
        ticker: str,
        handler: Optional[Callable[[WebSocketMessage], None]] = None
    ) -> str:
        """Subscribe to market status updates."""
        subscription_id = f"status_{ticker}_{datetime.now().timestamp()}"
        
        subscription = Subscription(
            id=subscription_id,
            type=SubscriptionType.MARKET_STATUS,
            params={"ticker": ticker},
            handler=handler
        )
        
        self.subscriptions[subscription_id] = subscription
        
        if handler:
            self.message_handlers[SubscriptionType.MARKET_STATUS].append(handler)
        
        if self.connected:
            await self._send_subscription(subscription)
        
        logger.info(f"Subscribed to market status: {ticker}")
        return subscription_id
    
    async def subscribe_fills(
        self, 
        handler: Optional[Callable[[WebSocketMessage], None]] = None
    ) -> str:
        """Subscribe to fill notifications (requires authentication)."""
        if not self.client.authenticated:
            raise RuntimeError("Authentication required for fill subscriptions")
        
        subscription_id = f"fills_{datetime.now().timestamp()}"
        
        subscription = Subscription(
            id=subscription_id,
            type=SubscriptionType.FILLS,
            params={},  # No additional params for fills
            handler=handler
        )
        
        self.subscriptions[subscription_id] = subscription
        
        if handler:
            self.message_handlers[SubscriptionType.FILLS].append(handler)
        
        if self.connected:
            await self._send_subscription(subscription)
        
        logger.info("Subscribed to fills")
        return subscription_id
    
    async def subscribe_orders(
        self, 
        handler: Optional[Callable[[WebSocketMessage], None]] = None
    ) -> str:
        """Subscribe to order status updates (requires authentication)."""
        if not self.client.authenticated:
            raise RuntimeError("Authentication required for order subscriptions")
        
        subscription_id = f"orders_{datetime.now().timestamp()}"
        
        subscription = Subscription(
            id=subscription_id,
            type=SubscriptionType.ORDERS,
            params={},
            handler=handler
        )
        
        self.subscriptions[subscription_id] = subscription
        
        if handler:
            self.message_handlers[SubscriptionType.ORDERS].append(handler)
        
        if self.connected:
            await self._send_subscription(subscription)
        
        logger.info("Subscribed to orders")
        return subscription_id
    
    async def unsubscribe(self, subscription_id: str):
        """
        Unsubscribe from a subscription.
        
        Args:
            subscription_id: Subscription ID to remove
        """
        if subscription_id not in self.subscriptions:
            logger.warning(f"Subscription not found: {subscription_id}")
            return
        
        subscription = self.subscriptions[subscription_id]
        
        # Remove from subscriptions
        del self.subscriptions[subscription_id]
        
        # Remove handler if it was registered
        if subscription.handler:
            handlers = self.message_handlers.get(subscription.type, [])
            if subscription.handler in handlers:
                handlers.remove(subscription.handler)
        
        # Send unsubscribe message if connected
        if self.connected:
            await self._send_unsubscription(subscription_id)
        
        logger.info(f"Unsubscribed from {subscription_id}")
    
    async def unsubscribe_all(self):
        """Unsubscribe from all subscriptions."""
        subscription_ids = list(self.subscriptions.keys())
        for subscription_id in subscription_ids:
            await self.unsubscribe(subscription_id)
        
        logger.info("Unsubscribed from all subscriptions")
    
    def add_handler(
        self, 
        subscription_type: SubscriptionType, 
        handler: Callable[[WebSocketMessage], None]
    ):
        """
        Add a global handler for a subscription type.
        
        Args:
            subscription_type: Type of subscription
            handler: Message handler function
        """
        self.message_handlers[subscription_type].append(handler)
        logger.info(f"Added handler for {subscription_type.value}")
    
    def remove_handler(
        self, 
        subscription_type: SubscriptionType, 
        handler: Callable[[WebSocketMessage], None]
    ):
        """Remove a global handler."""
        handlers = self.message_handlers.get(subscription_type, [])
        if handler in handlers:
            handlers.remove(handler)
            logger.info(f"Removed handler for {subscription_type.value}")
    
    # Context manager support
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
    
    # Status methods
    
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self.connected and self.websocket is not None
    
    def get_subscription_count(self) -> int:
        """Get number of active subscriptions."""
        return len(self.subscriptions)
    
    def get_subscriptions(self) -> List[str]:
        """Get list of subscription IDs."""
        return list(self.subscriptions.keys())
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get detailed connection status."""
        return {
            "connected": self.connected,
            "running": self.running,
            "reconnect_count": self.reconnect_count,
            "subscriptions": len(self.subscriptions),
            "last_seq_num": self.last_seq_num,
            "should_reconnect": self.should_reconnect
        }
