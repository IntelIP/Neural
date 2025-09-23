"""
WebSocket data source implementation for the Neural SDK.

This module provides a robust WebSocket client that handles real-time data
streaming with features like automatic reconnection, heartbeat management,
message queuing, and backpressure handling.
"""

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Union
import aiohttp
from aiohttp import ClientSession, ClientWebSocketResponse
import logging

from neural.data_collection.base import BaseDataSource, DataSourceConfig, ConnectionState
from neural.data_collection.exceptions import (
    ConnectionError,
    BufferOverflowError,
    TimeoutError,
    DataSourceError
)


# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class WebSocketConfig(DataSourceConfig):
    """
    Configuration specific to WebSocket data sources.
    
    Extends the base DataSourceConfig with WebSocket-specific parameters
    for connection management, heartbeat, and message handling.
    
    Attributes:
        url: WebSocket endpoint URL (wss:// or ws://)
        headers: HTTP headers to include in the connection request
        heartbeat_interval: Seconds between heartbeat pings (0 to disable)
        heartbeat_timeout: Seconds to wait for pong response
        message_queue_size: Maximum messages to queue
        compression: Whether to use WebSocket compression
        subprotocols: List of subprotocols to negotiate
        
    Example:
        >>> config = WebSocketConfig(
        ...     name="twitter_stream",
        ...     url="wss://api.twitter.com/stream",
        ...     headers={"Authorization": "Bearer token"},
        ...     heartbeat_interval=30
        ... )
    """
    url: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    heartbeat_interval: float = 30.0
    heartbeat_timeout: float = 10.0
    message_queue_size: int = 10000
    compression: bool = True
    subprotocols: List[str] = field(default_factory=list)
    
    def validate(self) -> None:
        """
        Validate WebSocket-specific configuration.
        
        Raises:
            ConfigurationError: If configuration is invalid
        """
        # Call parent validation
        super().validate()
        
        # Validate WebSocket-specific fields
        if not self.url:
            raise ConfigurationError("WebSocket URL cannot be empty")
        
        if not self.url.startswith(('ws://', 'wss://')):
            raise ConfigurationError(
                "WebSocket URL must start with ws:// or wss://",
                details={"url": self.url}
            )
        
        if self.heartbeat_interval < 0:
            raise ConfigurationError(
                "heartbeat_interval must be non-negative",
                details={"value": self.heartbeat_interval}
            )
        
        if self.heartbeat_timeout <= 0:
            raise ConfigurationError(
                "heartbeat_timeout must be positive",
                details={"value": self.heartbeat_timeout}
            )
        
        if self.message_queue_size <= 0:
            raise ConfigurationError(
                "message_queue_size must be positive",
                details={"value": self.message_queue_size}
            )


class WebSocketDataSource(BaseDataSource):
    """
    WebSocket implementation of a data source.
    
    This class provides a complete WebSocket client with features like:
    - Automatic reconnection with exponential backoff
    - Heartbeat/ping-pong to keep connection alive
    - Message queue with overflow protection
    - Subscription management
    - Graceful shutdown
    
    The class handles all WebSocket lifecycle events and provides a simple
    interface for receiving real-time data.
    
    Attributes:
        config: WebSocket configuration
        session: aiohttp client session
        websocket: Active WebSocket connection
        message_queue: Queue of received messages
        subscriptions: Active channel subscriptions
        
    Example:
        >>> config = WebSocketConfig(
        ...     name="market_data",
        ...     url="wss://api.example.com/stream"
        ... )
        >>> source = WebSocketDataSource(config)
        >>> 
        >>> # Register data handler
        >>> source.register_callback("data", on_market_data)
        >>> 
        >>> # Connect and subscribe
        >>> await source.connect()
        >>> await source.subscribe(["ticker.BTC", "trades.ETH"])
    """
    
    def __init__(self, config: WebSocketConfig):
        """
        Initialize the WebSocket data source.
        
        Args:
            config: WebSocket configuration
        """
        super().__init__(config)
        self.config: WebSocketConfig = config
        
        # Connection resources
        self.session: Optional[ClientSession] = None
        self.websocket: Optional[ClientWebSocketResponse] = None
        
        # Message handling
        self.message_queue: Deque[Dict[str, Any]] = deque(
            maxlen=config.message_queue_size
        )
        self.subscriptions: Set[str] = set()
        
        # Tasks
        self._receive_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        
        # Statistics
        self._last_message_time = time.time()
        self._last_heartbeat_time = time.time()
        
        logger.info(f"Initialized WebSocketDataSource: {config.name}")
    
    async def _connect_impl(self) -> bool:
        """
        Establish WebSocket connection.
        
        Returns:
            bool: True if connection successful
        """
        try:
            # Create session if not exists
            if not self.session:
                timeout = aiohttp.ClientTimeout(
                    total=self.config.timeout,
                    connect=self.config.timeout / 2
                )
                self.session = ClientSession(timeout=timeout)
            
            # Connect to WebSocket
            logger.debug(f"{self.config.name}: Connecting to {self.config.url}")
            
            self.websocket = await self.session.ws_connect(
                self.config.url,
                headers=self.config.headers,
                compress=self.config.compression,
                protocols=self.config.subprotocols if self.config.subprotocols else None,
                heartbeat=self.config.heartbeat_interval if self.config.heartbeat_interval > 0 else None
            )
            
            # Start receive task
            self._receive_task = asyncio.create_task(
                self._receive_messages(),
                name=f"{self.config.name}_receive"
            )
            
            # Start heartbeat task if configured
            if self.config.heartbeat_interval > 0:
                self._heartbeat_task = asyncio.create_task(
                    self._heartbeat_loop(),
                    name=f"{self.config.name}_heartbeat"
                )
            
            logger.info(f"{self.config.name}: WebSocket connected successfully")
            return True
            
        except asyncio.TimeoutError:
            logger.error(f"{self.config.name}: Connection timeout")
            raise TimeoutError(
                "WebSocket connection timeout",
                details={"url": self.config.url, "timeout": self.config.timeout}
            )
        except Exception as e:
            logger.error(f"{self.config.name}: Connection failed: {e}")
            raise ConnectionError(
                f"Failed to connect to WebSocket: {e}",
                details={"url": self.config.url}
            )
    
    async def _disconnect_impl(self) -> None:
        """
        Close WebSocket connection and cleanup resources.
        """
        # Cancel tasks
        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        
        # Close WebSocket
        if self.websocket and not self.websocket.closed:
            await self.websocket.close()
            logger.debug(f"{self.config.name}: WebSocket closed")
        
        # Close session
        if self.session and not self.session.closed:
            await self.session.close()
            # Small delay to allow graceful closure
            await asyncio.sleep(0.1)
            logger.debug(f"{self.config.name}: Session closed")
        
        # Clear resources
        self.websocket = None
        self.session = None
        self.subscriptions.clear()
        
        logger.info(f"{self.config.name}: Disconnected and cleaned up")
    
    async def _receive_messages(self) -> None:
        """
        Continuously receive messages from the WebSocket.
        
        This method runs in a separate task and processes incoming
        messages, triggering callbacks and handling errors.
        """
        logger.debug(f"{self.config.name}: Starting message receive loop")
        
        try:
            async for msg in self.websocket:
                try:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        # Parse and process text message
                        data = json.loads(msg.data) if isinstance(msg.data, str) else msg.data
                        await self._handle_message(data)
                        
                    elif msg.type == aiohttp.WSMsgType.BINARY:
                        # Process binary message
                        await self._handle_binary_message(msg.data)
                        
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        logger.error(
                            f"{self.config.name}: WebSocket error: "
                            f"{self.websocket.exception()}"
                        )
                        await self._trigger_callbacks("error", self.websocket.exception())
                        
                    elif msg.type == aiohttp.WSMsgType.CLOSE:
                        logger.info(f"{self.config.name}: WebSocket closed by server")
                        break
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"{self.config.name}: Failed to parse message: {e}")
                except Exception as e:
                    logger.error(f"{self.config.name}: Error processing message: {e}")
                    await self._trigger_callbacks("error", e)
                    
        except Exception as e:
            logger.error(f"{self.config.name}: Receive loop error: {e}")
            self.state = ConnectionState.ERROR
            await self._trigger_callbacks("error", e)
        
        logger.debug(f"{self.config.name}: Message receive loop ended")
    
    async def _handle_message(self, data: Dict[str, Any]) -> None:
        """
        Handle a received message.
        
        Args:
            data: Parsed message data
        """
        # Update statistics
        self._last_message_time = time.time()
        self._metrics["messages_received"] += 1
        
        # Add to queue if not full
        if len(self.message_queue) >= self.config.message_queue_size:
            # Queue is full, drop oldest message
            dropped = self.message_queue.popleft()
            logger.warning(
                f"{self.config.name}: Message queue full, dropping oldest message"
            )
            
            # Trigger error callback for dropped message
            error = BufferOverflowError(
                "Message queue overflow",
                details={
                    "queue_size": self.config.message_queue_size,
                    "dropped_message": dropped
                }
            )
            await self._trigger_callbacks("error", error)
        
        # Add message to queue
        self.message_queue.append(data)
        
        # Trigger data callback
        await self._trigger_callbacks("data", data)
        
        # Update processed count
        self._metrics["messages_processed"] += 1
    
    async def _handle_binary_message(self, data: bytes) -> None:
        """
        Handle a binary message.
        
        Args:
            data: Binary message data
        """
        # For now, just trigger callback with raw bytes
        # Subclasses can override for specific handling
        await self._trigger_callbacks("data", data)
        self._metrics["messages_received"] += 1
        self._metrics["messages_processed"] += 1
    
    async def _heartbeat_loop(self) -> None:
        """
        Send periodic heartbeat pings to keep connection alive.
        """
        logger.debug(
            f"{self.config.name}: Starting heartbeat loop "
            f"(interval={self.config.heartbeat_interval}s)"
        )
        
        try:
            while self.state == ConnectionState.CONNECTED:
                await asyncio.sleep(self.config.heartbeat_interval)
                
                if self.websocket and not self.websocket.closed:
                    try:
                        # Send ping
                        pong_waiter = await self.websocket.ping()
                        
                        # Wait for pong with timeout
                        await asyncio.wait_for(
                            pong_waiter,
                            timeout=self.config.heartbeat_timeout
                        )
                        
                        self._last_heartbeat_time = time.time()
                        logger.debug(f"{self.config.name}: Heartbeat successful")
                        
                    except asyncio.TimeoutError:
                        logger.warning(
                            f"{self.config.name}: Heartbeat timeout, "
                            f"connection may be dead"
                        )
                        # Connection might be dead, trigger reconnection
                        self.state = ConnectionState.ERROR
                        break
                    except Exception as e:
                        logger.error(f"{self.config.name}: Heartbeat error: {e}")
                        
        except asyncio.CancelledError:
            logger.debug(f"{self.config.name}: Heartbeat loop cancelled")
        except Exception as e:
            logger.error(f"{self.config.name}: Heartbeat loop error: {e}")
        
        logger.debug(f"{self.config.name}: Heartbeat loop ended")
    
    async def _subscribe_impl(self, channels: List[str]) -> bool:
        """
        Subscribe to WebSocket channels.
        
        Args:
            channels: List of channel names to subscribe to
            
        Returns:
            bool: True if subscription successful
        """
        if not self.websocket or self.websocket.closed:
            raise ConnectionError("WebSocket is not connected")
        
        # Default implementation sends a subscription message
        # Subclasses should override for specific protocols
        subscription_msg = {
            "action": "subscribe",
            "channels": channels
        }
        
        try:
            await self.websocket.send_json(subscription_msg)
            self.subscriptions.update(channels)
            
            logger.info(
                f"{self.config.name}: Subscribed to channels: {channels}"
            )
            return True
            
        except Exception as e:
            logger.error(f"{self.config.name}: Subscription failed: {e}")
            raise DataSourceError(
                f"Failed to subscribe to channels: {e}",
                details={"channels": channels}
            )
    
    async def unsubscribe(self, channels: List[str]) -> bool:
        """
        Unsubscribe from WebSocket channels.
        
        Args:
            channels: List of channel names to unsubscribe from
            
        Returns:
            bool: True if unsubscription successful
            
        Example:
            >>> await source.unsubscribe(["ticker.BTC"])
        """
        if not self.websocket or self.websocket.closed:
            raise ConnectionError("WebSocket is not connected")
        
        # Default implementation sends an unsubscription message
        unsubscribe_msg = {
            "action": "unsubscribe",
            "channels": channels
        }
        
        try:
            await self.websocket.send_json(unsubscribe_msg)
            self.subscriptions.difference_update(channels)
            
            logger.info(
                f"{self.config.name}: Unsubscribed from channels: {channels}"
            )
            return True
            
        except Exception as e:
            logger.error(f"{self.config.name}: Unsubscription failed: {e}")
            return False
    
    async def send_message(
        self,
        message: Union[str, Dict[str, Any], bytes]
    ) -> None:
        """
        Send a message through the WebSocket.
        
        Args:
            message: Message to send (string, dict, or bytes)
            
        Raises:
            ConnectionError: If not connected
            DataSourceError: If send fails
            
        Example:
            >>> await source.send_message({"type": "ping"})
        """
        if not self.websocket or self.websocket.closed:
            raise ConnectionError("WebSocket is not connected")
        
        try:
            if isinstance(message, dict):
                await self.websocket.send_json(message)
            elif isinstance(message, str):
                await self.websocket.send_str(message)
            elif isinstance(message, bytes):
                await self.websocket.send_bytes(message)
            else:
                raise ValueError(f"Unsupported message type: {type(message)}")
            
            logger.debug(f"{self.config.name}: Sent message")
            
        except Exception as e:
            logger.error(f"{self.config.name}: Failed to send message: {e}")
            raise DataSourceError(
                f"Failed to send message: {e}",
                details={"message_type": type(message).__name__}
            )
    
    def get_queue_size(self) -> int:
        """
        Get the current message queue size.
        
        Returns:
            Number of messages in queue
            
        Example:
            >>> size = source.get_queue_size()
            >>> print(f"Queue has {size} messages")
        """
        return len(self.message_queue)
    
    def clear_queue(self) -> None:
        """
        Clear all messages from the queue.
        
        Example:
            >>> source.clear_queue()
        """
        self.message_queue.clear()
        logger.debug(f"{self.config.name}: Message queue cleared")
    
    def get_subscriptions(self) -> Set[str]:
        """
        Get current channel subscriptions.
        
        Returns:
            Set of subscribed channel names
            
        Example:
            >>> subs = source.get_subscriptions()
            >>> print(f"Subscribed to: {subs}")
        """
        return self.subscriptions.copy()
    
    async def subscribe(self, channels: List[str]) -> bool:
        """
        Subscribe to channels (public interface).
        
        Args:
            channels: List of channel names
            
        Returns:
            bool: True if successful
            
        Example:
            >>> await source.subscribe(["trades", "orderbook"])
        """
        return await self._subscribe_impl(channels)