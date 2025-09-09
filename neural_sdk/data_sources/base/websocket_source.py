"""
Base WebSocket Data Source

Abstract base class for WebSocket-based data sources.
Provides connection management, reconnection logic, and event handling.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable, List, Set
from datetime import datetime, timedelta
from enum import Enum
import json
from dataclasses import dataclass, field
from typing import Dict, Optional
import aiohttp
from collections import deque

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """WebSocket connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    CLOSING = "closing"
    CLOSED = "closed"


@dataclass
class ConnectionConfig:
    """WebSocket connection configuration."""
    url: str
    api_key: Optional[str] = None
    heartbeat_interval: int = 30  # seconds
    reconnect_interval: int = 5  # seconds
    max_reconnect_attempts: int = 10
    connection_timeout: int = 30  # seconds
    message_queue_size: int = 10000
    max_subscriptions: int = 100
    # Optional headers to include when establishing the WS connection
    headers: Optional[Dict[str, str]] = None


@dataclass
class WebSocketStats:
    """WebSocket connection statistics."""
    connected_at: Optional[datetime] = None
    disconnected_at: Optional[datetime] = None
    messages_received: int = 0
    messages_sent: int = 0
    reconnect_attempts: int = 0
    last_error: Optional[str] = None
    last_heartbeat: Optional[datetime] = None
    subscriptions: Set[str] = field(default_factory=set)
    
    @property
    def uptime(self) -> Optional[timedelta]:
        """Get connection uptime."""
        if self.connected_at:
            end = self.disconnected_at or datetime.utcnow()
            return end - self.connected_at
        return None


class WebSocketDataSource(ABC):
    """
    Abstract base class for WebSocket data sources.
    
    Provides:
    - Automatic connection management
    - Reconnection with exponential backoff
    - Heartbeat/keepalive
    - Message queuing during disconnections
    - Event-driven architecture
    """
    
    def __init__(self, config: ConnectionConfig, name: str = "WebSocket"):
        """
        Initialize WebSocket data source.
        
        Args:
            config: Connection configuration
            name: Data source name for logging
        """
        self.config = config
        self.name = name
        self.state = ConnectionState.DISCONNECTED
        self.stats = WebSocketStats()
        
        # WebSocket connection
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Tasks
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._message_handler_task: Optional[asyncio.Task] = None
        self._reconnect_task: Optional[asyncio.Task] = None
        
        # Message queue for offline buffering
        self._message_queue = deque(maxlen=config.message_queue_size)
        
        # Event handlers
        self._event_handlers: Dict[str, List[Callable]] = {}
        
        # Subscriptions to restore on reconnect
        self._active_subscriptions: Set[str] = set()
        
        logger.info(f"{self.name} WebSocket initialized")
    
    # Abstract methods to implement
    
    @abstractmethod
    async def _authenticate(self) -> bool:
        """
        Authenticate WebSocket connection.
        
        Returns:
            True if authenticated successfully
        """
        pass
    
    @abstractmethod
    async def _subscribe_internal(self, subscription: str) -> bool:
        """
        Internal subscription implementation.
        
        Args:
            subscription: Subscription identifier
            
        Returns:
            True if subscribed successfully
        """
        pass
    
    @abstractmethod
    async def _unsubscribe_internal(self, subscription: str) -> bool:
        """
        Internal unsubscription implementation.
        
        Args:
            subscription: Subscription identifier
            
        Returns:
            True if unsubscribed successfully
        """
        pass
    
    @abstractmethod
    async def _process_message(self, message: Dict[str, Any]):
        """
        Process incoming WebSocket message.
        
        Args:
            message: Parsed message data
        """
        pass
    
    @abstractmethod
    async def _send_heartbeat(self) -> bool:
        """
        Send heartbeat/ping message.
        
        Returns:
            True if heartbeat sent successfully
        """
        pass
    
    # Connection management
    
    async def connect(self) -> bool:
        """
        Connect to WebSocket server.
        
        Returns:
            True if connected successfully
        """
        if self.state == ConnectionState.CONNECTED:
            logger.warning(f"{self.name} already connected")
            return True
        
        try:
            self.state = ConnectionState.CONNECTING
            logger.info(f"{self.name} connecting to {self.config.url}")
            
            # Create session
            if not self._session:
                self._session = aiohttp.ClientSession()
            
            # Connect with timeout
            timeout = aiohttp.ClientTimeout(total=self.config.connection_timeout)
            
            # Add authentication headers if needed
            headers = dict(self.config.headers or {})
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            
            self._ws = await self._session.ws_connect(
                self.config.url,
                headers=headers,
                timeout=timeout
            )
            
            # Authenticate if required
            if not await self._authenticate():
                await self.disconnect()
                return False
            
            self.state = ConnectionState.CONNECTED
            self.stats.connected_at = datetime.utcnow()
            self.stats.reconnect_attempts = 0
            
            # Start background tasks
            self._start_background_tasks()
            
            # Restore subscriptions
            await self._restore_subscriptions()
            
            logger.info(f"{self.name} connected successfully")
            await self._emit_event("connected", {"timestamp": datetime.utcnow()})
            
            return True
            
        except Exception as e:
            logger.error(f"{self.name} connection failed: {e}")
            self.stats.last_error = str(e)
            self.state = ConnectionState.DISCONNECTED
            await self._emit_event("error", {"error": str(e)})
            return False
    
    async def disconnect(self):
        """Disconnect from WebSocket server."""
        if self.state == ConnectionState.DISCONNECTED:
            return
        
        try:
            self.state = ConnectionState.CLOSING
            logger.info(f"{self.name} disconnecting")
            
            # Cancel background tasks
            self._cancel_background_tasks()
            
            # Close WebSocket
            if self._ws:
                await self._ws.close()
                self._ws = None
            
            # Close session
            if self._session:
                await self._session.close()
                self._session = None
            
            self.state = ConnectionState.CLOSED
            self.stats.disconnected_at = datetime.utcnow()
            
            logger.info(f"{self.name} disconnected")
            await self._emit_event("disconnected", {"timestamp": datetime.utcnow()})
            
        except Exception as e:
            logger.error(f"{self.name} disconnect error: {e}")
        finally:
            self.state = ConnectionState.DISCONNECTED
    
    async def reconnect(self):
        """Reconnect to WebSocket server."""
        await self.disconnect()
        await asyncio.sleep(self.config.reconnect_interval)
        return await self.connect()
    
    # Subscription management
    
    async def subscribe(self, subscription: str) -> bool:
        """
        Subscribe to data stream.
        
        Args:
            subscription: Subscription identifier
            
        Returns:
            True if subscribed successfully
        """
        if len(self._active_subscriptions) >= self.config.max_subscriptions:
            logger.error(f"{self.name} max subscriptions reached")
            return False
        
        if subscription in self._active_subscriptions:
            logger.warning(f"{self.name} already subscribed to {subscription}")
            return True
        
        if await self._subscribe_internal(subscription):
            self._active_subscriptions.add(subscription)
            self.stats.subscriptions.add(subscription)
            logger.info(f"{self.name} subscribed to {subscription}")
            return True
        
        return False
    
    async def unsubscribe(self, subscription: str) -> bool:
        """
        Unsubscribe from data stream.
        
        Args:
            subscription: Subscription identifier
            
        Returns:
            True if unsubscribed successfully
        """
        if subscription not in self._active_subscriptions:
            logger.warning(f"{self.name} not subscribed to {subscription}")
            return True
        
        if await self._unsubscribe_internal(subscription):
            self._active_subscriptions.discard(subscription)
            self.stats.subscriptions.discard(subscription)
            logger.info(f"{self.name} unsubscribed from {subscription}")
            return True
        
        return False
    
    # Message handling
    
    async def send(self, message: Dict[str, Any]) -> bool:
        """
        Send message through WebSocket.
        
        Args:
            message: Message to send
            
        Returns:
            True if sent successfully
        """
        if self.state != ConnectionState.CONNECTED:
            # Queue message if disconnected
            self._message_queue.append(message)
            logger.warning(f"{self.name} queued message (not connected)")
            return False
        
        try:
            if self._ws:
                await self._ws.send_json(message)
                self.stats.messages_sent += 1
                return True
        except Exception as e:
            logger.error(f"{self.name} send error: {e}")
            self.stats.last_error = str(e)
            # Trigger reconnection
            asyncio.create_task(self._handle_disconnection())
        
        return False
    
    # Event handling
    
    def on(self, event: str, handler: Callable):
        """
        Register event handler.
        
        Args:
            event: Event name
            handler: Event handler function
        """
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)
    
    async def _emit_event(self, event: str, data: Any):
        """
        Emit event to registered handlers.
        
        Args:
            event: Event name
            data: Event data
        """
        if event in self._event_handlers:
            for handler in self._event_handlers[event]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(data)
                    else:
                        handler(data)
                except Exception as e:
                    logger.error(f"{self.name} event handler error: {e}")
    
    # Background tasks
    
    def _start_background_tasks(self):
        """Start background tasks."""
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._message_handler_task = asyncio.create_task(self._message_handler_loop())
    
    def _cancel_background_tasks(self):
        """Cancel background tasks."""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        if self._message_handler_task:
            self._message_handler_task.cancel()
        if self._reconnect_task:
            self._reconnect_task.cancel()
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats."""
        while self.state == ConnectionState.CONNECTED:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)
                if await self._send_heartbeat():
                    self.stats.last_heartbeat = datetime.utcnow()
                else:
                    logger.warning(f"{self.name} heartbeat failed")
                    await self._handle_disconnection()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"{self.name} heartbeat error: {e}")
    
    async def _message_handler_loop(self):
        """Handle incoming messages."""
        while self.state == ConnectionState.CONNECTED:
            try:
                if self._ws:
                    msg = await self._ws.receive()
                    
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        self.stats.messages_received += 1
                        await self._process_message(data)
                        
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        logger.error(f"{self.name} WebSocket error: {msg.data}")
                        await self._handle_disconnection()
                        
                    elif msg.type == aiohttp.WSMsgType.CLOSED:
                        logger.info(f"{self.name} WebSocket closed")
                        await self._handle_disconnection()
                        break
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"{self.name} message handler error: {e}")
                await self._handle_disconnection()
    
    async def _handle_disconnection(self):
        """Handle unexpected disconnection."""
        if self.state != ConnectionState.CONNECTED:
            return
        
        self.state = ConnectionState.RECONNECTING
        await self._emit_event("disconnected", {"unexpected": True})
        
        # Start reconnection task
        if not self._reconnect_task or self._reconnect_task.done():
            self._reconnect_task = asyncio.create_task(self._reconnect_loop())
    
    async def _reconnect_loop(self):
        """Attempt to reconnect with exponential backoff."""
        backoff = self.config.reconnect_interval
        
        while self.stats.reconnect_attempts < self.config.max_reconnect_attempts:
            self.stats.reconnect_attempts += 1
            logger.info(f"{self.name} reconnection attempt {self.stats.reconnect_attempts}")
            
            if await self.reconnect():
                logger.info(f"{self.name} reconnected successfully")
                return
            
            # Exponential backoff
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 300)  # Max 5 minutes
        
        logger.error(f"{self.name} max reconnection attempts reached")
        await self._emit_event("max_reconnects_reached", {})
    
    async def _restore_subscriptions(self):
        """Restore subscriptions after reconnection."""
        for subscription in list(self._active_subscriptions):
            logger.info(f"{self.name} restoring subscription: {subscription}")
            await self._subscribe_internal(subscription)
    
    async def _process_queued_messages(self):
        """Process messages queued during disconnection."""
        while self._message_queue and self.state == ConnectionState.CONNECTED:
            message = self._message_queue.popleft()
            await self.send(message)
    
    # Status methods
    
    def get_state(self) -> ConnectionState:
        """Get current connection state."""
        return self.state
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            "state": self.state.value,
            "connected_at": self.stats.connected_at.isoformat() if self.stats.connected_at else None,
            "uptime": str(self.stats.uptime) if self.stats.uptime else None,
            "messages_received": self.stats.messages_received,
            "messages_sent": self.stats.messages_sent,
            "reconnect_attempts": self.stats.reconnect_attempts,
            "subscriptions": list(self.stats.subscriptions),
            "last_error": self.stats.last_error,
            "last_heartbeat": self.stats.last_heartbeat.isoformat() if self.stats.last_heartbeat else None
        }
    
    def is_connected(self) -> bool:
        """Check if connected."""
        return self.state == ConnectionState.CONNECTED
