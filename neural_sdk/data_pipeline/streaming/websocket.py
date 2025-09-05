"""
Kalshi WebSocket Infrastructure - WebSocket Client
Handles real-time market data streaming from Kalshi
With circuit breaker protection, flow control, and automatic reconnection
"""

import asyncio
import json
import logging
import time
from typing import List, Optional, Dict, Any

import websockets
from websockets.client import WebSocketClientProtocol

from ..config import KalshiConfig
from ..data_sources.kalshi.auth import KalshiAuth
from .handlers import MessageHandler, DefaultMessageHandler
from ..reliability.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitOpenException
from ..reliability.flow_controller import CreditBasedFlowController
from ..reliability.rate_limiter import TokenBucket

logger = logging.getLogger(__name__)


class KalshiWebSocket:
    """WebSocket client for Kalshi market data streaming"""
    
    def __init__(
        self,
        config: Optional[KalshiConfig] = None,
        auth: Optional[KalshiAuth] = None,
        message_handler: Optional[MessageHandler] = None,
        enable_circuit_breaker: bool = True,
        auto_reconnect: bool = True,
        enable_flow_control: bool = True
    ):
        """
        Initialize WebSocket client
        
        Args:
            config: Optional KalshiConfig instance
            auth: Optional KalshiAuth instance
            message_handler: Optional custom message handler
            enable_circuit_breaker: Enable circuit breaker protection
            auto_reconnect: Enable automatic reconnection
        """
        if config is None:
            from ..config import get_config
            config = get_config()
        
        self.config = config
        self.auth = auth or KalshiAuth(config)
        self.message_handler = message_handler or DefaultMessageHandler()
        
        self.ws: Optional[WebSocketClientProtocol] = None
        self.subscribed_markets: set = set()
        self.running = False
        self._heartbeat_task = None
        self._receive_task = None
        
        # Circuit breaker configuration
        self.enable_circuit_breaker = enable_circuit_breaker
        self.auto_reconnect = auto_reconnect
        
        if enable_circuit_breaker:
            breaker_config = CircuitBreakerConfig(
                name="kalshi_websocket",
                failure_threshold=3,  # 3 failures trigger open
                success_threshold=2,  # 2 successes to close
                timeout=30.0,
                half_open_interval=10.0,
                window_size=60
            )
            self.circuit_breaker = CircuitBreaker(breaker_config)
        else:
            self.circuit_breaker = None
        
        # Connection stats
        self.connection_attempts = 0
        self.last_connection_time: Optional[float] = None
        self.reconnect_task: Optional[asyncio.Task] = None
        
        # Flow control and rate limiting
        self.enable_flow_control = enable_flow_control
        if enable_flow_control:
            self.flow_controller = CreditBasedFlowController(
                initial_credits=1000,
                max_credits=5000,
                refill_rate=100,  # 100 credits/second
                window_size=500
            )
            # Set flow control callbacks
            self.flow_controller.on_pause = self._handle_flow_pause
            self.flow_controller.on_resume = self._handle_flow_resume
        else:
            self.flow_controller = None
        
        # Subscription rate limiter (10 subscriptions per second max)
        self.subscription_limiter = TokenBucket(
            capacity=10,
            refill_rate=1.0,  # 1 subscription per second
            burst_size=10
        )
    
    async def connect(self) -> None:
        """Connect to Kalshi WebSocket with circuit breaker protection"""
        if self.enable_circuit_breaker:
            try:
                await self.circuit_breaker.call(self._connect_internal)
            except CircuitOpenException:
                logger.warning("Circuit breaker is open, connection attempt blocked")
                if self.auto_reconnect:
                    await self._schedule_reconnect()
                raise
        else:
            await self._connect_internal()
    
    async def _connect_internal(self) -> None:
        """Internal connection logic"""
        try:
            self.connection_attempts += 1
            start_time = time.time()
            
            # Get authentication headers
            headers = self.auth.get_websocket_headers()
            
            # Connect with authentication
            # Use additional_headers instead of extra_headers for compatibility
            try:
                self.ws = await websockets.connect(
                    self.config.ws_url,
                    additional_headers=headers
                )
            except TypeError:
                # Fallback if additional_headers is not supported
                self.ws = await websockets.connect(
                    self.config.ws_url
                )
            
            self.running = True
            self.last_connection_time = time.time()
            connection_time = self.last_connection_time - start_time
            
            logger.info(
                f"Connected to Kalshi WebSocket at {self.config.ws_url} "
                f"(attempt {self.connection_attempts}, {connection_time:.2f}s)"
            )
            
            # Reset connection attempts on success
            self.connection_attempts = 0
            
            # Re-subscribe to markets if we had subscriptions
            if self.subscribed_markets:
                markets = list(self.subscribed_markets)
                self.subscribed_markets.clear()
                await self.subscribe_markets(markets)
            
            # Start heartbeat and receive tasks
            self._heartbeat_task = asyncio.create_task(self._heartbeat())
            self._receive_task = asyncio.create_task(self._receive_messages())
            
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            
            # Schedule reconnect if enabled
            if self.auto_reconnect:
                await self._schedule_reconnect()
            
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from WebSocket"""
        self.running = False
        
        # Cancel tasks
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        if self._receive_task:
            self._receive_task.cancel()
        
        # Close WebSocket
        if self.ws:
            await self.ws.close()
            self.ws = None
        
        logger.info("Disconnected from Kalshi WebSocket")
    
    async def subscribe_markets(self, tickers: List[str]) -> None:
        """
        Subscribe to market tickers for real-time updates
        
        Args:
            tickers: List of market tickers to subscribe to
        """
        if not self.ws:
            raise RuntimeError("WebSocket not connected")
        
        # Apply subscription rate limiting
        for ticker in tickers:
            if not self.subscription_limiter.try_consume(1):
                # Wait for rate limit
                await self.subscription_limiter.consume(1, timeout=5.0)
        
        # Build subscription message
        message = {
            "id": 1,
            "cmd": "subscribe",
            "params": {
                "channels": ["ticker", "orderbook_delta", "trade"],
                "market_tickers": tickers
            }
        }
        
        # Send subscription
        await self.ws.send(json.dumps(message))
        
        # Track subscriptions
        self.subscribed_markets.update(tickers)
        
        logger.info(f"Subscribed to markets: {tickers}")
    
    async def unsubscribe_markets(self, tickers: List[str]) -> None:
        """
        Unsubscribe from market tickers
        
        Args:
            tickers: List of market tickers to unsubscribe from
        """
        if not self.ws:
            raise RuntimeError("WebSocket not connected")
        
        # Build unsubscribe message
        message = {
            "id": 2,
            "cmd": "unsubscribe",
            "params": {
                "channels": ["ticker", "orderbook_delta", "trade"],
                "market_tickers": tickers
            }
        }
        
        # Send unsubscribe
        await self.ws.send(json.dumps(message))
        
        # Update tracked subscriptions
        for ticker in tickers:
            self.subscribed_markets.discard(ticker)
        
        logger.info(f"Unsubscribed from markets: {tickers}")
    
    async def _heartbeat(self) -> None:
        """Send periodic heartbeat to keep connection alive"""
        while self.running:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)
                
                if self.ws:
                    # Send ping
                    pong_waiter = await self.ws.ping()
                    await asyncio.wait_for(pong_waiter, timeout=10)
                    logger.debug("Heartbeat sent and pong received")
                    
                    # Record successful heartbeat if circuit breaker enabled
                    if self.circuit_breaker:
                        self.circuit_breaker.metrics.successful_calls += 1
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Heartbeat failed: {e}")
                
                # Record failure if circuit breaker enabled
                if self.circuit_breaker:
                    await self.circuit_breaker._on_failure()
                
                # Connection might be lost, trigger reconnect
                if self.running:
                    if self.auto_reconnect:
                        await self._schedule_reconnect()
                    else:
                        await self._reconnect()
    
    async def _receive_messages(self) -> None:
        """Receive and process messages from WebSocket"""
        while self.running:
            try:
                if not self.ws:
                    await asyncio.sleep(1)
                    continue
                
                # Receive message
                message = await self.ws.recv()
                
                # Apply flow control if enabled
                if self.flow_controller:
                    # Check if we have credits to process
                    if not self.flow_controller.consume_credits(1):
                        logger.warning("Message dropped due to flow control")
                        continue
                    
                    # Track processing start time
                    start_time = time.time()
                
                # Parse JSON
                data = json.loads(message)
                
                # Process message through handler
                await self.message_handler.handle_message(data)
                
                # Update flow control metrics
                if self.flow_controller:
                    processing_time_ms = (time.time() - start_time) * 1000
                    self.flow_controller.release_credits(1)
                    self.flow_controller.update_metrics(processing_time_ms)
                
            except asyncio.CancelledError:
                break
            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket connection closed")
                if self.running:
                    await self._reconnect()
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse message: {e}")
            except Exception as e:
                logger.error(f"Error receiving message: {e}")
    
    async def _reconnect(self) -> None:
        """Reconnect to WebSocket with exponential backoff"""
        if not self.running:
            return
        
        for attempt in range(self.config.reconnect_attempts):
            try:
                logger.info(f"Reconnection attempt {attempt + 1}/{self.config.reconnect_attempts}")
                
                # Disconnect cleanly
                if self.ws:
                    await self.ws.close()
                    self.ws = None
                
                # Wait with exponential backoff
                delay = self.config.reconnect_delay * (2 ** attempt)
                await asyncio.sleep(delay)
                
                # Reconnect
                await self.connect()
                
                # Resubscribe to markets
                if self.subscribed_markets:
                    await self.subscribe_markets(list(self.subscribed_markets))
                
                logger.info("Reconnection successful")
                return
                
            except Exception as e:
                logger.error(f"Reconnection attempt {attempt + 1} failed: {e}")
        
        logger.error("All reconnection attempts failed")
        self.running = False
    
    async def _schedule_reconnect(self):
        """Schedule reconnection with exponential backoff"""
        if self.reconnect_task and not self.reconnect_task.done():
            return  # Already scheduled
        
        self.reconnect_task = asyncio.create_task(self._reconnect())
    
    async def run_forever(self) -> None:
        """Run the WebSocket client until stopped"""
        try:
            await self.connect()
            
            # Wait until stopped
            while self.running:
                await asyncio.sleep(1)
                
        finally:
            await self.disconnect()
    
    async def _handle_flow_pause(self):
        """Handle flow control pause event"""
        logger.warning("Flow control PAUSED - too many messages")
        # Could send a pause message to server if protocol supports it
    
    async def _handle_flow_resume(self):
        """Handle flow control resume event"""
        logger.info("Flow control RESUMED")
        # Could send a resume message to server if protocol supports it
    
    def get_flow_stats(self) -> Optional[Dict[str, Any]]:
        """Get flow control statistics"""
        if self.flow_controller:
            return self.flow_controller.get_stats()
        return None
    
    def get_rate_limit_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        return self.subscription_limiter.get_stats()
    
    def convert_price(self, centi_cents: int) -> float:
        """
        Convert price from centi-cents to dollars
        
        Args:
            centi_cents: Price in centi-cents (1/10000 of a dollar)
        
        Returns:
            Price in dollars
        """
        return round(centi_cents / 10000, self.config.price_precision)
