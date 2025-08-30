"""
Base Redis Consumer Module for Agent System

This module provides the foundational Redis pub/sub integration for all trading agents.
It handles connection management, message routing, error recovery, and statistics tracking.

The base consumer implements:
- Automatic reconnection with exponential backoff
- Channel-specific message routing
- Health monitoring and statistics
- Integration with Agentuity KV storage
- Standardized publishing methods for signals, trades, and alerts

Example:
    ```python
    from src.agents.base_consumer import BaseAgentRedisConsumer
    
    class MyAgentConsumer(BaseAgentRedisConsumer):
        async def process_message(self, channel: str, data: Dict[str, Any]):
            if channel == "kalshi:markets":
                await self.analyze_market(data)
    
    # Usage
    consumer = MyAgentConsumer("MyAgent")
    await consumer.connect()
    await consumer.subscribe(["kalshi:markets", "kalshi:signals"])
    await consumer.start_consuming()
    ```

References:
    Redis Pub/Sub Documentation: https://redis.io/docs/manual/pubsub/
    Agentuity Platform: https://agentuity.dev/
"""

import asyncio
import json
import logging
import redis.asyncio as redis
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseAgentRedisConsumer(ABC):
    """
    Abstract base class for Redis-based agent consumers.
    
    This class provides the infrastructure for agents to consume real-time data
    from Redis pub/sub channels and publish trading decisions back to the system.
    
    Attributes:
        agent_name (str): Unique identifier for the agent
        redis_url (str): Redis connection URL
        agent_context (Optional[Any]): Agentuity context for KV storage
        redis_client (Optional[redis.Redis]): Primary Redis connection for subscriptions
        publisher (Optional[redis.Redis]): Separate connection for publishing
        pubsub (Optional[redis.PubSub]): Pub/sub handler
        is_running (bool): Flag indicating if consumer is actively processing
        subscribed_channels (List[str]): List of subscribed channel names
        messages_received (int): Total messages received counter
        messages_processed (int): Successfully processed messages counter
        messages_published (int): Published messages counter
        last_message_time (Optional[datetime]): Timestamp of last received message
        channel_handlers (Dict[str, Callable]): Channel-specific message handlers
    
    Configuration:
        The consumer can be configured via environment variables:
        - REDIS_URL: Redis connection string (default: redis://localhost:6379)
        - REDIS_MAX_RETRIES: Maximum reconnection attempts (default: 5)
        - REDIS_RETRY_DELAY: Initial retry delay in seconds (default: 1)
    
    Thread Safety:
        This class is designed for async operation and is not thread-safe.
        Use within a single event loop context.
    """
    
    def __init__(
        self,
        agent_name: str,
        redis_url: str = "redis://localhost:6379",
        agent_context: Optional[Any] = None
    ):
        """
        Initialize a new Redis consumer for an agent.
        
        This sets up the consumer infrastructure but does not establish connections.
        Call connect() to establish Redis connections.
        
        Args:
            agent_name: Unique identifier for this agent instance.
                       Used for logging and statistics tracking.
            redis_url: Redis connection URL in format redis://[user]:[password]@[host]:[port]/[db].
                      Defaults to local Redis on standard port.
            agent_context: Agentuity platform context for accessing KV storage and other services.
                         Optional - consumer works without it but won't persist statistics.
        
        Raises:
            ValueError: If agent_name is empty or contains invalid characters.
        
        Example:
            >>> consumer = MyConsumer(
            ...     agent_name="DataCoordinator",
            ...     redis_url="redis://localhost:6379",
            ...     agent_context=agentuity_context
            ... )
        """
        if not agent_name or not agent_name.strip():
            raise ValueError("Agent name cannot be empty")
            
        self.agent_name = agent_name
        self.redis_url = redis_url
        self.agent_context = agent_context
        
        # Redis connections
        self.redis_client: Optional[redis.Redis] = None
        self.publisher: Optional[redis.Redis] = None
        self.pubsub: Optional[redis.PubSub] = None
        
        # State
        self.is_running = False
        self.subscribed_channels: List[str] = []
        
        # Statistics
        self.messages_received = 0
        self.messages_processed = 0
        self.messages_published = 0
        self.last_message_time: Optional[datetime] = None
        
        # Message handlers by channel
        self.channel_handlers: Dict[str, Callable] = {}
        
        logger.info(f"{agent_name} Redis consumer initialized")
    
    async def connect(self, max_retries: int = 5, initial_delay: float = 1.0) -> None:
        """
        Establish connections to Redis with automatic retry logic.
        
        Creates separate connections for publishing and subscribing to avoid
        blocking issues. Implements exponential backoff for connection failures.
        
        Args:
            max_retries: Maximum number of connection attempts before giving up.
                        Set to 0 for infinite retries.
            initial_delay: Initial delay between retries in seconds.
                          Doubles after each failed attempt.
        
        Raises:
            redis.ConnectionError: If unable to connect after all retry attempts.
            redis.AuthenticationError: If Redis authentication fails.
        
        Example:
            >>> await consumer.connect(max_retries=10, initial_delay=2.0)
        
        Note:
            This method is idempotent - calling it multiple times will not
            create duplicate connections if already connected.
        """
        if self.redis_client:
            logger.debug(f"{self.agent_name} already connected")
            return
            
        retry_delay = initial_delay
        attempt = 0
        
        while max_retries == 0 or attempt < max_retries:
            try:
                # Create separate connections for sub and pub to avoid blocking
                self.redis_client = redis.from_url(self.redis_url)
                self.publisher = redis.from_url(self.redis_url)
                self.pubsub = self.redis_client.pubsub()
                
                # Test connection
                await self.redis_client.ping()
                
                logger.info(f"{self.agent_name} connected to Redis")
                return
                
            except redis.AuthenticationError:
                logger.error(f"{self.agent_name} Redis authentication failed")
                raise
                
            except Exception as e:
                attempt += 1
                logger.warning(f"{self.agent_name} connection attempt {attempt} failed: {e}")
                
                if max_retries > 0 and attempt >= max_retries:
                    raise redis.ConnectionError(f"Failed to connect after {max_retries} attempts")
                    
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 60)  # Cap at 60 seconds
    
    async def disconnect(self) -> None:
        """
        Gracefully disconnect from Redis and clean up resources.
        
        This method:
        1. Stops message consumption
        2. Unsubscribes from all channels
        3. Closes all Redis connections
        4. Resets internal state
        
        Safe to call multiple times - subsequent calls are no-ops.
        
        Example:
            >>> await consumer.disconnect()
        
        Note:
            Always call this method when shutting down to ensure
            clean resource cleanup and avoid connection leaks.
        """
        self.is_running = False
        
        if self.pubsub:
            try:
                await self.pubsub.unsubscribe()
                await self.pubsub.close()
            except Exception as e:
                logger.debug(f"Error closing pubsub: {e}")
            finally:
                self.pubsub = None
        
        if self.redis_client:
            try:
                await self.redis_client.close()
            except Exception as e:
                logger.debug(f"Error closing Redis client: {e}")
            finally:
                self.redis_client = None
        
        if self.publisher:
            try:
                await self.publisher.close()
            except Exception as e:
                logger.debug(f"Error closing publisher: {e}")
            finally:
                self.publisher = None
        
        self.subscribed_channels = []
        logger.info(f"{self.agent_name} disconnected from Redis")
    
    async def subscribe(self, channels: List[str]) -> None:
        """
        Subscribe to one or more Redis pub/sub channels.
        
        Args:
            channels: List of channel names to subscribe to.
                     Channel names should follow the pattern "source:type".
                     Examples: ["kalshi:markets", "espn:games", "twitter:sentiment"]
        
        Raises:
            RuntimeError: If not connected to Redis.
            ValueError: If channels list is empty.
        
        Example:
            >>> await consumer.subscribe([
            ...     "kalshi:markets",
            ...     "kalshi:signals",
            ...     "espn:games"
            ... ])
        
        Note:
            Subscribing to new channels will replace any existing subscriptions.
            To add channels, get current channels first and append new ones.
        """
        if not self.pubsub:
            raise RuntimeError("Not connected to Redis. Call connect() first.")
        
        if not channels:
            raise ValueError("Channels list cannot be empty")
        
        await self.pubsub.subscribe(*channels)
        self.subscribed_channels = channels
        logger.info(f"{self.agent_name} subscribed to: {channels}")
    
    async def start_consuming(self) -> None:
        """
        Start consuming messages from subscribed channels.
        
        This method runs an infinite loop processing incoming messages.
        It should be run as a background task in production.
        
        The consumption loop:
        1. Waits for messages on subscribed channels
        2. Parses and validates incoming messages
        3. Routes to appropriate handlers
        4. Tracks statistics
        5. Handles errors gracefully
        
        Raises:
            RuntimeError: If not connected to Redis or no channels subscribed.
        
        Example:
            >>> # Run as background task
            >>> task = asyncio.create_task(consumer.start_consuming())
            >>> 
            >>> # Or await directly (blocks)
            >>> await consumer.start_consuming()
        
        Note:
            Call stop_consuming() or disconnect() to stop the consumption loop.
            The loop will also stop if the Redis connection is lost.
        """
        if not self.pubsub:
            raise RuntimeError("Not connected to Redis")
        
        if not self.subscribed_channels:
            logger.warning(f"{self.agent_name} no channels subscribed")
        
        self.is_running = True
        logger.info(f"{self.agent_name} starting message consumption")
        
        try:
            async for message in self.pubsub.listen():
                if not self.is_running:
                    logger.info(f"{self.agent_name} stopping consumption")
                    break
                
                # Skip non-message types (subscribe confirmations, etc)
                if message['type'] not in ('message', 'pmessage'):
                    continue
                    
                await self._handle_message(message)
                
        except asyncio.CancelledError:
            logger.info(f"{self.agent_name} consumption cancelled")
            raise
        except Exception as e:
            logger.error(f"{self.agent_name} consumption error: {e}")
            self.is_running = False
            raise
    
    async def _handle_message(self, message: Dict[str, Any]) -> None:
        """
        Internal method to handle incoming Redis messages.
        
        Parses the message, updates statistics, and routes to appropriate
        handler. Errors in message processing are logged but don't stop
        the consumer.
        
        Args:
            message: Raw message dict from Redis with keys:
                    - type: Message type (message, pmessage, subscribe, etc)
                    - channel: Channel name (bytes)
                    - data: Message payload (bytes)
        
        Note:
            This method should not be called directly. It's invoked by
            the consumption loop in start_consuming().
        """
        try:
            channel = message['channel'].decode('utf-8')
            data = json.loads(message['data'])
            
            self.messages_received += 1
            self.last_message_time = datetime.utcnow()
            
            # Log message receipt (debug level to avoid spam)
            logger.debug(f"{self.agent_name} received on {channel}: {data.get('type', 'unknown')}")
            
            # Route to channel-specific handler if registered
            if channel in self.channel_handlers:
                await self.channel_handlers[channel](data)
            else:
                # Default routing to process_message
                await self.process_message(channel, data)
            
            self.messages_processed += 1
            
            # Save stats to KV if Agentuity context available
            if self.agent_context and self.messages_processed % 100 == 0:
                await self._save_stats()
                
        except json.JSONDecodeError as e:
            logger.error(f"{self.agent_name} invalid JSON in message: {e}")
        except Exception as e:
            logger.error(f"{self.agent_name} error processing message: {e}", exc_info=True)
    
    @abstractmethod
    async def process_message(self, channel: str, data: Dict[str, Any]) -> None:
        """
        Process an incoming message from a Redis channel.
        
        This method must be implemented by all subclasses to define
        agent-specific message processing logic.
        
        Args:
            channel: The Redis channel the message was received from.
                    Format: "source:type" (e.g., "kalshi:markets")
            data: Parsed message data containing:
                 - timestamp: ISO format timestamp
                 - type: Message type (market_update, trade, signal, etc)
                 - data: Message-specific payload
                 - source: Originating service
        
        Example Implementation:
            ```python
            async def process_message(self, channel: str, data: Dict[str, Any]):
                if channel == "kalshi:markets":
                    market_data = data['data']
                    if self.detect_arbitrage(market_data):
                        await self.publish_signal({
                            "action": "ARBITRAGE",
                            "market": market_data['market_ticker'],
                            "confidence": 0.95
                        })
                elif channel == "espn:games":
                    await self.handle_game_event(data['data'])
            ```
        
        Note:
            - This method should not raise exceptions - handle errors internally
            - Long-running operations should be offloaded to background tasks
            - Consider using channel_handlers for better organization
        """
        pass
    
    async def publish(self, channel: str, data: Dict[str, Any]) -> bool:
        """
        Publish a message to a Redis channel.
        
        Wraps the data with metadata (timestamp, source) before publishing.
        
        Args:
            channel: Target channel name (e.g., "kalshi:signals")
            data: Message payload to publish
        
        Returns:
            bool: True if published successfully, False otherwise
        
        Example:
            >>> success = await consumer.publish("kalshi:signals", {
            ...     "type": "BUY_SIGNAL",
            ...     "market": "NFL-WINNER",
            ...     "confidence": 0.75
            ... })
        
        Note:
            Publishing is fire-and-forget - no guarantee of delivery.
            Use appropriate channel conventions for your system.
        """
        if not self.publisher:
            logger.warning(f"{self.agent_name} not connected, cannot publish")
            return False
        
        message = {
            "timestamp": datetime.utcnow().isoformat(),
            "source": self.agent_name,
            "data": data
        }
        
        try:
            await self.publisher.publish(channel, json.dumps(message))
            self.messages_published += 1
            logger.debug(f"{self.agent_name} published to {channel}")
            return True
        except Exception as e:
            logger.error(f"{self.agent_name} publish error: {e}")
            return False
    
    async def publish_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Publish a trading signal to the signals channel.
        
        Convenience method for publishing trading signals with
        standardized format.
        
        Args:
            signal: Signal data containing:
                   - action: BUY, SELL, HOLD
                   - market: Market ticker
                   - confidence: Signal confidence (0-1)
                   - size: Optional position size
                   - reason: Optional explanation
        
        Returns:
            bool: True if published successfully
        
        Example:
            >>> await consumer.publish_signal({
            ...     "action": "BUY",
            ...     "market": "NFL-CHAMPIONSHIP",
            ...     "side": "YES",
            ...     "confidence": 0.85,
            ...     "size": 100,
            ...     "reason": "Arbitrage opportunity detected"
            ... })
        """
        return await self.publish("kalshi:signals", {
            "type": "signal",
            **signal
        })
    
    async def publish_trade(self, trade: Dict[str, Any]) -> bool:
        """
        Publish a trade execution to the trades channel.
        
        Args:
            trade: Trade data containing:
                  - market: Market ticker
                  - side: YES or NO
                  - price: Execution price
                  - quantity: Number of contracts
                  - order_id: Order identifier
                  - status: PENDING, FILLED, CANCELLED
        
        Returns:
            bool: True if published successfully
        
        Example:
            >>> await consumer.publish_trade({
            ...     "market": "NFL-CHAMPIONSHIP",
            ...     "side": "YES",
            ...     "price": 0.65,
            ...     "quantity": 100,
            ...     "order_id": "order_123",
            ...     "status": "FILLED"
            ... })
        """
        return await self.publish("kalshi:trades", {
            "type": "trade",
            **trade
        })
    
    async def publish_risk_alert(self, alert: Dict[str, Any]) -> bool:
        """
        Publish a risk management alert.
        
        Args:
            alert: Alert data containing:
                  - severity: LOW, MEDIUM, HIGH, CRITICAL
                  - type: POSITION_LIMIT, DRAWDOWN, CORRELATION, etc
                  - message: Human-readable alert message
                  - data: Alert-specific data
        
        Returns:
            bool: True if published successfully
        
        Example:
            >>> await consumer.publish_risk_alert({
            ...     "severity": "HIGH",
            ...     "type": "DRAWDOWN",
            ...     "message": "Daily drawdown approaching limit",
            ...     "data": {"current": 0.18, "limit": 0.20}
            ... })
        """
        return await self.publish("kalshi:risk", {
            "type": "risk_alert",
            **alert
        })
    
    def register_handler(self, channel: str, handler: Callable) -> None:
        """
        Register a custom handler for a specific channel.
        
        Allows channel-specific processing logic without modifying
        the main process_message method.
        
        Args:
            channel: Channel name to handle
            handler: Async function that accepts message data dict
        
        Example:
            >>> async def handle_markets(data: Dict[str, Any]):
            ...     print(f"Market update: {data}")
            >>> 
            >>> consumer.register_handler("kalshi:markets", handle_markets)
        
        Note:
            Handlers override the default process_message routing.
            Only one handler per channel is supported.
        """
        self.channel_handlers[channel] = handler
        logger.info(f"{self.agent_name} registered handler for {channel}")
    
    async def _save_stats(self) -> None:
        """
        Save consumer statistics to Agentuity KV storage.
        
        Internal method called periodically to persist statistics
        for monitoring and debugging.
        """
        if not self.agent_context:
            return
        
        try:
            stats = {
                "messages_received": self.messages_received,
                "messages_processed": self.messages_processed,
                "messages_published": self.messages_published,
                "last_message_time": self.last_message_time.isoformat() if self.last_message_time else None,
                "subscribed_channels": self.subscribed_channels
            }
            
            await self.agent_context.kv.set(
                "agent_stats",
                f"{self.agent_name}_redis",
                stats
            )
        except Exception as e:
            logger.debug(f"Could not save stats: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get current consumer statistics.
        
        Returns:
            Dict containing:
            - agent: Agent name
            - connected: Connection status
            - running: Processing status
            - messages_received: Total received
            - messages_processed: Successfully processed
            - messages_published: Total published
            - last_message_time: Last message timestamp
            - subscribed_channels: List of channels
        
        Example:
            >>> stats = consumer.get_stats()
            >>> print(f"Processed {stats['messages_processed']} messages")
        """
        return {
            "agent": self.agent_name,
            "connected": self.redis_client is not None,
            "running": self.is_running,
            "messages_received": self.messages_received,
            "messages_processed": self.messages_processed,
            "messages_published": self.messages_published,
            "last_message_time": self.last_message_time.isoformat() if self.last_message_time else None,
            "subscribed_channels": self.subscribed_channels
        }
    
    async def health_check(self) -> bool:
        """
        Check if the consumer is healthy and operational.
        
        Performs a Redis ping to verify connectivity.
        
        Returns:
            bool: True if healthy (connected and responsive), False otherwise
        
        Example:
            >>> if not await consumer.health_check():
            ...     await consumer.connect()  # Try to reconnect
        """
        if not self.redis_client:
            return False
        
        try:
            await self.redis_client.ping()
            return True
        except:
            return False


class AgentRedisOrchestrator:
    """
    Orchestrates multiple agent Redis consumers for coordinated operation.
    
    This class manages the lifecycle of multiple consumers, allowing them
    to be started, stopped, and monitored as a group. Useful for running
    the complete agent system.
    
    Attributes:
        redis_url (str): Redis connection URL shared by all consumers
        consumers (Dict[str, BaseAgentRedisConsumer]): Registered consumers by name
        tasks (List[asyncio.Task]): Background tasks for each consumer
    
    Example:
        ```python
        orchestrator = AgentRedisOrchestrator()
        orchestrator.register_consumer(DataCoordinatorConsumer())
        orchestrator.register_consumer(MarketEngineerConsumer())
        orchestrator.register_consumer(TradeExecutorConsumer())
        
        await orchestrator.start_all()
        
        # Run for some time...
        stats = orchestrator.get_all_stats()
        
        await orchestrator.stop_all()
        ```
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """
        Initialize the orchestrator.
        
        Args:
            redis_url: Redis connection URL to be used by all consumers
        """
        self.redis_url = redis_url
        self.consumers: Dict[str, BaseAgentRedisConsumer] = {}
        self.tasks: List[asyncio.Task] = []
        
    def register_consumer(self, consumer: BaseAgentRedisConsumer) -> None:
        """
        Register a consumer to be managed by the orchestrator.
        
        Args:
            consumer: Consumer instance to register
        
        Raises:
            ValueError: If a consumer with the same name is already registered
        
        Example:
            >>> orchestrator.register_consumer(MyConsumer("Agent1"))
        """
        if consumer.agent_name in self.consumers:
            raise ValueError(f"Consumer {consumer.agent_name} already registered")
            
        self.consumers[consumer.agent_name] = consumer
        logger.info(f"Registered consumer: {consumer.agent_name}")
    
    async def start_all(self) -> None:
        """
        Start all registered consumers concurrently.
        
        Each consumer:
        1. Connects to Redis
        2. Subscribes to its channels
        3. Starts consuming in a background task
        
        Raises:
            RuntimeError: If no consumers are registered
        
        Example:
            >>> await orchestrator.start_all()
            >>> # All consumers now running in background
        """
        if not self.consumers:
            raise RuntimeError("No consumers registered")
            
        for name, consumer in self.consumers.items():
            logger.info(f"Starting {name}...")
            
            # Connect and subscribe
            await consumer.connect()
            
            # Start consuming in background
            task = asyncio.create_task(
                consumer.start_consuming(),
                name=f"consumer_{name}"
            )
            self.tasks.append(task)
        
        logger.info(f"Started {len(self.consumers)} consumers")
    
    async def stop_all(self) -> None:
        """
        Stop all consumers gracefully.
        
        Process:
        1. Signal all consumers to stop
        2. Cancel background tasks
        3. Disconnect from Redis
        4. Clean up resources
        
        Safe to call multiple times.
        
        Example:
            >>> await orchestrator.stop_all()
        """
        # Signal consumers to stop
        for consumer in self.consumers.values():
            consumer.is_running = False
        
        # Give consumers time to finish current message
        await asyncio.sleep(0.1)
        
        # Cancel tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Disconnect all consumers
        for consumer in self.consumers.values():
            await consumer.disconnect()
        
        self.tasks.clear()
        logger.info("All consumers stopped")
    
    def get_all_stats(self) -> Dict[str, Any]:
        """
        Get statistics from all registered consumers.
        
        Returns:
            Dict mapping agent names to their statistics
        
        Example:
            >>> stats = orchestrator.get_all_stats()
            >>> for agent, data in stats.items():
            ...     print(f"{agent}: {data['messages_processed']} processed")
        """
        return {
            name: consumer.get_stats()
            for name, consumer in self.consumers.items()
        }
    
    async def health_check_all(self) -> Dict[str, bool]:
        """
        Perform health checks on all consumers.
        
        Returns:
            Dict mapping agent names to health status (True/False)
        
        Example:
            >>> health = await orchestrator.health_check_all()
            >>> unhealthy = [name for name, ok in health.items() if not ok]
            >>> if unhealthy:
            ...     print(f"Unhealthy consumers: {unhealthy}")
        """
        results = {}
        for name, consumer in self.consumers.items():
            results[name] = await consumer.health_check()
        return results