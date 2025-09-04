"""
Redis Publisher - Bridges WebSocket streams to Redis pub/sub
Simple, focused, no bloat - just publishes market data for agents
"""

import json
import logging
from typing import Dict, Any, Optional
import redis.asyncio as redis
from datetime import datetime

logger = logging.getLogger(__name__)


class RedisPublisher:
    """Publishes WebSocket events to Redis for agent consumption"""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        channel_prefix: str = "kalshi"
    ):
        """
        Initialize Redis publisher
        
        Args:
            redis_url: Redis connection URL
            channel_prefix: Prefix for channel names
        """
        self.redis_url = redis_url
        self.channel_prefix = channel_prefix
        self.redis_client: Optional[redis.Redis] = None
        self.is_connected = False
        
        # Track publishing stats
        self.messages_published = 0
        self.last_publish_time = None
        
    async def connect(self):
        """Connect to Redis"""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            self.is_connected = True
            logger.info(f"Connected to Redis at {self.redis_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from Redis"""
        if self.redis_client:
            await self.redis_client.close()
            self.is_connected = False
            logger.info("Disconnected from Redis")
    
    async def publish_market_update(self, market_data: Dict[str, Any]):
        """
        Publish market update to Redis
        
        Args:
            market_data: Market data from Kalshi WebSocket
        """
        if not self.is_connected:
            logger.warning("Not connected to Redis, skipping publish")
            return
        
        channel = f"{self.channel_prefix}:markets"
        
        # Add metadata
        message = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "market_update",
            "data": market_data
        }
        
        try:
            await self.redis_client.publish(
                channel,
                json.dumps(message)
            )
            self.messages_published += 1
            self.last_publish_time = datetime.utcnow()
            
            logger.debug(f"Published market update to {channel}")
            
        except Exception as e:
            logger.error(f"Failed to publish market update: {e}")
    
    async def publish_trade(self, trade_data: Dict[str, Any]):
        """
        Publish trade event to Redis
        
        Args:
            trade_data: Trade data from Kalshi
        """
        if not self.is_connected:
            return
        
        channel = f"{self.channel_prefix}:trades"
        
        message = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "trade",
            "data": trade_data
        }
        
        try:
            await self.redis_client.publish(
                channel,
                json.dumps(message)
            )
            self.messages_published += 1
            logger.debug(f"Published trade to {channel}")
            
        except Exception as e:
            logger.error(f"Failed to publish trade: {e}")
    
    async def publish_orderbook(self, orderbook_data: Dict[str, Any]):
        """
        Publish orderbook update to Redis
        
        Args:
            orderbook_data: Orderbook data from Kalshi
        """
        if not self.is_connected:
            return
        
        channel = f"{self.channel_prefix}:orderbook"
        
        message = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "orderbook",
            "data": orderbook_data
        }
        
        try:
            await self.redis_client.publish(
                channel,
                json.dumps(message)
            )
            self.messages_published += 1
            
        except Exception as e:
            logger.error(f"Failed to publish orderbook: {e}")
    
    async def publish_espn_update(self, game_data: Dict[str, Any]):
        """
        Publish ESPN game update to Redis
        
        Args:
            game_data: Game data from ESPN
        """
        if not self.is_connected:
            return
        
        channel = "espn:games"
        
        message = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "game_update",
            "data": game_data
        }
        
        try:
            await self.redis_client.publish(
                channel,
                json.dumps(message)
            )
            self.messages_published += 1
            logger.debug(f"Published ESPN update to {channel}")
            
        except Exception as e:
            logger.error(f"Failed to publish ESPN update: {e}")
    
    async def publish_signal(self, signal_data: Dict[str, Any]):
        """
        Publish trading signal to Redis
        
        Args:
            signal_data: Trading signal for agents
        """
        if not self.is_connected:
            return
        
        channel = f"{self.channel_prefix}:signals"
        
        message = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "signal",
            "data": signal_data
        }
        
        try:
            await self.redis_client.publish(
                channel,
                json.dumps(message)
            )
            logger.info(f"Published signal: {signal_data.get('action', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Failed to publish signal: {e}")

    async def publish_twitter_sentiment(self, sentiment_data: Dict[str, Any]):
        """
        Publish Twitter sentiment event to Redis

        Args:
            sentiment_data: Aggregated sentiment or shift event
        """
        if not self.is_connected:
            return

        channel = "twitter:sentiment"

        message = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "sentiment",
            "data": sentiment_data
        }

        try:
            await self.redis_client.publish(
                channel,
                json.dumps(message)
            )
            self.messages_published += 1
            logger.debug(f"Published twitter sentiment to {channel}")
        except Exception as e:
            logger.error(f"Failed to publish twitter sentiment: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get publisher statistics"""
        return {
            "connected": self.is_connected,
            "messages_published": self.messages_published,
            "last_publish": self.last_publish_time.isoformat() if self.last_publish_time else None
        }


class RedisStreamBridge:
    """
    Bridges WebSocket streams to Redis
    Integrates with existing StreamManager
    """
    
    def __init__(self, stream_manager, redis_publisher: RedisPublisher):
        """
        Initialize bridge
        
        Args:
            stream_manager: StreamManager instance
            redis_publisher: RedisPublisher instance
        """
        self.stream_manager = stream_manager
        self.publisher = redis_publisher
        self.is_running = False
        
    async def start(self):
        """Start bridging events to Redis"""
        await self.publisher.connect()
        self.is_running = True
        
        # Subscribe to stream manager events
        self.stream_manager.on_market_update = self.handle_market_update
        self.stream_manager.on_trade = self.handle_trade
        self.stream_manager.on_orderbook = self.handle_orderbook
        self.stream_manager.on_espn_update = self.handle_espn_update
        # Twitter events are bridged via StreamManager _emit_event
        
        logger.info("Redis bridge started")
    
    async def stop(self):
        """Stop bridging"""
        self.is_running = False
        await self.publisher.disconnect()
        logger.info("Redis bridge stopped")
    
    async def handle_market_update(self, data: Dict[str, Any]):
        """Handle market update from stream"""
        if self.is_running:
            await self.publisher.publish_market_update(data)
    
    async def handle_trade(self, data: Dict[str, Any]):
        """Handle trade event from stream"""
        if self.is_running:
            await self.publisher.publish_trade(data)
    
    async def handle_orderbook(self, data: Dict[str, Any]):
        """Handle orderbook update from stream"""
        if self.is_running:
            await self.publisher.publish_orderbook(data)
    
    async def handle_espn_update(self, data: Dict[str, Any]):
        """Handle ESPN update from stream"""
        if self.is_running:
            await self.publisher.publish_espn_update(data)

    async def publish_signal(self, signal: Dict[str, Any]):
        """
        Publish trading signal
        
        Args:
            signal: Trading signal with action, market, quantity, etc.
        """
        if self.is_running:
            await self.publisher.publish_signal(signal)

    async def handle_twitter_sentiment(self, data: Dict[str, Any]):
        """Handle Twitter sentiment from stream"""
        if self.is_running:
            await self.publisher.publish_twitter_sentiment(data)
