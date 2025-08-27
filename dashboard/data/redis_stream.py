"""Redis stream handler for real-time data updates."""

import os
import json
import redis
import asyncio
import logging
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
from decimal import Decimal
import streamlit as st

logger = logging.getLogger(__name__)


class RedisStreamHandler:
    """Handles real-time data streaming from Redis pub/sub channels."""
    
    def __init__(self, redis_url: Optional[str] = None):
        """Initialize Redis stream handler.
        
        Args:
            redis_url: Redis connection URL. If not provided, uses environment variable.
        """
        self.redis_url = redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379')
        self.redis_client = None
        self.pubsub = None
        self.subscriptions = {}
        self.callbacks = {}
        self._running = False
        self._connect()
    
    def _connect(self):
        """Establish Redis connection."""
        try:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            self.pubsub = self.redis_client.pubsub()
            self.redis_client.ping()
            logger.info("Connected to Redis successfully")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def subscribe(self, channel: str, callback: Optional[Callable] = None):
        """Subscribe to a Redis channel.
        
        Args:
            channel: Channel name to subscribe to
            callback: Optional callback function to handle messages
        """
        try:
            self.pubsub.subscribe(channel)
            if callback:
                self.callbacks[channel] = callback
            self.subscriptions[channel] = True
            logger.info(f"Subscribed to channel: {channel}")
        except Exception as e:
            logger.error(f"Failed to subscribe to {channel}: {e}")
            raise
    
    def unsubscribe(self, channel: str):
        """Unsubscribe from a Redis channel.
        
        Args:
            channel: Channel name to unsubscribe from
        """
        try:
            self.pubsub.unsubscribe(channel)
            self.subscriptions.pop(channel, None)
            self.callbacks.pop(channel, None)
            logger.info(f"Unsubscribed from channel: {channel}")
        except Exception as e:
            logger.error(f"Failed to unsubscribe from {channel}: {e}")
    
    def publish(self, channel: str, data: Dict[str, Any]):
        """Publish data to a Redis channel.
        
        Args:
            channel: Channel name to publish to
            data: Data dictionary to publish
        """
        try:
            message = json.dumps(data, default=str)
            self.redis_client.publish(channel, message)
            logger.debug(f"Published to {channel}: {data}")
        except Exception as e:
            logger.error(f"Failed to publish to {channel}: {e}")
            raise
    
    def send_emergency_stop(self):
        """Send emergency stop command to all agents."""
        stop_command = {
            'command': 'EMERGENCY_STOP',
            'timestamp': datetime.utcnow().isoformat(),
            'source': 'dashboard',
            'reason': 'User initiated emergency stop'
        }
        
        # Publish to control channel
        self.publish('agent:control', stop_command)
        
        # Also publish to individual agent channels
        agents = ['DataCoordinator', 'StrategyAnalyst', 'MarketEngineer', 'TradeExecutor', 'RiskManager']
        for agent in agents:
            self.publish(f'agent:{agent}:control', stop_command)
        
        logger.warning("Emergency stop command sent to all agents")
    
    def process_message(self, message: Dict[str, Any]):
        """Process incoming Redis message.
        
        Args:
            message: Redis message dictionary
        """
        if message['type'] not in ['message', 'pmessage']:
            return
        
        channel = message['channel']
        data = message['data']
        
        try:
            # Parse JSON data
            if isinstance(data, str):
                data = json.loads(data)
            
            # Call registered callback if exists
            if channel in self.callbacks:
                self.callbacks[channel](data)
            
            # Store in Streamlit session state for UI updates
            if 'redis_data' not in st.session_state:
                st.session_state.redis_data = {}
            
            st.session_state.redis_data[channel] = {
                'data': data,
                'timestamp': datetime.utcnow()
            }
            
            logger.debug(f"Processed message from {channel}")
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message from {channel}: {e}")
        except Exception as e:
            logger.error(f"Error processing message from {channel}: {e}")
    
    def start_listening(self):
        """Start listening to subscribed channels."""
        self._running = True
        logger.info("Started Redis stream listener")
        
        try:
            while self._running:
                message = self.pubsub.get_message(timeout=1.0)
                if message:
                    self.process_message(message)
        except KeyboardInterrupt:
            logger.info("Redis listener interrupted")
        except Exception as e:
            logger.error(f"Redis listener error: {e}")
        finally:
            self.stop_listening()
    
    def stop_listening(self):
        """Stop listening to channels."""
        self._running = False
        if self.pubsub:
            self.pubsub.close()
        logger.info("Stopped Redis stream listener")
    
    def get_latest_data(self, channel: str) -> Optional[Dict[str, Any]]:
        """Get latest data from a channel stored in session state.
        
        Args:
            channel: Channel name
            
        Returns:
            Latest data from the channel or None
        """
        if 'redis_data' in st.session_state and channel in st.session_state.redis_data:
            return st.session_state.redis_data[channel]['data']
        return None
    
    def subscribe_to_trading_channels(self):
        """Subscribe to all trading-related channels."""
        channels = [
            'kalshi:trades',      # Trade execution events
            'kalshi:positions',   # Position updates
            'kalshi:markets',     # Market price updates
            'kalshi:signals',     # Trading signals
            'agent:status',       # Agent status updates
            'agent:control',      # Control commands
            'pnl:updates',        # P&L updates
            'risk:alerts'         # Risk management alerts
        ]
        
        for channel in channels:
            self.subscribe(channel)
        
        logger.info(f"Subscribed to {len(channels)} trading channels")


class RedisDataProcessor:
    """Processes Redis data for dashboard display."""
    
    @staticmethod
    def process_trade_update(data: Dict[str, Any]) -> Dict[str, Any]:
        """Process trade update from Redis.
        
        Args:
            data: Trade data from Redis
            
        Returns:
            Processed trade data for dashboard
        """
        return {
            'trade_id': data.get('trade_id'),
            'market_ticker': data.get('market_ticker'),
            'side': data.get('side'),
            'quantity': data.get('quantity'),
            'price': float(data.get('price', 0)),
            'total_cost': float(data.get('total_cost', 0)),
            'realized_pnl': float(data.get('realized_pnl', 0)),
            'status': data.get('status', 'pending'),
            'timestamp': data.get('timestamp', datetime.utcnow().isoformat()),
            'strategy': data.get('strategy')
        }
    
    @staticmethod
    def process_position_update(data: Dict[str, Any]) -> Dict[str, Any]:
        """Process position update from Redis.
        
        Args:
            data: Position data from Redis
            
        Returns:
            Processed position data for dashboard
        """
        entry_price = float(data.get('entry_price', 0))
        current_price = float(data.get('current_price', entry_price))
        quantity = int(data.get('quantity', 0))
        
        # Calculate unrealized P&L
        unrealized_pnl = (current_price - entry_price) * quantity
        unrealized_pnl_pct = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
        
        return {
            'position_id': data.get('position_id'),
            'market_ticker': data.get('market_ticker'),
            'side': data.get('side'),
            'quantity': quantity,
            'entry_price': entry_price,
            'current_price': current_price,
            'market_value': current_price * quantity,
            'unrealized_pnl': unrealized_pnl,
            'unrealized_pnl_pct': unrealized_pnl_pct,
            'kelly_fraction': float(data.get('kelly_fraction', 0))
        }
    
    @staticmethod
    def process_market_update(data: Dict[str, Any]) -> Dict[str, Any]:
        """Process market update from Redis.
        
        Args:
            data: Market data from Redis
            
        Returns:
            Processed market data for dashboard
        """
        return {
            'market_ticker': data.get('ticker'),
            'yes_price': float(data.get('yes_price', 0)),
            'no_price': float(data.get('no_price', 0)),
            'volume': int(data.get('volume', 0)),
            'open_interest': int(data.get('open_interest', 0)),
            'last_trade_price': float(data.get('last_trade_price', 0)),
            'timestamp': data.get('timestamp', datetime.utcnow().isoformat())
        }
    
    @staticmethod
    def process_agent_status(data: Dict[str, Any]) -> Dict[str, Any]:
        """Process agent status update from Redis.
        
        Args:
            data: Agent status data from Redis
            
        Returns:
            Processed agent status for dashboard
        """
        return {
            'agent_name': data.get('agent_name'),
            'status': data.get('status', 'unknown'),
            'messages_processed': int(data.get('messages_processed', 0)),
            'errors_count': int(data.get('errors_count', 0)),
            'last_heartbeat': data.get('last_heartbeat', datetime.utcnow().isoformat()),
            'uptime_seconds': int(data.get('uptime_seconds', 0))
        }