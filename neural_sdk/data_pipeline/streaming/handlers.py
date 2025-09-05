"""
Kalshi WebSocket Infrastructure - Message Handlers
Process different types of WebSocket messages
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable
from datetime import datetime

logger = logging.getLogger(__name__)


class MessageHandler(ABC):
    """Abstract base class for WebSocket message handlers"""
    
    @abstractmethod
    async def handle_message(self, message: Dict[str, Any]) -> None:
        """
        Handle a WebSocket message
        
        Args:
            message: Parsed JSON message from WebSocket
        """
        pass


class DefaultMessageHandler(MessageHandler):
    """Default message handler that logs and processes standard message types"""
    
    def __init__(self):
        """Initialize the default message handler"""
        self.ticker_callback: Optional[Callable] = None
        self.trade_callback: Optional[Callable] = None
        self.orderbook_callback: Optional[Callable] = None
        self.error_callback: Optional[Callable] = None
        
        # Statistics
        self.message_count = 0
        self.last_message_time = None
    
    async def handle_message(self, message: Dict[str, Any]) -> None:
        """
        Handle incoming WebSocket message
        
        Args:
            message: Parsed JSON message
        """
        self.message_count += 1
        self.last_message_time = datetime.now()
        
        # Get message type
        msg_type = message.get('type')
        
        if msg_type == 'ticker':
            await self._handle_ticker(message)
        elif msg_type == 'trade':
            await self._handle_trade(message)
        elif msg_type == 'orderbook_delta':
            await self._handle_orderbook(message)
        elif msg_type == 'subscribed':
            await self._handle_subscribed(message)
        elif msg_type == 'error':
            await self._handle_error(message)
        else:
            logger.debug(f"Received message type: {msg_type}")
    
    async def _handle_ticker(self, message: Dict[str, Any]) -> None:
        """Handle ticker update message"""
        data = message.get('msg', {})
        market_ticker = data.get('market_ticker')
        
        # Convert prices from centi-cents to dollars
        price_data = {
            'market_ticker': market_ticker,
            'yes_price': self._convert_price(data.get('yes_ask')),
            'no_price': self._convert_price(data.get('no_ask')),
            'yes_bid': self._convert_price(data.get('yes_bid')),
            'no_bid': self._convert_price(data.get('no_bid')),
            'volume': data.get('volume'),
            'open_interest': data.get('open_interest'),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Ticker update for {market_ticker}: Yes=${price_data['yes_price']:.4f}, No=${price_data['no_price']:.4f}")
        
        if self.ticker_callback:
            await self.ticker_callback(price_data)
    
    async def _handle_trade(self, message: Dict[str, Any]) -> None:
        """Handle trade execution message"""
        data = message.get('msg', {})
        market_ticker = data.get('market_ticker')
        
        trade_data = {
            'market_ticker': market_ticker,
            'trade_id': data.get('trade_id'),
            'price': self._convert_price(data.get('yes_price') or data.get('no_price')),
            'size': data.get('count'),
            'side': data.get('taker_side'),
            'timestamp': data.get('ts_millis')
        }
        
        logger.info(f"Trade on {market_ticker}: {trade_data['size']} @ ${trade_data['price']:.4f}")
        
        if self.trade_callback:
            await self.trade_callback(trade_data)
    
    async def _handle_orderbook(self, message: Dict[str, Any]) -> None:
        """Handle orderbook update message"""
        data = message.get('msg', {})
        market_ticker = data.get('market_ticker')
        
        orderbook_data = {
            'market_ticker': market_ticker,
            'yes_bids': self._convert_orderbook_levels(data.get('yes_bids', [])),
            'yes_asks': self._convert_orderbook_levels(data.get('yes_asks', [])),
            'no_bids': self._convert_orderbook_levels(data.get('no_bids', [])),
            'no_asks': self._convert_orderbook_levels(data.get('no_asks', [])),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.debug(f"Orderbook update for {market_ticker}")
        
        if self.orderbook_callback:
            await self.orderbook_callback(orderbook_data)
    
    async def _handle_subscribed(self, message: Dict[str, Any]) -> None:
        """Handle subscription confirmation"""
        data = message.get('msg', {})
        markets = data.get('markets', [])
        channels = data.get('channels', [])
        
        logger.info(f"Subscription confirmed for {len(markets)} markets on channels: {channels}")
    
    async def _handle_error(self, message: Dict[str, Any]) -> None:
        """Handle error message"""
        error = message.get('error', {})
        code = error.get('code')
        msg = error.get('message')
        
        logger.error(f"WebSocket error {code}: {msg}")
        
        if self.error_callback:
            await self.error_callback(error)
    
    def _convert_price(self, centi_cents: Optional[int]) -> Optional[float]:
        """Convert price from centi-cents to dollars"""
        if centi_cents is None:
            return None
        return round(centi_cents / 10000, 4)
    
    def _convert_orderbook_levels(self, levels: list) -> list:
        """Convert orderbook price levels from centi-cents"""
        converted = []
        for level in levels:
            converted.append([
                self._convert_price(level[0]),  # price
                level[1]  # quantity
            ])
        return converted
    
    def set_ticker_callback(self, callback: Callable) -> None:
        """Set callback for ticker updates"""
        self.ticker_callback = callback
    
    def set_trade_callback(self, callback: Callable) -> None:
        """Set callback for trade updates"""
        self.trade_callback = callback
    
    def set_orderbook_callback(self, callback: Callable) -> None:
        """Set callback for orderbook updates"""
        self.orderbook_callback = callback
    
    def set_error_callback(self, callback: Callable) -> None:
        """Set callback for errors"""
        self.error_callback = callback
    
    def get_stats(self) -> Dict[str, Any]:
        """Get handler statistics"""
        return {
            'message_count': self.message_count,
            'last_message_time': self.last_message_time.isoformat() if self.last_message_time else None
        }