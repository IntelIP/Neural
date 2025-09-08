"""
Market Stream Module

Provides market-specific streaming capabilities.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MarketUpdate:
    """Represents a market data update."""
    ticker: str
    timestamp: datetime
    update_type: str
    data: Dict[str, Any]
    

class MarketStream:
    """
    Base class for market-specific data streams.
    """
    
    def __init__(self, market_ticker: str, websocket_client=None):
        """
        Initialize market stream.
        
        Args:
            market_ticker: Market ticker to stream
            websocket_client: WebSocket client instance
        """
        self.market_ticker = market_ticker
        self.websocket = websocket_client
        self.handlers = {
            'ticker': [],
            'orderbook': [],
            'trade': [],
            'lifecycle': []
        }
        self._streaming = False
        
    async def start(self):
        """Start streaming market data."""
        if not self.websocket:
            raise ValueError("WebSocket client not initialized")
            
        await self.websocket.subscribe(self.market_ticker)
        self._streaming = True
        
        # Start processing messages
        asyncio.create_task(self._process_messages())
        
    async def stop(self):
        """Stop streaming market data."""
        self._streaming = False
        if self.websocket:
            await self.websocket.unsubscribe(self.market_ticker)
            
    async def _process_messages(self):
        """Process incoming messages."""
        try:
            async for message in self.websocket.stream():
                if not self._streaming:
                    break
                    
                # Process message based on type
                if message.get('ticker') == self.market_ticker:
                    await self._handle_message(message)
        except Exception as e:
            logger.error(f"Error processing messages: {e}")
            
    async def _handle_message(self, message: Dict[str, Any]):
        """Handle incoming message."""
        msg_type = message.get('type', 'unknown')
        
        # Create market update
        update = MarketUpdate(
            ticker=self.market_ticker,
            timestamp=datetime.now(),
            update_type=msg_type,
            data=message
        )
        
        # Call appropriate handlers
        if msg_type in self.handlers:
            for handler in self.handlers[msg_type]:
                try:
                    await handler(update)
                except Exception as e:
                    logger.error(f"Handler error: {e}")
                    
    def on_ticker(self, handler: Callable):
        """Register ticker update handler."""
        self.handlers['ticker'].append(handler)
        
    def on_orderbook(self, handler: Callable):
        """Register orderbook update handler."""
        self.handlers['orderbook'].append(handler)
        
    def on_trade(self, handler: Callable):
        """Register trade update handler."""
        self.handlers['trade'].append(handler)
        
    def on_lifecycle(self, handler: Callable):
        """Register lifecycle update handler."""
        self.handlers['lifecycle'].append(handler)


class NFLMarketStream(MarketStream):
    """
    NFL-specific market stream with enhanced features.
    """
    
    def __init__(self, game_id: str, websocket_client=None):
        """
        Initialize NFL market stream.
        
        Args:
            game_id: NFL game identifier
            websocket_client: WebSocket client instance
        """
        # Convert game ID to market ticker format
        market_ticker = self._convert_game_id(game_id)
        super().__init__(market_ticker, websocket_client)
        self.game_id = game_id
        self.teams = self._parse_teams(game_id)
        
    def _convert_game_id(self, game_id: str) -> str:
        """Convert NFL game ID to market ticker."""
        # Example conversion logic
        # This would be customized based on actual ticker format
        return f"KXNFLGAME-{game_id}"
        
    def _parse_teams(self, game_id: str) -> Dict[str, str]:
        """Parse team information from game ID."""
        # Example parsing logic
        parts = game_id.split('-') if '-' in game_id else [game_id]
        if len(parts) >= 2:
            return {
                'home': parts[-1],
                'away': parts[-2] if len(parts) > 1 else 'UNKNOWN'
            }
        return {'home': 'UNKNOWN', 'away': 'UNKNOWN'}
        
    async def get_game_stats(self) -> Dict[str, Any]:
        """Get current game statistics."""
        return {
            'game_id': self.game_id,
            'ticker': self.market_ticker,
            'teams': self.teams,
            'streaming': self._streaming
        }
        
    async def track_momentum(self, window_size: int = 10):
        """
        Track momentum indicators for the game.
        
        Args:
            window_size: Number of updates to consider for momentum
        """
        momentum_data = []
        
        async def momentum_handler(update: MarketUpdate):
            momentum_data.append(update)
            if len(momentum_data) > window_size:
                momentum_data.pop(0)
                
            # Calculate momentum metrics
            if len(momentum_data) >= 2:
                # Example momentum calculation
                latest = momentum_data[-1].data.get('price', 0)
                previous = momentum_data[-2].data.get('price', 0)
                momentum = latest - previous
                
                logger.info(f"Momentum for {self.game_id}: {momentum:.4f}")
                
        self.on_ticker(momentum_handler)