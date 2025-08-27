"""
Custom MessageHandler for KalshiWebSocket
Handles incoming WebSocket messages and routes them appropriately
"""

import logging
import json
from typing import Dict, Any, Optional, Callable
from .streaming.handlers import MessageHandler

logger = logging.getLogger(__name__)


class StreamManagerMessageHandler(MessageHandler):
    """
    Custom message handler for StreamManager's KalshiWebSocket connection
    Routes messages to appropriate callback functions
    """
    
    def __init__(self, on_message: Optional[Callable] = None, on_error: Optional[Callable] = None):
        """
        Initialize the message handler with callbacks
        
        Args:
            on_message: Callback for handling messages
            on_error: Callback for handling errors
        """
        super().__init__()
        self.on_message_callback = on_message
        self.on_error_callback = on_error
        self.message_count = 0
        self.error_count = 0
    
    async def handle_message(self, message: Dict[str, Any]) -> None:
        """
        Handle incoming WebSocket message
        
        Args:
            message: The decoded message dictionary
        """
        try:
            self.message_count += 1
            
            # Log message type
            msg_type = message.get("type", "unknown")
            logger.debug(f"Received message type: {msg_type}, count: {self.message_count}")
            
            # Call the callback if provided
            if self.on_message_callback:
                await self.on_message_callback(message)
            else:
                # Default handling if no callback provided
                if msg_type == "ticker":
                    await self.handle_ticker(message)
                elif msg_type == "trade":
                    await self.handle_trade(message)
                elif msg_type == "orderbook_snapshot":
                    await self.handle_orderbook(message)
                elif msg_type == "error":
                    await self.handle_error(message)
                else:
                    logger.warning(f"Unhandled message type: {msg_type}")
                    
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            self.error_count += 1
            if self.on_error_callback:
                await self.on_error_callback(e)
    
    async def handle_ticker(self, message: Dict[str, Any]) -> None:
        """Handle ticker update messages"""
        ticker = message.get("ticker", {})
        market_ticker = ticker.get("market_ticker", "unknown")
        yes_price = ticker.get("yes_price")
        no_price = ticker.get("no_price")
        
        logger.info(f"Ticker update for {market_ticker}: YES={yes_price}, NO={no_price}")
    
    async def handle_trade(self, message: Dict[str, Any]) -> None:
        """Handle trade messages"""
        trade = message.get("trade", {})
        market_ticker = trade.get("market_ticker", "unknown")
        price = trade.get("price")
        count = trade.get("count")
        
        logger.info(f"Trade on {market_ticker}: {count} @ {price}")
    
    async def handle_orderbook(self, message: Dict[str, Any]) -> None:
        """Handle orderbook snapshot messages"""
        orderbook = message.get("orderbook", {})
        market_ticker = orderbook.get("market_ticker", "unknown")
        
        logger.debug(f"Orderbook snapshot for {market_ticker}")
    
    async def handle_error(self, message: Dict[str, Any]) -> None:
        """Handle error messages"""
        error = message.get("error", {})
        error_msg = error.get("message", "Unknown error")
        error_code = error.get("code", "UNKNOWN")
        
        logger.error(f"WebSocket error [{error_code}]: {error_msg}")
        self.error_count += 1
        
        if self.on_error_callback:
            await self.on_error_callback(Exception(f"[{error_code}] {error_msg}"))
    
    def get_stats(self) -> Dict[str, int]:
        """Get handler statistics"""
        return {
            "messages_received": self.message_count,
            "errors_encountered": self.error_count
        }