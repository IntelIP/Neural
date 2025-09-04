"""
Neural SDK WebSocket Client

User-friendly wrapper around the data pipeline WebSocket infrastructure.
Provides a clean API for real-time market data streaming.
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional

from ..core.config import SDKConfig
from ..core.exceptions import ConnectionError, SDKError

# Import from data pipeline
try:
    from neural_sdk.data_pipeline.streaming.websocket import KalshiWebSocket
    from neural_sdk.data_pipeline.streaming.handlers import DefaultMessageHandler
    from neural_sdk.data_pipeline.data_sources.kalshi.market_discovery import (
        KalshiMarketDiscovery,
        SportMarket
    )
    from neural_sdk.data_pipeline.sports_config import Sport
except ImportError as e:
    raise SDKError(
        f"Data pipeline not available: {e}. "
        "Make sure data pipeline is properly installed."
    ) from e

logger = logging.getLogger(__name__)


class NeuralWebSocket:
    """
    Neural SDK WebSocket client for real-time market data streaming.
    
    This class provides a user-friendly interface to the underlying
    data pipeline WebSocket infrastructure.
    
    Example:
        ```python
        from neural_sdk import NeuralSDK
        
        sdk = NeuralSDK.from_env()
        websocket = sdk.create_websocket()
        
        @websocket.on_market_data
        async def handle_price_update(market_data):
            print(f"Price update: {market_data['ticker']} = {market_data['yes_price']}")
        
        await websocket.connect()
        await websocket.subscribe_markets(['NFL-*'])
        await websocket.run_forever()
        ```
    """
    
    def __init__(self, config: SDKConfig):
        """
        Initialize Neural WebSocket client.
        
        Args:
            config: SDK configuration object
        """
        self.config = config
        self._ws_client: Optional[KalshiWebSocket] = None
        self._market_discovery: Optional[KalshiMarketDiscovery] = None
        
        # Event handlers
        self._market_data_handlers: List[Callable] = []
        self._trade_handlers: List[Callable] = []
        self._connection_handlers: List[Callable] = []
        self._error_handlers: List[Callable] = []
        
        # Connection state
        self._connected = False
        self._subscribed_markets: set = set()
        
        logger.info("Neural WebSocket client initialized")
    
    async def connect(self) -> None:
        """
        Connect to the WebSocket server.
        
        Raises:
            ConnectionError: If connection fails
            SDKError: If configuration is invalid
        """
        try:
            # Initialize data pipeline WebSocket client
            self._ws_client = KalshiWebSocket()
            self._market_discovery = KalshiMarketDiscovery()
            
            # Set up message handler
            handler = DefaultMessageHandler()
            handler.set_ticker_callback(self._handle_market_data)
            handler.set_trade_callback(self._handle_trade_data)
            handler.set_error_callback(self._handle_error)
            
            self._ws_client.message_handler = handler
            
            # Connect
            await self._ws_client.connect()
            self._connected = True
            
            logger.info("✅ Neural WebSocket connected successfully")
            
            # Notify connection handlers
            for handler in self._connection_handlers:
                try:
                    await handler({"status": "connected"})
                except Exception as e:
                    logger.error(f"Error in connection handler: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to connect WebSocket: {e}")
            raise ConnectionError(f"WebSocket connection failed: {e}") from e
    
    async def disconnect(self) -> None:
        """Disconnect from the WebSocket server."""
        if self._ws_client:
            await self._ws_client.disconnect()
            self._connected = False
            logger.info("Neural WebSocket disconnected")
            
            # Notify connection handlers
            for handler in self._connection_handlers:
                try:
                    await handler({"status": "disconnected"})
                except Exception as e:
                    logger.error(f"Error in connection handler: {e}")
    
    async def subscribe_markets(self, tickers: List[str]) -> None:
        """
        Subscribe to market tickers for real-time updates.
        
        Args:
            tickers: List of market tickers to subscribe to
            
        Raises:
            ConnectionError: If not connected
            SDKError: If subscription fails
        """
        if not self._connected or not self._ws_client:
            raise ConnectionError("WebSocket not connected. Call connect() first.")
        
        try:
            await self._ws_client.subscribe_markets(tickers)
            self._subscribed_markets.update(tickers)
            
            logger.info(f"✅ Subscribed to {len(tickers)} markets")
            
        except Exception as e:
            logger.error(f"Failed to subscribe to markets: {e}")
            raise SDKError(f"Market subscription failed: {e}") from e
    
    async def subscribe_nfl_game(self, game_id: str) -> None:
        """
        Subscribe to all markets for a specific NFL game.
        
        Args:
            game_id: Game identifier (e.g., "25SEP04DALPHI")
            
        Raises:
            SDKError: If game markets not found or subscription fails
        """
        if not self._market_discovery:
            raise SDKError("Market discovery not initialized. Call connect() first.")
        
        try:
            # Find all markets for this game
            nfl_markets = await self._market_discovery.discover_nfl_markets()
            game_markets = [
                market.ticker for market in nfl_markets 
                if game_id.upper() in market.ticker.upper()
            ]
            
            if not game_markets:
                raise SDKError(f"No markets found for game: {game_id}")
            
            await self.subscribe_markets(game_markets)
            logger.info(f"✅ Subscribed to {len(game_markets)} markets for game {game_id}")
            
        except Exception as e:
            logger.error(f"Failed to subscribe to game {game_id}: {e}")
            raise SDKError(f"Game subscription failed: {e}") from e
    
    async def subscribe_nfl_team(self, team_code: str) -> None:
        """
        Subscribe to all markets for a specific NFL team.
        
        Args:
            team_code: Team code (e.g., "PHI", "KC", "SF")
            
        Raises:
            SDKError: If team markets not found or subscription fails
        """
        if not self._market_discovery:
            raise SDKError("Market discovery not initialized. Call connect() first.")
        
        try:
            team_markets = await self._market_discovery.find_team_markets(Sport.NFL, team_code)
            
            if not team_markets:
                raise SDKError(f"No markets found for team: {team_code}")
            
            market_tickers = [market.ticker for market in team_markets]
            await self.subscribe_markets(market_tickers)
            
            logger.info(f"✅ Subscribed to {len(market_tickers)} markets for team {team_code}")
            
        except Exception as e:
            logger.error(f"Failed to subscribe to team {team_code}: {e}")
            raise SDKError(f"Team subscription failed: {e}") from e
    
    def on_market_data(self, func: Callable) -> Callable:
        """
        Decorator to register market data handler.
        
        Args:
            func: Handler function that processes market data updates
            
        Returns:
            Decorated handler function
            
        Example:
            ```python
            @websocket.on_market_data
            async def handle_price_update(market_data):
                print(f"Price: {market_data['yes_price']}")
            ```
        """
        self._market_data_handlers.append(func)
        logger.info(f"Registered market data handler: {func.__name__}")
        return func
    
    def on_trade(self, func: Callable) -> Callable:
        """
        Decorator to register trade execution handler.
        
        Args:
            func: Handler function that processes trade executions
            
        Returns:
            Decorated handler function
        """
        self._trade_handlers.append(func)
        logger.info(f"Registered trade handler: {func.__name__}")
        return func
    
    def on_connection(self, func: Callable) -> Callable:
        """
        Decorator to register connection status handler.
        
        Args:
            func: Handler function that processes connection events
            
        Returns:
            Decorated handler function
        """
        self._connection_handlers.append(func)
        logger.info(f"Registered connection handler: {func.__name__}")
        return func
    
    def on_error(self, func: Callable) -> Callable:
        """
        Decorator to register error handler.
        
        Args:
            func: Handler function that processes errors
            
        Returns:
            Decorated handler function
        """
        self._error_handlers.append(func)
        logger.info(f"Registered error handler: {func.__name__}")
        return func
    
    async def _handle_market_data(self, market_data: Dict[str, Any]) -> None:
        """Handle incoming market data updates."""
        # Call all registered market data handlers
        for handler in self._market_data_handlers:
            try:
                await handler(market_data)
            except Exception as e:
                logger.error(f"Error in market data handler {handler.__name__}: {e}")
    
    async def _handle_trade_data(self, trade_data: Dict[str, Any]) -> None:
        """Handle incoming trade execution data."""
        # Call all registered trade handlers
        for handler in self._trade_handlers:
            try:
                await handler(trade_data)
            except Exception as e:
                logger.error(f"Error in trade handler {handler.__name__}: {e}")
    
    async def _handle_error(self, error_data: Dict[str, Any]) -> None:
        """Handle WebSocket errors."""
        # Call all registered error handlers
        for handler in self._error_handlers:
            try:
                await handler(error_data)
            except Exception as e:
                logger.error(f"Error in error handler {handler.__name__}: {e}")
    
    async def run_forever(self) -> None:
        """
        Run the WebSocket client until stopped.
        
        This method will block until the WebSocket is disconnected
        or an error occurs.
        """
        if not self._connected:
            raise ConnectionError("WebSocket not connected. Call connect() first.")
        
        try:
            logger.info("🚀 Neural WebSocket running... (Press Ctrl+C to stop)")
            
            while self._connected:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        finally:
            await self.disconnect()
    
    def get_subscribed_markets(self) -> List[str]:
        """Get list of currently subscribed markets."""
        return list(self._subscribed_markets)
    
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._connected
    
    def get_status(self) -> Dict[str, Any]:
        """Get WebSocket client status."""
        return {
            "connected": self._connected,
            "subscribed_markets": len(self._subscribed_markets),
            "market_data_handlers": len(self._market_data_handlers),
            "trade_handlers": len(self._trade_handlers),
            "connection_handlers": len(self._connection_handlers),
            "error_handlers": len(self._error_handlers)
        }
