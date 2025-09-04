"""
Neural SDK Market Streaming Utilities

Specialized streaming classes for different market types and sports.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from .websocket import NeuralWebSocket
from ..core.config import SDKConfig
from ..core.exceptions import SDKError

logger = logging.getLogger(__name__)


class MarketStream:
    """
    Base class for market-specific streaming functionality.
    
    Provides common functionality for streaming different types of markets.
    """
    
    def __init__(self, config: SDKConfig):
        """
        Initialize market stream.
        
        Args:
            config: SDK configuration object
        """
        self.config = config
        self.websocket = NeuralWebSocket(config)
        
        # Market tracking
        self.active_markets: Dict[str, Dict] = {}
        self.price_history: Dict[str, List[Dict]] = {}
        
        # Set up default handlers
        self._setup_default_handlers()
    
    def _setup_default_handlers(self):
        """Set up default market data handlers."""
        
        @self.websocket.on_market_data
        async def track_market_data(market_data: Dict[str, Any]):
            """Track market data updates."""
            ticker = market_data.get('market_ticker')
            if ticker:
                # Update active markets
                self.active_markets[ticker] = market_data
                
                # Add to price history
                if ticker not in self.price_history:
                    self.price_history[ticker] = []
                
                self.price_history[ticker].append({
                    'timestamp': datetime.now(),
                    'yes_price': market_data.get('yes_price'),
                    'no_price': market_data.get('no_price'),
                    'volume': market_data.get('volume', 0)
                })
                
                # Keep only last 100 price points
                if len(self.price_history[ticker]) > 100:
                    self.price_history[ticker].pop(0)
    
    async def connect(self):
        """Connect to WebSocket."""
        await self.websocket.connect()
    
    async def disconnect(self):
        """Disconnect from WebSocket."""
        await self.websocket.disconnect()
    
    def get_market_data(self, ticker: str) -> Optional[Dict]:
        """Get current market data for a ticker."""
        return self.active_markets.get(ticker)
    
    def get_price_history(self, ticker: str) -> List[Dict]:
        """Get price history for a ticker."""
        return self.price_history.get(ticker, [])
    
    def get_active_markets(self) -> List[str]:
        """Get list of active market tickers."""
        return list(self.active_markets.keys())


class NFLMarketStream(MarketStream):
    """
    Specialized streaming class for NFL markets.
    
    Provides NFL-specific functionality like game tracking,
    team filtering, and game state analysis.
    """
    
    def __init__(self, config: SDKConfig):
        """Initialize NFL market stream."""
        super().__init__(config)
        
        # NFL-specific tracking
        self.games: Dict[str, Dict] = {}
        self.teams: Dict[str, List[str]] = {}  # team -> list of market tickers
        
        # Set up NFL-specific handlers
        self._setup_nfl_handlers()
    
    def _setup_nfl_handlers(self):
        """Set up NFL-specific market data handlers."""
        
        @self.websocket.on_market_data
        async def track_nfl_games(market_data: Dict[str, Any]):
            """Track NFL game data."""
            ticker = market_data.get('market_ticker', '')
            
            if 'NFL' in ticker.upper() or 'KXNFL' in ticker.upper():
                game_id = self._extract_game_id(ticker)
                if game_id:
                    if game_id not in self.games:
                        self.games[game_id] = {
                            'markets': {},
                            'home_team': self._extract_home_team(ticker),
                            'away_team': self._extract_away_team(ticker),
                            'last_update': datetime.now()
                        }
                    
                    self.games[game_id]['markets'][ticker] = market_data
                    self.games[game_id]['last_update'] = datetime.now()
                    
                    # Track by teams
                    home_team = self.games[game_id]['home_team']
                    away_team = self.games[game_id]['away_team']
                    
                    for team in [home_team, away_team]:
                        if team and team != 'UNK':
                            if team not in self.teams:
                                self.teams[team] = []
                            if ticker not in self.teams[team]:
                                self.teams[team].append(ticker)
    
    async def subscribe_to_game(self, game_id: str):
        """
        Subscribe to all markets for a specific NFL game.
        
        Args:
            game_id: Game identifier (e.g., "25SEP04DALPHI")
        """
        await self.websocket.subscribe_nfl_game(game_id)
        logger.info(f"🏈 Subscribed to NFL game: {game_id}")
    
    async def subscribe_to_team(self, team_code: str):
        """
        Subscribe to all markets for a specific NFL team.
        
        Args:
            team_code: Team code (e.g., "PHI", "KC", "SF")
        """
        await self.websocket.subscribe_nfl_team(team_code)
        logger.info(f"🏈 Subscribed to NFL team: {team_code}")
    
    async def subscribe_to_all_nfl(self):
        """Subscribe to all active NFL markets."""
        try:
            # This would need to be implemented to get all NFL markets
            # For now, we'll use a pattern-based subscription
            await self.websocket.subscribe_markets(['KXNFLGAME*'])  # Pattern matching
            logger.info("🏈 Subscribed to all NFL markets")
        except Exception as e:
            logger.error(f"Failed to subscribe to all NFL markets: {e}")
            raise SDKError(f"NFL subscription failed: {e}") from e
    
    def get_game_data(self, game_id: str) -> Optional[Dict]:
        """Get data for a specific game."""
        return self.games.get(game_id)
    
    def get_team_markets(self, team_code: str) -> List[str]:
        """Get market tickers for a specific team."""
        return self.teams.get(team_code.upper(), [])
    
    def get_active_games(self) -> List[str]:
        """Get list of active game IDs."""
        return list(self.games.keys())
    
    def get_game_win_probability(self, game_id: str) -> Optional[float]:
        """
        Calculate win probability for home team in a game.
        
        Args:
            game_id: Game identifier
            
        Returns:
            Win probability (0.0 to 1.0) or None if not available
        """
        game_data = self.games.get(game_id)
        if not game_data:
            return None
        
        # Look for winner markets
        for ticker, market_data in game_data['markets'].items():
            if 'WINNER' in ticker.upper():
                yes_price = market_data.get('yes_price')
                if yes_price is not None:
                    return yes_price
        
        return None
    
    def get_game_summary(self, game_id: str) -> Optional[Dict]:
        """
        Get comprehensive summary for a game.
        
        Args:
            game_id: Game identifier
            
        Returns:
            Game summary dictionary or None if game not found
        """
        game_data = self.games.get(game_id)
        if not game_data:
            return None
        
        win_prob = self.get_game_win_probability(game_id)
        
        return {
            'game_id': game_id,
            'home_team': game_data['home_team'],
            'away_team': game_data['away_team'],
            'markets_count': len(game_data['markets']),
            'win_probability': win_prob,
            'last_update': game_data['last_update'].isoformat()
        }
    
    def _extract_game_id(self, ticker: str) -> Optional[str]:
        """Extract game identifier from ticker."""
        if "KXNFLGAME" in ticker:
            parts = ticker.split("-")
            if len(parts) >= 3:
                return f"{parts[1]}-{parts[2]}"
        return None
    
    def _extract_home_team(self, ticker: str) -> str:
        """Extract home team from ticker (simplified)."""
        if "KXNFLGAME" in ticker:
            parts = ticker.split("-")
            if len(parts) >= 4:
                return parts[-1][:3]
        return "UNK"
    
    def _extract_away_team(self, ticker: str) -> str:
        """Extract away team from ticker (simplified)."""
        if "KXNFLGAME" in ticker:
            parts = ticker.split("-")
            if len(parts) >= 4:
                game_part = parts[2]
                if len(game_part) > 6:
                    return game_part[-6:-3]
        return "UNK"
