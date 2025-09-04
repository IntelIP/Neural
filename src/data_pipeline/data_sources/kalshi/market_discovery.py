"""
Kalshi Market Discovery Module
Discover active markets for sports using Kalshi API
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any

from .client import KalshiClient
from ...sports_config import Sport, get_sport_config, get_series_ticker

logger = logging.getLogger(__name__)


@dataclass
class SportMarket:
    """Represents a discovered sport market"""
    ticker: str
    title: str
    sport: Sport
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    game_date: Optional[datetime] = None
    market_type: str = "game_winner"
    yes_bid: Optional[float] = None
    yes_ask: Optional[float] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None
    
    @property
    def display_name(self) -> str:
        """Human readable market name"""
        if self.home_team and self.away_team:
            return f"{self.away_team} @ {self.home_team}"
        return self.title


class KalshiMarketDiscovery:
    """Discover sports markets using Kalshi API"""
    
    def __init__(self, client: Optional[KalshiClient] = None):
        """
        Initialize market discovery
        
        Args:
            client: Optional KalshiClient instance
        """
        self.client = client or KalshiClient()
    
    async def discover_sport_markets(self, sport: Sport) -> List[SportMarket]:
        """
        Discover all active markets for a sport
        
        Args:
            sport: Sport to discover markets for
        
        Returns:
            List of discovered markets
        """
        sport_config = get_sport_config(sport)
        
        if not sport_config.season_active:
            logger.warning(f"{sport_config.display_name} season not currently active")
            return []
        
        try:
            # Get markets by series ticker
            response = self.client.get('/markets', params={
                'series_ticker': sport_config.series_ticker,
                'limit': 200,
                'status': 'open'
            })
            
            markets = response.get('markets', [])
            logger.info(f"Found {len(markets)} active {sport_config.display_name} markets")
            
            return [self._parse_market(market, sport) for market in markets]
            
        except Exception as e:
            logger.error(f"Failed to discover {sport_config.display_name} markets: {e}")
            return []
    
    def _parse_market(self, market_data: Dict[str, Any], sport: Sport) -> SportMarket:
        """Parse market data from API response"""
        ticker = market_data.get('ticker', '')
        title = market_data.get('title', '')
        
        # Extract team names from ticker (NFL specific)
        home_team, away_team = self._extract_teams_from_ticker(ticker, sport)
        
        # Parse game date if available
        game_date = None
        if 'close_time' in market_data:
            try:
                game_date = datetime.fromisoformat(market_data['close_time'].replace('Z', '+00:00'))
            except (ValueError, TypeError):
                pass
        
        return SportMarket(
            ticker=ticker,
            title=title,
            sport=sport,
            home_team=home_team,
            away_team=away_team,
            game_date=game_date,
            yes_bid=market_data.get('yes_bid', 0) / 100 if market_data.get('yes_bid') else None,
            yes_ask=market_data.get('yes_ask', 0) / 100 if market_data.get('yes_ask') else None,
            volume=market_data.get('volume'),
            open_interest=market_data.get('open_interest')
        )
    
    def _extract_teams_from_ticker(self, ticker: str, sport: Sport) -> tuple[Optional[str], Optional[str]]:
        """Extract team names from market ticker"""
        if sport == Sport.NFL and 'KXNFLGAME' in ticker:
            # Example: KXNFLGAME-25SEP04DALPHI-PHI
            try:
                parts = ticker.split('-')
                if len(parts) >= 3:
                    team_part = parts[2]  # PHI, DAL, etc.
                    game_part = parts[1]  # 25SEP04DALPHI
                    
                    # Extract away team (last 3 chars of team_part)
                    away_team = team_part
                    
                    # Extract home team from game_part (chars before away_team)
                    if game_part.endswith(away_team.upper()):
                        home_part = game_part[:-len(away_team)]
                        if len(home_part) >= 3:
                            home_team = home_part[-3:]  # Last 3 chars
                            return home_team, away_team
                    
            except (IndexError, ValueError):
                pass
        
        return None, None
    
    async def discover_nfl_markets(self) -> List[SportMarket]:
        """Discover NFL markets"""
        return await self.discover_sport_markets(Sport.NFL)
    
    async def discover_cfp_markets(self) -> List[SportMarket]:
        """Discover CFP markets"""
        return await self.discover_sport_markets(Sport.CFP)
    
    async def find_team_markets(self, sport: Sport, team_name: str) -> List[SportMarket]:
        """
        Find markets for a specific team
        
        Args:
            sport: Sport to search in
            team_name: Team name (e.g., 'PHI', 'DAL')
        
        Returns:
            List of markets involving the team
        """
        all_markets = await self.discover_sport_markets(sport)
        
        team_markets = []
        for market in all_markets:
            if (market.home_team and team_name.upper() in market.home_team.upper()) or \
               (market.away_team and team_name.upper() in market.away_team.upper()) or \
               team_name.upper() in market.ticker.upper():
                team_markets.append(market)
        
        return team_markets
    
    async def find_game_markets(self, sport: Sport, home_team: str, away_team: str) -> List[SportMarket]:
        """
        Find markets for a specific game
        
        Args:
            sport: Sport to search in
            home_team: Home team name
            away_team: Away team name
        
        Returns:
            List of markets for the game
        """
        all_markets = await self.discover_sport_markets(sport)
        
        game_markets = []
        for market in all_markets:
            if ((market.home_team and home_team.upper() in market.home_team.upper()) and
                (market.away_team and away_team.upper() in market.away_team.upper())) or \
               (home_team.upper() in market.ticker.upper() and away_team.upper() in market.ticker.upper()):
                game_markets.append(market)
        
        return game_markets


# Convenience functions
async def discover_nfl_markets() -> List[SportMarket]:
    """Quick way to discover NFL markets"""
    discovery = KalshiMarketDiscovery()
    return await discovery.discover_nfl_markets()


async def discover_cfp_markets() -> List[SportMarket]:
    """Quick way to discover CFP markets"""
    discovery = KalshiMarketDiscovery()
    return await discovery.discover_cfp_markets()


async def find_eagles_cowboys_markets() -> List[SportMarket]:
    """Find Eagles vs Cowboys markets specifically"""
    discovery = KalshiMarketDiscovery()
    return await discovery.find_game_markets(Sport.NFL, "PHI", "DAL")