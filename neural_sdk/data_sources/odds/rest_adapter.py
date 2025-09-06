"""
The Odds API REST Adapter

Provides access to comprehensive sports betting odds from multiple bookmakers.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging
from collections import defaultdict

from ..base.rest_source import RESTDataSource
from ..base.auth_strategies import APIKeyAuth
from .models import (
    OddsData, BookmakerOdds, MarketOdds, MarketOutcome,
    Sport, Market, OddsFormat, Bookmaker, LineMovement
)

logger = logging.getLogger(__name__)


class OddsAPIAdapter(RESTDataSource):
    """
    REST adapter for The Odds API.
    
    Provides real-time and historical sports betting odds from
    major US bookmakers including DraftKings, FanDuel, BetMGM, etc.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Odds API adapter.
        
        Args:
            api_key: The Odds API key (or set ODDS_API_KEY env var)
        """
        import os
        
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("ODDS_API_KEY", "78a75adadb135d123eecb7900b57c82e")
        
        if not self.api_key:
            raise ValueError("Odds API key required. Set ODDS_API_KEY or pass api_key parameter")
        
        # No auth strategy needed - using query params
        super().__init__(
            base_url="https://api.the-odds-api.com",
            name="OddsAPI",
            auth_strategy=None,  # Auth via query params
            timeout=30,
            cache_ttl=300,  # 5 minute cache to preserve quota
            rate_limit=10,  # Conservative rate limit
            max_retries=3
        )
        
        # Track API usage
        self.requests_remaining = 500
        self.requests_used = 0
        
        # Line movement tracking
        self.line_movements: Dict[str, LineMovement] = {}
        
        logger.info(f"Odds API adapter initialized")
    
    async def fetch(self, endpoint: str, params: Optional[Dict] = None, **kwargs) -> Dict:
        """
        Override fetch to add API key and track usage.
        """
        # Add API key to params
        if params is None:
            params = {}
        params["apiKey"] = self.api_key
        
        # Make request
        response = await super().fetch(endpoint, params, **kwargs)
        
        # Track usage from headers if available
        if hasattr(self, '_last_response_headers'):
            headers = self._last_response_headers
            if "x-requests-remaining" in headers:
                self.requests_remaining = int(headers["x-requests-remaining"])
            if "x-requests-used" in headers:
                self.requests_used = int(headers["x-requests-used"])
        
        return response
    
    async def validate_response(self, response) -> bool:
        """
        Validate Odds API response and extract headers.
        """
        # Store headers for usage tracking
        self._last_response_headers = dict(response.headers)
        
        if response.status_code == 200:
            return True
        elif response.status_code == 401:
            logger.error("Invalid Odds API key")
        elif response.status_code == 429:
            logger.error("Odds API quota exceeded")
        elif response.status_code == 500:
            logger.error("Odds API server error")
        
        return False
    
    async def transform_response(self, data: Any, endpoint: str) -> Dict:
        """
        Transform Odds API response to standardized format.
        
        Args:
            data: Raw API response
            endpoint: The endpoint that was called
            
        Returns:
            Standardized response
        """
        return {
            "source": "odds_api",
            "endpoint": endpoint,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": {
                "requests_remaining": self.requests_remaining,
                "requests_used": self.requests_used
            }
        }
    
    # Sports Methods
    
    async def get_sports(self, all_sports: bool = False) -> List[Dict]:
        """
        Get list of available sports.
        
        Args:
            all_sports: Include inactive sports
            
        Returns:
            List of sports with keys and descriptions
        """
        params = {}
        if all_sports:
            params["all"] = "true"
        
        result = await self.fetch("/v4/sports/", params=params)
        return result if isinstance(result, list) else []
    
    # Odds Methods
    
    async def get_odds(
        self,
        sport: str,
        regions: str = "us",
        markets: Optional[str] = None,
        odds_format: str = "american",
        bookmakers: Optional[str] = None,
        event_ids: Optional[str] = None,
        commence_time_from: Optional[str] = None,
        commence_time_to: Optional[str] = None
    ) -> List[OddsData]:
        """
        Get current odds for a sport.
        
        Args:
            sport: Sport key (e.g., "americanfootball_nfl")
            regions: Comma-separated regions (us, uk, au, eu)
            markets: Comma-separated markets (h2h, spreads, totals)
            odds_format: american or decimal
            bookmakers: Comma-separated bookmaker keys
            event_ids: Comma-separated event IDs
            commence_time_from: ISO format start time
            commence_time_to: ISO format end time
            
        Returns:
            List of OddsData objects
        """
        params = {
            "regions": regions,
            "oddsFormat": odds_format
        }
        
        # Default markets if not specified
        if markets:
            params["markets"] = markets
        else:
            params["markets"] = "h2h,spreads,totals"
        
        if bookmakers:
            params["bookmakers"] = bookmakers
        if event_ids:
            params["eventIds"] = event_ids
        if commence_time_from:
            params["commenceTimeFrom"] = commence_time_from
        if commence_time_to:
            params["commenceTimeTo"] = commence_time_to
        
        result = await self.fetch(f"/v4/sports/{sport}/odds/", params=params)
        
        # Parse response into OddsData objects
        odds_list = []
        if isinstance(result, list):
            for game in result:
                odds_list.append(self._parse_odds_response(game))
        
        return odds_list
    
    async def get_nfl_odds(self, markets: str = "h2h,spreads,totals") -> List[OddsData]:
        """Get NFL odds."""
        return await self.get_odds(Sport.NFL.value, markets=markets)
    
    async def get_nba_odds(self, markets: str = "h2h,spreads,totals") -> List[OddsData]:
        """Get NBA odds."""
        return await self.get_odds(Sport.NBA.value, markets=markets)
    
    async def get_ncaaf_odds(self, markets: str = "h2h,spreads,totals") -> List[OddsData]:
        """Get college football odds."""
        return await self.get_odds(Sport.NCAAF.value, markets=markets)
    
    # Scores Methods
    
    async def get_scores(
        self,
        sport: str,
        days_from: int = 1,
        date_format: str = "iso"
    ) -> List[Dict]:
        """
        Get scores for recent and live games.
        
        Args:
            sport: Sport key
            days_from: Number of days back (1-3)
            date_format: iso or unix
            
        Returns:
            List of games with scores
        """
        params = {
            "daysFrom": str(days_from),
            "dateFormat": date_format
        }
        
        result = await self.fetch(f"/v4/sports/{sport}/scores/", params=params)
        return result if isinstance(result, list) else []
    
    # Historical Methods
    
    async def get_historical_odds(
        self,
        sport: str,
        date: str,
        regions: str = "us",
        markets: str = "h2h",
        odds_format: str = "american",
        bookmakers: Optional[str] = None
    ) -> List[OddsData]:
        """
        Get historical odds snapshot.
        
        Args:
            sport: Sport key
            date: Date in YYYY-MM-DD format
            regions: Comma-separated regions
            markets: Comma-separated markets
            odds_format: american or decimal
            bookmakers: Comma-separated bookmaker keys
            
        Returns:
            List of historical OddsData
        """
        params = {
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format,
            "date": date
        }
        
        if bookmakers:
            params["bookmakers"] = bookmakers
        
        result = await self.fetch(
            f"/v4/historical/sports/{sport}/odds/",
            params=params
        )
        
        # Parse response
        odds_list = []
        if isinstance(result, dict) and "data" in result:
            for game in result["data"]:
                odds_list.append(self._parse_odds_response(game))
        
        return odds_list
    
    # Analysis Methods
    
    async def get_best_odds(
        self,
        sport: str,
        market: str = "h2h"
    ) -> Dict[str, Dict]:
        """
        Get best odds for all games in a sport.
        
        Args:
            sport: Sport key
            market: Market type
            
        Returns:
            Dictionary of game IDs to best odds
        """
        odds_list = await self.get_odds(sport, markets=market)
        
        best_odds = {}
        for game in odds_list:
            best = game.get_best_odds(market)
            if best:
                best_odds[game.id] = {
                    "game": f"{game.away_team} @ {game.home_team}",
                    "commence_time": game.commence_time,
                    "best_odds": best
                }
        
        return best_odds
    
    async def find_arbitrage_opportunities(
        self,
        sport: str
    ) -> List[Dict[str, Any]]:
        """
        Find arbitrage opportunities in a sport.
        
        Args:
            sport: Sport key
            
        Returns:
            List of arbitrage opportunities
        """
        odds_list = await self.get_odds(sport, markets="h2h")
        
        opportunities = []
        for game in odds_list:
            arb = game.find_arbitrage()
            if arb and arb["exists"]:
                opportunities.append({
                    "game_id": game.id,
                    "game": f"{game.away_team} @ {game.home_team}",
                    "commence_time": game.commence_time,
                    "arbitrage": arb
                })
        
        return opportunities
    
    def track_line_movement(
        self,
        game_id: str,
        bookmaker: str,
        market_type: str,
        team: str,
        price: float,
        point: Optional[float] = None
    ):
        """
        Track line movement for a game.
        
        Args:
            game_id: Unique game identifier
            bookmaker: Bookmaker name
            market_type: Market type
            team: Team name
            price: Current odds
            point: Spread or total
        """
        key = f"{game_id}:{bookmaker}:{market_type}:{team}"
        
        if key not in self.line_movements:
            self.line_movements[key] = LineMovement(
                game_id=game_id,
                market_type=market_type,
                bookmaker=bookmaker,
                team=team
            )
        
        self.line_movements[key].add_movement(price, point)
    
    async def monitor_line_movements(
        self,
        sport: str,
        interval_minutes: int = 5,
        duration_hours: int = 1
    ) -> Dict[str, List[LineMovement]]:
        """
        Monitor line movements over time.
        
        Args:
            sport: Sport to monitor
            interval_minutes: Check interval
            duration_hours: Total monitoring duration
            
        Returns:
            Line movements grouped by game
        """
        import asyncio
        
        end_time = datetime.utcnow() + timedelta(hours=duration_hours)
        movements_by_game = defaultdict(list)
        
        while datetime.utcnow() < end_time:
            odds_list = await self.get_odds(sport)
            
            for game in odds_list:
                for bookmaker in game.bookmakers:
                    # Track moneyline
                    h2h = bookmaker.get_market("h2h")
                    if h2h:
                        for outcome in h2h.outcomes:
                            self.track_line_movement(
                                game.id,
                                bookmaker.title,
                                "h2h",
                                outcome.name,
                                outcome.price
                            )
                    
                    # Track spreads
                    spreads = bookmaker.get_market("spreads")
                    if spreads:
                        for outcome in spreads.outcomes:
                            self.track_line_movement(
                                game.id,
                                bookmaker.title,
                                "spreads",
                                outcome.name,
                                outcome.price,
                                outcome.point
                            )
            
            # Wait for next interval
            await asyncio.sleep(interval_minutes * 60)
        
        # Group movements by game
        for key, movement in self.line_movements.items():
            game_id = movement.game_id
            movements_by_game[game_id].append(movement)
        
        return dict(movements_by_game)
    
    # Helper Methods
    
    def _parse_odds_response(self, data: Dict) -> OddsData:
        """Parse API response into OddsData object."""
        # Parse bookmakers
        bookmakers = []
        for bm in data.get("bookmakers", []):
            markets = []
            for market in bm.get("markets", []):
                outcomes = []
                for outcome in market.get("outcomes", []):
                    outcomes.append(MarketOutcome(
                        name=outcome["name"],
                        price=outcome["price"],
                        point=outcome.get("point")
                    ))
                
                markets.append(MarketOdds(
                    key=market["key"],
                    last_update=datetime.fromisoformat(
                        market["last_update"].replace("Z", "+00:00")
                    ),
                    outcomes=outcomes
                ))
            
            bookmakers.append(BookmakerOdds(
                key=bm["key"],
                title=bm["title"],
                last_update=datetime.fromisoformat(
                    bm["last_update"].replace("Z", "+00:00")
                ),
                markets=markets
            ))
        
        return OddsData(
            id=data["id"],
            sport_key=data["sport_key"],
            sport_title=data.get("sport_title", data["sport_key"]),
            commence_time=datetime.fromisoformat(
                data["commence_time"].replace("Z", "+00:00")
            ),
            home_team=data["home_team"],
            away_team=data["away_team"],
            bookmakers=bookmakers
        )
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get API usage statistics."""
        return {
            "requests_used": self.requests_used,
            "requests_remaining": self.requests_remaining,
            "cache_stats": self.get_stats()
        }
    
    # Specialized Methods for Trading
    
    async def compare_with_kalshi(
        self,
        sport: str,
        kalshi_markets: List[Dict]
    ) -> List[Dict[str, Any]]:
        """
        Compare sportsbook odds with Kalshi prediction markets.
        
        Args:
            sport: Sport to compare
            kalshi_markets: List of Kalshi market data
            
        Returns:
            Comparison analysis with arbitrage opportunities
        """
        odds_list = await self.get_odds(sport)
        comparisons = []
        
        for game in odds_list:
            # Find matching Kalshi market
            for km in kalshi_markets:
                if self._matches_game(game, km):
                    consensus = game.get_consensus_moneyline()
                    
                    # Convert Kalshi prices to American odds
                    kalshi_home = self._probability_to_american(
                        km.get("yes_ask", 0) / 100
                    )
                    kalshi_away = self._probability_to_american(
                        1 - (km.get("yes_ask", 0) / 100)
                    )
                    
                    comparisons.append({
                        "game": f"{game.away_team} @ {game.home_team}",
                        "sportsbook_odds": consensus,
                        "kalshi_odds": {
                            game.home_team: kalshi_home,
                            game.away_team: kalshi_away
                        },
                        "kalshi_ticker": km.get("ticker"),
                        "discrepancy": abs(consensus[game.home_team] - kalshi_home)
                    })
        
        return comparisons
    
    def _matches_game(self, odds_game: OddsData, kalshi_market: Dict) -> bool:
        """Check if Kalshi market matches odds game."""
        title = kalshi_market.get("title", "").lower()
        return (
            odds_game.home_team.lower() in title or
            odds_game.away_team.lower() in title
        )
    
    def _probability_to_american(self, prob: float) -> float:
        """Convert probability to American odds."""
        if prob >= 0.5:
            return -100 * prob / (1 - prob)
        else:
            return 100 * (1 - prob) / prob