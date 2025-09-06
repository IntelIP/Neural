"""
ESPN REST API Adapter

Provides unified interface for ESPN sports data.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, date
import logging

from ..base.rest_source import RESTDataSource
from ..base.auth_strategies import NoAuth
from neural_sdk.data_pipeline.data_sources.espn.client import ESPNClient
from neural_sdk.data_pipeline.data_sources.espn.processor import PlayByPlayProcessor

logger = logging.getLogger(__name__)


class ESPNRESTAdapter(RESTDataSource):
    """
    REST adapter for ESPN API.
    
    Provides sports data including scores, stats, play-by-play,
    and team information through a unified interface.
    """
    
    # Sport path mappings
    SPORT_PATHS = {
        "nfl": "football/nfl",
        "nba": "basketball/nba",
        "cfb": "football/college-football",
        "college-football": "football/college-football",
        "mlb": "baseball/mlb",
        "nhl": "hockey/nhl"
    }
    
    def __init__(self):
        """Initialize ESPN REST adapter."""
        # ESPN doesn't require authentication
        super().__init__(
            base_url="http://site.api.espn.com/apis/site/v2/sports",
            name="ESPNREST",
            auth_strategy=NoAuth(),
            timeout=30,
            cache_ttl=30,  # Cache for 30 seconds
            rate_limit=10,  # ESPN is less strict on rate limits
            max_retries=3
        )
        
        # Use existing processor for play-by-play
        self.processor = PlayByPlayProcessor()
        
        logger.info("ESPN REST adapter initialized")
    
    async def validate_response(self, response) -> bool:
        """
        Validate ESPN API response.
        
        Args:
            response: HTTP response object
            
        Returns:
            True if valid, False otherwise
        """
        if response.status_code == 200:
            return True
        
        if response.status_code == 404:
            logger.warning("ESPN resource not found")
        elif response.status_code == 429:
            logger.warning("ESPN rate limit exceeded")
        elif response.status_code >= 500:
            logger.error(f"ESPN server error: {response.status_code}")
        
        return False
    
    async def transform_response(self, data: Any, endpoint: str) -> Dict:
        """
        Transform ESPN response to standardized format.
        
        Args:
            data: Raw ESPN response
            endpoint: The endpoint that was called
            
        Returns:
            Standardized response
        """
        return {
            "source": "espn",
            "endpoint": endpoint,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": {
                "sport": self._extract_sport_from_endpoint(endpoint)
            }
        }
    
    def _extract_sport_from_endpoint(self, endpoint: str) -> Optional[str]:
        """Extract sport from endpoint path."""
        for sport, path in self.SPORT_PATHS.items():
            if path in endpoint:
                return sport
        return None
    
    def _get_sport_path(self, sport: str) -> str:
        """Get API path for sport."""
        return self.SPORT_PATHS.get(sport.lower(), sport.lower())
    
    # Scoreboard Methods
    
    async def get_scoreboard(
        self,
        sport: str,
        date_str: Optional[str] = None,
        week: Optional[int] = None,
        groups: Optional[str] = None
    ) -> Dict:
        """
        Get scoreboard for a sport.
        
        Args:
            sport: Sport name (nfl, nba, cfb, etc.)
            date_str: Date in YYYYMMDD format
            week: Week number (for NFL/CFB)
            groups: Conference/division filter
            
        Returns:
            Scoreboard data
        """
        sport_path = self._get_sport_path(sport)
        endpoint = f"/{sport_path}/scoreboard"
        
        params = {}
        if date_str:
            params["dates"] = date_str
        if week:
            params["week"] = week
        if groups:
            params["groups"] = groups
        
        return await self.fetch(endpoint, params=params)
    
    async def get_game_summary(self, sport: str, game_id: str) -> Dict:
        """
        Get detailed game summary.
        
        Args:
            sport: Sport name
            game_id: ESPN game ID
            
        Returns:
            Game summary data
        """
        sport_path = self._get_sport_path(sport)
        endpoint = f"/{sport_path}/summary"
        
        return await self.fetch(endpoint, params={"event": game_id})
    
    # Play-by-Play Methods
    
    async def get_play_by_play(self, sport: str, game_id: str) -> Dict:
        """
        Get play-by-play data for a game.
        
        Args:
            sport: Sport name
            game_id: ESPN game ID
            
        Returns:
            Play-by-play data with processed events
        """
        # Get raw play-by-play
        result = await self.get_game_summary(sport, game_id)
        
        # Process with existing processor
        if "data" in result and "drives" in result["data"]:
            processed = self.processor.process_game(result["data"])
            result["processed"] = processed
        
        return result
    
    # Team Methods
    
    async def get_teams(self, sport: str, limit: int = 100) -> Dict:
        """
        Get teams for a sport.
        
        Args:
            sport: Sport name
            limit: Maximum number of teams
            
        Returns:
            Teams data
        """
        sport_path = self._get_sport_path(sport)
        endpoint = f"/{sport_path}/teams"
        
        return await self.fetch(endpoint, params={"limit": limit})
    
    async def get_team(self, sport: str, team_id: str) -> Dict:
        """
        Get single team information.
        
        Args:
            sport: Sport name
            team_id: ESPN team ID
            
        Returns:
            Team data
        """
        sport_path = self._get_sport_path(sport)
        endpoint = f"/{sport_path}/teams/{team_id}"
        
        return await self.fetch(endpoint)
    
    async def get_team_roster(self, sport: str, team_id: str) -> Dict:
        """
        Get team roster.
        
        Args:
            sport: Sport name
            team_id: ESPN team ID
            
        Returns:
            Roster data
        """
        sport_path = self._get_sport_path(sport)
        endpoint = f"/{sport_path}/teams/{team_id}/roster"
        
        return await self.fetch(endpoint)
    
    # NFL-Specific Methods
    
    async def get_nfl_games(self, week: Optional[int] = None) -> Dict:
        """
        Get NFL games.
        
        Args:
            week: NFL week number
            
        Returns:
            NFL games data
        """
        params = {}
        if week:
            params["week"] = week
        
        result = await self.get_scoreboard("nfl", **params)
        
        # Extract game information
        if "data" in result and "events" in result["data"]:
            games = []
            for event in result["data"]["events"]:
                game_info = {
                    "id": event.get("id"),
                    "name": event.get("name"),
                    "short_name": event.get("shortName"),
                    "date": event.get("date"),
                    "status": event.get("status", {}).get("type", {}).get("name"),
                    "completed": event.get("status", {}).get("type", {}).get("completed", False)
                }
                
                # Add competition details
                if "competitions" in event and event["competitions"]:
                    comp = event["competitions"][0]
                    game_info["venue"] = comp.get("venue", {}).get("fullName")
                    game_info["attendance"] = comp.get("attendance")
                    
                    # Add competitor information
                    if "competitors" in comp:
                        for competitor in comp["competitors"]:
                            team_type = "home" if competitor.get("homeAway") == "home" else "away"
                            game_info[f"{team_type}_team"] = {
                                "id": competitor.get("id"),
                                "name": competitor.get("team", {}).get("displayName"),
                                "abbreviation": competitor.get("team", {}).get("abbreviation"),
                                "score": competitor.get("score"),
                                "record": competitor.get("records", [{}])[0].get("summary") if competitor.get("records") else None
                            }
                
                games.append(game_info)
            
            result["games"] = games
        
        return result
    
    async def get_cfb_games(self, week: Optional[int] = None, group: Optional[str] = None) -> Dict:
        """
        Get college football games.
        
        Args:
            week: CFB week number
            group: Conference filter (e.g., "80" for Big Ten)
            
        Returns:
            CFB games data
        """
        params = {}
        if week:
            params["week"] = week
        if group:
            params["groups"] = group
        
        result = await self.get_scoreboard("cfb", **params)
        
        # Process similar to NFL games
        if "data" in result and "events" in result["data"]:
            games = []
            for event in result["data"]["events"]:
                game_info = {
                    "id": event.get("id"),
                    "name": event.get("name"),
                    "short_name": event.get("shortName"),
                    "date": event.get("date"),
                    "status": event.get("status", {}).get("type", {}).get("name")
                }
                
                # Add teams and scores
                if "competitions" in event and event["competitions"]:
                    comp = event["competitions"][0]
                    if "competitors" in comp:
                        for competitor in comp["competitors"]:
                            team_type = "home" if competitor.get("homeAway") == "home" else "away"
                            game_info[f"{team_type}_team"] = {
                                "name": competitor.get("team", {}).get("displayName"),
                                "score": competitor.get("score"),
                                "rank": competitor.get("curatedRank", {}).get("current")
                            }
                
                games.append(game_info)
            
            result["games"] = games
        
        return result
    
    # Batch Operations
    
    async def get_multiple_games(self, sport: str, game_ids: List[str]) -> Dict:
        """
        Get multiple games in parallel.
        
        Args:
            sport: Sport name
            game_ids: List of game IDs
            
        Returns:
            Dictionary of game data by ID
        """
        sport_path = self._get_sport_path(sport)
        
        requests = [
            {"endpoint": f"/{sport_path}/summary", "params": {"event": game_id}}
            for game_id in game_ids
        ]
        
        results = await self.batch_fetch(requests)
        
        # Map results to game IDs
        game_data = {}
        for game_id, result in zip(game_ids, results):
            if not isinstance(result, Exception):
                game_data[game_id] = result
            else:
                logger.error(f"Failed to fetch game {game_id}: {result}")
                game_data[game_id] = None
        
        return {
            "source": "espn",
            "sport": sport,
            "data": game_data,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # Odds Methods
    
    async def get_odds(self, sport: str, game_id: str) -> Dict:
        """
        Get betting odds for a game.
        
        Args:
            sport: Sport name
            game_id: ESPN game ID
            
        Returns:
            Odds data
        """
        sport_path = self._get_sport_path(sport)
        endpoint = f"/{sport_path}/odds"
        
        return await self.fetch(endpoint, params={"event": game_id})
    
    # Health Check
    
    async def health_check(self) -> bool:
        """
        Check ESPN API health.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            result = await self.get_scoreboard("nfl")
            return "data" in result
        except Exception as e:
            logger.error(f"ESPN health check failed: {e}")
            return False