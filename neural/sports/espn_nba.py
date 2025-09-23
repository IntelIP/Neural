"""
ESPN NBA client implementation.

This module provides NBA-specific functionality for fetching data from ESPN's API,
including scores, standings, player stats, and more.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
import logging

from neural.sports.espn_base import ESPNClient, ESPNConfig


logger = logging.getLogger(__name__)


class ESPNNBAConfig(ESPNConfig):
    """NBA-specific ESPN configuration."""
    
    def __init__(self, **kwargs):
        kwargs.setdefault("name", "espn_nba")
        kwargs.setdefault("sport", "basketball")
        kwargs.setdefault("league", "nba")
        kwargs.setdefault("default_season", "2025-26")
        super().__init__(**kwargs)


class ESPNNBA(ESPNClient):
    """
    ESPN NBA API client.
    
    Provides methods for fetching NBA-specific data including games, teams,
    rosters, standings, player stats, and more.
    
    Example:
        >>> nba = ESPNNBA()
        >>> scores = await nba.get_scoreboard(dates="20251225")  # Christmas games
        >>> standings = await nba.get_standings()
    """
    
    def __init__(self, config: Optional[ESPNNBAConfig] = None):
        """
        Initialize NBA client.
        
        Args:
            config: NBA-specific configuration
        """
        if config is None:
            config = ESPNNBAConfig()
        
        super().__init__(config)
        self.config: ESPNNBAConfig = config
    
    async def get_scoreboard(
        self,
        dates: Optional[str] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Get NBA scoreboard data.
        
        Args:
            dates: Date in YYYYMMDD format or date range
            limit: Maximum number of games to return
            
        Returns:
            Scoreboard data with games and scores
            
        Example:
            >>> # Get today's games
            >>> scores = await nba.get_scoreboard()
            >>> 
            >>> # Get specific date
            >>> scores = await nba.get_scoreboard(dates="20251225")
            >>> 
            >>> # Get date range
            >>> scores = await nba.get_scoreboard(dates="20251201-20251207")
        """
        params = {"limit": limit}
        
        if dates:
            params["dates"] = dates
        else:
            # Default to today
            params["dates"] = datetime.now().strftime("%Y%m%d")
        
        return await super().get_scoreboard(**params)
    
    async def get_standings(
        self,
        season: Optional[str] = None,
        group: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get NBA standings.
        
        Args:
            season: Season (e.g., "2025-26")
            group: Optional group filter ("conference" or "division")
            
        Returns:
            Current NBA standings by conference and division
        """
        endpoint = f"{self._base_path}/standings"
        
        params = {}
        if season:
            # NBA seasons span two years, extract the first year
            params["season"] = season.split("-")[0] if "-" in season else season
        elif self.config.default_season:
            params["season"] = self.config.default_season.split("-")[0] if "-" in str(self.config.default_season) else self.config.default_season
        
        if group:
            params["group"] = group
        
        return await self.get(
            endpoint,
            params=params if params else None,
            cache_ttl=3600  # 1 hour for standings
        )
    
    async def get_player_stats(
        self,
        player_id: str,
        season: Optional[str] = None,
        seasontype: int = 2
    ) -> Dict[str, Any]:
        """
        Get player statistics.
        
        Args:
            player_id: Player ID
            season: Season (e.g., "2025-26")
            seasontype: Season type (1=preseason, 2=regular, 3=playoffs)
            
        Returns:
            Player statistics for the specified season
        """
        # Use the core API for player stats
        base_url = "https://sports.core.api.espn.com"
        
        season_year = season.split("-")[0] if season and "-" in season else season
        if not season_year:
            season_year = self.config.default_season.split("-")[0] if "-" in str(self.config.default_season) else self.config.default_season
        
        endpoint = f"/v2/sports/basketball/leagues/nba/seasons/{season_year}/types/{seasontype}/athletes/{player_id}/statistics"
        
        # Temporarily change base URL for this request
        original_base = self.config.base_url
        self.config.base_url = base_url
        
        try:
            result = await self.get(
                endpoint,
                cache_ttl=3600  # 1 hour for stats
            )
            return result
        finally:
            self.config.base_url = original_base
    
    async def get_team_roster(
        self,
        team_id: str,
        include_stats: bool = False
    ) -> Dict[str, Any]:
        """
        Get team roster with optional statistics.
        
        Args:
            team_id: Team ID or abbreviation (e.g., "LAL" for Lakers)
            include_stats: Include player statistics
            
        Returns:
            Team roster with player details
        """
        if include_stats:
            endpoint = f"{self._base_path}/teams/{team_id}"
            params = {"enable": "roster,stats"}
            
            return await self.get(
                endpoint,
                params=params,
                cache_ttl=self.config.cache_roster_ttl
            )
        else:
            return await super().get_roster(team_id)
    
    async def get_team_schedule(
        self,
        team_id: str,
        season: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get team schedule for a season.
        
        Args:
            team_id: Team ID or abbreviation
            season: Season (e.g., "2025-26")
            
        Returns:
            Complete team schedule with game details
        """
        endpoint = f"{self._base_path}/teams/{team_id}/schedule"
        
        params = {}
        if season:
            # Use the first year of the season
            params["season"] = season.split("-")[0] if "-" in season else season
        elif self.config.default_season:
            params["season"] = self.config.default_season.split("-")[0] if "-" in str(self.config.default_season) else self.config.default_season
        
        return await self.get(
            endpoint,
            params=params if params else None,
            cache_ttl=self.config.cache_schedule_ttl
        )
    
    async def get_team_statistics(
        self,
        team_id: str,
        season: Optional[str] = None,
        seasontype: int = 2
    ) -> Dict[str, Any]:
        """
        Get team statistics.
        
        Args:
            team_id: Team ID or abbreviation
            season: Season (e.g., "2025-26")
            seasontype: Season type (1=preseason, 2=regular, 3=playoffs)
            
        Returns:
            Team statistics for the specified season
        """
        # Use the core API for team stats
        base_url = "https://sports.core.api.espn.com"
        
        season_year = season.split("-")[0] if season and "-" in season else season
        if not season_year:
            season_year = self.config.default_season.split("-")[0] if "-" in str(self.config.default_season) else self.config.default_season
        
        endpoint = f"/v2/sports/basketball/leagues/nba/seasons/{season_year}/types/{seasontype}/teams/{team_id}/statistics"
        
        # Temporarily change base URL for this request
        original_base = self.config.base_url
        self.config.base_url = base_url
        
        try:
            result = await self.get(
                endpoint,
                cache_ttl=3600  # 1 hour for stats
            )
            return result
        finally:
            self.config.base_url = original_base
    
    async def get_game_summary(self, game_id: str) -> Dict[str, Any]:
        """
        Get detailed game summary.
        
        Args:
            game_id: ESPN game/event ID
            
        Returns:
            Detailed game information including play-by-play if available
        """
        endpoint = f"{self._base_path}/summary"
        params = {"event": game_id}
        
        return await self.get(
            endpoint,
            params=params,
            cache_ttl=60  # 1 minute for live games
        )
    
    async def get_playoffs_bracket(
        self,
        season: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get NBA playoffs bracket.
        
        Args:
            season: Season (e.g., "2025-26")
            
        Returns:
            Playoffs bracket and matchups
        """
        # Get playoffs standings
        standings = await self.get_standings(season)
        
        # Filter for playoff teams (top 8 from each conference or play-in)
        playoff_teams = {
            "eastern": [],
            "western": []
        }
        
        if "children" in standings:
            for conference in standings["children"]:
                conf_name = conference.get("name", "").lower()
                if "eastern" in conf_name:
                    # Get top 10 teams (including play-in)
                    for division in conference.get("children", []):
                        for entry in division.get("standings", {}).get("entries", []):
                            if len(playoff_teams["eastern"]) < 10:
                                playoff_teams["eastern"].append(entry)
                elif "western" in conf_name:
                    # Get top 10 teams (including play-in)
                    for division in conference.get("children", []):
                        for entry in division.get("standings", {}).get("entries", []):
                            if len(playoff_teams["western"]) < 10:
                                playoff_teams["western"].append(entry)
        
        return {
            "playoffs": playoff_teams,
            "standings": standings
        }
    
    async def get_all_teams(self) -> Dict[str, Any]:
        """
        Get all NBA teams.
        
        Returns:
            List of all 30 NBA teams with details
        """
        return await super().get_teams()
    
    async def get_team(self, team_id: str) -> Dict[str, Any]:
        """
        Get specific team information.
        
        Args:
            team_id: Team ID or abbreviation (e.g., "LAL", "13")
            
        Returns:
            Team information including roster and stats
        """
        return await super().get_team(team_id)
    
    async def get_news(
        self,
        team_id: Optional[str] = None,
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        Get NBA news.
        
        Args:
            team_id: Optional team ID to filter news
            limit: Maximum number of articles
            
        Returns:
            Latest NBA news articles
        """
        return await super().get_news(team_id, limit)
    
    async def get_transactions(
        self,
        team_id: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get recent transactions (trades, signings, waivers).
        
        Args:
            team_id: Optional team ID to filter
            days: Number of days to look back
            
        Returns:
            Recent transaction data
        """
        # This endpoint might not be directly available
        # Could be parsed from news or a specific transactions endpoint
        endpoint = f"{self._base_path}/transactions"
        
        params = {}
        if team_id:
            params["team"] = team_id
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        params["dates"] = f"{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}"
        
        try:
            return await self.get(
                endpoint,
                params=params,
                cache_ttl=3600  # 1 hour for transactions
            )
        except Exception as e:
            logger.warning(f"Transactions endpoint not available: {e}")
            return {"error": "Transactions data not available"}
    
    async def get_injuries(self, team_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get injury report.
        
        Args:
            team_id: Optional team ID to filter
            
        Returns:
            Current injury report
        """
        if team_id:
            # Use the core API for team injuries
            base_url = "https://sports.core.api.espn.com"
            endpoint = f"/v2/sports/basketball/leagues/nba/teams/{team_id}/injuries"
        else:
            # Get league-wide injuries
            endpoint = f"{self._base_path}/injuries"
            base_url = None
        
        if base_url:
            # Temporarily change base URL for this request
            original_base = self.config.base_url
            self.config.base_url = base_url
            
            try:
                result = await self.get(
                    endpoint,
                    cache_ttl=300  # 5 minutes for injury data
                )
                return result
            finally:
                self.config.base_url = original_base
        else:
            return await self.get(
                endpoint,
                cache_ttl=300  # 5 minutes for injury data
            )
    
    # Helper methods
    def get_current_season(self) -> str:
        """
        Get current NBA season.
        
        Returns:
            Current season string (e.g., "2025-26")
        """
        now = datetime.now()
        
        # NBA season typically runs October to June
        if now.month >= 10:
            # New season started
            return f"{now.year}-{str(now.year + 1)[2:]}"
        elif now.month <= 6:
            # Still in previous season
            return f"{now.year - 1}-{str(now.year)[2:]}"
        else:
            # Off-season (July-September)
            return f"{now.year}-{str(now.year + 1)[2:]}"
    
    def is_playoffs(self) -> bool:
        """
        Check if currently in playoffs.
        
        Returns:
            True if currently in NBA playoffs period
        """
        now = datetime.now()
        # NBA playoffs typically April-June
        return 4 <= now.month <= 6
    
    def get_team_abbreviations(self) -> Dict[str, str]:
        """
        Get mapping of team abbreviations to full names.
        
        Returns:
            Dictionary of abbreviations to team names
        """
        # Common NBA team abbreviations
        return {
            "ATL": "Atlanta Hawks",
            "BOS": "Boston Celtics",
            "BKN": "Brooklyn Nets",
            "CHA": "Charlotte Hornets",
            "CHI": "Chicago Bulls",
            "CLE": "Cleveland Cavaliers",
            "DAL": "Dallas Mavericks",
            "DEN": "Denver Nuggets",
            "DET": "Detroit Pistons",
            "GSW": "Golden State Warriors",
            "HOU": "Houston Rockets",
            "IND": "Indiana Pacers",
            "LAC": "LA Clippers",
            "LAL": "Los Angeles Lakers",
            "MEM": "Memphis Grizzlies",
            "MIA": "Miami Heat",
            "MIL": "Milwaukee Bucks",
            "MIN": "Minnesota Timberwolves",
            "NOP": "New Orleans Pelicans",
            "NYK": "New York Knicks",
            "OKC": "Oklahoma City Thunder",
            "ORL": "Orlando Magic",
            "PHI": "Philadelphia 76ers",
            "PHX": "Phoenix Suns",
            "POR": "Portland Trail Blazers",
            "SAC": "Sacramento Kings",
            "SAS": "San Antonio Spurs",
            "TOR": "Toronto Raptors",
            "UTA": "Utah Jazz",
            "WAS": "Washington Wizards"
        }