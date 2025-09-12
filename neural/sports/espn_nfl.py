"""
ESPN NFL client implementation.

This module provides NFL-specific functionality for fetching data from ESPN's API,
including scores, rosters, schedules, injuries, and more.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import logging

from neural.sports.espn_base import ESPNClient, ESPNConfig


logger = logging.getLogger(__name__)


class ESPNNFLConfig(ESPNConfig):
    """NFL-specific ESPN configuration."""
    
    def __init__(self, **kwargs):
        kwargs.setdefault("name", "espn_nfl")
        kwargs.setdefault("sport", "football")
        kwargs.setdefault("league", "nfl")
        kwargs.setdefault("default_season", 2025)
        super().__init__(**kwargs)


class ESPNNFL(ESPNClient):
    """
    ESPN NFL API client.
    
    Provides methods for fetching NFL-specific data including games, teams,
    rosters, injuries, depth charts, and more.
    
    Example:
        >>> nfl = ESPNNFL()
        >>> scores = await nfl.get_scoreboard(week=10)
        >>> roster = await nfl.get_roster("GB", include_stats=True)
    """
    
    def __init__(self, config: Optional[ESPNNFLConfig] = None):
        """
        Initialize NFL client.
        
        Args:
            config: NFL-specific configuration
        """
        if config is None:
            config = ESPNNFLConfig()
        
        super().__init__(config)
        self.config: ESPNNFLConfig = config
    
    async def get_scoreboard(
        self,
        dates: Optional[str] = None,
        week: Optional[int] = None,
        seasontype: Optional[int] = None,
        year: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get NFL scoreboard data.
        
        Args:
            dates: Date in YYYYMMDD format or date range
            week: Week number (1-18 for regular season)
            seasontype: Season type (1=preseason, 2=regular, 3=postseason)
            year: Season year (defaults to current season)
            
        Returns:
            Scoreboard data with games and scores
            
        Example:
            >>> # Get current week
            >>> scores = await nfl.get_scoreboard()
            >>> 
            >>> # Get specific week
            >>> scores = await nfl.get_scoreboard(week=10, seasontype=2)
            >>> 
            >>> # Get specific date
            >>> scores = await nfl.get_scoreboard(dates="20251215")
        """
        params = {}
        
        if dates:
            params["dates"] = dates
        elif week and seasontype:
            # Build date parameter for specific week
            year = year or self.config.default_season
            params["dates"] = str(year)
            params["seasontype"] = seasontype
            params["week"] = week
        elif year:
            params["dates"] = str(year)
            if seasontype:
                params["seasontype"] = seasontype
        
        return await super().get_scoreboard(**params)
    
    async def get_team_roster(
        self,
        team_id: str,
        include_stats: bool = False,
        include_projections: bool = False
    ) -> Dict[str, Any]:
        """
        Get detailed team roster with optional stats and projections.
        
        Args:
            team_id: Team ID or abbreviation (e.g., "GB" for Packers)
            include_stats: Include player statistics
            include_projections: Include player projections
            
        Returns:
            Team roster with player details
        """
        if include_stats or include_projections:
            endpoint = f"{self._base_path}/teams/{team_id}"
            
            enable_params = ["roster"]
            if include_stats:
                enable_params.append("stats")
            if include_projections:
                enable_params.append("projection")
            
            params = {"enable": ",".join(enable_params)}
            
            return await self.get(
                endpoint,
                params=params,
                cache_ttl=self.config.cache_roster_ttl
            )
        else:
            return await super().get_roster(team_id)
    
    async def get_depth_chart(
        self,
        team_id: str,
        season: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get team depth chart.
        
        Args:
            team_id: Team ID or abbreviation
            season: Season year (defaults to current)
            
        Returns:
            Depth chart data showing player positions and rankings
        """
        season = season or self.config.default_season
        
        # Use the core API for depth charts
        base_url = "https://sports.core.api.espn.com"
        endpoint = f"/v2/sports/football/leagues/nfl/seasons/{season}/teams/{team_id}/depthcharts"
        
        # Temporarily change base URL for this request
        original_base = self.config.base_url
        self.config.base_url = base_url
        
        try:
            result = await self.get(
                endpoint,
                cache_ttl=self.config.cache_roster_ttl
            )
            return result
        finally:
            self.config.base_url = original_base
    
    async def get_injuries(self, team_id: str) -> Dict[str, Any]:
        """
        Get team injury report.
        
        Args:
            team_id: Team ID or abbreviation
            
        Returns:
            Current injury report for the team
        """
        # Use the core API for injuries
        base_url = "https://sports.core.api.espn.com"
        endpoint = f"/v2/sports/football/leagues/nfl/teams/{team_id}/injuries"
        
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
    
    async def get_team_statistics(
        self,
        team_id: str,
        season: Optional[int] = None,
        seasontype: int = 2
    ) -> Dict[str, Any]:
        """
        Get team statistics.
        
        Args:
            team_id: Team ID or abbreviation
            season: Season year (defaults to current)
            seasontype: Season type (1=preseason, 2=regular, 3=postseason)
            
        Returns:
            Team statistics for the specified season
        """
        season = season or self.config.default_season
        
        # Use the core API for statistics
        base_url = "https://sports.core.api.espn.com"
        endpoint = f"/v2/sports/football/leagues/nfl/seasons/{season}/types/{seasontype}/teams/{team_id}/statistics"
        
        # Temporarily change base URL for this request
        original_base = self.config.base_url
        self.config.base_url = base_url
        
        try:
            result = await self.get(
                endpoint,
                cache_ttl=self.config.cache_schedule_ttl
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
    
    async def get_team_schedule(
        self,
        team_id: str,
        season: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get team schedule for a season.
        
        Args:
            team_id: Team ID or abbreviation
            season: Season year (defaults to current)
            
        Returns:
            Complete team schedule with game details
        """
        return await super().get_schedule(team_id, season)
    
    async def get_all_teams(self) -> Dict[str, Any]:
        """
        Get all NFL teams.
        
        Returns:
            List of all 32 NFL teams with details
        """
        return await super().get_teams()
    
    async def get_news(
        self,
        team_id: Optional[str] = None,
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        Get NFL news.
        
        Args:
            team_id: Optional team ID to filter news
            limit: Maximum number of articles
            
        Returns:
            Latest NFL news articles
        """
        return await super().get_news(team_id, limit)
    
    async def get_standings(
        self,
        season: Optional[int] = None,
        seasontype: int = 2
    ) -> Dict[str, Any]:
        """
        Get NFL standings.
        
        Args:
            season: Season year (defaults to current)
            seasontype: Season type (1=preseason, 2=regular, 3=postseason)
            
        Returns:
            Current NFL standings by division
        """
        endpoint = f"{self._base_path}/standings"
        
        params = {}
        if season:
            params["season"] = season
        else:
            params["season"] = self.config.default_season
        
        params["seasontype"] = seasontype
        
        return await self.get(
            endpoint,
            params=params,
            cache_ttl=self.config.cache_schedule_ttl
        )
    
    async def get_playoff_picture(
        self,
        season: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get current playoff picture.
        
        Args:
            season: Season year (defaults to current)
            
        Returns:
            Playoff standings and scenarios
        """
        # This is typically part of standings with additional processing
        standings = await self.get_standings(season, seasontype=2)
        
        # Extract playoff-relevant information
        if "children" in standings:
            playoff_info = {
                "afc": {},
                "nfc": {},
                "wildcard": {"afc": [], "nfc": []}
            }
            
            for conference in standings.get("children", []):
                conf_name = conference.get("abbreviation", "").lower()
                if conf_name in ["afc", "nfc"]:
                    # Process division winners and wildcard teams
                    for division in conference.get("children", []):
                        division_standings = division.get("standings", {}).get("entries", [])
                        if division_standings:
                            # Division winner
                            playoff_info[conf_name][division["name"]] = division_standings[0]
                            
                            # Potential wildcard teams
                            for team in division_standings[1:]:
                                if len(playoff_info["wildcard"][conf_name]) < 3:
                                    playoff_info["wildcard"][conf_name].append(team)
            
            standings["playoff_picture"] = playoff_info
        
        return standings
    
    # Convenience methods for common team abbreviations
    async def get_team_by_city(self, city: str) -> Dict[str, Any]:
        """
        Get team by city name.
        
        Args:
            city: City name (e.g., "Green Bay", "Dallas")
            
        Returns:
            Team information
        """
        # This would require a mapping of cities to team IDs
        # For now, pass through to get_team
        return await self.get_team(city)
    
    def get_current_week(self) -> int:
        """
        Calculate current NFL week based on date.
        
        Returns:
            Current week number
        """
        # NFL season typically starts first week of September
        # This is a simplified calculation
        now = datetime.now()
        
        # Rough calculation (would need refinement for production)
        if now.month < 9:
            # Off-season or preseason
            return 0
        elif now.month == 9:
            # Weeks 1-4
            return (now.day // 7) + 1
        elif now.month == 10:
            # Weeks 5-8
            return 4 + (now.day // 7) + 1
        elif now.month == 11:
            # Weeks 9-12
            return 8 + (now.day // 7) + 1
        elif now.month == 12:
            # Weeks 13-16
            return 12 + (now.day // 7) + 1
        elif now.month == 1:
            # Weeks 17-18 and playoffs
            return 17 + (now.day // 14)
        else:
            # Post-season
            return 0