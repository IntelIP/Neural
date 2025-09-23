"""
Base ESPN client implementation using Neural SDK.

This module provides the foundation for all ESPN sports data clients,
leveraging the Neural SDK's Data Collection Infrastructure.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import logging

from neural.data_collection import (
    RestDataSource,
    RestConfig,
    TransformStage,
    DataPipeline
)


logger = logging.getLogger(__name__)


@dataclass
class ESPNConfig(RestConfig):
    """
    Configuration for ESPN API clients.
    
    Attributes:
        sport: Sport type (football, basketball, etc.)
        league: League identifier (nfl, college-football, nba, etc.)
        default_season: Default season/year for queries
        cache_scoreboard_ttl: TTL for scoreboard data (seconds)
        cache_roster_ttl: TTL for roster data (seconds)
        cache_schedule_ttl: TTL for schedule data (seconds)
    """
    sport: str = "football"
    league: str = "nfl"
    default_season: Union[int, str] = 2025
    cache_scoreboard_ttl: int = 30  # 30 seconds for live data
    cache_roster_ttl: int = 3600    # 1 hour for roster data
    cache_schedule_ttl: int = 86400  # 24 hours for schedule data
    
    def __post_init__(self):
        """Set default base URL and name if not provided."""
        # Set default name based on sport/league
        if not self.name:
            self.name = f"espn_{self.sport}_{self.league}"
        
        if not self.base_url:
            self.base_url = "https://site.api.espn.com/apis/site/v2/sports/"
        
        # Set conservative rate limit if not specified
        # Note: RestConfig uses rate_limit_requests (requests per second)
        if not hasattr(self, 'rate_limit_requests') or self.rate_limit_requests == 10.0:
            self.rate_limit_requests = 1.67  # ~100 requests per minute


class ESPNDataNormalizer(TransformStage):
    """
    Transform stage to normalize ESPN API responses.
    
    This stage standardizes the data format across different ESPN endpoints
    to provide a consistent interface for downstream processing.
    """
    
    async def process(self, data: Any) -> Any:
        """
        Normalize ESPN data format.
        
        Args:
            data: Raw ESPN API response
            
        Returns:
            Normalized data structure
        """
        if not isinstance(data, dict):
            return data
        
        # Normalize common fields
        normalized = {
            "timestamp": datetime.utcnow().isoformat(),
            "source": "espn",
            "data": data
        }
        
        # Extract and normalize specific data types
        if "events" in data:
            # Scoreboard data
            normalized["type"] = "scoreboard"
            normalized["events"] = self._normalize_events(data["events"])
        elif "team" in data:
            # Team data
            normalized["type"] = "team"
            normalized["team"] = self._normalize_team(data["team"])
        elif "athletes" in data:
            # Roster data
            normalized["type"] = "roster"
            normalized["athletes"] = self._normalize_athletes(data["athletes"])
        elif "items" in data:
            # News data
            normalized["type"] = "news"
            normalized["items"] = data["items"]
        
        return normalized
    
    def _normalize_events(self, events: List[Dict]) -> List[Dict]:
        """Normalize event/game data."""
        normalized_events = []
        
        for event in events:
            normalized_event = {
                "id": event.get("id"),
                "date": event.get("date"),
                "name": event.get("name"),
                "shortName": event.get("shortName"),
                "status": event.get("status", {}).get("type", {}).get("name"),
                "completed": event.get("status", {}).get("type", {}).get("completed", False)
            }
            
            # Extract competitions
            if "competitions" in event and event["competitions"]:
                competition = event["competitions"][0]
                normalized_event["competition"] = {
                    "id": competition.get("id"),
                    "venue": competition.get("venue", {}).get("fullName"),
                    "attendance": competition.get("attendance"),
                    "broadcast": competition.get("broadcasts", [{}])[0].get("names", []) if competition.get("broadcasts") else []
                }
                
                # Extract competitors/teams
                if "competitors" in competition:
                    normalized_event["competitors"] = []
                    for competitor in competition["competitors"]:
                        team_data = {
                            "id": competitor.get("id"),
                            "homeAway": competitor.get("homeAway"),
                            "winner": competitor.get("winner"),
                            "score": competitor.get("score"),
                            "team": {
                                "id": competitor.get("team", {}).get("id"),
                                "name": competitor.get("team", {}).get("name"),
                                "abbreviation": competitor.get("team", {}).get("abbreviation"),
                                "displayName": competitor.get("team", {}).get("displayName"),
                                "logo": competitor.get("team", {}).get("logo")
                            }
                        }
                        
                        # Add records if available
                        if "records" in competitor:
                            team_data["records"] = competitor["records"]
                        
                        normalized_event["competitors"].append(team_data)
            
            normalized_events.append(normalized_event)
        
        return normalized_events
    
    def _normalize_team(self, team: Dict) -> Dict:
        """Normalize team data."""
        return {
            "id": team.get("id"),
            "abbreviation": team.get("abbreviation"),
            "displayName": team.get("displayName"),
            "shortDisplayName": team.get("shortDisplayName"),
            "name": team.get("name"),
            "nickname": team.get("nickname"),
            "location": team.get("location"),
            "color": team.get("color"),
            "alternateColor": team.get("alternateColor"),
            "logos": team.get("logos", []),
            "links": team.get("links", [])
        }
    
    def _normalize_athletes(self, athletes: List[Dict]) -> List[Dict]:
        """Normalize athlete/player data."""
        normalized_athletes = []
        
        for athlete in athletes:
            normalized_athlete = {
                "id": athlete.get("id"),
                "firstName": athlete.get("firstName"),
                "lastName": athlete.get("lastName"),
                "fullName": athlete.get("fullName"),
                "displayName": athlete.get("displayName"),
                "jersey": athlete.get("jersey"),
                "position": athlete.get("position", {}).get("abbreviation"),
                "positionName": athlete.get("position", {}).get("displayName"),
                "height": athlete.get("height"),
                "weight": athlete.get("weight"),
                "age": athlete.get("age"),
                "birthDate": athlete.get("dateOfBirth"),
                "headshot": athlete.get("headshot", {}).get("href") if athlete.get("headshot") else None
            }
            normalized_athletes.append(normalized_athlete)
        
        return normalized_athletes


class ESPNClient(RestDataSource):
    """
    Base ESPN API client.
    
    This class provides common functionality for all ESPN sports clients,
    built on top of the Neural SDK's RestDataSource.
    """
    
    def __init__(self, config: Optional[ESPNConfig] = None):
        """
        Initialize ESPN client.
        
        Args:
            config: ESPN-specific configuration
        """
        if config is None:
            config = ESPNConfig(name="espn_client")
        
        super().__init__(config)
        self.config: ESPNConfig = config
        
        # Build base path for this sport/league (no leading slash since base_url has trailing slash)
        self._base_path = f"{self.config.sport}/{self.config.league}"
        
        logger.info(
            f"Initialized ESPN client for {self.config.sport}/{self.config.league}"
        )
    
    async def get_scoreboard(
        self,
        dates: Optional[str] = None,
        limit: int = 1000,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Get scoreboard data.
        
        Args:
            dates: Date in YYYYMMDD format or date range YYYYMMDD-YYYYMMDD
            limit: Maximum number of results
            **kwargs: Additional query parameters
            
        Returns:
            Scoreboard data
        """
        endpoint = f"{self._base_path}/scoreboard"
        
        params = {"limit": limit}
        if dates:
            params["dates"] = dates
        params.update(kwargs)
        
        return await self.get(
            endpoint,
            params=params,
            cache_ttl=self.config.cache_scoreboard_ttl
        )
    
    async def get_teams(self) -> Dict[str, Any]:
        """
        Get all teams.
        
        Returns:
            List of all teams in the league
        """
        endpoint = f"{self._base_path}/teams"
        
        return await self.get(
            endpoint,
            cache_ttl=self.config.cache_roster_ttl
        )
    
    async def get_team(self, team_id: str) -> Dict[str, Any]:
        """
        Get specific team information.
        
        Args:
            team_id: Team ID or abbreviation
            
        Returns:
            Team information
        """
        endpoint = f"{self._base_path}/teams/{team_id}"
        
        return await self.get(
            endpoint,
            cache_ttl=self.config.cache_roster_ttl
        )
    
    async def get_roster(
        self,
        team_id: str,
        enable_stats: bool = False
    ) -> Dict[str, Any]:
        """
        Get team roster.
        
        Args:
            team_id: Team ID or abbreviation
            enable_stats: Include player statistics
            
        Returns:
            Team roster data
        """
        endpoint = f"{self._base_path}/teams/{team_id}/roster"
        
        params = {}
        if enable_stats:
            # For detailed roster with stats, use different endpoint
            endpoint = f"{self._base_path}/teams/{team_id}"
            params["enable"] = "roster,stats"
        
        return await self.get(
            endpoint,
            params=params if params else None,
            cache_ttl=self.config.cache_roster_ttl
        )
    
    async def get_schedule(
        self,
        team_id: str,
        season: Optional[Union[int, str]] = None
    ) -> Dict[str, Any]:
        """
        Get team schedule.
        
        Args:
            team_id: Team ID or abbreviation
            season: Season year (defaults to current season)
            
        Returns:
            Team schedule data
        """
        endpoint = f"{self._base_path}/teams/{team_id}/schedule"
        
        params = {}
        if season:
            params["season"] = season
        elif self.config.default_season:
            params["season"] = self.config.default_season
        
        return await self.get(
            endpoint,
            params=params if params else None,
            cache_ttl=self.config.cache_schedule_ttl
        )
    
    async def get_news(
        self,
        team_id: Optional[str] = None,
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        Get news articles.
        
        Args:
            team_id: Optional team ID to filter news
            limit: Maximum number of articles
            
        Returns:
            News articles data
        """
        if team_id:
            endpoint = f"{self._base_path}/teams/{team_id}/news"
        else:
            endpoint = f"{self._base_path}/news"
        
        params = {"limit": limit}
        
        return await self.get(
            endpoint,
            params=params,
            cache_ttl=300  # 5 minutes for news
        )
    
    async def get_standings(self) -> Dict[str, Any]:
        """
        Get league standings.
        
        Returns:
            Current standings data
        """
        endpoint = f"{self._base_path}/standings"
        
        return await self.get(
            endpoint,
            cache_ttl=self.config.cache_schedule_ttl
        )
    
    def create_pipeline(self) -> DataPipeline:
        """
        Create a data pipeline with ESPN-specific transformations.
        
        Returns:
            Configured DataPipeline instance
        """
        pipeline = DataPipeline()
        
        # Add normalization stage
        pipeline.add_stage(ESPNDataNormalizer())
        
        return pipeline