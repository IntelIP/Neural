"""
ESPN API Client
Enhanced client for fetching play-by-play data with retry logic
"""

import logging
from typing import Dict, Any, List
from datetime import datetime
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

from .processor import PlayByPlayProcessor

logger = logging.getLogger(__name__)


class ESPNClient:
    """Enhanced ESPN API client with play-by-play focus"""
    
    BASE_URL = "http://site.api.espn.com/apis/site/v2/sports"
    
    # Sport path mappings
    SPORT_PATHS = {
        "nfl": "football/nfl",
        "nba": "basketball/nba",
        "cfb": "football/college-football",
        "college-football": "football/college-football",
        "mlb": "baseball/mlb",
        "nhl": "hockey/nhl"
    }
    
    def __init__(self, timeout: int = 30):
        """
        Initialize ESPN client
        
        Args:
            timeout: Request timeout in seconds
        """
        self.client = httpx.AsyncClient(timeout=timeout)
        self.processor = PlayByPlayProcessor()
        self._cache = {}
        self._cache_ttl = 5  # Cache TTL in seconds
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError))
    )
    async def _make_request(self, url: str) -> Dict[str, Any]:
        """
        Make HTTP request with retry logic
        
        Args:
            url: Full URL to request
            
        Returns:
            JSON response data
        """
        # Check cache
        cache_key = url
        if cache_key in self._cache:
            cached_data, cached_time = self._cache[cache_key]
            if (datetime.now() - cached_time).total_seconds() < self._cache_ttl:
                return cached_data
        
        # Make request
        response = await self.client.get(url)
        response.raise_for_status()
        data = response.json()
        
        # Update cache
        self._cache[cache_key] = (data, datetime.now())
        
        return data
    
    async def get_scoreboard(self, sport: str = "nfl") -> Dict[str, Any]:
        """
        Get current scoreboard for a sport
        
        Args:
            sport: Sport identifier
            
        Returns:
            Scoreboard data with all current games
        """
        sport_path = self.SPORT_PATHS.get(sport, sport)
        url = f"{self.BASE_URL}/{sport_path}/scoreboard"
        return await self._make_request(url)
    
    async def get_game_summary(self, game_id: str, sport: str = "nfl") -> Dict[str, Any]:
        """
        Get detailed game summary with play-by-play
        
        Args:
            game_id: ESPN game ID
            sport: Sport identifier
            
        Returns:
            Complete game summary including drives and plays
        """
        sport_path = self.SPORT_PATHS.get(sport, sport)
        url = f"{self.BASE_URL}/{sport_path}/summary?event={game_id}"
        return await self._make_request(url)
    
    async def get_play_by_play(self, game_id: str, sport: str = "nfl") -> Dict[str, Any]:
        """
        Get structured play-by-play data
        
        Args:
            game_id: ESPN game ID
            sport: Sport identifier
            
        Returns:
            Structured play-by-play data
        """
        summary = await self.get_game_summary(game_id, sport)
        
        # Extract game state
        game_state = self.processor.extract_game_state(summary)
        
        # Process drives
        all_drives = []
        drives_data = summary.get('drives', {})
        
        # Process previous drives
        for drive_data in drives_data.get('previous', []):
            drive = self.processor.process_drive(drive_data, game_state)
            all_drives.append(drive)
        
        # Process current drive
        current_drive = drives_data.get('current')
        if current_drive:
            drive = self.processor.process_drive(current_drive, game_state)
            all_drives.append(drive)
        
        # Extract all plays
        all_plays = []
        for drive in all_drives:
            all_plays.extend(drive.plays)
        
        return {
            'game_state': game_state,
            'drives': all_drives,
            'plays': all_plays,
            'total_plays': len(all_plays),
            'scoring_plays': [p for p in all_plays if p.is_scoring()],
            'high_impact_plays': [p for p in all_plays if p.is_high_impact()]
        }
    
    async def get_live_games(self, sport: str = "nfl") -> List[Dict[str, Any]]:
        """
        Get all currently live games
        
        Args:
            sport: Sport identifier
            
        Returns:
            List of live games with basic info
        """
        scoreboard = await self.get_scoreboard(sport)
        live_games = []
        
        for event in scoreboard.get('events', []):
            status = event.get('status', {}).get('type', {}).get('state')
            if status == 'in':  # Game is live
                game_info = {
                    'id': event['id'],
                    'name': event.get('name', ''),
                    'shortName': event.get('shortName', ''),
                    'status': status,
                    'period': event.get('status', {}).get('period', 0),
                    'clock': event.get('status', {}).get('displayClock', ''),
                    'competitors': []
                }
                
                # Extract competitor info
                for competitor in event.get('competitions', [{}])[0].get('competitors', []):
                    game_info['competitors'].append({
                        'team': competitor.get('team', {}).get('displayName', ''),
                        'score': competitor.get('score', 0),
                        'homeAway': competitor.get('homeAway', '')
                    })
                
                live_games.append(game_info)
        
        return live_games
    
    async def get_win_probability_history(self, game_id: str, sport: str = "nfl") -> List[Dict[str, float]]:
        """
        Get win probability history for a game
        
        Args:
            game_id: ESPN game ID
            sport: Sport identifier
            
        Returns:
            List of win probability data points
        """
        summary = await self.get_game_summary(game_id, sport)
        win_prob_data = summary.get('winprobability', [])
        
        history = []
        for data_point in win_prob_data:
            history.append({
                'play_id': data_point.get('playId'),
                'home_win_percentage': data_point.get('homeWinPercentage', 0),
                'away_win_percentage': 100 - data_point.get('homeWinPercentage', 0),
                'play_text': data_point.get('text', '')
            })
        
        return history
    
    async def get_injuries(self, game_id: str, sport: str = "nfl") -> List[Dict[str, Any]]:
        """
        Get injury information for a game
        
        Args:
            game_id: ESPN game ID
            sport: Sport identifier
            
        Returns:
            List of injuries with player and status info
        """
        summary = await self.get_game_summary(game_id, sport)
        injuries = []
        
        for team_injuries in summary.get('injuries', {}).values():
            for injury in team_injuries:
                injuries.append({
                    'player': injury.get('athlete', {}).get('displayName', 'Unknown'),
                    'position': injury.get('athlete', {}).get('position', {}).get('abbreviation', ''),
                    'status': injury.get('status', 'Unknown'),
                    'description': injury.get('description', ''),
                    'team': injury.get('team', {}).get('displayName', '')
                })
        
        return injuries
    
    async def get_scoring_plays(self, game_id: str, sport: str = "nfl") -> List[Dict[str, Any]]:
        """
        Get all scoring plays from a game
        
        Args:
            game_id: ESPN game ID
            sport: Sport identifier
            
        Returns:
            List of scoring plays with details
        """
        summary = await self.get_game_summary(game_id, sport)
        scoring_plays = []
        
        for play in summary.get('scoringPlays', []):
            scoring_plays.append({
                'id': play.get('id'),
                'text': play.get('text', ''),
                'quarter': play.get('period', {}).get('number', 0),
                'clock': play.get('clock', {}).get('displayValue', ''),
                'team': play.get('team', {}).get('displayName', ''),
                'score_value': play.get('scoreValue', 0),
                'home_score': play.get('homeScore', 0),
                'away_score': play.get('awayScore', 0)
            })
        
        return scoring_plays
    
    async def get_game_leaders(self, game_id: str, sport: str = "nfl") -> Dict[str, Any]:
        """
        Get statistical leaders for a game
        
        Args:
            game_id: ESPN game ID
            sport: Sport identifier
            
        Returns:
            Dictionary of game leaders by category
        """
        summary = await self.get_game_summary(game_id, sport)
        leaders = summary.get('leaders', [])
        
        game_leaders = {}
        for leader_category in leaders:
            category = leader_category.get('name', '')
            game_leaders[category] = []
            
            for leader in leader_category.get('leaders', []):
                game_leaders[category].append({
                    'player': leader.get('athlete', {}).get('displayName', ''),
                    'team': leader.get('team', {}).get('displayName', ''),
                    'value': leader.get('value', 0),
                    'displayValue': leader.get('displayValue', '')
                })
        
        return game_leaders
    
    async def get_game_odds(self, game_id: str, sport: str = "nfl") -> Dict[str, Any]:
        """
        Get betting odds for a game
        
        Args:
            game_id: ESPN game ID
            sport: Sport identifier
            
        Returns:
            Current odds information
        """
        summary = await self.get_game_summary(game_id, sport)
        odds_data = summary.get('pickcenter', [])
        
        if odds_data:
            current_odds = odds_data[0]
            return {
                'spread': current_odds.get('spread', 0),
                'over_under': current_odds.get('overUnder', 0),
                'home_money_line': current_odds.get('homeMoneyLine', 0),
                'away_money_line': current_odds.get('awayMoneyLine', 0),
                'provider': current_odds.get('provider', {}).get('name', '')
            }
        
        return {}
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()