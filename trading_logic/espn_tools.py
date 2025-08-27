"""
ESPN Free API Tools
Real-time game data for market reaction analysis
No API key required - completely free
"""

import httpx
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio


class ESPNTools:
    """
    Free ESPN API endpoints for real-time sports data.
    Perfect for correlating game events with Kalshi market movements.
    """
    
    BASE_URL = "http://site.api.espn.com/apis/site/v2/sports"
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def get_nfl_scoreboard(self) -> Dict[str, Any]:
        """Get all current NFL games with scores and status."""
        url = f"{self.BASE_URL}/football/nfl/scoreboard"
        response = await self.client.get(url)
        return response.json()
    
    async def get_nba_scoreboard(self) -> Dict[str, Any]:
        """Get all current NBA games with scores and status."""
        url = f"{self.BASE_URL}/basketball/nba/scoreboard"
        response = await self.client.get(url)
        return response.json()
    
    async def get_cfb_scoreboard(self) -> Dict[str, Any]:
        """Get all current College Football games."""
        url = f"{self.BASE_URL}/football/college-football/scoreboard"
        response = await self.client.get(url)
        return response.json()
    
    async def get_game_summary(self, game_id: str, sport: str = "nfl") -> Dict[str, Any]:
        """
        Get detailed game summary including play-by-play and win probability.
        This is the MOST VALUABLE endpoint for Kalshi correlation.
        
        Args:
            game_id: ESPN game ID
            sport: 'nfl', 'nba', or 'college-football'
            
        Returns:
            Complete game data with plays, injuries, win probability
        """
        sport_map = {
            "nfl": "football/nfl",
            "nba": "basketball/nba",
            "cfb": "football/college-football",
            "college-football": "football/college-football"
        }
        
        sport_path = sport_map.get(sport, sport)
        url = f"{self.BASE_URL}/{sport_path}/summary?event={game_id}"
        
        response = await self.client.get(url)
        return response.json()
    
    async def get_win_probability(self, game_id: str, sport: str = "nfl") -> Optional[Dict[str, float]]:
        """
        Extract win probability from game summary.
        Direct correlation to Kalshi market prices!
        """
        summary = await self.get_game_summary(game_id, sport)
        
        if "winprobability" in summary:
            latest = summary["winprobability"][-1] if isinstance(summary["winprobability"], list) else summary["winprobability"]
            return {
                "home_win_pct": latest.get("homeWinPercentage", 0),
                "away_win_pct": latest.get("awayWinPercentage", 0),
                "timestamp": datetime.now().isoformat()
            }
        return None
    
    async def get_game_events(self, game_id: str, sport: str = "nfl") -> List[Dict[str, Any]]:
        """
        Extract market-moving events from game.
        Injuries, touchdowns, turnovers, etc.
        """
        summary = await self.get_game_summary(game_id, sport)
        events = []
        
        # Extract injuries (huge market movers)
        if "injuries" in summary:
            for team_injuries in summary.get("injuries", {}).values():
                for injury in team_injuries:
                    events.append({
                        "type": "injury",
                        "player": injury.get("athlete", {}).get("displayName"),
                        "status": injury.get("status"),
                        "impact": "HIGH",  # Injuries always high impact
                        "timestamp": datetime.now().isoformat()
                    })
        
        # Extract scoring plays
        if "scoringPlays" in summary:
            for play in summary["scoringPlays"]:
                events.append({
                    "type": "scoring",
                    "description": play.get("text"),
                    "impact": "MEDIUM",
                    "timestamp": play.get("wallclock")
                })
        
        # Extract big plays from play-by-play
        if "drives" in summary:
            for drive in summary.get("drives", {}).get("previous", []):
                for play in drive.get("plays", []):
                    # Turnovers, big gains, etc.
                    if any(word in play.get("text", "").lower() for word in ["intercepted", "fumble", "touchdown"]):
                        events.append({
                            "type": "big_play",
                            "description": play.get("text"),
                            "impact": "HIGH" if "intercepted" in play.get("text", "").lower() else "MEDIUM",
                            "timestamp": play.get("wallclock")
                        })
        
        return events
    
    async def monitor_games_for_events(self, sport: str = "nfl", interval: int = 30) -> None:
        """
        Monitor all active games for market-moving events.
        Perfect for real-time Kalshi correlation.
        """
        while True:
            scoreboard = await self.get_nfl_scoreboard() if sport == "nfl" else await self.get_nba_scoreboard()
            
            for game in scoreboard.get("events", []):
                game_id = game["id"]
                status = game.get("status", {}).get("type", {}).get("state")
                
                if status == "in":  # Game is live
                    events = await self.get_game_events(game_id, sport)
                    win_prob = await self.get_win_probability(game_id, sport)
                    
                    # This is where you'd correlate with Kalshi prices
                    if events or win_prob:
                        yield {
                            "game_id": game_id,
                            "events": events,
                            "win_probability": win_prob,
                            "teams": f"{game['competitions'][0]['competitors'][0]['team']['displayName']} vs {game['competitions'][0]['competitors'][1]['team']['displayName']}"
                        }
            
            await asyncio.sleep(interval)
    
    async def close(self):
        """Clean up HTTP client."""
        await self.client.aclose()


# Example usage for Kalshi correlation
async def correlate_with_kalshi():
    """
    Example of how to use ESPN data with Kalshi markets.
    """
    espn = ESPNTools()
    
    # Get current NFL games
    scoreboard = await espn.get_nfl_scoreboard()
    
    for game in scoreboard.get("events", []):
        game_id = game["id"]
        
        # Get win probability (correlates with Kalshi prices)
        win_prob = await espn.get_win_probability(game_id)
        
        # Get market-moving events
        events = await espn.get_game_events(game_id)
        
        # Find injuries (biggest market movers)
        injuries = [e for e in events if e["type"] == "injury"]
        
        if injuries:
            print(f"MARKET ALERT: Injury detected!")
            print(f"Current win probability: {win_prob}")
            print(f"Check Kalshi market for arbitrage opportunity")
    
    await espn.close()