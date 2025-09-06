"""
College Football Market Discovery Service
Discovers college football markets using the Kalshi Events and Markets API
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, date

from .client import KalshiClient

logger = logging.getLogger(__name__)


class CFBMarketDiscovery:
    """Service for discovering College Football markets through the Events → Markets hierarchy"""
    
    # Major conferences
    CONFERENCES = {
        'SEC': ['Alabama', 'Georgia', 'LSU', 'Florida', 'Tennessee', 'Auburn', 'Texas A&M', 
                'Mississippi', 'South Carolina', 'Arkansas', 'Kentucky', 'Missouri', 'Vanderbilt'],
        'Big Ten': ['Ohio State', 'Michigan', 'Penn State', 'Wisconsin', 'Iowa', 'Minnesota',
                    'Michigan State', 'Indiana', 'Northwestern', 'Illinois', 'Purdue', 'Nebraska',
                    'Maryland', 'Rutgers'],
        'ACC': ['Clemson', 'Florida State', 'Miami', 'North Carolina', 'NC State', 'Virginia Tech',
                'Virginia', 'Louisville', 'Pittsburgh', 'Syracuse', 'Boston College', 'Wake Forest',
                'Duke', 'Georgia Tech'],
        'Big 12': ['Oklahoma', 'Texas', 'Oklahoma State', 'Baylor', 'TCU', 'Kansas State',
                   'Iowa State', 'West Virginia', 'Kansas', 'Texas Tech'],
        'Pac-12': ['USC', 'UCLA', 'Oregon', 'Washington', 'Stanford', 'California', 'Utah',
                   'Arizona State', 'Arizona', 'Colorado', 'Washington State', 'Oregon State'],
        'Independent': ['Notre Dame', 'BYU', 'Army', 'Navy', 'Air Force']
    }
    
    def __init__(self, client: Optional[KalshiClient] = None):
        """
        Initialize CFB market discovery service
        
        Args:
            client: Optional KalshiClient instance
        """
        self.client = client or KalshiClient()
        self.cfb_game_series = "KXNCAAFGAME"
        self.cfb_championship_series = "KXNCAAF"
        
    def get_all_cfb_events(self, status: str = "open") -> List[Dict[str, Any]]:
        """
        Get all College Football events
        
        Args:
            status: Event status filter (open, closed, settled)
            
        Returns:
            List of CFB events
        """
        try:
            logger.info(f"Fetching CFB events with status: {status}")
            
            all_events = []
            cursor = None
            
            while True:
                params = {
                    'limit': 200,
                    'status': status,
                    'series_ticker': self.cfb_game_series,
                    'with_nested_markets': True
                }
                
                if cursor:
                    params['cursor'] = cursor
                
                response = self.client.get('/events', params=params)
                events = response.get('events', [])
                all_events.extend(events)
                
                cursor = response.get('cursor')
                if not cursor or len(events) < 200:
                    break
            
            logger.info(f"Found {len(all_events)} CFB events")
            return all_events
            
        except Exception as e:
            logger.error(f"Error fetching CFB events: {e}")
            return []
    
    def get_events_by_date(self, target_date: Optional[date] = None, status: str = "open") -> List[Dict[str, Any]]:
        """
        Get CFB events for a specific date
        
        Args:
            target_date: Date to filter events (None for today)
            status: Event status filter
            
        Returns:
            List of events for the specified date
        """
        if target_date is None:
            target_date = datetime.now().date()
        
        # Get all events
        all_events = self.get_all_cfb_events(status)
        
        # Filter by date
        date_events = []
        next_day = target_date + timedelta(days=1)
        
        for event in all_events:
            # Check expected expiration time
            exp_time_str = event.get('expected_expiration_time')
            if exp_time_str:
                try:
                    exp_time = datetime.fromisoformat(exp_time_str.replace('Z', '+00:00'))
                    event_date = exp_time.date()
                    
                    # Games typically expire shortly after they end
                    # So check if the expiration is on target date or next day
                    if target_date <= event_date <= next_day:
                        date_events.append(event)
                except Exception as e:
                    logger.debug(f"Error parsing date for event: {e}")
        
        logger.info(f"Found {len(date_events)} events for {target_date}")
        return date_events
    
    def get_team_events(self, team_name: str, status: str = "open") -> List[Dict[str, Any]]:
        """
        Get all events for a specific team
        
        Args:
            team_name: Team name to search for
            status: Event status filter
            
        Returns:
            List of events involving the team
        """
        all_events = self.get_all_cfb_events(status)
        
        team_events = []
        team_upper = team_name.upper()
        
        for event in all_events:
            title = event.get('title', '')
            
            # Check if team is in the title
            if team_upper in title.upper():
                team_events.append(event)
                logger.info(f"Found team event: {event.get('ticker')} - {title}")
        
        logger.info(f"Found {len(team_events)} events for team {team_name}")
        return team_events
    
    def get_conference_events(self, conference: str, status: str = "open") -> List[Dict[str, Any]]:
        """
        Get all events for teams in a specific conference
        
        Args:
            conference: Conference name (SEC, Big Ten, ACC, etc.)
            status: Event status filter
            
        Returns:
            List of events for conference teams
        """
        if conference not in self.CONFERENCES:
            logger.warning(f"Unknown conference: {conference}")
            return []
        
        conference_teams = self.CONFERENCES[conference]
        all_events = self.get_all_cfb_events(status)
        
        conference_events = []
        
        for event in all_events:
            title = event.get('title', '').upper()
            
            # Check if any conference team is in the title
            for team in conference_teams:
                if team.upper() in title:
                    conference_events.append(event)
                    break
        
        logger.info(f"Found {len(conference_events)} events for {conference} conference")
        return conference_events
    
    def get_game_markets(self, home_team: str, away_team: str, status: str = "open") -> List[str]:
        """
        Get market tickers for a specific game
        
        Args:
            home_team: Home team name or abbreviation
            away_team: Away team name or abbreviation
            status: Market status filter
            
        Returns:
            List of market tickers for the game
        """
        logger.info(f"Getting markets for {away_team} @ {home_team}")
        
        # Get all events
        all_events = self.get_all_cfb_events(status)
        
        # Find matching game
        home_upper = home_team.upper()
        away_upper = away_team.upper()
        
        for event in all_events:
            title = event.get('title', '').upper()
            
            # Check if both teams are in the title
            if home_upper in title and away_upper in title:
                # Extract market tickers from nested markets
                markets = event.get('markets', [])
                tickers = [m.get('ticker') for m in markets if m.get('ticker')]
                
                logger.info(f"Found {len(tickers)} markets for {away_team} @ {home_team}")
                return tickers
        
        logger.info(f"No markets found for {away_team} @ {home_team}")
        return []
    
    def get_events_with_markets(self, status: str = "open") -> Dict[str, List[str]]:
        """
        Get all events with their associated market tickers
        
        Args:
            status: Event status filter
            
        Returns:
            Dictionary mapping event titles to market tickers
        """
        events = self.get_all_cfb_events(status)
        
        event_markets = {}
        
        for event in events:
            title = event.get('title', 'Unknown')
            markets = event.get('markets', [])
            
            if markets:
                tickers = [m.get('ticker') for m in markets if m.get('ticker')]
                event_markets[title] = tickers
        
        return event_markets
    
    def get_championship_markets(self) -> List[Dict[str, Any]]:
        """
        Get College Football Championship/Playoff markets
        
        Returns:
            List of championship markets
        """
        try:
            logger.info("Fetching CFB Championship markets")
            
            params = {
                'series_ticker': self.cfb_championship_series,
                'status': 'open',
                'limit': 200
            }
            
            response = self.client.get('/markets', params=params)
            markets = response.get('markets', [])
            
            logger.info(f"Found {len(markets)} championship markets")
            return markets
            
        except Exception as e:
            logger.error(f"Error fetching championship markets: {e}")
            return []
    
    def format_game_info(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format event information for display
        
        Args:
            event: Event data
            
        Returns:
            Formatted game information
        """
        markets = event.get('markets', [])
        
        # Extract team names from markets
        teams = set()
        for market in markets:
            if market.get('yes_sub_title'):
                teams.add(market['yes_sub_title'])
        
        # Parse date
        exp_time_str = event.get('expected_expiration_time')
        game_date = None
        if exp_time_str:
            try:
                exp_time = datetime.fromisoformat(exp_time_str.replace('Z', '+00:00'))
                game_date = exp_time.date()
            except:
                pass
        
        return {
            'title': event.get('title', 'Unknown Game'),
            'ticker': event.get('ticker'),
            'teams': list(teams),
            'date': game_date,
            'market_count': len(markets),
            'markets': [m.get('ticker') for m in markets if m.get('ticker')]
        }