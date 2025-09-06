"""
NFL Market Discovery Service
Discovers actual NFL markets using the Kalshi Events and Markets API
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from .client import KalshiClient

logger = logging.getLogger(__name__)


class NFLMarketDiscovery:
    """Service for discovering NFL markets through the Events → Markets hierarchy"""
    
    def __init__(self, client: Optional[KalshiClient] = None):
        """
        Initialize NFL market discovery service
        
        Args:
            client: Optional KalshiClient instance
        """
        self.client = client or KalshiClient()
        self.nfl_series_ticker = "KXNFLGAME"
        
    def get_all_nfl_events(self, status: str = "open") -> List[Dict[str, Any]]:
        """
        Get all NFL events
        
        Args:
            status: Event status filter (open, closed, settled)
            
        Returns:
            List of NFL events
        """
        try:
            logger.info(f"Fetching NFL events with status: {status}")
            
            all_events = []
            cursor = None
            
            while True:
                params = {
                    'limit': 200,
                    'status': status,
                    'series_ticker': self.nfl_series_ticker,
                    'with_nested_markets': True  # Include markets in response
                }
                
                if cursor:
                    params['cursor'] = cursor
                
                response = self.client.get('/events', params=params)
                events = response.get('events', [])
                all_events.extend(events)
                
                cursor = response.get('cursor')
                if not cursor:
                    break
            
            logger.info(f"Found {len(all_events)} NFL events")
            return all_events
            
        except Exception as e:
            logger.error(f"Failed to get NFL events: {e}")
            return []
    
    def get_team_game_events(self, team_code: str, status: str = "open") -> List[Dict[str, Any]]:
        """
        Get events for a specific NFL team
        
        Args:
            team_code: Team code (e.g., 'KC', 'LAC', 'PHI')
            status: Event status filter
            
        Returns:
            List of events involving the team
        """
        all_events = self.get_all_nfl_events(status)
        team_events = []
        
        team_code_upper = team_code.upper()
        
        for event in all_events:
            event_ticker = event.get('event_ticker', '')
            title = event.get('title', '')
            
            # Check if team is in event ticker or title
            if team_code_upper in event_ticker.upper() or team_code_upper in title.upper():
                team_events.append(event)
                logger.info(f"Found team event: {event_ticker} - {title}")
        
        logger.info(f"Found {len(team_events)} events for team {team_code}")
        return team_events
    
    def get_event_markets(self, event_ticker: str) -> List[str]:
        """
        Get all market tickers for a specific event
        
        Args:
            event_ticker: Event ticker to get markets for
            
        Returns:
            List of market tickers
        """
        try:
            logger.info(f"Fetching markets for event: {event_ticker}")
            
            response = self.client.get_markets(
                event_ticker=event_ticker,
                status='open',
                limit=200
            )
            
            markets = response.get('markets', [])
            market_tickers = [market.get('ticker') for market in markets if market.get('ticker')]
            
            logger.info(f"Found {len(market_tickers)} markets for event {event_ticker}")
            for ticker in market_tickers[:5]:  # Log first 5
                logger.debug(f"  - {ticker}")
            
            return market_tickers
            
        except Exception as e:
            logger.error(f"Failed to get markets for event {event_ticker}: {e}")
            return []
    
    def get_team_markets(self, team_code: str, status: str = "open") -> List[str]:
        """
        Complete flow to get all market tickers for a team
        
        Args:
            team_code: Team code (e.g., 'KC', 'LAC')
            status: Status filter for events/markets
            
        Returns:
            List of exact market tickers
        """
        logger.info(f"Getting all markets for team: {team_code}")
        
        # Step 1: Get events for the team
        team_events = self.get_team_game_events(team_code, status)
        
        if not team_events:
            logger.warning(f"No events found for team {team_code}")
            return []
        
        # Step 2: Get markets for each event
        all_market_tickers = []
        
        for event in team_events:
            event_ticker = event.get('event_ticker')
            
            # Check if markets are already included (with_nested_markets=True)
            if 'markets' in event:
                markets = event.get('markets', [])
                tickers = [m.get('ticker') for m in markets if m.get('ticker')]
                all_market_tickers.extend(tickers)
                logger.info(f"Event {event_ticker} has {len(tickers)} nested markets")
            else:
                # Fetch markets separately
                tickers = self.get_event_markets(event_ticker)
                all_market_tickers.extend(tickers)
        
        # Remove duplicates
        unique_tickers = list(set(all_market_tickers))
        
        logger.info(f"Found {len(unique_tickers)} unique markets for team {team_code}")
        return unique_tickers
    
    def get_game_markets(self, home_team: str, away_team: str, status: str = "open") -> List[str]:
        """
        Get markets for a specific game between two teams
        
        Args:
            home_team: Home team code
            away_team: Away team code
            status: Market status filter
            
        Returns:
            List of market tickers for the game
        """
        logger.info(f"Getting markets for {away_team} @ {home_team}")
        
        # Get markets for both teams
        home_markets = set(self.get_team_markets(home_team, status))
        away_markets = set(self.get_team_markets(away_team, status))
        
        # Find intersection (markets that involve both teams)
        game_markets = list(home_markets & away_markets)
        
        if not game_markets:
            logger.warning(f"No specific game markets found for {away_team} @ {home_team}")
            # Return union instead if no intersection
            game_markets = list(home_markets | away_markets)
        
        logger.info(f"Found {len(game_markets)} markets for {away_team} @ {home_team}")
        return game_markets
    
    def discover_all_nfl_markets(self, status: str = "open") -> Dict[str, List[str]]:
        """
        Discover all NFL markets organized by event
        
        Args:
            status: Market status filter
            
        Returns:
            Dictionary mapping event_ticker to list of market tickers
        """
        logger.info("Discovering all NFL markets...")
        
        events = self.get_all_nfl_events(status)
        markets_by_event = {}
        
        for event in events:
            event_ticker = event.get('event_ticker')
            
            if 'markets' in event:
                # Markets included in response
                markets = event.get('markets', [])
                tickers = [m.get('ticker') for m in markets if m.get('ticker')]
            else:
                # Fetch markets separately
                tickers = self.get_event_markets(event_ticker)
            
            if tickers:
                markets_by_event[event_ticker] = tickers
        
        total_markets = sum(len(tickers) for tickers in markets_by_event.values())
        logger.info(f"Discovered {total_markets} total markets across {len(markets_by_event)} events")
        
        return markets_by_event
    
    def search_markets_by_pattern(self, pattern: str) -> List[str]:
        """
        Search for markets containing a pattern (for backwards compatibility)
        Note: This fetches all markets first, so it's less efficient than team-specific methods
        
        Args:
            pattern: Pattern to search for in market tickers
            
        Returns:
            List of matching market tickers
        """
        logger.info(f"Searching for markets matching pattern: {pattern}")
        
        all_markets = self.discover_all_nfl_markets()
        matching_tickers = []
        
        pattern_upper = pattern.upper()
        
        for event_ticker, market_tickers in all_markets.items():
            for ticker in market_tickers:
                if pattern_upper in ticker.upper():
                    matching_tickers.append(ticker)
        
        logger.info(f"Found {len(matching_tickers)} markets matching pattern {pattern}")
        return matching_tickers