"""
DraftKings Sportsbook Adapter
Fetches live odds from DraftKings to identify pricing discrepancies
"""

import aiohttp
import asyncio
import json
from typing import Dict, Any, Optional, AsyncGenerator, List
from datetime import datetime
import logging

from ..core.base_adapter import (
    DataSourceAdapter,
    DataSourceMetadata,
    StandardizedEvent,
    EventType,
    SignalStrength
)

logger = logging.getLogger(__name__)


class DraftKingsAdapter(DataSourceAdapter):
    """
    DraftKings sportsbook odds adapter
    Monitors live odds and line movements for sports events
    """
    
    BASE_URL = "https://sportsbook-us-nh.draftkings.com/sites/US-NH-SB/api/v5"
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DraftKings adapter
        
        Config should include:
        - sports: List of sports to monitor (NFL, NBA, etc.)
        - poll_interval: Seconds between API calls
        - min_odds_change: Minimum odds change to trigger event (e.g., 0.03)
        """
        super().__init__(config)
        self.session: Optional[aiohttp.ClientSession] = None
        self.sports = config.get('sports', ['NFL', 'NBA', 'MLB'])
        self.poll_interval = config.get('poll_interval', 5)
        self.min_odds_change = config.get('min_odds_change', 0.03)
        
        # Track previous odds to detect changes
        self.previous_odds: Dict[str, Dict] = {}
        
    def get_metadata(self) -> DataSourceMetadata:
        """Return DraftKings adapter metadata"""
        return DataSourceMetadata(
            name="DraftKings",
            version="1.0.0",
            author="Neural Trading Platform",
            description="Live odds from DraftKings Sportsbook",
            source_type="sportsbook",
            latency_ms=500,
            reliability=0.95,
            requires_auth=False,
            rate_limits={"requests_per_second": 10},
            supported_sports=["NFL", "NBA", "MLB", "NHL", "NCAAF", "NCAAB"],
            supported_markets=["moneyline", "spread", "total"]
        )
    
    async def connect(self) -> bool:
        """Establish connection to DraftKings API"""
        try:
            # Create HTTP session
            self.session = aiohttp.ClientSession(
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; NeuralTradingPlatform/1.0)"
                }
            )
            
            # Test connection
            async with self.session.get(f"{self.BASE_URL}/eventgroups") as response:
                if response.status == 200:
                    self.logger.info("Connected to DraftKings API")
                    return True
                else:
                    self.logger.error(f"DraftKings API returned status {response.status}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Failed to connect to DraftKings: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Close connection to DraftKings"""
        if self.session:
            await self.session.close()
            self.session = None
        self.logger.info("Disconnected from DraftKings")
    
    async def validate_connection(self) -> bool:
        """Check if connection is still valid"""
        if not self.session:
            return False
        
        try:
            async with self.session.get(
                f"{self.BASE_URL}/eventgroups",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                return response.status == 200
        except:
            return False
    
    async def stream(self) -> AsyncGenerator[StandardizedEvent, None]:
        """Stream odds updates from DraftKings"""
        while self.is_connected:
            try:
                # Fetch current odds for all sports
                for sport in self.sports:
                    events = await self._fetch_sport_events(sport)
                    
                    # Process each event
                    for event in events:
                        # Check for odds changes
                        odds_event = self._check_odds_change(event)
                        if odds_event:
                            yield odds_event
                            self._increment_event_count()
                
                # Wait before next poll
                await asyncio.sleep(self.poll_interval)
                
            except Exception as e:
                self.logger.error(f"Error in DraftKings stream: {e}")
                self._increment_error_count()
                await asyncio.sleep(self.poll_interval)
    
    async def _fetch_sport_events(self, sport: str) -> List[Dict]:
        """
        Fetch live events for a specific sport
        
        Args:
            sport: Sport identifier (NFL, NBA, etc.)
            
        Returns:
            List of event dictionaries
        """
        try:
            # Get event group ID for sport
            group_id = await self._get_event_group_id(sport)
            if not group_id:
                return []
            
            # Fetch events
            url = f"{self.BASE_URL}/eventgroups/{group_id}/events"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('events', [])
                    
        except Exception as e:
            self.logger.error(f"Error fetching {sport} events: {e}")
        
        return []
    
    async def _get_event_group_id(self, sport: str) -> Optional[str]:
        """Get DraftKings event group ID for a sport"""
        # Simplified - in production would cache these
        sport_map = {
            "NFL": "88808",
            "NBA": "42648",
            "MLB": "84240",
            "NHL": "42133",
            "NCAAF": "87637",
            "NCAAB": "92483"
        }
        return sport_map.get(sport)
    
    def _check_odds_change(self, event: Dict) -> Optional[StandardizedEvent]:
        """
        Check if odds have changed significantly
        
        Args:
            event: DraftKings event data
            
        Returns:
            StandardizedEvent if significant change detected
        """
        event_id = event.get('eventId')
        if not event_id:
            return None
        
        # Extract current odds
        current_odds = self._extract_odds(event)
        if not current_odds:
            return None
        
        # Check against previous odds
        if event_id in self.previous_odds:
            prev_odds = self.previous_odds[event_id]
            
            # Calculate change
            for market_type, current_value in current_odds.items():
                prev_value = prev_odds.get(market_type, 0)
                change = abs(current_value - prev_value)
                
                # Significant change detected
                if change >= self.min_odds_change:
                    # Update stored odds
                    self.previous_odds[event_id] = current_odds
                    
                    # Create event
                    return self._create_odds_event(
                        event, 
                        market_type,
                        prev_value,
                        current_value,
                        change
                    )
        
        # Store odds for future comparison
        self.previous_odds[event_id] = current_odds
        return None
    
    def _extract_odds(self, event: Dict) -> Dict[str, float]:
        """Extract odds from DraftKings event data"""
        odds = {}
        
        # Get offers (markets)
        for offer in event.get('offers', []):
            # Get outcomes (sides of the bet)
            for outcome in offer.get('outcomes', []):
                # Convert American odds to decimal probability
                american_odds = outcome.get('oddsAmerican')
                if american_odds:
                    decimal_prob = self._american_to_probability(american_odds)
                    
                    # Store by market type and team
                    market_key = f"{offer.get('label', 'unknown')}_{outcome.get('label', 'unknown')}"
                    odds[market_key] = decimal_prob
        
        return odds
    
    def _american_to_probability(self, american_odds: str) -> float:
        """
        Convert American odds to implied probability
        
        Args:
            american_odds: American odds string (e.g., "-110", "+150")
            
        Returns:
            Implied probability (0.0 to 1.0)
        """
        try:
            odds = int(american_odds)
            
            if odds > 0:
                # Positive odds
                return 100 / (odds + 100)
            else:
                # Negative odds
                return abs(odds) / (abs(odds) + 100)
                
        except:
            return 0.0
    
    def _create_odds_event(
        self,
        event: Dict,
        market_type: str,
        prev_value: float,
        current_value: float,
        change: float
    ) -> StandardizedEvent:
        """Create standardized event for odds change"""
        
        # Calculate confidence based on change magnitude
        confidence = min(1.0, change / 0.10)  # 10% change = max confidence
        
        # Determine impact
        if change >= 0.08:
            impact = "critical"
        elif change >= 0.05:
            impact = "high"
        elif change >= 0.03:
            impact = "medium"
        else:
            impact = "low"
        
        return StandardizedEvent(
            source="DraftKings",
            event_type=EventType.ODDS_CHANGE,
            timestamp=datetime.now(),
            game_id=str(event.get('eventId')),
            data={
                "event_name": event.get('name', 'Unknown'),
                "market_type": market_type,
                "previous_odds": prev_value,
                "current_odds": current_value,
                "change": change,
                "direction": "up" if current_value > prev_value else "down",
                "sport": event.get('eventGroupName'),
                "start_time": event.get('startDate')
            },
            confidence=confidence,
            impact=impact,
            metadata={
                "teams": event.get('teamName1', '') + ' vs ' + event.get('teamName2', ''),
                "event_status": event.get('eventStatus', {}).get('state')
            }
        )
    
    def transform(self, raw_data: Any) -> Optional[StandardizedEvent]:
        """
        Transform raw DraftKings data to standardized event
        
        Args:
            raw_data: Raw data from DraftKings
            
        Returns:
            StandardizedEvent or None
        """
        # This would be used if we had a different data flow
        # For now, transformation happens in _create_odds_event
        return None


class DraftKingsMonitor:
    """
    Helper class for monitoring specific games
    """
    
    def __init__(self, adapter: DraftKingsAdapter):
        self.adapter = adapter
        self.tracked_games: Dict[str, Dict] = {}
    
    async def track_game(self, game_id: str, kalshi_market_id: str):
        """
        Track a specific game and correlate with Kalshi market
        
        Args:
            game_id: DraftKings game ID
            kalshi_market_id: Corresponding Kalshi market ID
        """
        self.tracked_games[game_id] = {
            "kalshi_market": kalshi_market_id,
            "start_time": datetime.now(),
            "events_count": 0
        }
    
    async def analyze_divergence(
        self, 
        dk_odds: float, 
        kalshi_price: float
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze divergence between DraftKings and Kalshi
        
        Args:
            dk_odds: DraftKings implied probability
            kalshi_price: Kalshi market price
            
        Returns:
            Trading signal if divergence is significant
        """
        divergence = abs(dk_odds - kalshi_price)
        
        if divergence > 0.05:  # 5% divergence threshold
            return {
                "signal": "DIVERGENCE",
                "dk_odds": dk_odds,
                "kalshi_price": kalshi_price,
                "divergence": divergence,
                "action": "BUY" if dk_odds > kalshi_price else "SELL",
                "confidence": min(1.0, divergence / 0.10)
            }
        
        return None