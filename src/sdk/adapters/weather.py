"""
Weather Data Adapter
Monitors weather conditions at game venues for impact on gameplay
"""

import aiohttp
import asyncio
from typing import Dict, Any, Optional, AsyncGenerator, List
from datetime import datetime, timedelta
import logging
import math

from ..core.base_adapter import (
    DataSourceAdapter,
    DataSourceMetadata,
    StandardizedEvent,
    EventType
)

logger = logging.getLogger(__name__)


class WeatherAdapter(DataSourceAdapter):
    """
    Weather conditions monitor for sports venues
    Tracks wind, precipitation, temperature that affect game outcomes
    """
    
    # OpenWeatherMap API (free tier available)
    BASE_URL = "https://api.openweathermap.org/data/2.5"
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Weather adapter
        
        Config should include:
        - api_key: OpenWeatherMap API key
        - stadiums: List of stadium locations to monitor
        - update_interval: Seconds between weather checks
        - thresholds: Weather thresholds for alerts
        """
        super().__init__(config)
        
        self.api_key = config.get('api_key')
        self.stadiums = config.get('stadiums', self._get_default_stadiums())
        self.update_interval = config.get('update_interval', 300)  # 5 minutes
        
        # Weather thresholds for significant events
        self.thresholds = config.get('thresholds', {
            'wind_speed': 15,  # mph - affects passing game
            'precipitation': 0.1,  # inches/hour - affects ball handling
            'temperature_change': 10,  # degrees F - rapid change
            'visibility': 1  # miles - affects deep passes
        })
        
        self.session: Optional[aiohttp.ClientSession] = None
        self.previous_conditions: Dict[str, Dict] = {}
        
    def get_metadata(self) -> DataSourceMetadata:
        """Return Weather adapter metadata"""
        return DataSourceMetadata(
            name="Weather",
            version="1.0.0",
            author="Neural Trading Platform",
            description="Real-time weather conditions at sports venues",
            source_type="environmental",
            latency_ms=2000,
            reliability=0.99,
            requires_auth=True,
            rate_limits={"requests_per_minute": 60},
            supported_sports=["NFL", "MLB", "Soccer", "NCAAF"],
            supported_markets=["weather_impact"]
        )
    
    def _get_default_stadiums(self) -> List[Dict[str, Any]]:
        """Get default NFL stadium locations"""
        return [
            {"name": "Arrowhead Stadium", "team": "KC", "lat": 39.0489, "lon": -94.4839, "outdoor": True},
            {"name": "Highmark Stadium", "team": "BUF", "lat": 42.7738, "lon": -78.7870, "outdoor": True},
            {"name": "Lambeau Field", "team": "GB", "lat": 44.5013, "lon": -88.0622, "outdoor": True},
            {"name": "Soldier Field", "team": "CHI", "lat": 41.8623, "lon": -87.6167, "outdoor": True},
            {"name": "MetLife Stadium", "team": "NYG/NYJ", "lat": 40.8135, "lon": -74.0745, "outdoor": True},
            {"name": "Gillette Stadium", "team": "NE", "lat": 42.0909, "lon": -71.2643, "outdoor": True},
            {"name": "Mile High", "team": "DEN", "lat": 39.7439, "lon": -105.0201, "outdoor": True},
            {"name": "Lumen Field", "team": "SEA", "lat": 47.5952, "lon": -122.3316, "outdoor": True}
        ]
    
    async def connect(self) -> bool:
        """Connect to Weather API"""
        try:
            self.session = aiohttp.ClientSession()
            
            # Test API key with a sample request
            test_url = f"{self.BASE_URL}/weather"
            params = {
                "lat": 40.7128,
                "lon": -74.0060,
                "appid": self.api_key,
                "units": "imperial"
            }
            
            async with self.session.get(test_url, params=params) as response:
                if response.status == 200:
                    self.logger.info("Connected to Weather API")
                    return True
                else:
                    self.logger.error(f"Weather API error: {response.status}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Failed to connect to Weather API: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Weather API"""
        if self.session:
            await self.session.close()
            self.session = None
        self.logger.info("Disconnected from Weather API")
    
    async def validate_connection(self) -> bool:
        """Validate Weather API connection"""
        if not self.session:
            return False
        
        try:
            # Simple API test
            test_url = f"{self.BASE_URL}/weather"
            params = {
                "q": "London",
                "appid": self.api_key
            }
            
            async with self.session.get(
                test_url, 
                params=params,
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                return response.status == 200
        except:
            return False
    
    async def stream(self) -> AsyncGenerator[StandardizedEvent, None]:
        """Stream weather updates for tracked stadiums"""
        while self.is_connected:
            try:
                # Check weather for each stadium
                for stadium in self.stadiums:
                    # Skip indoor stadiums
                    if not stadium.get('outdoor', True):
                        continue
                    
                    # Get current conditions
                    conditions = await self._fetch_weather(stadium)
                    if not conditions:
                        continue
                    
                    # Check for significant changes
                    events = self._analyze_conditions(stadium, conditions)
                    for event in events:
                        yield event
                        self._increment_event_count()
                
                # Wait before next update
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in weather stream: {e}")
                self._increment_error_count()
                await asyncio.sleep(self.update_interval)
    
    async def _fetch_weather(self, stadium: Dict) -> Optional[Dict]:
        """
        Fetch weather data for a stadium location
        
        Args:
            stadium: Stadium information with lat/lon
            
        Returns:
            Weather conditions dictionary
        """
        try:
            # Current weather
            current_url = f"{self.BASE_URL}/weather"
            params = {
                "lat": stadium['lat'],
                "lon": stadium['lon'],
                "appid": self.api_key,
                "units": "imperial"  # Fahrenheit, mph
            }
            
            async with self.session.get(current_url, params=params) as response:
                if response.status != 200:
                    return None
                
                data = await response.json()
                
                # Extract relevant conditions
                conditions = {
                    "temperature": data['main']['temp'],
                    "feels_like": data['main']['feels_like'],
                    "humidity": data['main']['humidity'],
                    "wind_speed": data['wind']['speed'],
                    "wind_direction": data['wind'].get('deg', 0),
                    "wind_gust": data['wind'].get('gust', data['wind']['speed']),
                    "precipitation": self._calculate_precipitation(data),
                    "visibility": data.get('visibility', 10000) / 1609.34,  # Convert to miles
                    "pressure": data['main']['pressure'],
                    "weather": data['weather'][0]['main'] if data.get('weather') else "Clear",
                    "description": data['weather'][0]['description'] if data.get('weather') else "",
                    "timestamp": datetime.now()
                }
                
                return conditions
                
        except Exception as e:
            self.logger.error(f"Error fetching weather for {stadium['name']}: {e}")
            return None
    
    def _calculate_precipitation(self, data: Dict) -> float:
        """Calculate precipitation rate from weather data"""
        # Check for rain
        if 'rain' in data:
            return data['rain'].get('1h', 0) / 25.4  # Convert mm to inches
        # Check for snow
        if 'snow' in data:
            return data['snow'].get('1h', 0) / 25.4 * 0.1  # Snow to water ratio
        return 0.0
    
    def _analyze_conditions(
        self, 
        stadium: Dict, 
        conditions: Dict
    ) -> List[StandardizedEvent]:
        """
        Analyze weather conditions for significant events
        
        Args:
            stadium: Stadium information
            conditions: Current weather conditions
            
        Returns:
            List of weather events
        """
        events = []
        stadium_id = stadium['name']
        
        # Get previous conditions
        previous = self.previous_conditions.get(stadium_id, {})
        
        # Check wind impact
        if conditions['wind_speed'] > self.thresholds['wind_speed']:
            impact = self._calculate_wind_impact(conditions['wind_speed'])
            events.append(StandardizedEvent(
                source="Weather",
                event_type=EventType.WEATHER_UPDATE,
                timestamp=datetime.now(),
                game_id=f"{stadium['team']}_home",
                data={
                    "condition": "high_wind",
                    "wind_speed": conditions['wind_speed'],
                    "wind_gust": conditions['wind_gust'],
                    "wind_direction": conditions['wind_direction'],
                    "impact": impact,
                    "stadium": stadium['name'],
                    "team": stadium['team']
                },
                confidence=0.95,
                impact="high" if conditions['wind_speed'] > 25 else "medium",
                metadata={
                    "affects": ["passing_game", "field_goals", "punts"],
                    "advantage": "running_game"
                }
            ))
        
        # Check precipitation
        if conditions['precipitation'] > self.thresholds['precipitation']:
            events.append(StandardizedEvent(
                source="Weather",
                event_type=EventType.WEATHER_UPDATE,
                timestamp=datetime.now(),
                game_id=f"{stadium['team']}_home",
                data={
                    "condition": "precipitation",
                    "rate": conditions['precipitation'],
                    "type": conditions['weather'],
                    "description": conditions['description'],
                    "stadium": stadium['name'],
                    "team": stadium['team']
                },
                confidence=0.95,
                impact="high",
                metadata={
                    "affects": ["ball_handling", "footing", "visibility"],
                    "advantage": "defense"
                }
            ))
        
        # Check temperature changes
        if previous:
            temp_change = abs(conditions['temperature'] - previous.get('temperature', conditions['temperature']))
            if temp_change > self.thresholds['temperature_change']:
                events.append(StandardizedEvent(
                    source="Weather",
                    event_type=EventType.WEATHER_UPDATE,
                    timestamp=datetime.now(),
                    game_id=f"{stadium['team']}_home",
                    data={
                        "condition": "temperature_change",
                        "current_temp": conditions['temperature'],
                        "previous_temp": previous.get('temperature'),
                        "change": temp_change,
                        "stadium": stadium['name'],
                        "team": stadium['team']
                    },
                    confidence=0.90,
                    impact="medium",
                    metadata={
                        "affects": ["player_stamina", "ball_pressure"],
                        "direction": "cooling" if conditions['temperature'] < previous.get('temperature', 0) else "warming"
                    }
                ))
        
        # Check visibility (fog, heavy rain/snow)
        if conditions['visibility'] < self.thresholds['visibility']:
            events.append(StandardizedEvent(
                source="Weather",
                event_type=EventType.WEATHER_UPDATE,
                timestamp=datetime.now(),
                game_id=f"{stadium['team']}_home",
                data={
                    "condition": "low_visibility",
                    "visibility": conditions['visibility'],
                    "weather": conditions['weather'],
                    "stadium": stadium['name'],
                    "team": stadium['team']
                },
                confidence=0.85,
                impact="high",
                metadata={
                    "affects": ["passing_game", "deep_routes"],
                    "advantage": "short_passing_running"
                }
            ))
        
        # Store current conditions for next comparison
        self.previous_conditions[stadium_id] = conditions
        
        return events
    
    def _calculate_wind_impact(self, wind_speed: float) -> Dict[str, Any]:
        """
        Calculate impact of wind on game elements
        
        Args:
            wind_speed: Wind speed in mph
            
        Returns:
            Impact analysis
        """
        # Based on research about wind effects on football
        passing_reduction = min(wind_speed * 1.5, 40)  # Up to 40% reduction
        fg_accuracy_reduction = min(wind_speed * 2, 50)  # Up to 50% reduction
        punt_variance = wind_speed * 0.5  # Yards of variance
        
        return {
            "passing_yards_reduction": f"{passing_reduction:.0f}%",
            "field_goal_accuracy_reduction": f"{fg_accuracy_reduction:.0f}%",
            "punt_distance_variance": f"±{punt_variance:.0f} yards",
            "recommended_strategy": "establish_run" if wind_speed > 20 else "balanced"
        }
    
    def transform(self, raw_data: Any) -> Optional[StandardizedEvent]:
        """Transform raw weather data to standardized event"""
        # Transformation happens in _analyze_conditions
        return None


class WeatherImpactAnalyzer:
    """
    Analyzes weather impact on specific game scenarios
    """
    
    def __init__(self):
        self.sport_impacts = {
            "NFL": {
                "wind": {"threshold": 15, "impact": "high"},
                "rain": {"threshold": 0.1, "impact": "medium"},
                "snow": {"threshold": 0.05, "impact": "high"},
                "temperature": {"threshold": 32, "impact": "low"}
            },
            "MLB": {
                "wind": {"threshold": 10, "impact": "medium"},
                "rain": {"threshold": 0.05, "impact": "high"},
                "temperature": {"threshold": 95, "impact": "low"},
                "humidity": {"threshold": 70, "impact": "medium"}
            }
        }
    
    def calculate_total_impact(
        self, 
        conditions: Dict, 
        sport: str
    ) -> Dict[str, Any]:
        """
        Calculate total weather impact on game
        
        Args:
            conditions: Weather conditions
            sport: Sport type
            
        Returns:
            Impact assessment
        """
        if sport not in self.sport_impacts:
            return {"impact": "unknown", "score": 0}
        
        impacts = self.sport_impacts[sport]
        total_score = 0
        factors = []
        
        # Check each weather factor
        if conditions.get('wind_speed', 0) > impacts['wind']['threshold']:
            score = self._score_impact(
                conditions['wind_speed'], 
                impacts['wind']['threshold'],
                impacts['wind']['impact']
            )
            total_score += score
            factors.append(f"Wind: {score:.1f}")
        
        if conditions.get('precipitation', 0) > impacts.get('rain', {}).get('threshold', 1):
            score = self._score_impact(
                conditions['precipitation'],
                impacts['rain']['threshold'],
                impacts['rain']['impact']
            )
            total_score += score
            factors.append(f"Rain: {score:.1f}")
        
        # Determine overall impact
        if total_score > 7:
            overall = "severe"
        elif total_score > 4:
            overall = "significant"
        elif total_score > 2:
            overall = "moderate"
        else:
            overall = "minimal"
        
        return {
            "overall_impact": overall,
            "impact_score": total_score,
            "factors": factors,
            "betting_recommendation": self._get_betting_recommendation(overall, sport)
        }
    
    def _score_impact(
        self, 
        value: float, 
        threshold: float, 
        impact_level: str
    ) -> float:
        """Calculate impact score"""
        multiplier = {
            "high": 3.0,
            "medium": 2.0,
            "low": 1.0
        }.get(impact_level, 1.0)
        
        return (value / threshold) * multiplier
    
    def _get_betting_recommendation(
        self, 
        impact: str, 
        sport: str
    ) -> str:
        """Get betting recommendation based on weather impact"""
        if sport == "NFL":
            if impact == "severe":
                return "Strong UNDER, favor running teams"
            elif impact == "significant":
                return "Lean UNDER, watch for adjustments"
            else:
                return "Weather neutral"
        
        elif sport == "MLB":
            if impact == "severe":
                return "Game likely postponed"
            elif impact == "significant":
                return "UNDER, reduced scoring"
            else:
                return "Minimal impact"
        
        return "Assess individually"