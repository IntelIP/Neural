"""
Weather REST API Adapter

Integrates OpenWeatherMap API for weather data and sports impact analysis.
"""

import os
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import logging

from ..base.rest_source import RESTDataSource
from ..base.auth_strategies import APIKeyAuth
from .models import WeatherData, WeatherCondition, WeatherImpact

logger = logging.getLogger(__name__)


class WeatherRESTAdapter(RESTDataSource):
    """
    REST adapter for OpenWeatherMap API.
    
    Provides weather data for sports venues and impact analysis
    for prediction market trading.
    """
    
    # NFL Stadium coordinates (outdoor stadiums only)
    NFL_STADIUMS = {
        "Highmark Stadium": (42.7738, -78.7870, "Buffalo Bills"),  # Buffalo
        "Gillette Stadium": (42.0909, -71.2643, "New England Patriots"),
        "MetLife Stadium": (40.8135, -74.0745, "NY Giants/Jets"),
        "M&T Bank Stadium": (39.2780, -76.6227, "Baltimore Ravens"),
        "Paycor Stadium": (39.0954, -84.5160, "Cincinnati Bengals"),
        "Cleveland Browns Stadium": (41.5061, -81.6995, "Cleveland Browns"),
        "Heinz Field": (40.4468, -80.0158, "Pittsburgh Steelers"),
        "TIAA Bank Field": (30.3239, -81.6373, "Jacksonville Jaguars"),
        "Nissan Stadium": (36.1665, -86.7713, "Tennessee Titans"),
        "Empower Field": (39.7439, -105.0201, "Denver Broncos"),
        "Arrowhead Stadium": (39.0489, -94.4839, "Kansas City Chiefs"),
        "Lambeau Field": (44.5013, -88.0622, "Green Bay Packers"),
        "Soldier Field": (41.8623, -87.6167, "Chicago Bears"),
        "Bank of America Stadium": (35.2258, -80.8528, "Carolina Panthers"),
        "Raymond James Stadium": (27.9759, -82.5033, "Tampa Bay Buccaneers"),
        "Lumen Field": (47.5952, -122.3316, "Seattle Seahawks"),
        "Levi's Stadium": (37.4033, -121.9694, "San Francisco 49ers"),
        "Lincoln Financial Field": (39.9008, -75.1675, "Philadelphia Eagles"),
        "FedExField": (38.9076, -76.8645, "Washington Commanders"),
        "Acrisure Stadium": (40.4468, -80.0158, "Pittsburgh Steelers")
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Weather REST adapter.
        
        Args:
            api_key: OpenWeatherMap API key (or from OPENWEATHER_API_KEY env)
        """
        self.api_key = api_key or os.getenv("OPENWEATHER_API_KEY")
        if not self.api_key:
            logger.warning("No OpenWeatherMap API key provided. Weather features limited.")
        
        # Use API key auth in query parameters
        auth_strategy = APIKeyAuth(
            api_key=self.api_key,
            header_name="appid",
            in_header=False
        ) if self.api_key else None
        
        super().__init__(
            base_url="https://api.openweathermap.org/data/2.5",
            name="WeatherREST",
            auth_strategy=auth_strategy,
            timeout=10,
            cache_ttl=600,  # Cache weather for 10 minutes
            rate_limit=60,  # 60 calls per minute for free tier
            max_retries=3
        )
        
        logger.info("Weather REST adapter initialized")
    
    async def validate_response(self, response) -> bool:
        """
        Validate OpenWeatherMap API response.
        
        Args:
            response: HTTP response object
            
        Returns:
            True if valid, False otherwise
        """
        if response.status_code == 200:
            return True
        
        if response.status_code == 401:
            logger.error("OpenWeatherMap authentication failed - check API key")
        elif response.status_code == 404:
            logger.warning("Location not found")
        elif response.status_code == 429:
            logger.warning("OpenWeatherMap rate limit exceeded")
        
        return False
    
    async def transform_response(self, data: Any, endpoint: str) -> Dict:
        """
        Transform weather response to standardized format.
        
        Args:
            data: Raw OpenWeatherMap response
            endpoint: The endpoint that was called
            
        Returns:
            Standardized response
        """
        return {
            "source": "openweathermap",
            "endpoint": endpoint,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": {
                "units": "imperial"  # We use imperial units for US sports
            }
        }
    
    def _parse_weather_condition(self, weather_id: int) -> WeatherCondition:
        """Parse OpenWeatherMap weather ID to condition enum."""
        if weather_id < 300:
            return WeatherCondition.STORM
        elif weather_id < 600:
            return WeatherCondition.RAIN
        elif weather_id < 700:
            return WeatherCondition.SNOW
        elif weather_id < 800:
            return WeatherCondition.FOG
        elif weather_id == 800:
            return WeatherCondition.CLEAR
        elif weather_id < 900:
            return WeatherCondition.CLOUDY
        else:
            return WeatherCondition.EXTREME
    
    def _parse_weather_response(self, data: Dict) -> WeatherData:
        """Parse OpenWeatherMap response to WeatherData."""
        main = data.get("main", {})
        wind = data.get("wind", {})
        weather = data.get("weather", [{}])[0]
        coord = data.get("coord", {})
        sys = data.get("sys", {})
        
        # Parse precipitation
        rain = data.get("rain", {})
        snow = data.get("snow", {})
        precipitation = rain.get("1h", 0) + snow.get("1h", 0)
        
        return WeatherData(
            latitude=coord.get("lat", 0),
            longitude=coord.get("lon", 0),
            city=data.get("name"),
            temperature=main.get("temp", 0),
            feels_like=main.get("feels_like", 0),
            humidity=main.get("humidity", 0),
            wind_speed=wind.get("speed", 0),
            wind_direction=wind.get("deg", 0),
            wind_gust=wind.get("gust"),
            precipitation=precipitation / 25.4 if precipitation else 0,  # Convert mm to inches
            rain=rain.get("1h", 0) / 25.4 if rain.get("1h") else 0,
            snow=snow.get("1h", 0) / 25.4 if snow.get("1h") else 0,
            condition=self._parse_weather_condition(weather.get("id", 800)),
            description=weather.get("description", ""),
            visibility=data.get("visibility", 10000) / 1609.34,  # Convert meters to miles
            pressure=main.get("pressure", 1013),
            cloud_cover=data.get("clouds", {}).get("all", 0),
            timestamp=datetime.fromtimestamp(data.get("dt", 0)),
            sunrise=datetime.fromtimestamp(sys.get("sunrise", 0)) if sys.get("sunrise") else None,
            sunset=datetime.fromtimestamp(sys.get("sunset", 0)) if sys.get("sunset") else None
        )
    
    # Weather Data Methods
    
    async def get_weather_by_coords(
        self,
        lat: float,
        lon: float,
        venue: Optional[str] = None
    ) -> Dict:
        """
        Get current weather by coordinates.
        
        Args:
            lat: Latitude
            lon: Longitude
            venue: Optional venue name
            
        Returns:
            Weather data
        """
        if not self.api_key:
            return {"error": "No API key configured"}
        
        params = {
            "lat": lat,
            "lon": lon,
            "units": "imperial",
            "appid": self.api_key
        }
        
        result = await self.fetch("/weather", params=params, use_cache=True)
        
        if "data" in result:
            weather_data = self._parse_weather_response(result["data"])
            if venue:
                weather_data.venue = venue
            
            result["parsed"] = weather_data.to_dict()
        
        return result
    
    async def get_weather_by_city(self, city: str, state: Optional[str] = None) -> Dict:
        """
        Get current weather by city name.
        
        Args:
            city: City name
            state: State code (US only)
            
        Returns:
            Weather data
        """
        if not self.api_key:
            return {"error": "No API key configured"}
        
        query = f"{city},{state},US" if state else city
        
        params = {
            "q": query,
            "units": "imperial",
            "appid": self.api_key
        }
        
        result = await self.fetch("/weather", params=params, use_cache=True)
        
        if "data" in result:
            weather_data = self._parse_weather_response(result["data"])
            result["parsed"] = weather_data.to_dict()
        
        return result
    
    async def get_forecast(
        self,
        lat: float,
        lon: float,
        hours: int = 24
    ) -> Dict:
        """
        Get weather forecast.
        
        Args:
            lat: Latitude
            lon: Longitude
            hours: Hours to forecast (max 120)
            
        Returns:
            Forecast data
        """
        if not self.api_key:
            return {"error": "No API key configured"}
        
        params = {
            "lat": lat,
            "lon": lon,
            "units": "imperial",
            "cnt": min(hours // 3, 40),  # API returns 3-hour intervals
            "appid": self.api_key
        }
        
        result = await self.fetch("/forecast", params=params, use_cache=True)
        
        if "data" in result and "list" in result["data"]:
            forecasts = []
            for item in result["data"]["list"]:
                weather_data = self._parse_weather_response(item)
                forecasts.append(weather_data.to_dict())
            
            result["parsed"] = forecasts
        
        return result
    
    # NFL Stadium Weather Methods
    
    async def get_nfl_stadium_weather(self, stadium_name: str) -> Dict:
        """
        Get weather for NFL stadium.
        
        Args:
            stadium_name: Name of NFL stadium
            
        Returns:
            Weather data with impact analysis
        """
        if stadium_name not in self.NFL_STADIUMS:
            return {"error": f"Stadium '{stadium_name}' not found or is indoor"}
        
        lat, lon, team = self.NFL_STADIUMS[stadium_name]
        
        # Get current weather
        result = await self.get_weather_by_coords(lat, lon, venue=stadium_name)
        
        if "parsed" in result:
            # Add weather impact analysis
            weather_data = self._parse_weather_response(result["data"])
            weather_data.venue = stadium_name
            
            impact = WeatherImpact(
                weather_data=weather_data,
                sport="NFL",
                is_outdoor=True
            )
            impact.calculate_football_impact()
            
            result["impact"] = impact.to_dict()
            result["team"] = team
        
        return result
    
    async def get_all_nfl_stadium_weather(self) -> Dict:
        """
        Get weather for all outdoor NFL stadiums.
        
        Returns:
            Dictionary of weather data by stadium
        """
        stadium_weather = {}
        
        for stadium_name in self.NFL_STADIUMS:
            try:
                weather = await self.get_nfl_stadium_weather(stadium_name)
                stadium_weather[stadium_name] = weather
            except Exception as e:
                logger.error(f"Failed to get weather for {stadium_name}: {e}")
                stadium_weather[stadium_name] = {"error": str(e)}
        
        return {
            "source": "openweathermap",
            "sport": "NFL",
            "stadiums": stadium_weather,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # Game Weather Analysis
    
    async def analyze_game_weather(
        self,
        lat: float,
        lon: float,
        game_time: datetime,
        sport: str = "football",
        venue: Optional[str] = None
    ) -> Dict:
        """
        Analyze weather impact for a specific game.
        
        Args:
            lat: Venue latitude
            lon: Venue longitude
            game_time: Game start time
            sport: Sport type
            venue: Venue name
            
        Returns:
            Weather analysis with betting impact
        """
        # Get forecast if game is in future
        if game_time > datetime.utcnow():
            hours_until = (game_time - datetime.utcnow()).total_seconds() / 3600
            forecast = await self.get_forecast(lat, lon, min(int(hours_until), 120))
            
            # Find closest forecast to game time
            if "parsed" in forecast and forecast["parsed"]:
                # Use first forecast as approximation
                weather_dict = forecast["parsed"][0]
            else:
                return {"error": "Could not get forecast for game time"}
        else:
            # Get current weather for past/current games
            result = await self.get_weather_by_coords(lat, lon, venue)
            if "parsed" in result:
                weather_dict = result["parsed"]
            else:
                return {"error": "Could not get weather data"}
        
        # Create weather data object
        weather_data = WeatherData(
            latitude=lat,
            longitude=lon,
            venue=venue,
            temperature=weather_dict["temperature"]["actual"],
            feels_like=weather_dict["temperature"]["feels_like"],
            humidity=weather_dict["conditions"]["humidity"],
            wind_speed=weather_dict["wind"]["speed"],
            wind_direction=weather_dict["wind"]["direction"],
            wind_gust=weather_dict["wind"]["gust"],
            precipitation=weather_dict["precipitation"]["total"],
            rain=weather_dict["precipitation"]["rain"],
            snow=weather_dict["precipitation"]["snow"]
        )
        
        # Calculate impact
        impact = WeatherImpact(
            weather_data=weather_data,
            sport=sport,
            is_outdoor=True
        )
        
        if sport.lower() in ["football", "nfl", "cfb"]:
            impact.calculate_football_impact()
        
        return {
            "source": "openweathermap",
            "venue": venue,
            "game_time": game_time.isoformat(),
            "weather": weather_data.to_dict(),
            "impact": impact.to_dict(),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # Health Check
    
    async def health_check(self) -> bool:
        """
        Check OpenWeatherMap API health.
        
        Returns:
            True if healthy, False otherwise
        """
        if not self.api_key:
            logger.warning("No API key configured for weather service")
            return False
        
        try:
            # Test with a known location (New York)
            result = await self.get_weather_by_coords(40.7128, -74.0060)
            return "data" in result and "error" not in result
        except Exception as e:
            logger.error(f"Weather API health check failed: {e}")
            return False