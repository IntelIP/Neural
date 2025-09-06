"""
Weather Data Models

Data structures for weather information and sports impact analysis.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum


class WeatherCondition(Enum):
    """Weather condition categories."""
    CLEAR = "clear"
    CLOUDY = "cloudy"
    RAIN = "rain"
    SNOW = "snow"
    FOG = "fog"
    WIND = "wind"
    STORM = "storm"
    EXTREME = "extreme"


@dataclass
class WeatherData:
    """Weather data for a location."""
    
    # Location
    latitude: float
    longitude: float
    
    # Current conditions (required)
    temperature: float  # Fahrenheit
    feels_like: float
    humidity: float  # Percentage
    wind_speed: float  # MPH
    wind_direction: float  # Degrees
    
    # Optional location info
    city: Optional[str] = None
    venue: Optional[str] = None
    wind_gust: Optional[float] = None
    
    # Precipitation
    precipitation: float = 0.0  # Inches per hour
    rain: float = 0.0
    snow: float = 0.0
    
    # Conditions
    condition: WeatherCondition = WeatherCondition.CLEAR
    description: str = ""
    visibility: float = 10.0  # Miles
    pressure: float = 1013.0  # mb
    cloud_cover: float = 0.0  # Percentage
    
    # Timestamps
    timestamp: datetime = None
    sunrise: Optional[datetime] = None
    sunset: Optional[datetime] = None
    
    def __post_init__(self):
        """Initialize timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    @property
    def is_outdoor_friendly(self) -> bool:
        """Check if weather is suitable for outdoor sports."""
        return (
            self.condition not in [WeatherCondition.STORM, WeatherCondition.EXTREME] and
            self.wind_speed < 30 and
            self.precipitation < 0.5 and
            self.visibility > 0.5
        )
    
    @property
    def has_precipitation(self) -> bool:
        """Check if there's active precipitation."""
        return self.precipitation > 0 or self.rain > 0 or self.snow > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "location": {
                "latitude": self.latitude,
                "longitude": self.longitude,
                "city": self.city,
                "venue": self.venue
            },
            "temperature": {
                "actual": self.temperature,
                "feels_like": self.feels_like
            },
            "wind": {
                "speed": self.wind_speed,
                "direction": self.wind_direction,
                "gust": self.wind_gust
            },
            "precipitation": {
                "total": self.precipitation,
                "rain": self.rain,
                "snow": self.snow
            },
            "conditions": {
                "main": self.condition.value,
                "description": self.description,
                "humidity": self.humidity,
                "visibility": self.visibility,
                "pressure": self.pressure,
                "cloud_cover": self.cloud_cover
            },
            "outdoor_friendly": self.is_outdoor_friendly,
            "has_precipitation": self.has_precipitation,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }


@dataclass
class WeatherImpact:
    """
    Weather impact analysis for sports betting.
    
    Analyzes how weather conditions might affect game outcomes.
    """
    
    weather_data: WeatherData
    sport: str
    is_outdoor: bool = True
    
    # Impact scores (0-100, higher = more impact)
    passing_impact: float = 0.0
    rushing_impact: float = 0.0
    kicking_impact: float = 0.0
    scoring_impact: float = 0.0
    home_advantage_impact: float = 0.0
    
    # Overall impact
    total_impact: float = 0.0
    impact_summary: str = ""
    
    def calculate_football_impact(self):
        """Calculate weather impact for football games."""
        weather = self.weather_data
        
        # Wind impact on passing and kicking
        if weather.wind_speed > 20:
            self.passing_impact = min(weather.wind_speed * 2, 80)
            self.kicking_impact = min(weather.wind_speed * 2.5, 90)
        elif weather.wind_speed > 10:
            self.passing_impact = weather.wind_speed * 1.5
            self.kicking_impact = weather.wind_speed * 2
        
        # Precipitation impact
        if weather.has_precipitation:
            precip_factor = min(weather.precipitation * 50, 70)
            self.passing_impact += precip_factor * 0.7
            self.rushing_impact -= precip_factor * 0.3  # Rush advantage in rain
            self.kicking_impact += precip_factor * 0.5
            
            # Snow has different impact
            if weather.snow > 0:
                snow_factor = min(weather.snow * 30, 80)
                self.scoring_impact += snow_factor
                self.passing_impact += snow_factor * 0.3
        
        # Temperature impact
        if weather.temperature < 32:
            cold_factor = (32 - weather.temperature) * 1.5
            self.passing_impact += cold_factor * 0.3
            self.kicking_impact += cold_factor * 0.4
        elif weather.temperature > 90:
            heat_factor = (weather.temperature - 90) * 2
            self.scoring_impact -= heat_factor * 0.2  # More scoring in heat
        
        # Visibility impact
        if weather.visibility < 1:
            vis_factor = (1 - weather.visibility) * 50
            self.passing_impact += vis_factor
        
        # Home team advantage in bad weather
        if not weather.is_outdoor_friendly:
            self.home_advantage_impact = 20 + (self.total_impact * 0.3)
        
        # Calculate total impact
        self.total_impact = (
            self.passing_impact * 0.3 +
            self.rushing_impact * 0.2 +
            self.kicking_impact * 0.2 +
            self.scoring_impact * 0.2 +
            self.home_advantage_impact * 0.1
        )
        
        # Generate summary
        self._generate_summary()
    
    def _generate_summary(self):
        """Generate human-readable impact summary."""
        if self.total_impact < 10:
            self.impact_summary = "Minimal weather impact expected"
        elif self.total_impact < 25:
            self.impact_summary = "Slight weather impact on gameplay"
        elif self.total_impact < 50:
            self.impact_summary = "Moderate weather impact - expect adjusted play calling"
        elif self.total_impact < 75:
            self.impact_summary = "Significant weather impact - favors running game and defense"
        else:
            self.impact_summary = "Extreme weather impact - major gameplay disruption expected"
        
        # Add specific concerns
        concerns = []
        if self.passing_impact > 40:
            concerns.append("passing game affected")
        if self.kicking_impact > 40:
            concerns.append("field goals risky")
        if self.weather_data.wind_speed > 20:
            concerns.append(f"high winds ({self.weather_data.wind_speed:.0f} mph)")
        if self.weather_data.has_precipitation:
            concerns.append("wet conditions")
        
        if concerns:
            self.impact_summary += f" ({', '.join(concerns)})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "weather": self.weather_data.to_dict(),
            "sport": self.sport,
            "is_outdoor": self.is_outdoor,
            "impacts": {
                "passing": round(self.passing_impact, 1),
                "rushing": round(self.rushing_impact, 1),
                "kicking": round(self.kicking_impact, 1),
                "scoring": round(self.scoring_impact, 1),
                "home_advantage": round(self.home_advantage_impact, 1),
                "total": round(self.total_impact, 1)
            },
            "summary": self.impact_summary,
            "betting_considerations": self._get_betting_considerations()
        }
    
    def _get_betting_considerations(self) -> Dict[str, Any]:
        """Get betting-specific considerations."""
        considerations = {
            "favor_under": self.scoring_impact > 30,
            "favor_home": self.home_advantage_impact > 15,
            "favor_running": self.rushing_impact < self.passing_impact,
            "avoid_player_props_passing": self.passing_impact > 40,
            "avoid_field_goal_props": self.kicking_impact > 50
        }
        
        # Recommendations
        recs = []
        if considerations["favor_under"]:
            recs.append("Consider UNDER on total points")
        if considerations["favor_home"]:
            recs.append("Home team has weather advantage")
        if considerations["favor_running"]:
            recs.append("Running backs may exceed expectations")
        if considerations["avoid_player_props_passing"]:
            recs.append("Avoid passing yards props")
        
        considerations["recommendations"] = recs
        return considerations