"""
Sports prediction strategy templates.

Templates specifically designed for sports prediction markets,
incorporating sentiment analysis, weather data, and game-specific factors.
"""

from .base import SportsPredictionTemplate
from .sentiment_momentum import SentimentMomentumTemplate
from .weather_impact import WeatherImpactTemplate

__all__ = [
    'SportsPredictionTemplate',
    'SentimentMomentumTemplate',
    'WeatherImpactTemplate',
]