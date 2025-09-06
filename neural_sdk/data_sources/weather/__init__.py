"""
Weather Data Integration

Provides weather data for sports impact analysis.
"""

from .rest_adapter import WeatherRESTAdapter
from .models import WeatherData, WeatherCondition, WeatherImpact

__all__ = [
    'WeatherRESTAdapter',
    'WeatherData',
    'WeatherCondition',
    'WeatherImpact'
]