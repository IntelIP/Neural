"""
Strategy Templates for Neural SDK

Pre-built strategy templates that can be customized and used for trading.
Templates provide a quick way to get started with common trading patterns.
"""

from .sports import (
    SportsPredictionTemplate,
    SentimentMomentumTemplate,
    WeatherImpactTemplate,
)

__all__ = [
    'SportsPredictionTemplate',
    'SentimentMomentumTemplate',
    'WeatherImpactTemplate',
]