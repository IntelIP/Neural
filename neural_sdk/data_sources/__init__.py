"""
Neural SDK Data Sources

Unified data source framework for REST APIs and WebSocket streams.
"""

# Base infrastructure
from .base import (
    RESTDataSource,
    AuthStrategy,
    APIKeyAuth,
    BearerTokenAuth,
    RSASignatureAuth,
    NoAuth,
    RateLimiter,
    ResponseCache
)

# REST Adapters
from .kalshi.rest_adapter import KalshiRESTAdapter
from .espn.rest_adapter import ESPNRESTAdapter
from .weather import WeatherRESTAdapter, WeatherData, WeatherImpact

__all__ = [
    # Base classes
    'RESTDataSource',
    'AuthStrategy',
    'APIKeyAuth',
    'BearerTokenAuth',
    'RSASignatureAuth',
    'NoAuth',
    'RateLimiter',
    'ResponseCache',
    
    # REST Adapters
    'KalshiRESTAdapter',
    'ESPNRESTAdapter',
    'WeatherRESTAdapter',
    
    # Data models
    'WeatherData',
    'WeatherImpact'
]