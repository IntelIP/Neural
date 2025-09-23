"""
Base Data Source Infrastructure

Provides abstract base classes and utilities for building
data source integrations with both REST API and WebSocket support.
"""

from .rest_source import RESTDataSource
from .auth_strategies import (
    AuthStrategy,
    APIKeyAuth,
    BearerTokenAuth,
    RSASignatureAuth,
    NoAuth
)
from .rate_limiter import RateLimiter
from .cache import ResponseCache

__all__ = [
    'RESTDataSource',
    'AuthStrategy',
    'APIKeyAuth',
    'BearerTokenAuth',
    'RSASignatureAuth',
    'NoAuth',
    'RateLimiter',
    'ResponseCache'
]