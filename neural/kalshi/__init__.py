"""
Kalshi integration module for Neural SDK.

This module provides interfaces for Kalshi market data,
fee calculations, and order management.
"""

from neural.kalshi.markets import KalshiMarket
from neural.kalshi.fees import (
    calculate_kalshi_fee, 
    calculate_expected_value,
    calculate_kelly_fraction
)
from neural.kalshi.client import KalshiClient, get_kalshi_client

__all__ = [
    'KalshiMarket',
    'calculate_kalshi_fee',
    'calculate_expected_value',
    'calculate_kelly_fraction',
    'KalshiClient',
    'get_kalshi_client'
]

__version__ = '0.1.0'