"""
The Odds API Data Source

Provides comprehensive sports betting odds from multiple bookmakers.
"""

from .rest_adapter import OddsAPIAdapter
from .models import OddsData, BookmakerOdds, MarketOdds

__all__ = [
    "OddsAPIAdapter",
    "OddsData",
    "BookmakerOdds",
    "MarketOdds",
]