"""
ESPN API Integration Module
Real-time play-by-play data for market correlation
"""

from .client import ESPNClient
from .models import Play, Drive, GameState, PlayType, EventType
from .processor import PlayByPlayProcessor
from .stream import ESPNStreamAdapter

__all__ = [
    'ESPNClient',
    'Play',
    'Drive', 
    'GameState',
    'PlayType',
    'EventType',
    'PlayByPlayProcessor',
    'ESPNStreamAdapter'
]