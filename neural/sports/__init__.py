"""
Neural SDK Sports Data Integration.

This module provides specialized clients for fetching sports data from various APIs,
starting with ESPN's public endpoints for NFL, College Football, and NBA.
"""

from neural.sports.espn_base import ESPNClient, ESPNConfig
from neural.sports.espn_nfl import ESPNNFL
from neural.sports.espn_cfb import ESPNCFB
from neural.sports.espn_nba import ESPNNBA

__all__ = [
    "ESPNClient",
    "ESPNConfig",
    "ESPNNFL",
    "ESPNCFB",
    "ESPNNBA"
]