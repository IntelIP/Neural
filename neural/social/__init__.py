"""
Neural SDK Social Media Data Collection Module.

This module provides data collection interfaces for social media platforms,
starting with Twitter API integration for collecting raw social data
related to sports events, teams, and players.
"""

from neural.social.twitter_client import (
    TwitterClient,
    TwitterConfig
)

__all__ = [
    "TwitterClient",
    "TwitterConfig"
]