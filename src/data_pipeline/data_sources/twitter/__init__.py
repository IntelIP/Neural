"""
Twitter API Integration Module
Real-time sentiment analysis for market correlation
"""

from .client import TwitterWebSocketClient
from .models import Tweet, TweetSentiment, FilterRule, SentimentEvent, MarketImpact
from .sentiment import SentimentAnalyzer
from .filters import FilterManager
from .stream import TwitterStreamAdapter

__all__ = [
    'TwitterWebSocketClient',
    'Tweet',
    'TweetSentiment',
    'FilterRule',
    'SentimentEvent',
    'MarketImpact',
    'SentimentAnalyzer',
    'FilterManager',
    'TwitterStreamAdapter'
]