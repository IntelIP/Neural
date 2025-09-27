"""
Twitter API Data Source using Twitter-API.io

This module provides real-time Twitter data collection for sentiment analysis
in trading algorithms. It uses Twitter-API.io for simplified API access.
"""

import asyncio
import aiohttp
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, AsyncGenerator, Any
from dataclasses import dataclass

from .base import DataSource
from .rest_api import RestApiSource


@dataclass
class TwitterConfig:
    """Configuration for Twitter data collection."""
    api_key: str
    query: str = ""
    max_results: int = 100
    tweet_fields: List[str] = None
    user_fields: List[str] = None
    poll_interval: float = 30.0

    def __post_init__(self):
        if self.tweet_fields is None:
            self.tweet_fields = [
                'created_at', 'author_id', 'public_metrics',
                'context_annotations', 'lang', 'conversation_id'
            ]
        if self.user_fields is None:
            self.user_fields = ['username', 'verified', 'public_metrics']


class TwitterAPISource(DataSource):
    """
    Twitter data source using Twitter-API.io service.

    Provides real-time Twitter data collection with built-in rate limiting
    and error handling for sentiment analysis in trading algorithms.
    """

    BASE_URL = "https://twitter-api.io/api/v2"

    def __init__(self, config: TwitterConfig):
        super().__init__(name="twitter_api", config=config.__dict__)
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self._running = False

    async def connect(self) -> None:
        """Establish connection to Twitter API."""
        if not self.session:
            headers = {
                'Authorization': f'Bearer {self.config.api_key}',
                'Content-Type': 'application/json'
            }
            self.session = aiohttp.ClientSession(headers=headers)
        self._connected = True

    async def disconnect(self) -> None:
        """Close Twitter API connection."""
        if self.session:
            await self.session.close()
            self.session = None
        self._connected = False
        self._running = False

    async def search_tweets(self, query: str, max_results: int = 100) -> Dict[str, Any]:
        """
        Search for tweets matching the query.

        Args:
            query: Twitter search query
            max_results: Maximum number of tweets to return

        Returns:
            Twitter API response with tweet data
        """
        if not self.session:
            raise RuntimeError("Not connected to Twitter API")

        params = {
            'query': query,
            'max_results': min(max_results, 100),
            'tweet.fields': ','.join(self.config.tweet_fields),
            'user.fields': ','.join(self.config.user_fields),
            'expansions': 'author_id'
        }

        async with self.session.get(f"{self.BASE_URL}/tweets/search/recent", params=params) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise RuntimeError(f"Twitter API error {response.status}: {error_text}")

    async def get_game_tweets(self, teams: List[str], hashtags: List[str] = None) -> Dict[str, Any]:
        """
        Get tweets related to a specific game.

        Args:
            teams: List of team names/hashtags
            hashtags: Additional hashtags to include

        Returns:
            Filtered tweets related to the game
        """
        # Build query for game-specific tweets
        team_terms = [f'"{team}"' for team in teams]
        hashtag_terms = [f'#{tag}' for tag in (hashtags or [])]

        query_parts = []
        if team_terms:
            query_parts.append(f"({' OR '.join(team_terms)})")
        if hashtag_terms:
            query_parts.append(f"({' OR '.join(hashtag_terms)})")

        query = ' AND '.join(query_parts)

        # Add filters for quality and recency
        query += " -is:retweet lang:en"

        return await self.search_tweets(query, self.config.max_results)

    async def collect(self) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Continuously collect Twitter data.

        Yields:
            Processed tweet data with metadata
        """
        if not self._connected:
            await self.connect()

        self._running = True

        while self._running:
            try:
                # Use the configured query or default to general sports content
                query = self.config.query or "NFL OR NBA OR MLB -is:retweet lang:en"

                tweets_data = await self.search_tweets(query, self.config.max_results)

                # Process and yield tweets
                if 'data' in tweets_data:
                    for tweet in tweets_data['data']:
                        processed_tweet = self._process_tweet(tweet, tweets_data.get('includes', {}))
                        yield processed_tweet

                # Wait before next poll
                await asyncio.sleep(self.config.poll_interval)

            except Exception as e:
                print(f"Error collecting Twitter data: {e}")
                await asyncio.sleep(60)  # Wait longer on error

    def _process_tweet(self, tweet: Dict[str, Any], includes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process raw tweet data into structured format.

        Args:
            tweet: Raw tweet data from API
            includes: Additional data (users, media, etc.)

        Returns:
            Processed tweet with metadata
        """
        # Get author information
        author_id = tweet.get('author_id')
        author_info = {}

        if 'users' in includes:
            for user in includes['users']:
                if user['id'] == author_id:
                    author_info = {
                        'username': user.get('username'),
                        'verified': user.get('verified', False),
                        'followers': user.get('public_metrics', {}).get('followers_count', 0)
                    }
                    break

        # Extract metrics
        metrics = tweet.get('public_metrics', {})

        # Process datetime
        created_at = tweet.get('created_at')
        if created_at:
            created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))

        return {
            'id': tweet['id'],
            'text': tweet['text'],
            'created_at': created_at,
            'author_id': author_id,
            'author_info': author_info,
            'metrics': {
                'retweet_count': metrics.get('retweet_count', 0),
                'like_count': metrics.get('like_count', 0),
                'reply_count': metrics.get('reply_count', 0),
                'quote_count': metrics.get('quote_count', 0),
            },
            'lang': tweet.get('lang', 'en'),
            'context_annotations': tweet.get('context_annotations', []),
            'conversation_id': tweet.get('conversation_id'),
            'source': 'twitter',
            'timestamp': datetime.now()
        }


class GameTwitterSource(TwitterAPISource):
    """
    Specialized Twitter source for specific games.

    Focuses on collecting tweets related to a specific game or teams.
    """

    def __init__(
        self,
        api_key: str,
        teams: List[str],
        hashtags: List[str] = None,
        poll_interval: float = 15.0
    ):
        # Build game-specific query
        team_terms = [f'"{team}"' for team in teams]
        hashtag_terms = [f'#{tag}' for tag in (hashtags or [])]

        query_parts = []
        if team_terms:
            query_parts.append(f"({' OR '.join(team_terms)})")
        if hashtag_terms:
            query_parts.append(f"({' OR '.join(hashtag_terms)})")

        query = ' AND '.join(query_parts) if query_parts else ' OR '.join(teams)
        query += " -is:retweet lang:en"

        config = TwitterConfig(
            api_key=api_key,
            query=query,
            poll_interval=poll_interval,
            max_results=50  # More focused, so fewer results needed
        )

        super().__init__(config)
        self.teams = teams
        self.hashtags = hashtags or []
        self.name = f"twitter_game_{' '.join(teams).replace(' ', '_').lower()}"


# Factory function for easy setup
def create_twitter_source(
    api_key: Optional[str] = None,
    teams: List[str] = None,
    hashtags: List[str] = None,
    query: str = None,
    poll_interval: float = 30.0
) -> TwitterAPISource:
    """
    Create a Twitter data source with sensible defaults.

    Args:
        api_key: Twitter API key (uses TWITTER_API_KEY env var if not provided)
        teams: List of teams to track
        hashtags: List of hashtags to track
        query: Custom search query
        poll_interval: How often to poll for new tweets

    Returns:
        Configured TwitterAPISource
    """
    if api_key is None:
        api_key = os.getenv('TWITTER_API_KEY')
        if not api_key:
            raise ValueError("Twitter API key required. Set TWITTER_API_KEY env var or pass api_key parameter")

    if teams:
        return GameTwitterSource(
            api_key=api_key,
            teams=teams,
            hashtags=hashtags,
            poll_interval=poll_interval
        )
    else:
        config = TwitterConfig(
            api_key=api_key,
            query=query or "NFL OR NBA OR MLB -is:retweet lang:en",
            poll_interval=poll_interval
        )
        return TwitterAPISource(config)


# Example usage patterns
if __name__ == "__main__":
    async def example():
        # Example 1: Track specific game
        ravens_lions_source = create_twitter_source(
            teams=["Baltimore Ravens", "Detroit Lions"],
            hashtags=["RavensVsLions", "NFL"],
            poll_interval=15.0
        )

        # Example 2: General sports sentiment
        sports_source = create_twitter_source(
            query="NFL OR NBA -is:retweet lang:en",
            poll_interval=60.0
        )

        async with ravens_lions_source:
            async for tweet in ravens_lions_source.collect():
                print(f"Tweet: {tweet['text'][:100]}...")
                print(f"Engagement: {tweet['metrics']['like_count']} likes")
                break

    asyncio.run(example())