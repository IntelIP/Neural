"""
Twitter Search Module
Uses TwitterAPI.io's REST API for tweet searching and polling
"""

import os
import asyncio
import logging
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
import httpx

from .models import Tweet, Author, TweetMetrics
from .sentiment import SentimentAnalyzer

logger = logging.getLogger(__name__)


class TwitterSearchClient:
    """
    Client for searching tweets using TwitterAPI.io REST API
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Twitter search client
        
        Args:
            api_key: TwitterAPI.io API key
        """
        self.api_key = api_key or os.getenv('TWITTERAPI_KEY') or os.getenv('TWITTER_BEARER_TOKEN')
        if not self.api_key:
            raise ValueError("TwitterAPI.io API key required")
        
        self.base_url = "https://api.twitterapi.io"
        self.headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        # Track seen tweets to avoid duplicates
        self.seen_tweet_ids: Set[str] = set()
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = SentimentAnalyzer()
    
    async def search_tweets(
        self, 
        query: str, 
        query_type: str = "Latest",
        limit: int = 20,
        cursor: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search for tweets using advanced search endpoint
        
        Args:
            query: Search query (TwitterAPI.io syntax)
            query_type: Type of search (Latest, Top, Photos, Videos)
            limit: Number of tweets to return (max 20)
            cursor: Pagination cursor
            
        Returns:
            Dict containing tweets and metadata
        """
        url = f"{self.base_url}/twitter/tweet/advanced_search"
        
        params = {
            "query": query,
            "queryType": query_type,
            "limit": min(limit, 20)  # API max is 20
        }
        
        if cursor:
            params["cursor"] = cursor
        
        async with httpx.AsyncClient(timeout=30) as client:
            try:
                response = await client.get(url, headers=self.headers, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                # Parse tweets into model objects
                tweets = []
                for tweet_data in data.get('tweets', []):
                    tweet = self._parse_tweet(tweet_data)
                    if tweet:
                        tweets.append(tweet)
                
                return {
                    'tweets': tweets,
                    'cursor': data.get('cursor'),
                    'has_more': data.get('hasMore', False)
                }
                
            except httpx.HTTPStatusError as e:
                logger.error(f"API error: {e.response.status_code} - {e.response.text}")
                return {'tweets': [], 'cursor': None, 'has_more': False}
            except Exception as e:
                logger.error(f"Search error: {e}")
                return {'tweets': [], 'cursor': None, 'has_more': False}
    
    async def search_game_tweets(
        self,
        home_team: str,
        away_team: str,
        additional_keywords: Optional[List[str]] = None
    ) -> List[Tweet]:
        """
        Search for tweets about a specific game
        
        Args:
            home_team: Home team name
            away_team: Away team name
            additional_keywords: Extra keywords to include
            
        Returns:
            List of Tweet objects
        """
        # Build comprehensive query
        team_query = f"({home_team} OR {away_team})"
        
        if additional_keywords:
            keywords = " OR ".join(additional_keywords)
            query = f"{team_query} AND ({keywords})"
        else:
            query = team_query
        
        # Add common game-related terms
        query += " AND (game OR score OR win OR loss OR play OR touchdown OR field goal)"
        
        logger.info(f"Searching for game tweets: {query[:100]}...")
        
        result = await self.search_tweets(query, limit=20)
        return result['tweets']
    
    async def search_player_tweets(
        self,
        player_name: str,
        team: Optional[str] = None,
        keywords: Optional[List[str]] = None
    ) -> List[Tweet]:
        """
        Search for tweets about a specific player
        
        Args:
            player_name: Player's name
            team: Optional team name
            keywords: Optional keywords (injury, status, etc.)
            
        Returns:
            List of Tweet objects
        """
        query = f'"{player_name}"'
        
        if team:
            query += f" AND {team}"
        
        if keywords:
            keyword_query = " OR ".join(keywords)
            query += f" AND ({keyword_query})"
        
        logger.info(f"Searching for player tweets: {query}")
        
        result = await self.search_tweets(query, limit=20)
        return result['tweets']
    
    async def get_new_tweets(self, tweets: List[Tweet]) -> List[Tweet]:
        """
        Filter out previously seen tweets
        
        Args:
            tweets: List of tweets to filter
            
        Returns:
            List of new tweets only
        """
        new_tweets = []
        
        for tweet in tweets:
            if tweet.id not in self.seen_tweet_ids:
                self.seen_tweet_ids.add(tweet.id)
                new_tweets.append(tweet)
        
        return new_tweets
    
    async def poll_for_tweets(
        self,
        query: str,
        interval: int = 30,
        duration: int = 300,
        callback: Optional[Any] = None
    ) -> None:
        """
        Poll for new tweets at regular intervals
        
        Args:
            query: Search query
            interval: Polling interval in seconds
            duration: Total duration to poll in seconds
            callback: Optional callback for new tweets
        """
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=duration)
        poll_count = 0
        
        logger.info(f"Starting tweet polling: every {interval}s for {duration}s")
        
        while datetime.now() < end_time:
            poll_count += 1
            logger.info(f"Poll #{poll_count} at {datetime.now().strftime('%H:%M:%S')}")
            
            # Search for tweets
            result = await self.search_tweets(query)
            tweets = result['tweets']
            
            # Get only new tweets
            new_tweets = await self.get_new_tweets(tweets)
            
            if new_tweets:
                logger.info(f"Found {len(new_tweets)} new tweets")
                
                # Analyze sentiment for each tweet
                for tweet in new_tweets:
                    sentiment = self.sentiment_analyzer.analyze_tweet(tweet)
                    tweet.sentiment = sentiment
                    
                    # Log high-impact tweets
                    if sentiment.market_impact.value in ['high', 'critical']:
                        logger.info(f"High impact tweet from @{tweet.author.username}")
                        logger.info(f"  Text: {tweet.text[:100]}...")
                        logger.info(f"  Sentiment: {sentiment.score:+.2f}")
                
                # Call callback if provided
                if callback:
                    await callback(new_tweets)
            else:
                logger.info("No new tweets found")
            
            # Wait for next poll
            if datetime.now() < end_time:
                await asyncio.sleep(interval)
        
        logger.info(f"Polling complete. Total tweets collected: {len(self.seen_tweet_ids)}")
    
    def _parse_tweet(self, data: Dict[str, Any]) -> Optional[Tweet]:
        """
        Parse raw tweet data into Tweet model
        
        Args:
            data: Raw tweet data from API
            
        Returns:
            Tweet object or None if parsing fails
        """
        try:
            # Parse author
            author_data = data.get('author', {})
            author = Author(
                id=author_data.get('id', ''),
                username=author_data.get('userName', ''),
                name=author_data.get('name', ''),
                verified=author_data.get('isVerified', False) or author_data.get('isBlueVerified', False),
                followers_count=author_data.get('followers', 0),
                following_count=author_data.get('following', 0),
                tweet_count=author_data.get('statusesCount', 0),
                description=author_data.get('description')
            )
            
            # Parse metrics
            metrics = TweetMetrics(
                retweet_count=data.get('retweetCount', 0),
                reply_count=data.get('replyCount', 0),
                like_count=data.get('likeCount', 0),
                quote_count=data.get('quoteCount', 0),
                impression_count=data.get('viewCount')
            )
            
            # Parse created_at
            created_at_str = data.get('createdAt', '')
            try:
                # TwitterAPI.io format: "Sat Aug 30 02:30:11 +0000 2025"
                created_at = datetime.strptime(created_at_str, "%a %b %d %H:%M:%S %z %Y")
            except:
                created_at = datetime.now()
            
            # Create Tweet object
            tweet = Tweet(
                id=data.get('id', ''),
                text=data.get('text', ''),
                author=author,
                created_at=created_at,
                metrics=metrics,
                lang=data.get('lang', 'en'),
                conversation_id=data.get('conversationId'),
                in_reply_to_user_id=data.get('inReplyToUserId'),
                referenced_tweets=[],
                entities=data.get('entities', {}),
                context_annotations=[],
                possibly_sensitive=data.get('possiblySensitive', False)
            )
            
            return tweet
            
        except Exception as e:
            logger.error(f"Error parsing tweet: {e}")
            return None
    
    def clear_seen_tweets(self):
        """Clear the set of seen tweet IDs"""
        self.seen_tweet_ids.clear()
        logger.info("Cleared seen tweet IDs")


class GameMonitor:
    """
    Monitor tweets for a specific game
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize game monitor"""
        self.search_client = TwitterSearchClient(api_key)
        self.sentiment_analyzer = SentimentAnalyzer()
        self.game_tweets = []
        self.sentiment_history = []
    
    async def monitor_game(
        self,
        home_team: str,
        away_team: str,
        duration: int = 300,
        interval: int = 30
    ) -> Dict[str, Any]:
        """
        Monitor a game for the specified duration
        
        Args:
            home_team: Home team name
            away_team: Away team name
            duration: Monitoring duration in seconds
            interval: Polling interval in seconds
            
        Returns:
            Summary of monitoring results
        """
        logger.info(f"Starting game monitoring: {away_team} @ {home_team}")
        
        # Build search query
        query = f"({home_team} OR {away_team}) AND (game OR score OR touchdown OR win OR loss)"
        
        # Define callback for new tweets
        async def process_tweets(tweets: List[Tweet]):
            self.game_tweets.extend(tweets)
            
            # Calculate aggregate sentiment
            if tweets:
                sentiments = [t.sentiment.score for t in tweets if hasattr(t, 'sentiment')]
                if sentiments:
                    avg_sentiment = sum(sentiments) / len(sentiments)
                    self.sentiment_history.append({
                        'time': datetime.now(),
                        'sentiment': avg_sentiment,
                        'tweet_count': len(tweets)
                    })
                    
                    logger.info(f"Current sentiment: {avg_sentiment:+.2f} ({len(tweets)} tweets)")
        
        # Start polling
        await self.search_client.poll_for_tweets(
            query=query,
            interval=interval,
            duration=duration,
            callback=process_tweets
        )
        
        # Generate summary
        total_tweets = len(self.game_tweets)
        
        if self.sentiment_history:
            avg_sentiment = sum(s['sentiment'] for s in self.sentiment_history) / len(self.sentiment_history)
            sentiment_trend = "improving" if self.sentiment_history[-1]['sentiment'] > self.sentiment_history[0]['sentiment'] else "declining"
        else:
            avg_sentiment = 0
            sentiment_trend = "neutral"
        
        return {
            'total_tweets': total_tweets,
            'average_sentiment': avg_sentiment,
            'sentiment_trend': sentiment_trend,
            'monitoring_duration': duration,
            'polling_interval': interval
        }