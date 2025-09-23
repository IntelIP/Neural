"""
Twitter API client implementation using Neural SDK.

This module provides data collection from twitterapi.io,
a cost-effective Twitter API alternative supporting up to 200 QPS.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
import logging
import os
from decimal import Decimal

from neural.data_collection import (
    RestDataSource,
    RestConfig,
    TransformStage,
    DataPipeline
)


logger = logging.getLogger(__name__)


@dataclass
class TwitterConfig(RestConfig):
    """
    Configuration for Twitter API client.
    
    Attributes:
        base_url: API base URL (defaults to api.twitterapi.io)
        api_key: API key for authentication (loaded from env by default)
        rate_limit_requests: Maximum requests per second (200 QPS limit)
        cost_per_tweet: Cost per 1000 tweets in USD
        cost_per_user: Cost per 1000 user profiles in USD  
        cost_per_follower: Cost per 1000 followers in USD
        min_cost_per_request: Minimum charge per request in USD
        track_costs: Whether to track API usage costs
        cache_tweets_ttl: TTL for tweet data cache (seconds)
        cache_users_ttl: TTL for user data cache (seconds)
        cache_trends_ttl: TTL for trends data cache (seconds)
    """
    name: str = "twitter_api"  # Default name
    api_key: Optional[str] = None  # API key for authentication
    cost_per_tweet: Decimal = Decimal("0.15")  # Per 1000 tweets
    cost_per_user: Decimal = Decimal("0.18")   # Per 1000 user profiles
    cost_per_follower: Decimal = Decimal("0.15")  # Per 1000 followers
    min_cost_per_request: Decimal = Decimal("0.00015")
    track_costs: bool = True
    cache_tweets_ttl: int = 60  # 1 minute for recent tweets
    cache_users_ttl: int = 300  # 5 minutes for user data
    cache_trends_ttl: int = 300  # 5 minutes for trends
    
    def __post_init__(self):
        """Set Twitter-specific defaults."""
        if not self.base_url:
            self.base_url = "https://api.twitterapi.io"
        
        # Load API key from environment if not provided
        if not self.api_key:
            self.api_key = os.getenv("TWITTERAPI_IO_KEY")
            if not self.api_key:
                logger.warning(
                    "No Twitter API key found. Set TWITTERAPI_IO_KEY environment variable "
                    "or pass api_key to TwitterConfig"
                )
        
        # Twitter API supports up to 200 QPS
        if not hasattr(self, 'rate_limit_requests') or self.rate_limit_requests == 10.0:
            self.rate_limit_requests = 100.0  # Conservative limit


class TwitterDataNormalizer(TransformStage):
    """
    Transform stage to normalize Twitter API responses.
    
    Ensures consistent data format across different Twitter endpoints.
    """
    
    def __init__(self, name: str = "twitter_normalizer"):
        """Initialize the Twitter data normalizer."""
        super().__init__(name)
    
    async def process(self, data: Any) -> Any:
        """
        Normalize Twitter data format.
        
        Args:
            data: Raw Twitter API response
            
        Returns:
            Normalized data structure
        """
        if not isinstance(data, dict):
            return data
        
        normalized = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "twitter",
            "raw_data": data
        }
        
        # Identify data type and add metadata
        if "user" in data or "screen_name" in data:
            normalized["type"] = "user"
        elif "tweets" in data or "text" in data:
            normalized["type"] = "tweet"
        elif "trends" in data:
            normalized["type"] = "trends"
        elif "followers" in data or "following" in data:
            normalized["type"] = "network"
        else:
            normalized["type"] = "unknown"
        
        return normalized


class TwitterClient(RestDataSource):
    """
    Twitter API client for data collection.
    
    Provides methods for collecting tweets, user data, trends, and social graphs
    from the twitterapi.io service.
    
    Example:
        >>> twitter = TwitterClient()
        >>> user = await twitter.get_user_info("ESPN_NFL")
        >>> tweets = await twitter.search_tweets("Packers vs Commanders")
    """
    
    def __init__(self, config: Optional[TwitterConfig] = None):
        """
        Initialize Twitter client.
        
        Args:
            config: Twitter-specific configuration
        """
        if config is None:
            config = TwitterConfig()
        
        super().__init__(config)
        self.config: TwitterConfig = config
        
        # Cost tracking
        self._api_costs = {
            "total_cost": Decimal("0"),
            "requests_count": 0,
            "tweets_fetched": 0,
            "users_fetched": 0,
            "followers_fetched": 0
        }
        
        # Add API key to default headers if available
        if self.config.api_key:
            if not self.config.headers:
                self.config.headers = {}
            self.config.headers["x-api-key"] = self.config.api_key
            logger.info(f"Initialized Twitter client with API key authentication")
        else:
            logger.warning("Twitter client initialized without API key")
        
        logger.info(f"Base URL: {self.config.base_url}")
    
    def _track_cost(self, endpoint: str, count: int = 1) -> None:
        """
        Track API usage costs.
        
        Args:
            endpoint: API endpoint called
            count: Number of items fetched
        """
        if not self.config.track_costs:
            return
        
        cost = self.config.min_cost_per_request
        
        # Calculate cost based on endpoint type
        if "tweet" in endpoint or ("search" in endpoint and "user" not in endpoint):
            cost = max(cost, (Decimal(count) / 1000) * self.config.cost_per_tweet)
            self._api_costs["tweets_fetched"] += count
        elif "follower" in endpoint or "following" in endpoint:
            cost = max(cost, (Decimal(count) / 1000) * self.config.cost_per_follower)
            self._api_costs["followers_fetched"] += count
        elif "user" in endpoint or "profile" in endpoint:
            cost = max(cost, (Decimal(count) / 1000) * self.config.cost_per_user)
            self._api_costs["users_fetched"] += count
        
        self._api_costs["total_cost"] += cost
        self._api_costs["requests_count"] += 1
        
        logger.debug(f"API cost tracked: ${cost:.6f} for {endpoint}")
    
    async def get_user_info(self, username: str) -> Dict[str, Any]:
        """
        Get user profile information.
        
        Args:
            username: Twitter username (without @)
            
        Returns:
            User profile data
        """
        endpoint = "/twitter/user/info"
        params = {"userName": username}  # API uses camelCase
        
        result = await self.get(
            endpoint,
            params=params,
            cache_ttl=self.config.cache_users_ttl
        )
        
        # Handle nested response structure
        # API returns: {status, msg, data: {user info}}
        if "data" in result and isinstance(result["data"], dict):
            # Return the data directly for easier access
            user_data = result["data"]
            user_data["_status"] = result.get("status")
            user_data["_msg"] = result.get("msg")
            self._track_cost(endpoint, 1)
            return user_data
        else:
            self._track_cost(endpoint, 1)
            return result
    
    async def get_user_tweets(
        self,
        username: str,
        limit: int = 20,
        cursor: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get recent tweets from a user.
        
        Args:
            username: Twitter username
            limit: Maximum number of tweets to fetch
            cursor: Pagination cursor for next page
            
        Returns:
            User's recent tweets with pagination info
        """
        endpoint = "/twitter/user/last_tweets"
        params = {
            "userName": username,  # API uses camelCase
            "limit": limit
        }
        
        if cursor:
            params["cursor"] = cursor
        
        result = await self.get(
            endpoint,
            params=params,
            cache_ttl=self.config.cache_tweets_ttl
        )
        
        # Handle nested response structure
        # API returns: {data: {tweets: [...], pin_tweet: ...}, has_next_page, next_cursor}
        if "data" in result and isinstance(result["data"], dict):
            data = result["data"]
            tweets = data.get("tweets", [])
            
            # Flatten the response for easier use
            flattened = {
                "tweets": tweets,
                "pin_tweet": data.get("pin_tweet"),
                "has_next_page": result.get("has_next_page", False),
                "next_cursor": result.get("next_cursor"),
                "status": result.get("status"),
                "code": result.get("code")
            }
            
            # Track cost based on actual tweets returned
            self._track_cost(endpoint, len(tweets))
            
            return flattened
        else:
            # Fallback for unexpected structure
            tweet_count = len(result.get("tweets", []))
            self._track_cost(endpoint, tweet_count)
            return result
    
    async def search_tweets(
        self,
        query: str,
        limit: int = 20,
        cursor: Optional[str] = None,
        search_type: str = "Top"
    ) -> Dict[str, Any]:
        """
        Advanced search for tweets.
        
        Args:
            query: Search query (supports Twitter search operators)
            limit: Maximum results per page (max 20)
            cursor: Pagination cursor
            search_type: Type of search ("Top", "Latest", "People", "Photos", "Videos")
            
        Returns:
            Search results with tweets matching the query
        """
        endpoint = "/twitter/tweet/advanced_search"
        params = {
            "query": query,
            "limit": min(limit, 20),  # API max is 20
            "type": search_type
        }
        
        if cursor:
            params["cursor"] = cursor
        
        result = await self.get(
            endpoint,
            params=params,
            cache_ttl=30  # Short cache for search results
        )
        
        tweet_count = len(result.get("tweets", []))
        self._track_cost(endpoint, tweet_count)
        
        return result
    
    async def get_tweet(self, tweet_id: str) -> Dict[str, Any]:
        """
        Get single tweet by ID.
        
        Args:
            tweet_id: Tweet ID
            
        Returns:
            Tweet data
        """
        endpoint = "/twitter/tweets"
        params = {"ids": tweet_id}
        
        result = await self.get(
            endpoint,
            params=params,
            cache_ttl=self.config.cache_tweets_ttl
        )
        
        self._track_cost(endpoint, 1)
        return result
    
    async def get_tweets_bulk(self, tweet_ids: List[str]) -> Dict[str, Any]:
        """
        Get multiple tweets by IDs.
        
        Args:
            tweet_ids: List of tweet IDs
            
        Returns:
            Multiple tweet data
        """
        endpoint = "/twitter/tweets"
        params = {"ids": ",".join(tweet_ids)}
        
        result = await self.get(
            endpoint,
            params=params,
            cache_ttl=self.config.cache_tweets_ttl
        )
        
        self._track_cost(endpoint, len(tweet_ids))
        return result
    
    async def get_replies(
        self,
        tweet_id: str,
        cursor: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get replies to a tweet.
        
        Args:
            tweet_id: Tweet ID
            cursor: Pagination cursor
            
        Returns:
            Tweet replies
        """
        endpoint = "/twitter/tweet/replies"
        params = {"tweet_id": tweet_id}
        
        if cursor:
            params["cursor"] = cursor
        
        result = await self.get(
            endpoint,
            params=params,
            cache_ttl=60  # 1 minute cache
        )
        
        reply_count = len(result.get("replies", []))
        self._track_cost(endpoint, reply_count)
        
        return result
    
    async def get_quotes(
        self,
        tweet_id: str,
        cursor: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get quote tweets.
        
        Args:
            tweet_id: Tweet ID
            cursor: Pagination cursor
            
        Returns:
            Quote tweets
        """
        endpoint = "/twitter/tweet/quotes"
        params = {"tweet_id": tweet_id}
        
        if cursor:
            params["cursor"] = cursor
        
        result = await self.get(
            endpoint,
            params=params,
            cache_ttl=60
        )
        
        quote_count = len(result.get("quotes", []))
        self._track_cost(endpoint, quote_count)
        
        return result
    
    async def get_followers(
        self,
        username: str,
        limit: int = 100,
        cursor: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get user's followers.
        
        Args:
            username: Twitter username
            limit: Maximum followers to fetch
            cursor: Pagination cursor
            
        Returns:
            List of followers
        """
        endpoint = "/twitter/user/followers"
        params = {
            "userName": username,  # API uses camelCase
            "limit": limit
        }
        
        if cursor:
            params["cursor"] = cursor
        
        result = await self.get(
            endpoint,
            params=params,
            cache_ttl=self.config.cache_users_ttl
        )
        
        follower_count = len(result.get("followers", []))
        self._track_cost(endpoint, follower_count)
        
        return result
    
    async def get_following(
        self,
        username: str,
        limit: int = 100,
        cursor: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get users that a user follows.
        
        Args:
            username: Twitter username
            limit: Maximum following to fetch
            cursor: Pagination cursor
            
        Returns:
            List of users being followed
        """
        endpoint = "/twitter/user/followings"
        params = {
            "userName": username,  # API uses camelCase
            "limit": limit
        }
        
        if cursor:
            params["cursor"] = cursor
        
        result = await self.get(
            endpoint,
            params=params,
            cache_ttl=self.config.cache_users_ttl
        )
        
        following_count = len(result.get("following", []))
        self._track_cost(endpoint, following_count)
        
        return result
    
    async def get_trends(self, woeid: int = 1) -> Dict[str, Any]:
        """
        Get trending topics by location.
        
        Args:
            woeid: Where On Earth ID (1 = Worldwide, 23424977 = USA)
            
        Returns:
            Trending topics for the location
        """
        endpoint = "/twitter/trends"
        params = {"woeid": woeid}
        
        result = await self.get(
            endpoint,
            params=params,
            cache_ttl=self.config.cache_trends_ttl
        )
        
        self._track_cost(endpoint, 1)
        return result
    
    async def search_users(
        self,
        query: str,
        limit: int = 20,
        cursor: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search for users.
        
        Args:
            query: Search query
            limit: Maximum results
            cursor: Pagination cursor
            
        Returns:
            Users matching the search query
        """
        endpoint = "/twitter/user/search"
        params = {
            "query": query,
            "limit": limit
        }
        
        if cursor:
            params["cursor"] = cursor
        
        result = await self.get(
            endpoint,
            params=params,
            cache_ttl=self.config.cache_users_ttl
        )
        
        user_count = len(result.get("users", []))
        self._track_cost(endpoint, user_count)
        
        return result
    
    async def collect_paginated_data(
        self,
        method: str,
        max_pages: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Helper method to collect paginated data.
        
        Args:
            method: Method name to call (e.g., "search_tweets")
            max_pages: Maximum number of pages to fetch
            **kwargs: Arguments to pass to the method
            
        Returns:
            Combined results from all pages
        """
        all_results = []
        cursor = None
        
        for page in range(max_pages):
            # Get the method
            api_method = getattr(self, method)
            
            # Add cursor if available
            if cursor:
                kwargs["cursor"] = cursor
            
            # Fetch page
            result = await api_method(**kwargs)
            
            # Extract data based on method
            if "tweets" in result:
                all_results.extend(result["tweets"])
            elif "users" in result:
                all_results.extend(result["users"])
            elif "followers" in result:
                all_results.extend(result["followers"])
            elif "following" in result:
                all_results.extend(result["following"])
            else:
                all_results.append(result)
            
            # Check for next page
            cursor = result.get("next_cursor") or result.get("cursor")
            if not cursor:
                break
            
            logger.debug(f"Fetched page {page + 1} of {method}")
        
        return all_results
    
    def get_api_costs(self) -> Dict[str, Any]:
        """
        Get current API usage costs.
        
        Returns:
            Dictionary with cost tracking information
        """
        return {
            "total_cost": float(self._api_costs["total_cost"]),
            "requests_count": self._api_costs["requests_count"],
            "tweets_fetched": self._api_costs["tweets_fetched"],
            "users_fetched": self._api_costs["users_fetched"],
            "followers_fetched": self._api_costs["followers_fetched"],
            "average_cost_per_request": (
                float(self._api_costs["total_cost"] / self._api_costs["requests_count"])
                if self._api_costs["requests_count"] > 0 else 0
            )
        }
    
    def reset_cost_tracking(self) -> None:
        """Reset API cost tracking counters."""
        self._api_costs = {
            "total_cost": Decimal("0"),
            "requests_count": 0,
            "tweets_fetched": 0,
            "users_fetched": 0,
            "followers_fetched": 0
        }
        logger.info("API cost tracking reset")
    
    def create_pipeline(self) -> DataPipeline:
        """
        Create a data pipeline with Twitter-specific transformations.
        
        Returns:
            Configured DataPipeline instance
        """
        pipeline = DataPipeline()
        
        # Add normalization stage
        pipeline.add_stage(TwitterDataNormalizer())
        
        return pipeline