"""
Tests for Twitter API client.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from decimal import Decimal

from neural.social import TwitterClient, TwitterConfig
from neural.data_collection import ConnectionState


@pytest.fixture
def twitter_config():
    """Create test Twitter configuration."""
    return TwitterConfig(
        name="test_twitter",
        base_url="https://twitterapi.io",
        rate_limit_requests=10.0,
        cache_tweets_ttl=60,
        track_costs=True
    )


@pytest.fixture
def twitter_client(twitter_config):
    """Create Twitter client instance."""
    return TwitterClient(twitter_config)


class TestTwitterConfig:
    """Test TwitterConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TwitterConfig()
        
        assert config.name == "twitter_api"
        assert config.base_url == "https://twitterapi.io"
        assert config.rate_limit_requests == 100.0
        assert config.cost_per_tweet == Decimal("0.15")
        assert config.cost_per_user == Decimal("0.18")
        assert config.cost_per_follower == Decimal("0.15")
        assert config.min_cost_per_request == Decimal("0.00015")
        assert config.track_costs is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = TwitterConfig(
            name="custom_twitter",
            rate_limit_requests=50.0,
            cache_tweets_ttl=120,
            track_costs=False
        )
        
        assert config.name == "custom_twitter"
        assert config.rate_limit_requests == 50.0
        assert config.cache_tweets_ttl == 120
        assert config.track_costs is False


class TestTwitterClient:
    """Test TwitterClient class."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, twitter_client):
        """Test client initialization."""
        assert twitter_client.config.name == "test_twitter"
        assert twitter_client.state == ConnectionState.DISCONNECTED
        assert twitter_client._api_costs["total_cost"] == Decimal("0")
        assert twitter_client._api_costs["requests_count"] == 0
    
    @pytest.mark.asyncio
    async def test_get_user_info(self, twitter_client):
        """Test getting user information."""
        mock_response = {
            "user": {
                "id": "123",
                "screen_name": "ESPN_NFL",
                "name": "ESPN NFL",
                "followers_count": 1000000
            }
        }
        
        with patch.object(twitter_client, 'get', new=AsyncMock(return_value=mock_response)):
            result = await twitter_client.get_user_info("ESPN_NFL")
            
            assert result == mock_response
            assert twitter_client._api_costs["requests_count"] == 1
            assert twitter_client._api_costs["users_fetched"] == 1
    
    @pytest.mark.asyncio
    async def test_search_tweets(self, twitter_client):
        """Test searching tweets."""
        mock_response = {
            "tweets": [
                {"id": "1", "text": "Go Packers!"},
                {"id": "2", "text": "Commanders win!"}
            ],
            "cursor": "next_page_cursor"
        }
        
        with patch.object(twitter_client, 'get', new=AsyncMock(return_value=mock_response)):
            result = await twitter_client.search_tweets(
                query="Packers Commanders",
                limit=20
            )
            
            assert "tweets" in result
            assert len(result["tweets"]) == 2
            assert twitter_client._api_costs["tweets_fetched"] == 2
    
    @pytest.mark.asyncio
    async def test_get_user_tweets(self, twitter_client):
        """Test getting user tweets."""
        mock_response = {
            "tweets": [
                {"id": "1", "text": "Breaking news!"},
                {"id": "2", "text": "Game update!"},
                {"id": "3", "text": "Final score!"}
            ]
        }
        
        with patch.object(twitter_client, 'get', new=AsyncMock(return_value=mock_response)):
            result = await twitter_client.get_user_tweets("NFL", limit=3)
            
            assert len(result["tweets"]) == 3
            assert twitter_client._api_costs["tweets_fetched"] == 3
    
    @pytest.mark.asyncio
    async def test_get_followers(self, twitter_client):
        """Test getting followers."""
        mock_response = {
            "followers": [
                {"id": "1", "screen_name": "user1"},
                {"id": "2", "screen_name": "user2"}
            ]
        }
        
        with patch.object(twitter_client, 'get', new=AsyncMock(return_value=mock_response)):
            result = await twitter_client.get_followers("ESPN_NFL", limit=100)
            
            assert len(result["followers"]) == 2
            assert twitter_client._api_costs["followers_fetched"] == 2
    
    @pytest.mark.asyncio
    async def test_get_trends(self, twitter_client):
        """Test getting trends."""
        mock_response = {
            "trends": [
                {"name": "#NFL", "volume": 50000},
                {"name": "#SuperBowl", "volume": 30000}
            ]
        }
        
        with patch.object(twitter_client, 'get', new=AsyncMock(return_value=mock_response)):
            result = await twitter_client.get_trends(woeid=23424977)
            
            assert "trends" in result
            assert twitter_client._api_costs["requests_count"] == 1
    
    @pytest.mark.asyncio
    async def test_cost_tracking(self, twitter_client):
        """Test API cost tracking."""
        # Simulate multiple API calls
        twitter_client._track_cost("/twitter/tweet/search", 100)  # 100 tweets
        twitter_client._track_cost("/twitter/user/info", 5)  # 5 users
        twitter_client._track_cost("/twitter/user/followers", 200)  # 200 followers
        
        costs = twitter_client.get_api_costs()
        
        assert costs["requests_count"] == 3
        assert costs["tweets_fetched"] == 100
        assert costs["users_fetched"] == 5
        assert costs["followers_fetched"] == 200
        
        # Check cost calculations
        # 100 tweets: 0.1 * 0.15 = 0.015
        # 5 users: 0.005 * 0.18 = 0.0009
        # 200 followers: 0.2 * 0.15 = 0.03
        # But minimum is 0.00015 per request
        assert costs["total_cost"] > 0
    
    @pytest.mark.asyncio
    async def test_reset_cost_tracking(self, twitter_client):
        """Test resetting cost tracking."""
        # Add some costs
        twitter_client._track_cost("/twitter/tweet/search", 50)
        
        # Reset
        twitter_client.reset_cost_tracking()
        
        costs = twitter_client.get_api_costs()
        assert costs["total_cost"] == 0
        assert costs["requests_count"] == 0
        assert costs["tweets_fetched"] == 0
    
    @pytest.mark.asyncio
    async def test_collect_paginated_data(self, twitter_client):
        """Test paginated data collection."""
        # Mock responses for pagination
        page1 = {
            "tweets": [{"id": "1"}, {"id": "2"}],
            "next_cursor": "cursor_2"
        }
        page2 = {
            "tweets": [{"id": "3"}, {"id": "4"}],
            "next_cursor": "cursor_3"
        }
        page3 = {
            "tweets": [{"id": "5"}],
            "next_cursor": None  # No more pages
        }
        
        # Mock the search_tweets method to return different pages
        call_count = 0
        async def mock_search(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return page1
            elif call_count == 2:
                return page2
            else:
                return page3
        
        with patch.object(twitter_client, 'search_tweets', new=mock_search):
            all_tweets = await twitter_client.collect_paginated_data(
                method="search_tweets",
                max_pages=5,
                query="test"
            )
            
            assert len(all_tweets) == 5
            assert all_tweets[0]["id"] == "1"
            assert all_tweets[4]["id"] == "5"
    
    @pytest.mark.asyncio
    async def test_get_replies(self, twitter_client):
        """Test getting tweet replies."""
        mock_response = {
            "replies": [
                {"id": "r1", "text": "Great point!"},
                {"id": "r2", "text": "I disagree"}
            ]
        }
        
        with patch.object(twitter_client, 'get', new=AsyncMock(return_value=mock_response)):
            result = await twitter_client.get_replies("tweet123")
            
            assert len(result["replies"]) == 2
            assert twitter_client._api_costs["tweets_fetched"] == 2
    
    @pytest.mark.asyncio
    async def test_get_quotes(self, twitter_client):
        """Test getting quote tweets."""
        mock_response = {
            "quotes": [
                {"id": "q1", "text": "This is important"}
            ]
        }
        
        with patch.object(twitter_client, 'get', new=AsyncMock(return_value=mock_response)):
            result = await twitter_client.get_quotes("tweet123")
            
            assert len(result["quotes"]) == 1
            assert twitter_client._api_costs["tweets_fetched"] == 1
    
    @pytest.mark.asyncio
    async def test_search_users(self, twitter_client):
        """Test searching users."""
        mock_response = {
            "users": [
                {"id": "1", "screen_name": "user1"},
                {"id": "2", "screen_name": "user2"}
            ]
        }
        
        with patch.object(twitter_client, 'get', new=AsyncMock(return_value=mock_response)):
            result = await twitter_client.search_users("NFL", limit=20)
            
            assert len(result["users"]) == 2
            assert twitter_client._api_costs["users_fetched"] == 2
    
    @pytest.mark.asyncio
    async def test_get_tweets_bulk(self, twitter_client):
        """Test getting multiple tweets by IDs."""
        mock_response = {
            "tweets": [
                {"id": "1", "text": "Tweet 1"},
                {"id": "2", "text": "Tweet 2"},
                {"id": "3", "text": "Tweet 3"}
            ]
        }
        
        with patch.object(twitter_client, 'get', new=AsyncMock(return_value=mock_response)):
            result = await twitter_client.get_tweets_bulk(["1", "2", "3"])
            
            twitter_client.get.assert_called_once()
            call_args = twitter_client.get.call_args
            assert call_args[1]["params"]["ids"] == "1,2,3"
            assert twitter_client._api_costs["tweets_fetched"] == 3
    
    def test_cost_tracking_disabled(self):
        """Test when cost tracking is disabled."""
        config = TwitterConfig(track_costs=False)
        client = TwitterClient(config)
        
        # Track some costs
        client._track_cost("/twitter/tweet/search", 100)
        
        # Costs should not be tracked
        costs = client.get_api_costs()
        assert costs["total_cost"] == 0
        assert costs["tweets_fetched"] == 0


class TestTwitterDataNormalizer:
    """Test TwitterDataNormalizer transform stage."""
    
    @pytest.mark.asyncio
    async def test_normalize_user_data(self):
        """Test normalizing user data."""
        from neural.social.twitter_client import TwitterDataNormalizer
        
        normalizer = TwitterDataNormalizer()
        
        raw_data = {
            "user": {
                "id": "123",
                "screen_name": "test_user"
            }
        }
        
        result = await normalizer.process(raw_data)
        
        assert result["source"] == "twitter"
        assert result["type"] == "user"
        assert "timestamp" in result
        assert result["raw_data"] == raw_data
    
    @pytest.mark.asyncio
    async def test_normalize_tweet_data(self):
        """Test normalizing tweet data."""
        from neural.social.twitter_client import TwitterDataNormalizer
        
        normalizer = TwitterDataNormalizer()
        
        raw_data = {
            "tweets": [
                {"id": "1", "text": "Hello"}
            ]
        }
        
        result = await normalizer.process(raw_data)
        
        assert result["type"] == "tweet"
        assert result["source"] == "twitter"
    
    @pytest.mark.asyncio
    async def test_normalize_trends_data(self):
        """Test normalizing trends data."""
        from neural.social.twitter_client import TwitterDataNormalizer
        
        normalizer = TwitterDataNormalizer()
        
        raw_data = {
            "trends": [
                {"name": "#NFL"}
            ]
        }
        
        result = await normalizer.process(raw_data)
        
        assert result["type"] == "trends"
    
    @pytest.mark.asyncio
    async def test_normalize_non_dict(self):
        """Test normalizing non-dictionary data."""
        from neural.social.twitter_client import TwitterDataNormalizer
        
        normalizer = TwitterDataNormalizer()
        
        result = await normalizer.process("string_data")
        
        assert result == "string_data"