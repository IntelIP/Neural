"""
Unit tests for Kalshi API Client

Tests cover authentication, market data retrieval, order operations,
and error handling scenarios.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone
import json

from neural.trading.kalshi_client import (
    KalshiClient, KalshiConfig, Environment, MarketData, OrderRequest
)


class TestKalshiConfig:
    """Test Kalshi configuration class."""
    
    def test_default_config_creation(self):
        """Test creating config with defaults."""
        config = KalshiConfig()
        
        assert config.name == "kalshi_trading"
        assert config.environment == Environment.DEMO
        assert config.auto_refresh_token is True
        assert config.base_url == "https://demo-api.kalshi.co/trade-api/v2"
        assert "application/json" in config.headers["accept"]
    
    def test_production_config_creation(self):
        """Test creating production config."""
        config = KalshiConfig(
            environment=Environment.PRODUCTION,
            api_key="test_key",
            private_key="test_private_key"
        )
        
        assert config.environment == Environment.PRODUCTION
        assert config.base_url == "https://api.elections.kalshi.com/trade-api/v2"
        assert config.api_key == "test_key"
        assert config.private_key == "test_private_key"
    
    @patch.dict('os.environ', {
        'KALSHI_API_KEY': 'env_api_key',
        'KALSHI_PRIVATE_KEY_PATH': '/path/to/key.pem',
        'KALSHI_USER_ID': 'env_user_id'
    })
    def test_config_from_environment(self):
        """Test loading config from environment variables."""
        config = KalshiConfig()
        
        assert config.api_key == 'env_api_key'
        assert config.private_key_path == '/path/to/key.pem'
        assert config.user_id == 'env_user_id'


class TestMarketData:
    """Test MarketData data class."""
    
    def test_market_data_creation(self):
        """Test creating MarketData object."""
        market = MarketData(
            ticker="TEST-MARKET",
            title="Test Market",
            subtitle="Test Subtitle",
            yes_bid=45,
            yes_ask=47,
            volume=1000
        )
        
        assert market.ticker == "TEST-MARKET"
        assert market.title == "Test Market"
        assert market.yes_bid == 45
        assert market.yes_ask == 47
        assert market.volume == 1000
    
    def test_mid_price_calculation(self):
        """Test mid price calculation."""
        market = MarketData(
            ticker="TEST",
            title="Test",
            subtitle="Test",
            yes_bid=48,
            yes_ask=52
        )
        
        assert market.mid_price == 0.50  # (48 + 52) / 200
    
    def test_spread_calculation(self):
        """Test bid-ask spread calculation."""
        market = MarketData(
            ticker="TEST",
            title="Test",
            subtitle="Test",
            yes_bid=48,
            yes_ask=52
        )
        
        assert market.spread == 4  # 52 - 48


class TestOrderRequest:
    """Test OrderRequest data class."""
    
    def test_order_request_creation(self):
        """Test creating OrderRequest."""
        order = OrderRequest(
            ticker="TEST-MARKET",
            client_order_id="test_order_123",
            side="yes",
            action="buy",
            type="limit",
            count=100,
            yes_price=50
        )
        
        assert order.ticker == "TEST-MARKET"
        assert order.side == "yes"
        assert order.count == 100
        assert order.yes_price == 50
    
    def test_order_to_dict(self):
        """Test converting order to dictionary."""
        order = OrderRequest(
            ticker="TEST-MARKET",
            client_order_id="test_order_123",
            side="yes",
            action="buy",
            type="limit",
            count=100,
            yes_price=50
        )
        
        order_dict = order.to_dict()
        
        assert order_dict["ticker"] == "TEST-MARKET"
        assert order_dict["side"] == "yes"
        assert order_dict["count"] == 100
        assert order_dict["yes_price"] == 50


class TestKalshiClient:
    """Test KalshiClient class."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return KalshiConfig(
            environment=Environment.DEMO,
            api_key="test_api_key",
            private_key="-----BEGIN RSA PRIVATE KEY-----\ntest_key_content\n-----END RSA PRIVATE KEY-----"
        )
    
    @pytest.fixture
    def client(self, config):
        """Create test client."""
        return KalshiClient(config)
    
    def test_client_initialization(self, client, config):
        """Test client initialization."""
        assert client.config == config
        assert client.session is None
        assert client.authenticated is False
        assert client.access_token is None
    
    @pytest.mark.asyncio
    async def test_connect(self, client):
        """Test client connection."""
        with patch('aiohttp.ClientSession') as mock_session:
            await client.connect()
            
            assert client.session is not None
            mock_session.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_disconnect(self, client):
        """Test client disconnection."""
        # Set up mock session
        mock_session = AsyncMock()
        client.session = mock_session
        
        await client.disconnect()
        
        mock_session.close.assert_called_once()
        assert client.session is None
    
    @pytest.mark.asyncio
    async def test_authentication_success(self, client):
        """Test successful authentication."""
        # Mock the private key loading
        with patch('neural.trading.kalshi_client.load_pem_private_key') as mock_load_key, \
             patch('jwt.encode') as mock_jwt, \
             patch.object(client, 'session') as mock_session:
            
            # Setup mocks
            mock_load_key.return_value = Mock()
            mock_jwt.return_value = "test_jwt_token"
            
            # Mock successful response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "token": "test_access_token",
                "refresh_token": "test_refresh_token"
            }
            mock_session.post.return_value.__aenter__.return_value = mock_response
            
            result = await client.authenticate()
            
            assert result is True
            assert client.authenticated is True
            assert client.access_token == "test_access_token"
            assert client.refresh_token == "test_refresh_token"
    
    @pytest.mark.asyncio
    async def test_authentication_failure(self, client):
        """Test authentication failure."""
        with patch('neural.trading.kalshi_client.load_pem_private_key') as mock_load_key, \
             patch('jwt.encode') as mock_jwt, \
             patch.object(client, 'session') as mock_session:
            
            # Setup mocks
            mock_load_key.return_value = Mock()
            mock_jwt.return_value = "test_jwt_token"
            
            # Mock failed response
            mock_response = AsyncMock()
            mock_response.status = 401
            mock_session.post.return_value.__aenter__.return_value = mock_response
            
            result = await client.authenticate()
            
            assert result is False
            assert client.authenticated is False
    
    @pytest.mark.asyncio
    async def test_request_success(self, client):
        """Test successful API request."""
        # Setup mock session
        mock_session = AsyncMock()
        client.session = mock_session
        
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {"data": "test_data"}
        mock_session.request.return_value.__aenter__.return_value = mock_response
        
        result = await client.request("GET", "/test")
        
        assert result == {"data": "test_data"}
    
    @pytest.mark.asyncio
    async def test_request_401_with_refresh(self, client):
        """Test request with 401 response and token refresh."""
        # Setup authenticated client
        client.authenticated = True
        client.access_token = "old_token"
        
        # Setup mock session
        mock_session = AsyncMock()
        client.session = mock_session
        
        # Mock 401 response followed by successful retry
        mock_response_401 = AsyncMock()
        mock_response_401.status = 401
        
        mock_response_success = AsyncMock()
        mock_response_success.status = 200
        mock_response_success.json.return_value = {"data": "test_data"}
        
        mock_session.request.return_value.__aenter__.side_effect = [
            mock_response_401, mock_response_success
        ]
        
        # Mock refresh token success
        with patch.object(client, 'refresh_access_token', return_value=True):
            result = await client.request("GET", "/test")
            
            assert result == {"data": "test_data"}
    
    @pytest.mark.asyncio
    async def test_get_markets(self, client):
        """Test getting markets."""
        # Mock the request method
        with patch.object(client, 'request') as mock_request:
            mock_request.return_value = {
                "markets": [
                    {"ticker": "TEST1", "title": "Test Market 1"},
                    {"ticker": "TEST2", "title": "Test Market 2"}
                ]
            }
            
            result = await client.get_markets(limit=10, series_ticker="TEST")
            
            mock_request.assert_called_once_with(
                "GET", "/markets", 
                params={"limit": 10, "series_ticker": "TEST"}
            )
            assert len(result["markets"]) == 2
    
    @pytest.mark.asyncio
    async def test_get_market_orderbook(self, client):
        """Test getting market orderbook."""
        with patch.object(client, 'request') as mock_request:
            mock_request.return_value = {
                "orderbook": {
                    "yes": [{"price": 50, "quantity": 100}],
                    "no": [{"price": 50, "quantity": 100}]
                }
            }
            
            result = await client.get_market_orderbook("TEST-MARKET")
            
            mock_request.assert_called_once_with(
                "GET", "/markets/TEST-MARKET/orderbook",
                params={"depth": 100}
            )
            assert "orderbook" in result
    
    @pytest.mark.asyncio
    async def test_create_order_success(self, client):
        """Test successful order creation."""
        client.authenticated = True
        
        order = OrderRequest(
            ticker="TEST-MARKET",
            client_order_id="test_123",
            side="yes",
            action="buy",
            type="limit",
            count=100,
            yes_price=50
        )
        
        with patch.object(client, 'request') as mock_request:
            mock_request.return_value = {
                "order_id": "exchange_order_123",
                "status": "submitted"
            }
            
            result = await client.create_order(order)
            
            mock_request.assert_called_once_with(
                "POST", "/orders",
                json=order.to_dict()
            )
            assert result["order_id"] == "exchange_order_123"
    
    @pytest.mark.asyncio
    async def test_create_order_unauthenticated(self, client):
        """Test order creation without authentication."""
        client.authenticated = False
        
        order = OrderRequest(
            ticker="TEST-MARKET",
            client_order_id="test_123",
            side="yes",
            action="buy",
            type="limit",
            count=100
        )
        
        result = await client.create_order(order)
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_balance(self, client):
        """Test getting account balance."""
        client.authenticated = True
        
        with patch.object(client, 'request') as mock_request:
            mock_request.return_value = {
                "balance": 10000,
                "payout": 500
            }
            
            result = await client.get_balance()
            
            mock_request.assert_called_once_with("GET", "/balance")
            assert result["balance"] == 10000
    
    @pytest.mark.asyncio
    async def test_get_positions(self, client):
        """Test getting user positions."""
        client.authenticated = True
        
        with patch.object(client, 'request') as mock_request:
            mock_request.return_value = {
                "positions": [
                    {"ticker": "TEST1", "position": 100},
                    {"ticker": "TEST2", "position": -50}
                ]
            }
            
            result = await client.get_positions(limit=50)
            
            mock_request.assert_called_once_with(
                "GET", "/positions",
                params={"limit": 50}
            )
            assert len(result["positions"]) == 2
    
    @pytest.mark.asyncio
    async def test_cancel_order(self, client):
        """Test order cancellation."""
        client.authenticated = True
        
        with patch.object(client, 'request') as mock_request:
            mock_request.return_value = {
                "order_id": "test_order_123",
                "status": "cancelled"
            }
            
            result = await client.cancel_order("test_order_123")
            
            mock_request.assert_called_once_with(
                "DELETE", "/orders/test_order_123"
            )
            assert result["status"] == "cancelled"
    
    @pytest.mark.asyncio
    async def test_parse_market_data(self, client):
        """Test parsing market data from API response."""
        api_market = {
            "ticker": "TEST-MARKET",
            "title": "Test Market",
            "subtitle": "Test Subtitle",
            "yes_bid": 48,
            "yes_ask": 52,
            "volume": 1000,
            "status": "open",
            "close_ts": 1694563200,  # Timestamp
            "can_close_early": True
        }
        
        market_data = await client.parse_market_data(api_market)
        
        assert isinstance(market_data, MarketData)
        assert market_data.ticker == "TEST-MARKET"
        assert market_data.yes_bid == 48
        assert market_data.yes_ask == 52
        assert market_data.volume == 1000
        assert market_data.mid_price == 0.50
    
    @pytest.mark.asyncio
    async def test_find_cfb_markets(self, client):
        """Test finding CFB markets."""
        with patch.object(client, 'get_markets') as mock_get_markets, \
             patch.object(client, 'parse_market_data') as mock_parse:
            
            mock_get_markets.return_value = {
                "markets": [
                    {"ticker": "NCAAF-TEST1", "title": "CFB Game 1"},
                    {"ticker": "NCAAF-TEST2", "title": "CFB Game 2"}
                ]
            }
            
            mock_parse.side_effect = [
                MarketData("NCAAF-TEST1", "CFB Game 1", ""),
                MarketData("NCAAF-TEST2", "CFB Game 2", "")
            ]
            
            result = await client.find_cfb_markets()
            
            assert len(result) == 2
            assert all(isinstance(market, MarketData) for market in result)
    
    @pytest.mark.asyncio
    async def test_context_manager(self, client):
        """Test using client as async context manager."""
        with patch.object(client, 'connect') as mock_connect, \
             patch.object(client, 'disconnect') as mock_disconnect:
            
            async with client:
                pass
            
            mock_connect.assert_called_once()
            mock_disconnect.assert_called_once()


# Integration-style tests (still unit tests but testing component integration)
class TestKalshiClientIntegration:
    """Integration tests for Kalshi client components."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_market_data_flow(self):
        """Test complete market data retrieval flow."""
        config = KalshiConfig(environment=Environment.DEMO)
        client = KalshiClient(config)
        
        with patch.object(client, 'request') as mock_request:
            # Mock market data response
            mock_request.return_value = {
                "markets": [{
                    "ticker": "NCAAF-TEST",
                    "title": "Test CFB Game", 
                    "subtitle": "Team A vs Team B",
                    "yes_bid": 48,
                    "yes_ask": 52,
                    "volume": 1000,
                    "status": "open"
                }]
            }
            
            # Get markets
            markets_response = await client.get_markets(series_ticker="NCAAF")
            assert len(markets_response["markets"]) == 1
            
            # Parse market data
            market_data = await client.parse_market_data(markets_response["markets"][0])
            assert market_data.ticker == "NCAAF-TEST"
            assert market_data.mid_price == 0.50
            assert market_data.spread == 4
    
    @pytest.mark.asyncio
    async def test_error_handling_chain(self):
        """Test error handling across multiple operations."""
        config = KalshiConfig()
        client = KalshiClient(config)
        
        # Test authentication failure followed by request failure
        with patch.object(client, 'session') as mock_session:
            # Mock auth failure
            mock_response = AsyncMock()
            mock_response.status = 401
            mock_session.post.return_value.__aenter__.return_value = mock_response
            
            auth_result = await client.authenticate()
            assert auth_result is False
            
            # Mock request failure
            mock_session.request.return_value.__aenter__.return_value = mock_response
            
            request_result = await client.request("GET", "/markets")
            assert request_result is None
