"""
Kalshi Trading API Client

This module provides a comprehensive client for interacting with the Kalshi API
for both market data and trading operations.

Based on Kalshi API documentation:
- Base URL: https://api.elections.kalshi.com/trade-api/v2
- Demo URL: https://demo-api.kalshi.co/trade-api/v2
- WebSocket: wss://api.elections.kalshi.com/trade-api/ws/v2
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.serialization import load_pem_private_key

from neural.data_collection.base import DataSourceConfig

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Kalshi API environments."""
    PRODUCTION = "production"
    DEMO = "demo"


@dataclass
class KalshiConfig(DataSourceConfig):
    """
    Configuration for Kalshi API client.
    
    Attributes:
        environment: Production or demo environment
        api_key: Kalshi API key (loaded from env by default)
        private_key_path: Path to RSA private key file
        private_key: Private key content (alternative to file path)
        user_id: User ID for authentication
        auto_refresh_token: Automatically refresh authentication tokens
        max_retries: Maximum number of request retries
        timeout: Request timeout in seconds
    """
    name: str = "kalshi_trading"
    environment: Environment = Environment.DEMO
    api_key: Optional[str] = None
    private_key_path: Optional[str] = None
    private_key: Optional[str] = None
    user_id: Optional[str] = None
    auto_refresh_token: bool = True
    
    # Web-specific attributes
    base_url: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    rate_limit_requests: float = 100.0
    
    def __post_init__(self):
        """Set Kalshi-specific defaults."""
        # Set base URL based on environment
        if self.environment == Environment.PRODUCTION:
            self.base_url = "https://api.elections.kalshi.com/trade-api/v2"
        else:
            self.base_url = "https://demo-api.kalshi.co/trade-api/v2"
        
        # Load credentials from environment if not provided
        if not self.api_key:
            self.api_key = os.getenv("KALSHI_API_KEY")
        
        if not self.private_key_path and not self.private_key:
            self.private_key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
        
        if not self.user_id:
            self.user_id = os.getenv("KALSHI_USER_ID")
        
        # Set default headers
        self.headers.update({
            "accept": "application/json",
            "content-type": "application/json"
        })


@dataclass
class MarketData:
    """Kalshi market data structure."""
    ticker: str
    title: str
    subtitle: str
    yes_bid: Optional[int] = None
    yes_ask: Optional[int] = None 
    no_bid: Optional[int] = None
    no_ask: Optional[int] = None
    last_price: Optional[int] = None
    volume: int = 0
    open_interest: int = 0
    status: str = "unknown"
    close_time: Optional[datetime] = None
    can_close_early: bool = False
    expiration_time: Optional[datetime] = None
    
    @property
    def mid_price(self) -> Optional[float]:
        """Calculate mid price if bid/ask available."""
        if self.yes_bid is not None and self.yes_ask is not None:
            return (self.yes_bid + self.yes_ask) / 200.0  # Convert cents to probability
        return None
    
    @property
    def spread(self) -> Optional[int]:
        """Calculate bid-ask spread in cents."""
        if self.yes_bid is not None and self.yes_ask is not None:
            return self.yes_ask - self.yes_bid
        return None


@dataclass
class OrderRequest:
    """Order request structure."""
    ticker: str
    client_order_id: str
    side: str  # "yes" or "no"
    action: str  # "buy" or "sell"
    type: str  # "market" or "limit"
    count: int
    yes_price: Optional[int] = None  # Price in cents (1-99)
    no_price: Optional[int] = None
    expiration_ts: Optional[int] = None
    buy_max_cost: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API request."""
        data = {
            "ticker": self.ticker,
            "client_order_id": self.client_order_id,
            "side": self.side,
            "action": self.action,
            "type": self.type,
            "count": self.count
        }
        
        if self.yes_price is not None:
            data["yes_price"] = self.yes_price
        if self.no_price is not None:
            data["no_price"] = self.no_price
        if self.expiration_ts is not None:
            data["expiration_ts"] = self.expiration_ts
        if self.buy_max_cost is not None:
            data["buy_max_cost"] = self.buy_max_cost
            
        return data


class KalshiClient:
    """
    Kalshi API client for trading operations.
    
    Provides methods for:
    - Authentication and session management
    - Market data retrieval
    - Order placement and management
    - Position and portfolio tracking
    - Settlement and reconciliation
    """
    
    def __init__(self, config: Optional[KalshiConfig] = None):
        """
        Initialize Kalshi client.
        
        Args:
            config: Kalshi-specific configuration
        """
        self.config = config or KalshiConfig()
        self.session: Optional[aiohttp.ClientSession] = None
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.token_expires_at: Optional[datetime] = None
        self.authenticated = False
        
        # Load private key for JWT signing
        self.private_key_obj = None
        if self.config.private_key:
            self.private_key_obj = load_pem_private_key(
                self.config.private_key.encode(), 
                password=None
            )
        elif self.config.private_key_path and os.path.exists(self.config.private_key_path):
            with open(self.config.private_key_path, 'rb') as key_file:
                self.private_key_obj = load_pem_private_key(
                    key_file.read(),
                    password=None
                )
        
        logger.info(f"Initialized Kalshi client for {self.config.environment.value} environment")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
    
    async def connect(self):
        """Establish connection to Kalshi API."""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers=self.config.headers
            )
            logger.info("Kalshi session established")
            
            # Authenticate if credentials are available
            if self.config.api_key and self.private_key_obj:
                await self.authenticate()
    
    async def disconnect(self):
        """Close connection to Kalshi API."""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("Kalshi session closed")
    
    async def authenticate(self) -> bool:
        """
        Authenticate with Kalshi API using API key and private key.
        
        Returns:
            True if authentication successful
        """
        if not self.config.api_key or not self.private_key_obj:
            logger.error("API key and private key required for authentication")
            return False
        
        try:
            # Create JWT token for authentication
            now = datetime.now(timezone.utc)
            payload = {
                "iss": self.config.api_key,
                "iat": int(now.timestamp()),
                "exp": int((now.timestamp() + 3600))  # 1 hour expiration
            }
            
            token = jwt.encode(
                payload,
                self.private_key_obj,
                algorithm="RS256"
            )
            
            # Login request
            login_data = {
                "token": token
            }
            
            async with self.session.post(
                f"{self.config.base_url}/login",
                json=login_data
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    self.access_token = data.get("token")
                    self.refresh_token = data.get("refresh_token")
                    
                    # Update session headers with access token
                    self.session.headers.update({
                        "Authorization": f"Bearer {self.access_token}"
                    })
                    
                    self.authenticated = True
                    logger.info("Successfully authenticated with Kalshi API")
                    return True
                else:
                    logger.error(f"Authentication failed: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False
    
    async def refresh_access_token(self) -> bool:
        """Refresh access token using refresh token."""
        if not self.refresh_token:
            logger.error("No refresh token available")
            return False
        
        try:
            refresh_data = {
                "token": self.refresh_token
            }
            
            async with self.session.post(
                f"{self.config.base_url}/login/refresh",
                json=refresh_data
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    self.access_token = data.get("token")
                    
                    # Update session headers
                    self.session.headers.update({
                        "Authorization": f"Bearer {self.access_token}"
                    })
                    
                    logger.info("Access token refreshed successfully")
                    return True
                else:
                    logger.error(f"Token refresh failed: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Token refresh error: {e}")
            return False
    
    async def request(
        self, 
        method: str, 
        endpoint: str, 
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Make authenticated request to Kalshi API.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            **kwargs: Additional request parameters
            
        Returns:
            Response data or None if failed
        """
        if not self.session:
            await self.connect()
        
        url = f"{self.config.base_url}{endpoint}"
        
        try:
            async with self.session.request(method, url, **kwargs) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 401 and self.authenticated:
                    # Token expired, try to refresh
                    if await self.refresh_access_token():
                        # Retry the request
                        async with self.session.request(method, url, **kwargs) as retry_response:
                            if retry_response.status == 200:
                                return await retry_response.json()
                
                logger.error(f"API request failed: {response.status} {await response.text()}")
                return None
                
        except Exception as e:
            logger.error(f"Request error: {e}")
            return None
    
    # Market Data Methods
    
    async def get_markets(
        self, 
        limit: int = 1000,
        cursor: Optional[str] = None,
        event_ticker: Optional[str] = None,
        series_ticker: Optional[str] = None,
        max_close_ts: Optional[int] = None,
        min_close_ts: Optional[int] = None,
        status: Optional[str] = None,
        tickers: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get markets data.
        
        Args:
            limit: Maximum number of markets to return
            cursor: Pagination cursor
            event_ticker: Filter by event ticker
            series_ticker: Filter by series ticker  
            max_close_ts: Maximum close timestamp
            min_close_ts: Minimum close timestamp
            status: Market status filter
            tickers: Comma-separated list of tickers
            
        Returns:
            Markets data
        """
        params = {"limit": limit}
        
        if cursor:
            params["cursor"] = cursor
        if event_ticker:
            params["event_ticker"] = event_ticker
        if series_ticker:
            params["series_ticker"] = series_ticker
        if max_close_ts:
            params["max_close_ts"] = max_close_ts
        if min_close_ts:
            params["min_close_ts"] = min_close_ts
        if status:
            params["status"] = status
        if tickers:
            params["tickers"] = tickers
        
        return await self.request("GET", "/markets", params=params)
    
    async def get_market(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get specific market data.
        
        Args:
            ticker: Market ticker
            
        Returns:
            Market data
        """
        return await self.request("GET", f"/markets/{ticker}")
    
    async def get_market_orderbook(self, ticker: str, depth: int = 100) -> Optional[Dict[str, Any]]:
        """
        Get market orderbook.
        
        Args:
            ticker: Market ticker
            depth: Orderbook depth
            
        Returns:
            Orderbook data
        """
        params = {"depth": depth}
        return await self.request("GET", f"/markets/{ticker}/orderbook", params=params)
    
    async def get_market_history(
        self, 
        ticker: str,
        limit: int = 1000,
        cursor: Optional[str] = None,
        max_ts: Optional[int] = None,
        min_ts: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get market trading history.
        
        Args:
            ticker: Market ticker
            limit: Maximum number of trades
            cursor: Pagination cursor
            max_ts: Maximum timestamp
            min_ts: Minimum timestamp
            
        Returns:
            Trading history
        """
        params = {"limit": limit}
        
        if cursor:
            params["cursor"] = cursor
        if max_ts:
            params["max_ts"] = max_ts
        if min_ts:
            params["min_ts"] = min_ts
        
        return await self.request("GET", f"/markets/{ticker}/history", params=params)
    
    # Series and Events
    
    async def get_series(self, limit: int = 1000, cursor: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get series data."""
        params = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        return await self.request("GET", "/series", params=params)
    
    async def get_events(
        self, 
        limit: int = 1000,
        cursor: Optional[str] = None,
        series_ticker: Optional[str] = None,
        status: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get events data."""
        params = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        if series_ticker:
            params["series_ticker"] = series_ticker
        if status:
            params["status"] = status
        return await self.request("GET", "/events", params=params)
    
    # Trading Operations (require authentication)
    
    async def create_order(self, order: OrderRequest) -> Optional[Dict[str, Any]]:
        """
        Create a new order.
        
        Args:
            order: Order request details
            
        Returns:
            Order creation response
        """
        if not self.authenticated:
            logger.error("Authentication required for order placement")
            return None
        
        return await self.request("POST", "/orders", json=order.to_dict())
    
    async def get_orders(
        self,
        limit: int = 1000,
        cursor: Optional[str] = None,
        ticker: Optional[str] = None,
        status: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get user orders."""
        if not self.authenticated:
            logger.error("Authentication required")
            return None
        
        params = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        if ticker:
            params["ticker"] = ticker
        if status:
            params["status"] = status
        
        return await self.request("GET", "/orders", params=params)
    
    async def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get specific order."""
        if not self.authenticated:
            logger.error("Authentication required")
            return None
        
        return await self.request("GET", f"/orders/{order_id}")
    
    async def cancel_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Cancel an order."""
        if not self.authenticated:
            logger.error("Authentication required")
            return None
        
        return await self.request("DELETE", f"/orders/{order_id}")
    
    async def decrease_order(self, order_id: str, reduce_by: int) -> Optional[Dict[str, Any]]:
        """Decrease order size."""
        if not self.authenticated:
            logger.error("Authentication required")
            return None
        
        data = {"reduce_by": reduce_by}
        return await self.request("POST", f"/orders/{order_id}/decrease", json=data)
    
    # Portfolio and Positions
    
    async def get_balance(self) -> Optional[Dict[str, Any]]:
        """Get account balance."""
        if not self.authenticated:
            logger.error("Authentication required")
            return None
        
        return await self.request("GET", "/balance")
    
    async def get_positions(
        self,
        limit: int = 1000,
        cursor: Optional[str] = None,
        settlement_status: Optional[str] = None,
        ticker: Optional[str] = None,
        event_ticker: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get user positions."""
        if not self.authenticated:
            logger.error("Authentication required")
            return None
        
        params = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        if settlement_status:
            params["settlement_status"] = settlement_status
        if ticker:
            params["ticker"] = ticker
        if event_ticker:
            params["event_ticker"] = event_ticker
        
        return await self.request("GET", "/positions", params=params)
    
    async def get_portfolio_settlements(
        self,
        limit: int = 1000,
        cursor: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get portfolio settlements."""
        if not self.authenticated:
            logger.error("Authentication required")
            return None
        
        params = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        
        return await self.request("GET", "/portfolio/settlements", params=params)
    
    async def get_fills(
        self,
        limit: int = 1000,
        cursor: Optional[str] = None,
        order_id: Optional[str] = None,
        ticker: Optional[str] = None,
        max_ts: Optional[int] = None,
        min_ts: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """Get trade fills."""
        if not self.authenticated:
            logger.error("Authentication required")
            return None
        
        params = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        if order_id:
            params["order_id"] = order_id
        if ticker:
            params["ticker"] = ticker
        if max_ts:
            params["max_ts"] = max_ts
        if min_ts:
            params["min_ts"] = min_ts
        
        return await self.request("GET", "/fills", params=params)
    
    # Convenience Methods
    
    async def parse_market_data(self, market: Dict[str, Any]) -> MarketData:
        """Parse API market response into MarketData object."""
        return MarketData(
            ticker=market.get("ticker", ""),
            title=market.get("title", ""),
            subtitle=market.get("subtitle", ""),
            yes_bid=market.get("yes_bid"),
            yes_ask=market.get("yes_ask"),
            no_bid=market.get("no_bid"),
            no_ask=market.get("no_ask"),
            last_price=market.get("last_price"),
            volume=market.get("volume", 0),
            open_interest=market.get("open_interest", 0),
            status=market.get("status", "unknown"),
            close_time=datetime.fromtimestamp(market["close_ts"]) if market.get("close_ts") else None,
            can_close_early=market.get("can_close_early", False),
            expiration_time=datetime.fromtimestamp(market["expiration_ts"]) if market.get("expiration_ts") else None
        )
    
    async def find_cfb_markets(self, limit: int = 100) -> List[MarketData]:
        """Find CFB-related markets."""
        # Search for college football markets
        markets_data = await self.get_markets(
            limit=limit,
            series_ticker="NCAAF"  # College football series
        )
        
        if not markets_data or "markets" not in markets_data:
            return []
        
        cfb_markets = []
        for market in markets_data["markets"]:
            market_data = await self.parse_market_data(market)
            cfb_markets.append(market_data)
        
        return cfb_markets
    
    async def get_cfb_market_by_teams(self, home_team: str, away_team: str) -> Optional[MarketData]:
        """Find CFB market by team names."""
        cfb_markets = await self.find_cfb_markets()
        
        # Simple matching - in production would use more sophisticated matching
        for market in cfb_markets:
            title_lower = market.title.lower()
            if (home_team.lower() in title_lower and 
                away_team.lower() in title_lower):
                return market
        
        return None
