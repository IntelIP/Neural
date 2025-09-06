"""
Kalshi REST API Adapter

Adapts existing Kalshi client to the unified REST data source framework.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from ..base.rest_source import RESTDataSource
from ..base.auth_strategies import RSASignatureAuth
from neural_sdk.data_pipeline.data_sources.kalshi.client import KalshiClient

logger = logging.getLogger(__name__)


class KalshiRESTAdapter(RESTDataSource):
    """
    REST adapter for Kalshi API.
    
    Provides unified interface for Kalshi market data while
    leveraging existing authentication and client implementations.
    """
    
    def __init__(self, config: Optional[Any] = None):
        """
        Initialize Kalshi REST adapter.
        
        Args:
            config: KalshiConfig object or None to use environment
        """
        # Use existing Kalshi client for compatibility
        self.kalshi_client = KalshiClient(config)
        self.config = self.kalshi_client.config
        
        # Create RSA auth strategy
        auth_strategy = RSASignatureAuth(
            api_key_id=self.config.api_key_id,
            private_key_str=self.config.private_key
        )
        
        # Initialize base class
        super().__init__(
            base_url=self.config.api_base_url,
            name="KalshiREST",
            auth_strategy=auth_strategy,
            timeout=30,
            cache_ttl=10,  # Short cache for market data
            rate_limit=30,  # Kalshi allows ~30 requests/second
            max_retries=3
        )
        
        logger.info("Kalshi REST adapter initialized")
    
    async def validate_response(self, response) -> bool:
        """
        Validate Kalshi API response.
        
        Args:
            response: HTTP response object
            
        Returns:
            True if valid, False otherwise
        """
        if response.status_code == 200:
            return True
        
        if response.status_code == 401:
            logger.error("Kalshi authentication failed")
        elif response.status_code == 429:
            logger.warning("Kalshi rate limit exceeded")
        elif response.status_code >= 500:
            logger.error(f"Kalshi server error: {response.status_code}")
        
        return False
    
    async def transform_response(self, data: Any, endpoint: str) -> Dict:
        """
        Transform Kalshi response to standardized format.
        
        Args:
            data: Raw Kalshi response
            endpoint: The endpoint that was called
            
        Returns:
            Standardized response
        """
        return {
            "source": "kalshi",
            "endpoint": endpoint,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": {
                "environment": self.config.environment,
                "api_version": "v2"
            }
        }
    
    # Market Data Methods
    
    async def get_markets(
        self,
        limit: int = 100,
        status: Optional[str] = None,
        ticker: Optional[str] = None,
        series_ticker: Optional[str] = None,
        **kwargs
    ) -> Dict:
        """
        Get markets from Kalshi.
        
        Args:
            limit: Maximum number of markets to return
            status: Market status filter
            ticker: Specific market ticker
            series_ticker: Series ticker filter
            **kwargs: Additional filters
            
        Returns:
            Markets data
        """
        params = {
            "limit": limit,
            **kwargs
        }
        
        if status:
            params["status"] = status
        if ticker:
            params["ticker"] = ticker
        if series_ticker:
            params["series_ticker"] = series_ticker
        
        return await self.fetch("/markets", params=params)
    
    async def get_market(self, ticker: str) -> Dict:
        """
        Get single market by ticker.
        
        Args:
            ticker: Market ticker
            
        Returns:
            Market data
        """
        return await self.fetch(f"/markets/{ticker}")
    
    async def get_market_orderbook(self, ticker: str, depth: int = 10) -> Dict:
        """
        Get market orderbook.
        
        Args:
            ticker: Market ticker
            depth: Orderbook depth
            
        Returns:
            Orderbook data
        """
        return await self.fetch(
            f"/markets/{ticker}/orderbook",
            params={"depth": depth}
        )
    
    async def get_market_history(
        self,
        ticker: str,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        limit: int = 100
    ) -> Dict:
        """
        Get market price history.
        
        Args:
            ticker: Market ticker
            start_ts: Start timestamp
            end_ts: End timestamp
            limit: Maximum results
            
        Returns:
            Price history data
        """
        params = {"limit": limit}
        
        if start_ts:
            params["start_ts"] = start_ts
        if end_ts:
            params["end_ts"] = end_ts
        
        return await self.fetch(f"/markets/{ticker}/history", params=params)
    
    # Series Methods
    
    async def get_series(self, series_ticker: str) -> Dict:
        """
        Get series information.
        
        Args:
            series_ticker: Series ticker
            
        Returns:
            Series data
        """
        return await self.fetch(f"/series/{series_ticker}")
    
    # Event Methods
    
    async def get_events(
        self,
        limit: int = 100,
        status: Optional[str] = None,
        series_ticker: Optional[str] = None,
        with_nested_markets: bool = False,
        cursor: Optional[str] = None,
        **kwargs
    ) -> Dict:
        """
        Get events.
        
        Args:
            limit: Maximum number of events
            status: Event status filter
            series_ticker: Series ticker filter
            with_nested_markets: Include market details
            cursor: Pagination cursor
            **kwargs: Additional filters
            
        Returns:
            Events data
        """
        params = {
            "limit": limit,
            "with_nested_markets": with_nested_markets,
            **kwargs
        }
        
        if status:
            params["status"] = status
        if series_ticker:
            params["series_ticker"] = series_ticker
        if cursor:
            params["cursor"] = cursor
        
        return await self.fetch("/events", params=params)
    
    async def get_event(self, event_ticker: str) -> Dict:
        """
        Get single event.
        
        Args:
            event_ticker: Event ticker
            
        Returns:
            Event data
        """
        return await self.fetch(f"/events/{event_ticker}")

    # Internal helpers

    async def _paginate_events(
        self,
        status: Optional[str] = None,
        series_ticker: Optional[str] = None,
        with_nested_markets: bool = True,
        limit_per_page: int = 200,
        max_pages: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Fetch events with cursor pagination.

        Args:
            status: Optional event status (e.g., 'open')
            series_ticker: Optional series filter
            with_nested_markets: Include nested markets if supported
            limit_per_page: Page size per request
            max_pages: Maximum number of pages to fetch
            **kwargs: Additional filters

        Returns:
            List of events
        """
        events: List[Dict[str, Any]] = []
        cursor: Optional[str] = None

        for _ in range(max_pages):
            resp = await self.get_events(
                limit=limit_per_page,
                status=status,
                series_ticker=series_ticker,
                with_nested_markets=with_nested_markets,
                cursor=cursor,
                **kwargs,
            )

            data = resp.get("data", {}) or {}
            page_events = []
            if isinstance(data, dict):
                page_events = data.get("events", []) or data.get("data", {}).get("events", [])
                cursor = data.get("cursor") or data.get("next_cursor") or data.get("nextCursor")
            elif isinstance(data, list):
                page_events = data
                cursor = None

            if not page_events:
                break

            events.extend(page_events)
            if not cursor:
                break

        return events

    # Specialized Methods for Sports
    
    async def get_game_markets(self, sport: Optional[str] = None) -> Dict:
        """
        Get game/sports betting markets.
        
        Args:
            sport: Optional sport filter ("soccer", "nfl", "bundesliga", "epl")
            
        Returns:
            Game markets
        """
        # Prefer events with nested markets and paginate
        events = await self._paginate_events(
            status="open",
            with_nested_markets=True,
            limit_per_page=200,
            max_pages=5,
        )

        # Flatten markets from events (if present) and include event context
        flattened: List[Dict[str, Any]] = []
        for ev in events:
            ev_title = (ev.get("title") or "")
            ev_title_lower = ev_title.lower()
            ev_series = ev.get("series_ticker") or ev.get("series")
            ev_markets = ev.get("markets") or []
            for m in ev_markets:
                item = dict(m)
                item.setdefault("event_ticker", ev.get("ticker"))
                item.setdefault("event_title", ev_title)
                if ev_series:
                    item.setdefault("event_series_ticker", ev_series)
                flattened.append(item)

        # If no nested markets were returned, fall back to markets scan
        markets: List[Dict[str, Any]]
        if flattened:
            markets = flattened
        else:
            fallback = await self.get_markets(status="open", limit=500)
            data = fallback.get("data", {}) or {}
            markets = data.get("markets", []) if isinstance(data, dict) else []

        # Case-insensitive keywords
        keyword_list = ["vs", "winner", "win", "beat", "defeat", "match", "game"]
        game_markets: List[Dict[str, Any]] = []

        for market in markets:
            title = market.get("title") or market.get("event_title") or ""
            title_lower = title.lower()

            if any(kw in title_lower for kw in keyword_list):
                # Simple categorization by hints in title; best-effort only
                sport_label = "Other"
                league_label = "Various"
                tl = title_lower
                if any(k in tl for k in ["bayern", "dortmund", "hamburg", "munich"]):
                    sport_label = "Bundesliga"
                    league_label = "German Bundesliga"
                elif any(k in tl for k in ["liverpool", "chelsea", "manchester", "arsenal", "tottenham"]):
                    sport_label = "EPL"
                    league_label = "English Premier League"
                elif "nfl" in tl or "football" in tl:
                    sport_label = "NFL"
                    league_label = "National Football League"

                # Optional filter by requested sport
                if sport:
                    s = sport.lower()
                    if s == "soccer" and sport_label in ["Bundesliga", "EPL"]:
                        pass
                    elif s != sport_label.lower():
                        continue

                market["sport"] = sport_label
                market["league"] = league_label
                game_markets.append(market)

        # Return standardized response
        return await self.transform_response({"markets": game_markets}, "/events:game_markets")
    
    async def get_soccer_markets(self) -> Dict:
        """
        Get soccer/football betting markets.
        
        Returns:
            Soccer markets (Bundesliga, EPL, etc.)
        """
        return await self.get_game_markets(sport="soccer")
    
    async def get_nfl_markets(self, week: Optional[int] = None) -> Dict:
        """
        Get NFL-related markets.
        Uses events with nested markets when available.
        
        Args:
            week: NFL week number (for filtering)
            
        Returns:
            NFL-related markets
        """
        # Fetch events first, with nested markets
        events = await self._paginate_events(
            status="open",
            with_nested_markets=True,
            limit_per_page=200,
            max_pages=5,
        )

        markets: List[Dict[str, Any]] = []
        for ev in events:
            ev_title = (ev.get("title") or "")
            ev_markets = ev.get("markets") or []
            for m in ev_markets:
                item = dict(m)
                item.setdefault("event_title", ev_title)
                markets.append(item)

        # Fallback if no nested markets
        if not markets:
            fallback = await self.get_markets(status="open", limit=500)
            data = fallback.get("data", {}) or {}
            markets = data.get("markets", []) if isinstance(data, dict) else []

        nfl_markets: List[Dict[str, Any]] = []
        for market in markets:
            title = (market.get("title") or market.get("event_title") or "").lower()
            ticker = (market.get("ticker") or "").lower()
            if "nfl" in title or "football" in title or "nfl" in ticker:
                if week:
                    if f"week {week}" in title:
                        market["sport"] = "NFL"
                        market["league"] = "National Football League"
                        nfl_markets.append(market)
                else:
                    market["sport"] = "NFL"
                    market["league"] = "National Football League"
                    nfl_markets.append(market)

        return await self.transform_response({"markets": nfl_markets}, "/events:nfl_markets")
    
    async def get_cfb_markets(self, week: Optional[int] = None) -> Dict:
        """
        Get college football markets.
        Uses events with nested markets when available.
        
        Args:
            week: College football week number
            
        Returns:
            CFB markets
        """
        # Fetch events first, with nested markets
        events = await self._paginate_events(
            status="open",
            with_nested_markets=True,
            limit_per_page=200,
            max_pages=5,
        )

        markets: List[Dict[str, Any]] = []
        for ev in events:
            ev_title = (ev.get("title") or "")
            ev_markets = ev.get("markets") or []
            for m in ev_markets:
                item = dict(m)
                item.setdefault("event_title", ev_title)
                markets.append(item)

        # Fallback if no nested markets
        if not markets:
            fallback = await self.get_markets(status="open", limit=500)
            data = fallback.get("data", {}) or {}
            markets = data.get("markets", []) if isinstance(data, dict) else []

        cfb_keywords = ["cfb", "college football", "ncaa football", "bowl"]
        cfb_markets: List[Dict[str, Any]] = []
        for market in markets:
            title = (market.get("title") or market.get("event_title") or "").lower()
            if any(kw in title for kw in cfb_keywords):
                if week:
                    if f"week {week}" in title:
                        market["sport"] = "CFB"
                        market["league"] = "College Football"
                        cfb_markets.append(market)
                else:
                    market["sport"] = "CFB"
                    market["league"] = "College Football"
                    cfb_markets.append(market)

        return await self.transform_response({"markets": cfb_markets}, "/events:cfb_markets")
    
    # Batch Operations
    
    async def get_multiple_markets(self, tickers: List[str]) -> Dict:
        """
        Get multiple markets in parallel.
        
        Args:
            tickers: List of market tickers
            
        Returns:
            Dictionary of market data by ticker
        """
        requests = [
            {"endpoint": f"/markets/{ticker}"}
            for ticker in tickers
        ]
        
        results = await self.batch_fetch(requests)
        
        # Map results to tickers
        market_data = {}
        for ticker, result in zip(tickers, results):
            if not isinstance(result, Exception):
                market_data[ticker] = result
            else:
                logger.error(f"Failed to fetch market {ticker}: {result}")
                market_data[ticker] = None
        
        return {
            "source": "kalshi",
            "data": market_data,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # Portfolio Methods (if authenticated)
    
    async def get_portfolio(self) -> Dict:
        """
        Get user portfolio.
        
        Returns:
            Portfolio data
        """
        return await self.fetch("/portfolio")
    
    async def get_positions(self, limit: int = 100) -> Dict:
        """
        Get user positions.
        
        Args:
            limit: Maximum number of positions
            
        Returns:
            Positions data
        """
        return await self.fetch("/positions", params={"limit": limit})
    
    # Health Check
    
    async def health_check(self) -> bool:
        """
        Check Kalshi API health.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            result = await self.fetch("/markets", params={"limit": 1})
            return "data" in result
        except Exception as e:
            logger.error(f"Kalshi health check failed: {e}")
            return False
