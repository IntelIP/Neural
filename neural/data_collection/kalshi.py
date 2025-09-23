import asyncio
import requests
from typing import Dict, Any, Optional, List, AsyncGenerator, Union
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from .base import DataSource


class KalshiMarketsSource(DataSource):
    """
    Data source for Kalshi markets with support for both authenticated and public API access.
    Optimized for sports market collection with proper series handling.
    """

    # Sports event tickers - Using actual Kalshi event prefixes
    SPORTS_SERIES = {
        "NFL": "KXNFLGAME",  # Professional football games
        "NBA": "KXNBA",  # NBA games
        "MLB": "KXMLB",  # MLB games
        "NHL": "KXNHL",  # NHL games
        "NCAA Football": "KXNCAAFGAME",  # College football games
        "NCAA Basketball": "KXNCAAB",  # College basketball games
        "Soccer": "KXSOCCER",  # Soccer matches
        "Tennis": "KXTENNIS",  # Tennis matches
        "Golf": "KXGOLF",  # Golf tournaments
        "MMA": "KXMMA",  # MMA fights
        "Formula 1": "KXF1",  # Formula 1 races
        # Direct ticker mappings
        "KXNFLGAME": "KXNFLGAME",
        "KXNBA": "KXNBA",
        "KXNCAAFGAME": "KXNCAAFGAME"
    }

    def __init__(
        self,
        series_ticker: Optional[str] = None,
        event_ticker: Optional[str] = None,
        status: Optional[str] = None,  # Changed: No default filtering
        interval: float = 60.0,
        use_authenticated: bool = True,
        api_key_id: Optional[str] = None,
        private_key_pem: Optional[bytes] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Kalshi markets data source.

        Args:
            series_ticker: Filter by series/event ticker (e.g., 'KXNFLGAME', 'KXNBA')
            event_ticker: Filter by specific event ticker (e.g., 'KXNFLGAME-25SEP22DETBAL')
            status: Market status filter ('active', 'closed', 'settled', etc., or None for all)
            interval: Polling interval in seconds
            use_authenticated: Use authenticated API (True) or public API (False)
            api_key_id: Optional API key ID for authentication
            private_key_pem: Optional private key PEM for authentication
            config: Additional configuration
        """
        name = f"kalshi_markets_{series_ticker or 'all'}"
        super().__init__(name, config)
        self.series_ticker = series_ticker
        self.event_ticker = event_ticker
        self.status = status
        self.interval = interval
        self.use_authenticated = use_authenticated
        self.api_key_id = api_key_id
        self.private_key_pem = private_key_pem
        self._executor = ThreadPoolExecutor(max_workers=2)
        self.base_url = "https://api.elections.kalshi.com/trade-api/v2"
        self._client = None

        # DEBUG: Print initialization parameters
        print(f"\n🔧 DEBUG - KalshiMarketsSource initialized:")
        print(f"  series_ticker: {self.series_ticker}")
        print(f"  status: {self.status}")
        print(f"  use_authenticated: {self.use_authenticated}")
        print(f"  base_url: {self.base_url}")

    def _init_authenticated_client(self):
        """Initialize the authenticated TradingClient if credentials are available."""
        try:
            from neural.trading import TradingClient
            self._client = TradingClient(
                api_key_id=self.api_key_id,
                private_key_pem=self.private_key_pem
            )
            return True
        except Exception as e:
            print(f"Failed to initialize authenticated client: {e}")
            print("Falling back to public API...")
            self.use_authenticated = False
            return False

    async def _fetch_markets_authenticated(
        self,
        limit: int = 1000,
        cursor: Optional[str] = None
    ) -> Dict[str, Any]:
        """Fetch markets using authenticated API."""
        loop = asyncio.get_event_loop()

        def _fetch():
            params = {
                "limit": limit
            }
            if self.status:
                params["status"] = self.status
            if cursor:
                params["cursor"] = cursor
            if self.series_ticker:
                params["series_ticker"] = self.series_ticker
            if self.event_ticker:
                params["event_ticker"] = self.event_ticker

            return self._client.markets.get_markets(**params)

        return await loop.run_in_executor(self._executor, _fetch)

    async def _fetch_markets_public(
        self,
        limit: int = 100,
        cursor: Optional[str] = None
    ) -> Dict[str, Any]:
        """Fetch markets using public API."""
        loop = asyncio.get_event_loop()

        def _fetch():
            params = {
                "limit": min(limit, 100)  # Public API may have lower limits
            }
            if self.status:
                params["status"] = self.status
            if cursor:
                params["cursor"] = cursor
            if self.series_ticker:
                params["series_ticker"] = self.series_ticker
            if self.event_ticker:
                params["event_ticker"] = self.event_ticker

            # DEBUG: Print the exact request being made
            url = f"{self.base_url}/markets"
            print(f"\n🔍 DEBUG - Making API request:")
            print(f"  URL: {url}")
            print(f"  Params: {params}")

            response = requests.get(url, params=params)

            # DEBUG: Print response details
            print(f"  Response Status: {response.status_code}")

            response.raise_for_status()
            data = response.json()

            # DEBUG: Print response data structure
            print(f"  Response keys: {list(data.keys())}")
            print(f"  Markets returned: {len(data.get('markets', []))}")
            if data.get('markets'):
                first_market = data['markets'][0]
                print(f"  First market series: {first_market.get('series_ticker')}")
                print(f"  First market title: {first_market.get('title', 'N/A')[:50]}...")

            return data

        return await loop.run_in_executor(self._executor, _fetch)

    async def _fetch_events_authenticated(
        self,
        limit: int = 1000,
        cursor: Optional[str] = None,
        with_nested_markets: bool = True
    ) -> Dict[str, Any]:
        """Fetch events using authenticated API."""
        loop = asyncio.get_event_loop()

        def _fetch():
            params = {
                "limit": limit,
                "with_nested_markets": with_nested_markets
            }
            if self.status:
                params["status"] = self.status
            if cursor:
                params["cursor"] = cursor
            if self.series_ticker:
                params["series_ticker"] = self.series_ticker

            return self._client.events.get_events(**params)

        return await loop.run_in_executor(self._executor, _fetch)

    async def _fetch_events_public(
        self,
        limit: int = 100,
        cursor: Optional[str] = None,
        with_nested_markets: bool = True
    ) -> Dict[str, Any]:
        """Fetch events using public API."""
        loop = asyncio.get_event_loop()

        def _fetch():
            params = {
                "limit": min(limit, 100),
                "with_nested_markets": with_nested_markets
            }
            if self.status:
                params["status"] = self.status
            if cursor:
                params["cursor"] = cursor
            if self.series_ticker:
                params["series_ticker"] = self.series_ticker

            url = f"{self.base_url}/events"
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()

        return await loop.run_in_executor(self._executor, _fetch)

    async def _fetch_event_markets(
        self,
        event_ticker: str
    ) -> Dict[str, Any]:
        """Fetch markets for a specific event."""
        loop = asyncio.get_event_loop()

        def _fetch():
            if self.use_authenticated and self._client:
                # Use authenticated API
                return self._client.events.get_event(event_ticker=event_ticker)
            else:
                # Use public API
                url = f"{self.base_url}/events/{event_ticker}"
                response = requests.get(url)
                response.raise_for_status()
                return response.json()

        return await loop.run_in_executor(self._executor, _fetch)

    async def _fetch_all_markets_paginated(self) -> List[Dict[str, Any]]:
        """Fetch all markets with pagination support."""
        all_markets = []
        cursor = None
        page_count = 0

        print(f"\n📄 DEBUG - Starting pagination loop")

        while True:
            try:
                page_count += 1
                print(f"\n  Page {page_count}:")

                if self.use_authenticated and self._client:
                    data = await self._fetch_markets_authenticated(cursor=cursor)
                else:
                    data = await self._fetch_markets_public(cursor=cursor)

                markets = data.get("markets", [])
                print(f"    Markets on this page: {len(markets)}")
                all_markets.extend(markets)

                # Check for next page
                cursor = data.get("cursor")
                print(f"    Next cursor: {cursor[:20] + '...' if cursor else 'None'}")

                if not cursor or not markets:
                    print(f"    Stopping pagination (no cursor or no markets)")
                    break

            except Exception as e:
                print(f"Error fetching page: {e}")
                break

        print(f"\n📊 DEBUG - Pagination complete:")
        print(f"  Total pages: {page_count}")
        print(f"  Total markets: {len(all_markets)}")

        return all_markets

    def _process_market_data(self, markets: List[Dict[str, Any]]) -> pd.DataFrame:
        """Process raw market data into comprehensive DataFrame."""
        if not markets:
            return pd.DataFrame()

        # Extract relevant fields for DataFrame
        processed_data = []
        for market in markets:
            processed_data.append({
                # Identifiers
                'ticker': market.get('ticker'),
                'title': market.get('title'),
                'subtitle': market.get('subtitle'),
                'event_ticker': market.get('event_ticker'),
                'series_ticker': market.get('series_ticker'),

                # Pricing
                'yes_bid': market.get('yes_bid'),
                'yes_ask': market.get('yes_ask'),
                'no_bid': market.get('no_bid'),
                'no_ask': market.get('no_ask'),
                'last_price': market.get('last_price'),
                'previous_price': market.get('previous_price'),

                # Volume metrics
                'volume': market.get('volume'),
                'volume_24h': market.get('volume_24h'),
                'open_interest': market.get('open_interest'),

                # Market info
                'status': market.get('status'),
                'result': market.get('result'),
                'can_close_early': market.get('can_close_early'),
                'cap_count': market.get('cap_count'),

                # Timestamps
                'open_time': pd.to_datetime(market.get('open_time')) if market.get('open_time') else None,
                'close_time': pd.to_datetime(market.get('close_time')) if market.get('close_time') else None,
                'expiration_time': pd.to_datetime(market.get('expiration_time')) if market.get('expiration_time') else None,

                # Additional metadata
                'rules': market.get('rules_primary'),
                'fetch_time': datetime.now()
            })

        df = pd.DataFrame(processed_data)

        # Calculate derived metrics
        if not df.empty:
            # Calculate spread
            df['spread'] = df['yes_ask'] - df['yes_bid']

            # Calculate mid price
            df['mid_price'] = (df['yes_bid'] + df['yes_ask']) / 2

            # Add liquidity indicator
            df['liquidity_score'] = df['volume_24h'] * (1 - df['spread']/100)

        return df

    async def connect(self) -> None:
        """Initialize connection - setup authenticated client if needed."""
        if self.use_authenticated and not self._client:
            self._init_authenticated_client()
        self._connected = True

    async def disconnect(self) -> None:
        """Close connections and cleanup."""
        if self._client:
            self._client.close()
            self._client = None
        self._executor.shutdown(wait=True)
        self._connected = False

    async def collect(self) -> AsyncGenerator[pd.DataFrame, None]:
        """
        Fetch markets and yield as comprehensive DataFrame.
        For sports, fetches events first then gets associated markets.
        Yields DataFrame with all market data including pricing, volume, and metadata.
        """
        while self._connected:
            try:
                all_markets = []

                # Check if we're looking for sports (series_ticker starts with KX)
                is_sports_query = (self.series_ticker and
                                 (self.series_ticker.startswith('KX') or
                                  self.series_ticker in self.SPORTS_SERIES.values()))

                if is_sports_query:
                    # For sports, fetch events and their markets
                    print(f"Fetching sports events for {self.series_ticker}...")

                    cursor = None
                    while True:
                        try:
                            if self.use_authenticated and self._client:
                                data = await self._fetch_events_authenticated(cursor=cursor, with_nested_markets=True)
                            else:
                                data = await self._fetch_events_public(cursor=cursor, with_nested_markets=True)

                            events = data.get("events", [])
                            print(f"Found {len(events)} events")

                            # Extract markets from events
                            for event in events:
                                event_markets = event.get("markets", [])
                                # Add event info to each market
                                for market in event_markets:
                                    market["event_ticker"] = event.get("ticker")
                                    market["event_title"] = event.get("title")
                                all_markets.extend(event_markets)

                            cursor = data.get("cursor")
                            if not cursor or not events:
                                break

                        except Exception as e:
                            print(f"Error fetching events: {e}")
                            break

                    # If we have specific event ticker, also fetch its markets directly
                    if self.event_ticker:
                        try:
                            event_data = await self._fetch_event_markets(self.event_ticker)
                            if event_data and "markets" in event_data:
                                all_markets.extend(event_data["markets"])
                        except Exception as e:
                            print(f"Error fetching event markets: {e}")

                # Also fetch regular markets
                regular_markets = await self._fetch_all_markets_paginated()
                all_markets.extend(regular_markets)

                # Remove duplicates based on ticker
                unique_markets = {}
                for market in all_markets:
                    ticker = market.get('ticker')
                    if ticker and ticker not in unique_markets:
                        unique_markets[ticker] = market

                # Process into DataFrame
                df = self._process_market_data(list(unique_markets.values()))

                if not df.empty:
                    print(f"Fetched {len(df)} unique markets" +
                          (f" for series {self.series_ticker}" if self.series_ticker else ""))
                    yield df
                else:
                    print("No markets found matching criteria")

            except Exception as e:
                print(f"Error in market collection: {e}")
                # On error, try to fallback to public API if using authenticated
                if self.use_authenticated:
                    print("Retrying with public API...")
                    self.use_authenticated = False

            await asyncio.sleep(self.interval)


# Utility functions for easy sports market collection
async def get_sports_series(
    use_authenticated: bool = True,
    api_key_id: Optional[str] = None,
    private_key_pem: Optional[bytes] = None
) -> Dict[str, str]:
    """
    Get available sports series from Kalshi.

    Returns:
        Dictionary mapping sport names to series tickers
    """
    return KalshiMarketsSource.SPORTS_SERIES


async def get_markets_by_sport(
    sport: str,
    status: str = "open",
    use_authenticated: bool = True,
    api_key_id: Optional[str] = None,
    private_key_pem: Optional[bytes] = None
) -> pd.DataFrame:
    """
    Get all markets for a specific sport.

    Args:
        sport: Sport name (e.g., 'NFL', 'NBA') or series ticker
        status: Market status filter
        use_authenticated: Use authenticated API
        api_key_id: Optional API key
        private_key_pem: Optional private key

    Returns:
        DataFrame with market data for the specified sport
    """
    # Check if sport is a known name and convert to ticker
    series_ticker = KalshiMarketsSource.SPORTS_SERIES.get(sport, sport)

    source = KalshiMarketsSource(
        series_ticker=series_ticker,
        status=status,
        use_authenticated=use_authenticated,
        api_key_id=api_key_id,
        private_key_pem=private_key_pem,
        interval=float('inf')  # Single fetch
    )

    async with source:
        async for df in source.collect():
            return df  # Return first (and only) result
    return pd.DataFrame()


async def get_all_sports_markets(
    sports: Optional[List[str]] = None,
    status: str = "open",
    use_authenticated: bool = True,
    api_key_id: Optional[str] = None,
    private_key_pem: Optional[bytes] = None
) -> pd.DataFrame:
    """
    Get markets for multiple sports combined.

    Args:
        sports: List of sport names/tickers (None for all known sports)
        status: Market status filter
        use_authenticated: Use authenticated API
        api_key_id: Optional API key
        private_key_pem: Optional private key

    Returns:
        Combined DataFrame with markets from all specified sports
    """
    if sports is None:
        sports = list(KalshiMarketsSource.SPORTS_SERIES.values())

    all_dfs = []

    for sport in sports:
        try:
            df = await get_markets_by_sport(
                sport=sport,
                status=status,
                use_authenticated=use_authenticated,
                api_key_id=api_key_id,
                private_key_pem=private_key_pem
            )
            if not df.empty:
                all_dfs.append(df)
        except Exception as e:
            print(f"Error fetching {sport} markets: {e}")

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        # Remove duplicates based on ticker
        combined_df = combined_df.drop_duplicates(subset=['ticker'])
        return combined_df

    return pd.DataFrame()


async def search_markets(
    search_term: str,
    series_ticker: Optional[str] = None,
    status: str = "open",
    use_authenticated: bool = True,
    api_key_id: Optional[str] = None,
    private_key_pem: Optional[bytes] = None
) -> pd.DataFrame:
    """
    Search for markets containing specific terms.

    Args:
        search_term: Term to search for in market titles
        series_ticker: Optional series filter
        status: Market status filter
        use_authenticated: Use authenticated API
        api_key_id: Optional API key
        private_key_pem: Optional private key

    Returns:
        DataFrame with matching markets
    """
    source = KalshiMarketsSource(
        series_ticker=series_ticker,
        status=status,
        use_authenticated=use_authenticated,
        api_key_id=api_key_id,
        private_key_pem=private_key_pem,
        interval=float('inf')
    )

    async with source:
        async for df in source.collect():
            if not df.empty:
                # Filter by search term in title or subtitle
                mask = (
                    df['title'].str.contains(search_term, case=False, na=False) |
                    df['subtitle'].str.contains(search_term, case=False, na=False)
                )
                return df[mask]

    return pd.DataFrame()


async def get_game_markets(
    event_ticker: str,
    use_authenticated: bool = True,
    api_key_id: Optional[str] = None,
    private_key_pem: Optional[bytes] = None
) -> pd.DataFrame:
    """
    Get markets for a specific game event.

    Args:
        event_ticker: Event ticker (e.g., 'KXNFLGAME-25SEP22DETBAL')
        use_authenticated: Use authenticated API
        api_key_id: Optional API key
        private_key_pem: Optional private key

    Returns:
        DataFrame with markets for the specified game
    """
    source = KalshiMarketsSource(
        event_ticker=event_ticker,
        status=None,  # Get all statuses
        use_authenticated=use_authenticated,
        api_key_id=api_key_id,
        private_key_pem=private_key_pem,
        interval=float('inf')
    )

    async with source:
        async for df in source.collect():
            return df
    return pd.DataFrame()


async def get_live_sports(
    sports: Optional[List[str]] = None,
    use_authenticated: bool = True,
    api_key_id: Optional[str] = None,
    private_key_pem: Optional[bytes] = None
) -> pd.DataFrame:
    """
    Get all currently active/live sports markets.

    Args:
        sports: List of sport names/tickers (None for all known sports)
        use_authenticated: Use authenticated API
        api_key_id: Optional API key
        private_key_pem: Optional private key

    Returns:
        DataFrame with currently active sports markets
    """
    if sports is None:
        sports = ['KXNFLGAME', 'KXNBA', 'KXNCAAFGAME', 'KXMLB', 'KXNHL']

    all_dfs = []

    for sport in sports:
        # Convert sport name to ticker if needed
        ticker = KalshiMarketsSource.SPORTS_SERIES.get(sport, sport)

        source = KalshiMarketsSource(
            series_ticker=ticker,
            status='active',  # Only active markets
            use_authenticated=use_authenticated,
            api_key_id=api_key_id,
            private_key_pem=private_key_pem,
            interval=float('inf')
        )

        try:
            async with source:
                async for df in source.collect():
                    if not df.empty:
                        all_dfs.append(df)
                    break
        except Exception as e:
            print(f"Error fetching {sport} markets: {e}")

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['ticker'])
        return combined_df

    return pd.DataFrame()