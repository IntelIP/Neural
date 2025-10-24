from __future__ import annotations

import asyncio
import re
from collections.abc import Iterable
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import requests

from neural.auth.http_client import KalshiHTTPClient

_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
_SPORT_SERIES_MAP = {
    "NFL": "KXNFLGAME",
    "NBA": "KXNBA",
    "MLB": "KXMLB",
    "NHL": "KXNHL",
    "NCAAF": "KXNCAAFGAME",
    "CFB": "KXNCAAFGAME",
    "NCAA": "KXNCAAFGAME",
}


def _normalize_series(identifier: str | None) -> str | None:
    if identifier is None:
        return None
    if identifier.upper().startswith("KX"):
        return identifier
    return _SPORT_SERIES_MAP.get(identifier.upper(), identifier)


def _resolve_series_list(series: Iterable[str] | None) -> list[str]:
    if not series:
        return list(set(_SPORT_SERIES_MAP.values()))
    return [s for s in (_normalize_series(item) for item in series) if s]


async def _fetch_markets(
    params: dict[str, Any],
    *,
    use_authenticated: bool,
    api_key_id: str | None,
    private_key_pem: bytes | None,
) -> pd.DataFrame:
    def _request() -> dict[str, Any]:
        if use_authenticated:
            client = KalshiHTTPClient(api_key_id=api_key_id, private_key_pem=private_key_pem)
            try:
                return client.get("/markets", params=params)
            finally:
                client.close()
        url = f"{_BASE_URL}/markets"
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return dict(resp.json())

    payload = await asyncio.to_thread(_request)
    return pd.DataFrame(payload.get("markets", []))


class KalshiMarketsSource:
    """Fetch markets for a given Kalshi series ticker."""

    def __init__(
        self,
        *,
        series_ticker: str | None = None,
        status: str | None = "open",
        limit: int = 200,
        use_authenticated: bool = True,
        api_key_id: str | None = None,
        private_key_pem: bytes | None = None,
    ) -> None:
        self.series_ticker = _normalize_series(series_ticker)
        self.status = status
        self.limit = limit
        self.use_authenticated = use_authenticated
        self.api_key_id = api_key_id
        self.private_key_pem = private_key_pem

    async def fetch(self) -> pd.DataFrame:
        params: dict[str, Any] = {"limit": self.limit}
        if self.series_ticker:
            params["series_ticker"] = self.series_ticker
        if self.status is not None:
            params["status"] = self.status
        return await _fetch_markets(
            params,
            use_authenticated=self.use_authenticated,
            api_key_id=self.api_key_id,
            private_key_pem=self.private_key_pem,
        )

    async def fetch_historical_candlesticks(
        self,
        market_ticker: str,
        interval: int = 60,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        hours_back: int = 48,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV candlestick data for a specific market.

        Args:
            market_ticker: Market ticker (e.g., 'KXNFLGAME-25NOV02SEAWAS-WAS')
            interval: Time interval in minutes (1, 60, or 1440)
            start_date: Start date for data (optional)
            end_date: End date for data (optional)
            hours_back: Hours of data to fetch if dates not specified

        Returns:
            DataFrame with OHLCV data and metadata
        """
        from datetime import datetime, timedelta

        from neural.auth.http_client import KalshiHTTPClient

        # Set up time range
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(hours=hours_back)

        start_ts = int(start_date.timestamp())
        end_ts = int(end_date.timestamp())

        # Create HTTP client for historical data
        client = KalshiHTTPClient(api_key_id=self.api_key_id, private_key_pem=self.private_key_pem)

        try:
            # Use series ticker if available, otherwise extract from market ticker
            series_ticker = self.series_ticker
            if not series_ticker:
                # Extract series from market ticker (e.g., KXNFLGAME-25NOV02SEAWAS-WAS -> KXNFLGAME)
                if "-" in market_ticker:
                    series_ticker = market_ticker.split("-")[0]
                else:
                    series_ticker = market_ticker

            # Fetch candlestick data
            response = client.get_market_candlesticks(
                series_ticker=series_ticker,
                ticker=market_ticker,
                start_ts=start_ts,
                end_ts=end_ts,
                period_interval=interval,
            )

            candlesticks = response.get("candlesticks", [])

            if not candlesticks:
                print(f"No candlestick data found for {market_ticker}")
                return pd.DataFrame()

            # Process candlestick data
            processed_data = []
            for candle in candlesticks:
                price_data = candle.get("price", {})
                yes_bid = candle.get("yes_bid", {})
                yes_ask = candle.get("yes_ask", {})

                # Handle None values safely
                def safe_convert(value, default=0.0):
                    if value is None:
                        return default
                    return float(value) / 100.0  # Convert cents to dollars

                processed_data.append(
                    {
                        "timestamp": pd.to_datetime(candle.get("end_period_ts"), unit="s"),
                        "open": safe_convert(price_data.get("open")),
                        "high": safe_convert(price_data.get("high")),
                        "low": safe_convert(price_data.get("low")),
                        "close": safe_convert(price_data.get("close")),
                        "volume": candle.get("volume", 0),
                        "yes_bid": safe_convert(yes_bid.get("close")),
                        "yes_ask": safe_convert(yes_ask.get("close")),
                        "open_interest": candle.get("open_interest", 0),
                    }
                )

            df = pd.DataFrame(processed_data)
            df = df.sort_values("timestamp").reset_index(drop=True)

            print(f"✅ Fetched {len(df)} candlesticks for {market_ticker}")
            return df

        except Exception as e:
            print(f"❌ Error fetching historical data for {market_ticker}: {e}")
            return pd.DataFrame()
        finally:
            client.close()


async def get_sports_series(
    leagues: Iterable[str] | None = None,
    *,
    status: str | None = "open",
    limit: int = 200,
    use_authenticated: bool = True,
    api_key_id: str | None = None,
    private_key_pem: bytes | None = None,
) -> dict[str, list[dict[str, Any]]]:
    series_ids = _resolve_series_list(leagues)
    results: dict[str, list[dict[str, Any]]] = {}
    for series_id in series_ids:
        df = await get_markets_by_sport(
            series_id,
            status=status,
            limit=limit,
            use_authenticated=use_authenticated,
            api_key_id=api_key_id,
            private_key_pem=private_key_pem,
        )
        if not df.empty:
            records = df.to_dict(orient="records")
            results[series_id] = [{str(k): v for k, v in record.items()} for record in records]
    return results


async def get_markets_by_sport(
    sport: str,
    *,
    status: str | None = "open",
    limit: int = 200,
    use_authenticated: bool = True,
    api_key_id: str | None = None,
    private_key_pem: bytes | None = None,
) -> pd.DataFrame:
    series = _normalize_series(sport)
    params: dict[str, Any] = {"limit": limit}
    if series:
        params["series_ticker"] = series
    if status is not None:
        params["status"] = status
    return await _fetch_markets(
        params,
        use_authenticated=use_authenticated,
        api_key_id=api_key_id,
        private_key_pem=private_key_pem,
    )


async def get_all_sports_markets(
    sports: Iterable[str] | None = None,
    *,
    status: str | None = "open",
    limit: int = 200,
    use_authenticated: bool = True,
    api_key_id: str | None = None,
    private_key_pem: bytes | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for series in _resolve_series_list(sports):
        df = await get_markets_by_sport(
            series,
            status=status,
            limit=limit,
            use_authenticated=use_authenticated,
            api_key_id=api_key_id,
            private_key_pem=private_key_pem,
        )
        if not df.empty:
            frames.append(df)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


async def search_markets(
    query: str,
    *,
    status: str | None = None,
    limit: int = 200,
    use_authenticated: bool = True,
    api_key_id: str | None = None,
    private_key_pem: bytes | None = None,
) -> pd.DataFrame:
    params: dict[str, Any] = {"search": query, "limit": limit}
    if status is not None:
        params["status"] = status
    return await _fetch_markets(
        params,
        use_authenticated=use_authenticated,
        api_key_id=api_key_id,
        private_key_pem=private_key_pem,
    )


async def get_game_markets(
    event_ticker: str,
    *,
    status: str | None = None,
    use_authenticated: bool = True,
    api_key_id: str | None = None,
    private_key_pem: bytes | None = None,
) -> pd.DataFrame:
    params: dict[str, Any] = {"event_ticker": event_ticker}
    if status is not None:
        params["status"] = status
    return await _fetch_markets(
        params,
        use_authenticated=use_authenticated,
        api_key_id=api_key_id,
        private_key_pem=private_key_pem,
    )


async def get_live_sports(
    *,
    limit: int = 200,
    use_authenticated: bool = True,
    api_key_id: str | None = None,
    private_key_pem: bytes | None = None,
) -> pd.DataFrame:
    return await _fetch_markets(
        {"status": "live", "limit": limit},
        use_authenticated=use_authenticated,
        api_key_id=api_key_id,
        private_key_pem=private_key_pem,
    )


async def get_nfl_games(
    status: str = "open",
    limit: int = 50,
    use_authenticated: bool = True,
    api_key_id: str | None = None,
    private_key_pem: bytes | None = None,
) -> pd.DataFrame:
    """
    Get NFL games markets from Kalshi.

    Args:
        status: Market status filter (default: 'open')
        limit: Maximum markets to fetch (default: 50)
        use_authenticated: Use authenticated API
        api_key_id: Optional API key
        private_key_pem: Optional private key

    Returns:
        DataFrame with NFL markets, including parsed teams and game date
    """
    df = await get_markets_by_sport(
        sport="NFL",
        status=status,
        limit=limit,
        use_authenticated=use_authenticated,
        api_key_id=api_key_id,
        private_key_pem=private_key_pem,
    )

    if not df.empty:
        # Parse teams from title (common format: "Will the [Away] beat the [Home]?" or similar)
        def parse_teams(row):
            title = row["title"]
            match = re.search(
                r"Will the (\w+(?:\s\w+)?) beat the (\w+(?:\s\w+)?)\?", title, re.IGNORECASE
            )
            if match:
                away, home = match.groups()
                return pd.Series({"home_team": home, "away_team": away})
            # Fallback: extract from subtitle or ticker
            subtitle = row.get("subtitle", "")
            if " vs " in subtitle:
                teams = subtitle.split(" vs ")
                return pd.Series(
                    {
                        "home_team": teams[1].strip() if len(teams) > 1 else None,
                        "away_team": teams[0].strip(),
                    }
                )
            return pd.Series({"home_team": None, "away_team": None})

        team_df = df.apply(parse_teams, axis=1)
        df = pd.concat([df, team_df], axis=1)

        # Parse game date from ticker (format: KXNFLGAME-25SEP22DETBAL -> 25SEP22)
        def parse_game_date(ticker):
            match = re.search(r"-(\d{2}[A-Z]{3}\d{2})", ticker)
            if match:
                date_str = match.group(1)
                try:
                    # Assume YYMMMDD, convert to full year (e.g., 22 -> 2022)
                    year = (
                        int(date_str[-2:]) + 2000
                        if int(date_str[-2:]) < 50
                        else 1900 + int(date_str[-2:])
                    )
                    month_map = {
                        "JAN": 1,
                        "FEB": 2,
                        "MAR": 3,
                        "APR": 4,
                        "MAY": 5,
                        "JUN": 6,
                        "JUL": 7,
                        "AUG": 8,
                        "SEP": 9,
                        "OCT": 10,
                        "NOV": 11,
                        "DEC": 12,
                    }
                    month = month_map.get(date_str[2:5])
                    day = int(date_str[0:2])
                    return pd.to_datetime(f"{year}-{month:02d}-{day:02d}")
                except Exception:
                    pass
            return pd.NaT

        df["game_date"] = df["ticker"].apply(parse_game_date)

        # Bug Fix #4, #12: Filter using ticker (which exists) instead of series_ticker (which doesn't)
        # The series_ticker field doesn't exist in Kalshi API responses, use ticker or event_ticker instead
        nfl_mask = df["ticker"].str.contains("KXNFLGAME", na=False) | df["title"].str.contains(
            "NFL", case=False, na=False
        )
        df = df[nfl_mask]

    return df


async def get_nba_games(
    status: str = "open",
    limit: int = 50,
    use_authenticated: bool = True,
    api_key_id: str | None = None,
    private_key_pem: bytes | None = None,
) -> pd.DataFrame:
    """
    Get NBA games markets from Kalshi.

    Args:
        status: Market status filter (default: 'open')
        limit: Maximum markets to fetch (default: 50)
        use_authenticated: Use authenticated API
        api_key_id: Optional API key
        private_key_pem: Optional private key

    Returns:
        DataFrame with NBA markets, including parsed teams and game date
    """
    df = await get_markets_by_sport(
        sport="NBA",
        status=status,
        limit=limit,
        use_authenticated=use_authenticated,
        api_key_id=api_key_id,
        private_key_pem=private_key_pem,
    )

    if not df.empty:
        # Parse teams from title (NBA format: "Will the [Away] beat the [Home]?" or similar)
        def parse_teams(row):
            title = row["title"]
            match = re.search(
                r"Will the (\w+(?:\s\w+)?) beat the (\w+(?:\s\w+)?)\?", title, re.IGNORECASE
            )
            if match:
                away, home = match.groups()
                return pd.Series({"home_team": home, "away_team": away})
            # Fallback: extract from subtitle or ticker
            subtitle = row.get("subtitle", "")
            if " vs " in subtitle:
                teams = subtitle.split(" vs ")
                return pd.Series(
                    {
                        "home_team": teams[1].strip() if len(teams) > 1 else None,
                        "away_team": teams[0].strip(),
                    }
                )
            # NBA-specific: Try "at" format (Away at Home)
            if " at " in subtitle:
                teams = subtitle.split(" at ")
                return pd.Series(
                    {
                        "home_team": teams[1].strip() if len(teams) > 1 else None,
                        "away_team": teams[0].strip(),
                    }
                )
            return pd.Series({"home_team": None, "away_team": None})

        team_df = df.apply(parse_teams, axis=1)
        df = pd.concat([df, team_df], axis=1)

        # Parse game date from ticker (format: KXNBA-25OCT15LALGSW -> 25OCT15)
        def parse_game_date(ticker):
            match = re.search(r"-(\d{2}[A-Z]{3}\d{2})", ticker)
            if match:
                date_str = match.group(1)
                try:
                    # Assume YYMMMDD, convert to full year (e.g., 25 -> 2025)
                    year = (
                        int(date_str[-2:]) + 2000
                        if int(date_str[-2:]) < 50
                        else 1900 + int(date_str[-2:])
                    )
                    month_map = {
                        "JAN": 1,
                        "FEB": 2,
                        "MAR": 3,
                        "APR": 4,
                        "MAY": 5,
                        "JUN": 6,
                        "JUL": 7,
                        "AUG": 8,
                        "SEP": 9,
                        "OCT": 10,
                        "NOV": 11,
                        "DEC": 12,
                    }
                    month = month_map.get(date_str[2:5])
                    day = int(date_str[0:2])
                    return pd.to_datetime(f"{year}-{month:02d}-{day:02d}")
                except Exception:
                    pass
            return pd.NaT

        df["game_date"] = df["ticker"].apply(parse_game_date)

        # Filter for NBA games only
        nba_mask = df["ticker"].str.contains("KXNBA", na=False) | df["title"].str.contains(
            "NBA|Basketball", case=False, na=False
        )
        df = df[nba_mask]

    return df


async def get_cfb_games(
    status: str = "open",
    limit: int = 50,
    use_authenticated: bool = True,
    api_key_id: str | None = None,
    private_key_pem: bytes | None = None,
) -> pd.DataFrame:
    """
    Get College Football (CFB) games markets from Kalshi.

    Args:
        status: Market status filter (default: 'open')
        limit: Maximum markets to fetch (default: 50)
        use_authenticated: Use authenticated API
        api_key_id: Optional API key
        private_key_pem: Optional private key

    Returns:
        DataFrame with CFB markets, including parsed teams and game date
    """
    df = await get_markets_by_sport(
        sport="NCAA Football",
        status=status,
        limit=limit,
        use_authenticated=use_authenticated,
        api_key_id=api_key_id,
        private_key_pem=private_key_pem,
    )

    if not df.empty:
        # Parse teams similar to NFL
        def parse_teams(row):
            title = row["title"]
            match = re.search(
                r"Will the (\w+(?:\s\w+)?) beat the (\w+(?:\s\w+)?)\?", title, re.IGNORECASE
            )
            if match:
                away, home = match.groups()
                return pd.Series({"home_team": home, "away_team": away})
            subtitle = row.get("subtitle", "")
            if " vs " in subtitle:
                teams = subtitle.split(" vs ")
                return pd.Series(
                    {
                        "home_team": teams[1].strip() if len(teams) > 1 else None,
                        "away_team": teams[0].strip(),
                    }
                )
            return pd.Series({"home_team": None, "away_team": None})

        team_df = df.apply(parse_teams, axis=1)
        df = pd.concat([df, team_df], axis=1)

        # Parse game date from ticker
        def parse_game_date(ticker):
            match = re.search(r"-(\d{2}[A-Z]{3}\d{2})", ticker)
            if match:
                date_str = match.group(1)
                try:
                    year = (
                        int(date_str[-2:]) + 2000
                        if int(date_str[-2:]) < 50
                        else 1900 + int(date_str[-2:])
                    )
                    month_map = {
                        "JAN": 1,
                        "FEB": 2,
                        "MAR": 3,
                        "APR": 4,
                        "MAY": 5,
                        "JUN": 6,
                        "JUL": 7,
                        "AUG": 8,
                        "SEP": 9,
                        "OCT": 10,
                        "NOV": 11,
                        "DEC": 12,
                    }
                    month = month_map.get(date_str[2:5])
                    day = int(date_str[0:2])
                    return pd.to_datetime(f"{year}-{month:02d}-{day:02d}")
                except Exception:
                    pass
            return pd.NaT

        df["game_date"] = df["ticker"].apply(parse_game_date)

        # Bug Fix #4, #12: Filter using ticker (which exists) instead of series_ticker (which doesn't)
        # The series_ticker field doesn't exist in Kalshi API responses, use ticker or event_ticker instead
        cfb_mask = df["ticker"].str.contains("KXNCAAFGAME", na=False) | df["title"].str.contains(
            "NCAA|College Football", case=False, na=False
        )
        df = df[cfb_mask]

    return df


def filter_moneyline_markets(markets_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter DataFrame to only include moneyline/winner markets.

    Args:
        markets_df: DataFrame from any get_*_games() function

    Returns:
        Filtered DataFrame with only moneyline markets
    """
    if markets_df.empty:
        return markets_df

    # Patterns that indicate moneyline markets
    moneyline_patterns = [
        r"Will.*beat.*\?",
        r"Will.*win.*\?",
        r".*to win.*\?",
        r".*winner.*\?",
        r".*vs.*winner",
    ]

    # Combine patterns
    pattern = "|".join(moneyline_patterns)

    # Filter based on title
    moneyline_mask = markets_df["title"].str.contains(pattern, case=False, na=False)

    # Additional filtering: exclude prop bets, totals, spreads
    exclude_patterns = [
        r"total.*points",
        r"over.*under",
        r"spread",
        r"touchdown",
        r"yards",
        r"first.*score",
        r"player.*prop",
    ]

    exclude_pattern = "|".join(exclude_patterns)
    exclude_mask = markets_df["title"].str.contains(exclude_pattern, case=False, na=False)

    # Return markets that match moneyline patterns but don't match exclude patterns
    filtered_df = markets_df[moneyline_mask & ~exclude_mask].copy()

    return filtered_df


async def get_moneyline_markets(
    sport: str, status: str = "open", limit: int = 100, **kwargs
) -> pd.DataFrame:
    """
    Get only moneyline/winner markets for a specific sport.

    Args:
        sport: Sport identifier ("NFL", "NBA", "CFB", etc.)
        status: Market status filter
        limit: Maximum markets to fetch
        **kwargs: Additional arguments for sport-specific functions

    Returns:
        DataFrame with only moneyline markets, enhanced with metadata
    """
    # Route to appropriate sport function
    if sport.upper() == "NFL":
        markets = await get_nfl_games(status=status, limit=limit, **kwargs)
    elif sport.upper() == "NBA":
        markets = await get_nba_games(status=status, limit=limit, **kwargs)
    elif sport.upper() in ["CFB", "NCAAF"]:
        markets = await get_cfb_games(status=status, limit=limit, **kwargs)
    else:
        # Fallback to general markets
        markets = await get_markets_by_sport(sport, status=status, limit=limit, **kwargs)

    # Filter for moneylines only
    moneylines = filter_moneyline_markets(markets)

    # Add sport metadata
    if not moneylines.empty:
        moneylines = moneylines.copy()
        moneylines["sport"] = sport.upper()
        moneylines["market_type"] = "moneyline"

    return moneylines


class SportMarketCollector:
    """
    Unified interface for collecting sports market data across all supported leagues.

    Provides consistent API and data format regardless of sport.
    """

    def __init__(self, use_authenticated: bool = True, **auth_kwargs):
        """Initialize with authentication parameters"""
        self.use_authenticated = use_authenticated
        self.auth_kwargs = auth_kwargs

    async def get_games(
        self, sport: str, market_type: str = "moneyline", status: str = "open", **kwargs
    ) -> pd.DataFrame:
        """
        Universal method to get games for any sport.

        Args:
            sport: "NFL", "NBA", "CFB", "MLB", "NHL"
            market_type: "moneyline", "all", "props"
            status: "open", "closed", "settled"

        Returns:
            Standardized DataFrame with consistent columns across sports
        """
        kwargs.update(self.auth_kwargs)
        kwargs.update({"use_authenticated": self.use_authenticated, "status": status})

        if market_type == "moneyline":
            return await get_moneyline_markets(sport, **kwargs)
        else:
            # Get all markets for the sport
            if sport.upper() == "NFL":
                return await get_nfl_games(**kwargs)
            elif sport.upper() == "NBA":
                return await get_nba_games(**kwargs)
            elif sport.upper() in ["CFB", "NCAAF"]:
                return await get_cfb_games(**kwargs)
            else:
                return await get_markets_by_sport(sport, **kwargs)

    async def get_moneylines_only(self, sports: list[str], **kwargs) -> pd.DataFrame:
        """Convenience method for moneyline markets only"""
        all_moneylines = []

        for sport in sports:
            try:
                moneylines = await get_moneyline_markets(sport, **kwargs)
                if not moneylines.empty:
                    all_moneylines.append(moneylines)
            except Exception as e:
                print(f"Warning: Failed to fetch {sport} markets: {e}")
                continue

        if all_moneylines:
            return pd.concat(all_moneylines, ignore_index=True)
        else:
            return pd.DataFrame()

    async def get_todays_games(self, sports: list[str] = None) -> pd.DataFrame:
        """Get all games happening today across specified sports"""
        if sports is None:
            sports = ["NFL", "NBA", "CFB"]

        today = pd.Timestamp.now().date()
        all_games = await self.get_moneylines_only(sports)

        if not all_games.empty and "game_date" in all_games.columns:
            today_games = all_games[all_games["game_date"].dt.date == today]
            return today_games

        return all_games

    async def get_upcoming_games(self, days: int = 7, sports: list[str] = None) -> pd.DataFrame:
        """Get games in the next N days"""
        if sports is None:
            sports = ["NFL", "NBA", "CFB"]

        end_date = pd.Timestamp.now() + pd.Timedelta(days=days)
        all_games = await self.get_moneylines_only(sports)

        if not all_games.empty and "game_date" in all_games.columns:
            upcoming = all_games[all_games["game_date"] <= end_date]
            return upcoming.sort_values("game_date")

        return all_games
