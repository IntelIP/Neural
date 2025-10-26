from __future__ import annotations

import asyncio
from collections.abc import Iterable
from datetime import datetime
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
        raise NotImplementedError("This method is not yet implemented")

    async def get_moneylines_only(self, sports: list[str], **kwargs) -> pd.DataFrame:
        """Convenience method for moneyline markets only"""
        raise NotImplementedError("This method is not yet implemented")

    async def get_todays_games(self, sports: list[str] | None = None) -> pd.DataFrame:
        """Get all games happening today across specified sports"""
        if sports is None:
            sports = ["NFL", "NBA", "CFB"]

        today = pd.Timestamp.now().date()
        all_games = await self.get_moneylines_only(sports)

        if not all_games.empty and "game_date" in all_games.columns:
            today_games = all_games[all_games["game_date"].dt.date == today]
            return today_games

        return all_games

    async def fetch_historical_candlesticks(
        self,
        market_ticker: str,
        hours_back: int = 24,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        """
        Fetch historical candlestick data for a specific market.

        Args:
            market_ticker: Market ticker (e.g., "KXNFLGAME-1234")
            hours_back: Hours of data to fetch (ignored if dates provided)
            start_date: Start date for data
            end_date: End date for data

        Returns:
            DataFrame with candlestick data
        """
        # Build API URL and params
        path = f"/trade-api/v2/markets/{market_ticker}/candlesticks"
        params = {}

        if start_date and end_date:
            params["start_ts"] = int(start_date.timestamp())
            params["end_ts"] = int(end_date.timestamp())
        else:
            end_ts = int(pd.Timestamp.now().timestamp())
            start_ts = end_ts - (hours_back * 3600)
            params["start_ts"] = start_ts
            params["end_ts"] = end_ts

        # Make authenticated request
        response = self._make_request("GET", path, params=params)

        if not response.get("candlesticks"):
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(response["candlesticks"])

        # Convert timestamps
        df["timestamp"] = pd.to_datetime(df["ts"], unit="s")

        # Convert prices from cents to dollars
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce") / 100.0

        return df.sort_values("timestamp")

    def _make_request(self, method: str, path: str, params: dict | None = None) -> dict:
        """Make authenticated request to Kalshi API."""
        # This would use the http_client, but for now, mock it
        # Since this is for tests, assume http_client is available
        if hasattr(self, "http_client"):
            # For test compatibility
            return self.http_client.get(path, params=params or {})
        else:
            # Fallback for production
            from neural.auth.http_client import KalshiHTTPClient

            client = KalshiHTTPClient()
            return client.get(path, params=params or {})

    async def get_upcoming_games(
        self, days: int = 7, sports: list[str] | None = None
    ) -> pd.DataFrame:
        """Get games in the next N days"""
        if sports is None:
            sports = ["NFL", "NBA", "CFB"]

        end_date = pd.Timestamp.now() + pd.Timedelta(days=days)
        all_games = await self.get_moneylines_only(sports)

        if not all_games.empty and "game_date" in all_games.columns:
            upcoming = all_games[all_games["game_date"] <= end_date]
            return upcoming.sort_values("game_date")

        return all_games


# Alias for backward compatibility
KalshiMarketsSource = SportMarketCollector


def filter_moneyline_markets(markets: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError


def get_moneyline_markets(sport: str, **kwargs) -> pd.DataFrame:
    raise NotImplementedError


def get_nba_games(**kwargs) -> pd.DataFrame:
    raise NotImplementedError


def get_all_sports_markets(**kwargs) -> pd.DataFrame:
    raise NotImplementedError


def get_cfb_games(**kwargs) -> pd.DataFrame:
    raise NotImplementedError


def get_game_markets(**kwargs) -> pd.DataFrame:
    raise NotImplementedError


def get_live_sports(**kwargs) -> pd.DataFrame:
    raise NotImplementedError


def get_markets_by_sport(sport: str, **kwargs) -> pd.DataFrame:
    raise NotImplementedError


def get_nfl_games(**kwargs) -> pd.DataFrame:
    raise NotImplementedError


def get_sports_series(**kwargs) -> pd.DataFrame:
    raise NotImplementedError


def search_markets(**kwargs) -> pd.DataFrame:
    raise NotImplementedError
