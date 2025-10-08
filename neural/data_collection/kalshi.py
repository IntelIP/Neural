from __future__ import annotations
import re
from typing import Optional

import pandas as pd

import asyncio
from typing import Any, Dict, Iterable, List

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

def _resolve_series_list(series: Iterable[str] | None) -> List[str]:
    if not series:
        return list({v for v in _SPORT_SERIES_MAP.values()})
    return [s for s in (_normalize_series(item) for item in series) if s]

async def _fetch_markets(
    params: Dict[str, Any],
    *,
    use_authenticated: bool,
    api_key_id: Optional[str],
    private_key_pem: Optional[bytes],
) -> pd.DataFrame:
    def _request() -> Dict[str, Any]:
        if use_authenticated:
            client = KalshiHTTPClient(api_key_id=api_key_id, private_key_pem=private_key_pem)
            try:
                return client.get("/markets", params=params)
            finally:
                client.close()
        url = f"{_BASE_URL}/markets"
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    payload = await asyncio.to_thread(_request)
    return pd.DataFrame(payload.get("markets", []))

class KalshiMarketsSource:
    """Fetch markets for a given Kalshi series ticker."""

    def __init__(
        self,
        *,
        series_ticker: Optional[str] = None,
        status: Optional[str] = "open",
        limit: int = 200,
        use_authenticated: bool = True,
        api_key_id: Optional[str] = None,
        private_key_pem: Optional[bytes] = None,
    ) -> None:
        self.series_ticker = _normalize_series(series_ticker)
        self.status = status
        self.limit = limit
        self.use_authenticated = use_authenticated
        self.api_key_id = api_key_id
        self.private_key_pem = private_key_pem

    async def fetch(self) -> pd.DataFrame:
        params: Dict[str, Any] = {"limit": self.limit}
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

async def get_sports_series(
    leagues: Iterable[str] | None = None,
    *,
    status: Optional[str] = "open",
    limit: int = 200,
    use_authenticated: bool = True,
    api_key_id: Optional[str] = None,
    private_key_pem: Optional[bytes] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    series_ids = _resolve_series_list(leagues)
    results: Dict[str, List[Dict[str, Any]]] = {}
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
            results[series_id] = df.to_dict(orient="records")
    return results

async def get_markets_by_sport(
    sport: str,
    *,
    status: Optional[str] = "open",
    limit: int = 200,
    use_authenticated: bool = True,
    api_key_id: Optional[str] = None,
    private_key_pem: Optional[bytes] = None,
) -> pd.DataFrame:
    series = _normalize_series(sport)
    params: Dict[str, Any] = {"limit": limit}
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
    status: Optional[str] = "open",
    limit: int = 200,
    use_authenticated: bool = True,
    api_key_id: Optional[str] = None,
    private_key_pem: Optional[bytes] = None,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
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
    status: Optional[str] = None,
    limit: int = 200,
    use_authenticated: bool = True,
    api_key_id: Optional[str] = None,
    private_key_pem: Optional[bytes] = None,
) -> pd.DataFrame:
    params: Dict[str, Any] = {"search": query, "limit": limit}
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
    status: Optional[str] = None,
    use_authenticated: bool = True,
    api_key_id: Optional[str] = None,
    private_key_pem: Optional[bytes] = None,
) -> pd.DataFrame:
    params: Dict[str, Any] = {"event_ticker": event_ticker}
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
    api_key_id: Optional[str] = None,
    private_key_pem: Optional[bytes] = None,
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
    api_key_id: Optional[str] = None,
    private_key_pem: Optional[bytes] = None,
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
                except:
                    pass
            return row.get("open_time", pd.NaT)

        df["game_date"] = df["ticker"].apply(parse_game_date)

        # Filter to ensure NFL-specific
        nfl_mask = df["series_ticker"].str.contains("KXNFLGAME", na=False) | df[
            "title"
        ].str.contains("NFL", case=False, na=False)
        df = df[nfl_mask]

    return df


async def get_cfb_games(
    status: str = "open",
    limit: int = 50,
    use_authenticated: bool = True,
    api_key_id: Optional[str] = None,
    private_key_pem: Optional[bytes] = None,
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
                except:
                    pass
            return row.get("open_time", pd.NaT)

        df["game_date"] = df["ticker"].apply(parse_game_date)

        # Filter to ensure CFB-specific
        cfb_mask = df["series_ticker"].str.contains("KXNCAAFGAME", na=False) | df[
            "title"
        ].str.contains("NCAA|College Football", case=False, na=False)
        df = df[cfb_mask]

    return df
