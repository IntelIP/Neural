from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

import neural.trading.polymarket_us_adapter as adapter_module
from neural.trading.polymarket_us_adapter import PolymarketUSAdapter, _to_float, _to_prob


@dataclass
class FakeResponse:
    payload: dict[str, Any]
    status_code: int = 200
    raise_http_error: bool = True

    @property
    def text(self) -> str:
        return "{}"

    def json(self) -> dict[str, Any]:
        return self.payload

    def raise_for_status(self) -> None:
        if self.raise_http_error:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        data: str | None = None,
        headers: dict[str, Any] | None = None,
        timeout: int | None = None,
    ) -> FakeResponse:
        self.calls.append((method, url, {"params": params, "data": data, "headers": headers}))
        if method == "GET" and url.endswith("/v1/markets"):
            return FakeResponse(
                {
                    "markets": [
                        {
                            "id": "1",
                            "slug": "nfl-team-a-team-b",
                            "question": "Team A vs. Team B",
                            "category": "sports",
                            "active": True,
                            "closed": False,
                            "marketType": "moneyline",
                            "gameStartTime": "2026-03-10T23:00:00Z",
                            "marketSides": [
                                {
                                    "description": "Team A",
                                    "team": {"name": "Team A", "league": "nfl", "ordering": "away"},
                                },
                                {
                                    "description": "Team B",
                                    "team": {"name": "Team B", "league": "nfl", "ordering": "home"},
                                },
                            ],
                        },
                        {
                            "id": "2",
                            "slug": "candidate-x",
                            "question": "Will candidate X win?",
                            "category": "politics",
                            "active": True,
                            "closed": False,
                        },
                    ]
                }
            )

        if method == "GET" and url.endswith("/v1/market/id/1"):
            return FakeResponse(
                {
                    "market": {
                        "id": "1",
                        "slug": "nfl-team-a-team-b",
                        "question": "Team A vs. Team B",
                    }
                }
            )

        if method == "GET" and url.endswith("/v1/markets/nfl-team-a-team-b/bbo"):
            return FakeResponse(
                {
                    "marketData": {
                        "bestBid": 0.61,
                        "bestAsk": 0.63,
                        "lastTradePx": {"value": "0.62"},
                        "sharesTraded": "1234",
                    }
                }
            )

        if method == "GET" and url.endswith("/api/v1/markets/1/candles"):
            return FakeResponse(
                {
                    "candles": [
                        {
                            "timestamp": 1700000000000,
                            "open": 0.5,
                            "high": 0.6,
                            "low": 0.4,
                            "close": 0.55,
                        }
                    ]
                }
            )

        if method == "GET" and url.endswith("/api/v1/markets/1/trades"):
            return FakeResponse(
                {
                    "trades": [
                        {
                            "id": "t1",
                            "timestamp": 1700000000000,
                            "price": 0.55,
                            "size": 10,
                            "sequence": 1,
                        }
                    ],
                    "next_cursor": "next-1",
                }
            )

        if method == "GET" and url.endswith("/api/v1/markets/1/events"):
            return FakeResponse(
                {
                    "events": [
                        {"id": "e1", "timestamp": 1700000000000, "type": "fill", "sequence": 1}
                    ],
                    "next_cursor": None,
                }
            )

        if method == "GET" and url.endswith("/api/v1/portfolio/positions"):
            return FakeResponse(
                {
                    "positions": [
                        {
                            "market_id": "1",
                            "side": "yes",
                            "quantity": 10,
                            "entry_price": 0.6,
                            "current_price": 0.63,
                        }
                    ]
                }
            )

        if method == "GET" and url.endswith("/api/v1/markets/1/candles"):
            return FakeResponse({"candles": [{"timestamp": 1700000000000, "open": 0.5}]})

        return FakeResponse({}, status_code=404)

    def close(self) -> None:
        return None


def _new_adapter(session: FakeSession, **kwargs: Any) -> PolymarketUSAdapter:
    return PolymarketUSAdapter(
        api_key="k",
        api_secret=bytes(range(32)),
        passphrase="p",
        base_url="https://api.polymarket.us",
        session=session,
        **kwargs,
    )


def _new_public_adapter(
    session: FakeSession,
    monkeypatch: pytest.MonkeyPatch,
    **kwargs: Any,
) -> PolymarketUSAdapter:
    monkeypatch.setattr(adapter_module, "get_polymarket_us_credentials", lambda: {})
    return PolymarketUSAdapter(
        base_url="https://api.polymarket.us",
        session=session,
        **kwargs,
    )


def test_list_markets_filters_non_sports_by_default() -> None:
    session = FakeSession()
    adapter = _new_adapter(session)

    markets = adapter.list_markets(limit=20, sports_only=True)

    assert len(markets) == 1
    assert markets[0].market_id == "1"
    assert markets[0].sport == "nfl"


def test_get_quote_maps_book_shape() -> None:
    session = FakeSession()
    adapter = _new_adapter(session)

    quote = adapter.get_quote("1")

    assert quote.market_id == "1"
    assert quote.yes_bid == pytest.approx(0.61)
    assert quote.yes_ask == pytest.approx(0.63)
    assert quote.no_bid == pytest.approx(0.37)


def test_capabilities_include_streaming_in_pr3() -> None:
    session = FakeSession()
    adapter = _new_adapter(session)

    caps = adapter.capabilities().as_dict()
    assert caps == {"read": True, "paper": False, "live": False, "streaming": True}


def test_get_positions_returns_normalized_rows() -> None:
    session = FakeSession()
    adapter = _new_adapter(session)

    positions = adapter.get_positions()

    assert len(positions) == 1
    assert positions[0].market_id == "1"
    assert positions[0].quantity == 10


def test_get_candles_returns_rows() -> None:
    session = FakeSession()
    adapter = _new_adapter(session)

    rows = adapter.get_candles("1", interval="1m", limit=5)

    assert len(rows) == 1
    assert rows[0]["close"] == pytest.approx(0.55)


def test_trade_replay_and_event_replay_return_cursor() -> None:
    session = FakeSession()
    adapter = _new_adapter(session)

    trades = adapter.get_trade_replay("1", limit=25)
    events = adapter.get_market_events("1", limit=25)

    assert len(trades["items"]) == 1
    assert trades["next_cursor"] == "next-1"
    assert len(events["items"]) == 1
    assert events["next_cursor"] is None


def test_request_raises_when_error_response_does_not_raise_for_status() -> None:
    class _SoftFailSession(FakeSession):
        def request(
            self,
            method: str,
            url: str,
            *,
            params: dict[str, Any] | None = None,
            data: str | None = None,
            headers: dict[str, Any] | None = None,
            timeout: int | None = None,
        ) -> FakeResponse:
            del method, url, params, data, headers, timeout
            return FakeResponse({}, status_code=500, raise_http_error=False)

    adapter = _new_adapter(_SoftFailSession())
    with pytest.raises(RuntimeError, match="raise_for_status\\(\\) did not raise"):
        adapter.get_quote("1")


def test_request_raises_for_invalid_json_response() -> None:
    class _BadJsonResponse(FakeResponse):
        @property
        def text(self) -> str:
            return "{bad json"

        def json(self) -> dict[str, Any]:
            raise ValueError("invalid json")

    class _BadJsonSession(FakeSession):
        def request(
            self,
            method: str,
            url: str,
            *,
            params: dict[str, Any] | None = None,
            data: str | None = None,
            headers: dict[str, Any] | None = None,
            timeout: int | None = None,
        ) -> FakeResponse:
            del method, url, params, data, headers, timeout
            return _BadJsonResponse({})

    adapter = _new_adapter(_BadJsonSession(), max_retries=0)
    with pytest.raises(RuntimeError, match="not valid JSON"):
        adapter.list_markets()


def test_numeric_parsing_helpers_return_none_for_invalid_values() -> None:
    assert _to_prob("bad") is None
    assert _to_prob(object()) is None
    assert _to_prob(150) is None
    assert _to_float("bad") is None
    assert _to_float(object()) is None




def test_list_markets_paginates_until_sport_filter_is_satisfied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _PagedSession(FakeSession):
        def request(
            self,
            method: str,
            url: str,
            *,
            params: dict[str, Any] | None = None,
            data: str | None = None,
            headers: dict[str, Any] | None = None,
            timeout: int | None = None,
        ) -> FakeResponse:
            if method == "GET" and url.endswith("/v1/markets"):
                offset = int((params or {}).get("offset", 0))
                if offset == 0:
                    return FakeResponse(
                        {
                            "markets": [
                                {
                                    "id": "10",
                                    "slug": "nfl-a",
                                    "question": "NFL A vs. B",
                                    "category": "sports",
                                    "active": True,
                                    "closed": False,
                                    "marketSides": [
                                        {"team": {"name": "A", "league": "nfl", "ordering": "away"}},
                                        {"team": {"name": "B", "league": "nfl", "ordering": "home"}},
                                    ],
                                }
                            ]
                        }
                    )
                if offset == 1:
                    return FakeResponse(
                        {
                            "markets": [
                                {
                                    "id": "20",
                                    "slug": "nba-a",
                                    "question": "NBA A vs. B",
                                    "category": "sports",
                                    "active": True,
                                    "closed": False,
                                    "marketSides": [
                                        {"team": {"name": "A", "league": "nba", "ordering": "away"}},
                                        {"team": {"name": "B", "league": "nba", "ordering": "home"}},
                                    ],
                                }
                            ]
                        }
                    )
                return FakeResponse({"markets": []})
            return super().request(
                method,
                url,
                params=params,
                data=data,
                headers=headers,
                timeout=timeout,
            )

    adapter = _new_public_adapter(_PagedSession(), monkeypatch)
    markets = adapter.list_markets(sport="nba", limit=1)

    assert len(markets) == 1
    assert markets[0].sport == "nba"
    assert markets[0].market_id == "20"


def test_public_market_reads_work_without_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = FakeSession()
    adapter = _new_public_adapter(session, monkeypatch)

    markets = adapter.list_markets(limit=20, sports_only=True)
    quote = adapter.get_quote("1")

    assert len(markets) == 1
    assert markets[0].status == "open"
    assert quote.market_id == "1"
    assert quote.last_price == pytest.approx(0.62)
    first_headers = session.calls[0][2]["headers"] or {}
    assert "PM-ACCESS-KEY" not in first_headers
    assert "PM-ACCESS-SIGNATURE" not in first_headers



def test_private_polymarket_methods_still_require_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _new_public_adapter(FakeSession(), monkeypatch)

    with pytest.raises(ValueError, match="credentials are required for portfolio positions"):
        adapter.get_positions()
    with pytest.raises(ValueError, match="credentials are required for /api/v1/markets/1/candles"):
        adapter.get_candles("1")
    with pytest.raises(ValueError, match="credentials are required for /api/v1/markets/1/trades"):
        adapter.get_trade_replay("1")
    with pytest.raises(ValueError, match="credentials are required for /api/v1/markets/1/events"):
        adapter.get_market_events("1")
    with pytest.raises(ValueError, match="credentials are required for market websocket"):
        adapter.market_ws_client()
    with pytest.raises(ValueError, match="credentials are required for user websocket"):
        adapter.user_ws_client()


def test_get_candles_returns_normalized_rows() -> None:
    session = FakeSession()
    adapter = _new_adapter(session)
    rows = adapter.get_candles("1")
    assert rows
    assert rows[0]["timestamp"] == 1700000000000
    assert rows[0]["open"] == 0.5
