from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from neural.trading.polymarket_us_adapter import PolymarketUSAdapter


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
        if method == "GET" and url.endswith("/api/v1/sports/markets"):
            return FakeResponse(
                {
                    "data": [
                        {
                            "id": "MKT-SPORT-1",
                            "title": "Will Team A win?",
                            "category": "sports",
                            "sport": "nfl",
                            "yes_price": 0.62,
                        },
                        {
                            "id": "MKT-POL-1",
                            "title": "Will candidate X win?",
                            "category": "politics",
                            "yes_price": 0.51,
                        },
                    ]
                }
            )

        if method == "GET" and url.endswith("/api/v1/markets/MKT-SPORT-1/book"):
            return FakeResponse({"book": {"yes_bid": 0.61, "yes_ask": 0.63, "volume": 1234}})

        if method == "GET" and url.endswith("/api/v1/markets/MKT-SPORT-1/candles"):
            return FakeResponse(
                {
                    "candles": [
                        {"timestamp": 1700000000000, "open": 0.5, "high": 0.6, "low": 0.4, "close": 0.55}
                    ]
                }
            )

        if method == "GET" and url.endswith("/api/v1/markets/MKT-SPORT-1/trades"):
            return FakeResponse(
                {
                    "trades": [
                        {"id": "t1", "timestamp": 1700000000000, "price": 0.55, "size": 10, "sequence": 1}
                    ],
                    "next_cursor": "next-1",
                }
            )

        if method == "GET" and url.endswith("/api/v1/markets/MKT-SPORT-1/events"):
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
                            "market_id": "MKT-SPORT-1",
                            "side": "yes",
                            "quantity": 10,
                            "entry_price": 0.6,
                            "current_price": 0.63,
                        }
                    ]
                }
            )

        return FakeResponse({}, status_code=404)

    def close(self) -> None:
        return None


def _new_adapter(session: FakeSession) -> PolymarketUSAdapter:
    return PolymarketUSAdapter(
        api_key="k",
        api_secret=bytes(range(32)),
        passphrase="p",
        base_url="https://api.polymarket.us",
        session=session,
    )


def test_list_markets_filters_non_sports_by_default() -> None:
    session = FakeSession()
    adapter = _new_adapter(session)

    markets = adapter.list_markets(limit=20, sports_only=True)

    assert len(markets) == 1
    assert markets[0].market_id == "MKT-SPORT-1"


def test_get_quote_maps_book_shape() -> None:
    session = FakeSession()
    adapter = _new_adapter(session)

    quote = adapter.get_quote("MKT-SPORT-1")

    assert quote.market_id == "MKT-SPORT-1"
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
    assert positions[0].market_id == "MKT-SPORT-1"
    assert positions[0].quantity == 10


def test_get_candles_returns_rows() -> None:
    session = FakeSession()
    adapter = _new_adapter(session)

    rows = adapter.get_candles("MKT-SPORT-1", interval="1m", limit=5)

    assert len(rows) == 1
    assert rows[0]["close"] == pytest.approx(0.55)


def test_trade_replay_and_event_replay_return_cursor() -> None:
    session = FakeSession()
    adapter = _new_adapter(session)

    trades = adapter.get_trade_replay("MKT-SPORT-1", limit=25)
    events = adapter.get_market_events("MKT-SPORT-1", limit=25)

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
        adapter.get_quote("MKT-SPORT-1")
