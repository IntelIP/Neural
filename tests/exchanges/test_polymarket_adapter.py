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

        if method == "GET" and url.endswith("/api/v1/markets/MKT-SPORT-1/candles"):
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


def test_capabilities_are_read_only_in_pr2() -> None:
    session = FakeSession()
    adapter = _new_adapter(session)

    caps = adapter.capabilities().as_dict()
    assert caps == {"read": True, "paper": False, "live": False, "streaming": False}


def test_get_positions_returns_normalized_rows() -> None:
    session = FakeSession()
    adapter = _new_adapter(session)

    positions = adapter.get_positions()

    assert len(positions) == 1
    assert positions[0].market_id == "MKT-SPORT-1"
    assert positions[0].quantity == 10


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


def test_adapter_raises_clear_error_for_missing_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(adapter_module, "get_polymarket_us_credentials", lambda: {})
    with pytest.raises(ValueError, match="credentials are required"):
        PolymarketUSAdapter(base_url="https://api.polymarket.us")


def test_get_candles_returns_normalized_rows() -> None:
    session = FakeSession()
    adapter = _new_adapter(session)
    rows = adapter.get_candles("MKT-SPORT-1")
    assert rows == [{"timestamp": 1700000000000, "open": 0.5}]
