from __future__ import annotations

from typing import Any

from neural.exchanges.types import NormalizedOrderRequest, TradingPolicy
from neural.trading.kalshi_adapter import KalshiAdapter


class _Markets:
    def get_markets(self, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        return {
            "markets": [
                {
                    "ticker": "KX-1",
                    "title": "Will Team A win?",
                    "status": "open",
                    "yes_ask": 62,
                    "no_ask": 38,
                    "category": "sports",
                }
            ]
        }

    def get_market(self, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        return {"market": {"ticker": "KX-1", "yes_bid": 61, "yes_ask": 63}}


class _Exchange:
    def create_order(self, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        return {"order": {"id": "ord-k-1", "status": "submitted"}}

    def cancel_order(self, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        return {"status": "cancelled"}

    def get_order(self, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        return {"order": {"id": "ord-k-1", "status": "filled"}}


class _Portfolio:
    def get_positions(self) -> dict[str, Any]:
        return {
            "positions": [
                {
                    "market_id": "KX-1",
                    "side": "yes",
                    "quantity": 5,
                    "entry_price": 0.5,
                }
            ]
        }


class _Client:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.markets = _Markets()
        self.exchange = _Exchange()
        self.portfolio = _Portfolio()

    def close(self) -> None:
        return None


def test_kalshi_adapter_contract_methods() -> None:
    adapter = KalshiAdapter(
        api_key_id="abc",
        private_key_pem=b"pem",
        client_factory=lambda **kwargs: _Client(**kwargs),
    )

    assert adapter.capabilities().as_dict()["read"] is True

    markets = adapter.list_markets()
    assert markets[0].market_id == "KX-1"

    quote = adapter.get_quote("KX-1")
    assert quote.yes_ask == 0.63

    order = NormalizedOrderRequest(market_id="KX-1", side="buy_yes", quantity=2, price=0.61)
    result = adapter.place_order(order, policy=TradingPolicy(live_enabled=True))
    assert result.order_id == "ord-k-1"

    status = adapter.get_order_status("ord-k-1")
    assert status.status == "filled"

    positions = adapter.get_positions()
    assert positions[0].quantity == 5
