from __future__ import annotations

import asyncio
import base64
from typing import Any

import pytest

from neural.trading.client import TradingClient


class _DummyClient:
    def __init__(self, **kwargs: Any) -> None:  # noqa: ARG002
        self.portfolio = object()
        self.markets = object()
        self.exchange = object()


class _FakePaperClient:
    def __init__(self) -> None:
        self.close_calls: list[dict[str, Any]] = []

    def close_position(
        self, market_id: str, side: str, quantity: int | None = None
    ) -> dict[str, Any]:
        self.close_calls.append({"market_id": market_id, "side": side, "quantity": quantity})
        return {"success": True, "closed_quantity": quantity}

    async def place_order(
        self,
        market_id: str,
        side: str,
        quantity: int,
        order_type: str = "market",
        price: float | None = None,
    ) -> dict[str, Any]:
        del order_type, price
        return {"success": True, "market_id": market_id, "side": side, "quantity": quantity}


def _fake_creds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KALSHI_API_KEY_ID", "abc123")
    monkeypatch.setenv("KALSHI_PRIVATE_KEY_BASE64", base64.b64encode(b"KEY").decode())


def test_paper_sell_uses_requested_quantity(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_creds(monkeypatch)
    client = TradingClient(client_factory=lambda **_: _DummyClient(), paper_trading=True)
    fake_paper = _FakePaperClient()
    client._paper_client = fake_paper

    out = client.place_order(market_id="MKT-1", side="sell_yes", quantity=3, paper=True)

    assert out["success"] is True
    assert out["closed_quantity"] == 3
    assert fake_paper.close_calls == [{"market_id": "MKT-1", "side": "yes", "quantity": 3}]


def test_paper_buy_works_inside_running_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_creds(monkeypatch)
    client = TradingClient(client_factory=lambda **_: _DummyClient(), paper_trading=True)
    client._paper_client = _FakePaperClient()

    async def _call_sync_api() -> dict[str, Any]:
        return client.place_order(market_id="MKT-2", side="buy_yes", quantity=2, paper=True)

    out = asyncio.run(_call_sync_api())
    assert out["success"] is True
    assert out["quantity"] == 2
