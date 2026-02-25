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

    def close_position(self, market_id: str, side: str, quantity: int | None = None) -> dict[str, Any]:
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


class _FakeLiveAdapter:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def place_order(self, order: Any, *, policy: Any = None) -> dict[str, Any]:
        self.calls.append({"order": order, "policy": policy})
        return {"success": True, "order_id": "LIVE-1"}
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


def test_paper_buy_sync_raises_inside_running_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_creds(monkeypatch)
    client = TradingClient(client_factory=lambda **_: _DummyClient(), paper_trading=True)
    client._paper_client = _FakePaperClient()

    async def _call_sync_api() -> None:
        with pytest.raises(RuntimeError, match="place_order_async"):
            client.place_order(market_id="MKT-2", side="buy_yes", quantity=2, paper=True)

    asyncio.run(_call_sync_api())


def test_paper_buy_async_works_inside_running_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_creds(monkeypatch)
    client = TradingClient(client_factory=lambda **_: _DummyClient(), paper_trading=True)
    client._paper_client = _FakePaperClient()

    async def _call_async_api() -> dict[str, Any]:
        return await client.place_order_async(market_id="MKT-2", side="buy_yes", quantity=2, paper=True)

    out = asyncio.run(_call_async_api())
    assert out["success"] is True
    assert out["quantity"] == 2


def test_place_order_async_passes_policy_keyword(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_creds(monkeypatch)
    client = TradingClient(client_factory=lambda **_: _DummyClient(), paper_trading=False)
    live_adapter = _FakeLiveAdapter()
    client._adapter = live_adapter

    async def _call_async_api() -> dict[str, Any]:
        return await client.place_order_async(market_id="MKT-3", side="buy_yes", quantity=1, paper=False)

    out = asyncio.run(_call_async_api())
    assert out["success"] is True
    assert live_adapter.calls
    assert live_adapter.calls[0]["policy"] is client.trading_policy
