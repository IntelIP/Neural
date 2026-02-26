from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from neural.auth.signers.polymarket_us import PolymarketUSSigner
from neural.trading.polymarket_us_adapter import (
    PolymarketUSMarketWebSocketClient,
)


@dataclass
class FakeConnection:
    recv_values: list[Any]
    sent_values: list[str] = field(default_factory=list)
    closed: bool = False

    async def recv(self) -> Any:
        if not self.recv_values:
            raise RuntimeError("No more messages")
        value = self.recv_values.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    async def send(self, payload: str) -> None:
        self.sent_values.append(payload)

    async def close(self) -> None:
        self.closed = True


class FakeConnectFactory:
    def __init__(self, conns: list[FakeConnection]) -> None:
        self._conns = conns
        self.calls: list[dict[str, Any]] = []

    async def __call__(
        self,
        url: str,
        *,
        extra_headers: dict[str, str],
        open_timeout: int,
    ) -> FakeConnection:
        self.calls.append(
            {"url": url, "extra_headers": extra_headers, "open_timeout": open_timeout}
        )
        if not self._conns:
            raise RuntimeError("No available websocket connections")
        return self._conns.pop(0)


def _new_signer() -> PolymarketUSSigner:
    return PolymarketUSSigner(api_key="k", api_secret=bytes(range(32)), passphrase="p")


async def test_reconnect_restores_subscriptions() -> None:
    conn1 = FakeConnection(recv_values=[RuntimeError("dropped")])
    conn2 = FakeConnection(
        recv_values=['{"channel":"markets","market_id":"MKT-1","sequence":1,"price":0.61}']
    )
    connect_factory = FakeConnectFactory([conn1, conn2])

    client = PolymarketUSMarketWebSocketClient(
        url="wss://ws.polymarket.us/markets",
        signer=_new_signer(),
        ws_connect=connect_factory,
        max_retries=2,
        backoff_base_s=0.0,
        backoff_max_s=0.0,
    )

    await client.connect()
    await client.subscribe_markets(["MKT-1"])
    assert conn1.sent_values

    stream = client.listen()
    message = await asyncio.wait_for(anext(stream), timeout=1.0)
    await stream.aclose()
    await client.disconnect()

    assert message["market_id"] == "MKT-1"
    assert len(connect_factory.calls) == 2
    assert conn2.sent_values


async def test_reconnect_resets_sequence_tracking() -> None:
    conn1 = FakeConnection(
        recv_values=[
            '{"channel":"markets","market_id":"MKT-1","sequence":5,"price":0.60}',
            RuntimeError("dropped"),
        ]
    )
    conn2 = FakeConnection(
        recv_values=['{"channel":"markets","market_id":"MKT-1","sequence":1,"price":0.61}']
    )
    connect_factory = FakeConnectFactory([conn1, conn2])

    client = PolymarketUSMarketWebSocketClient(
        url="wss://ws.polymarket.us/markets",
        signer=_new_signer(),
        ws_connect=connect_factory,
        max_retries=2,
        backoff_base_s=0.0,
        backoff_max_s=0.0,
    )

    stream = client.listen()
    first = await asyncio.wait_for(anext(stream), timeout=1.0)
    second = await asyncio.wait_for(anext(stream), timeout=1.0)
    await stream.aclose()
    await client.disconnect()

    assert first["sequence"] == 5
    assert second["sequence"] == 1
    assert "_sequence_gap" not in second


def test_sequence_rules_dedupe_and_gap() -> None:
    client = PolymarketUSMarketWebSocketClient(
        url="wss://ws.polymarket.us/markets",
        signer=_new_signer(),
    )

    first = client._apply_sequence_rules(
        {"channel": "markets", "market_id": "MKT-1", "sequence": 1}
    )
    duplicate = client._apply_sequence_rules(
        {"channel": "markets", "market_id": "MKT-1", "sequence": 1}
    )
    gap = client._apply_sequence_rules({"channel": "markets", "market_id": "MKT-1", "sequence": 3})

    assert first is not None
    assert duplicate is None
    assert gap is not None
    assert gap["_sequence_gap"] == {"expected": 2, "received": 3}


def test_parser_skips_invalid_payloads() -> None:
    client = PolymarketUSMarketWebSocketClient(
        url="wss://ws.polymarket.us/markets",
        signer=_new_signer(),
    )

    assert client._parse_message("not-json") is None
    parsed = client._parse_message(json.dumps({"type": "tick", "sequence": 1}).encode("utf-8"))
    assert parsed == {"type": "tick", "sequence": 1}
