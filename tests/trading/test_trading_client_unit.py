import base64
from typing import Any

import pytest

from neural.trading.client import TradingClient


class DummySubApi:
    def echo(self, x: Any) -> Any:
        return {"value": x}


class DummyClient:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.portfolio = DummySubApi()
        self.markets = DummySubApi()
        self.exchange = DummySubApi()
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_trading_client_injectable_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    # Provide fake credentials via env
    monkeypatch.setenv("KALSHI_API_KEY_ID", "abc123")
    monkeypatch.setenv("KALSHI_PRIVATE_KEY_BASE64", base64.b64encode(b"FAKE KEY").decode())

    # Inject a factory returning our DummyClient
    def factory(**kwargs: Any) -> DummyClient:  # type: ignore[override]
        return DummyClient(**kwargs)

    client = TradingClient(client_factory=factory)

    # Verify kwargs passed to factory
    assert client._client.kwargs["base_url"].startswith("https://")
    assert client._client.kwargs["api_key_id"] == "abc123"
    assert client._client.kwargs["private_key_pem"] == b"FAKE KEY"

    # Verify proxies forward and serialize
    assert client.portfolio.echo([1, 2, 3]) == {"value": [1, 2, 3]}
    assert client.markets.echo({"a": 1}) == {"value": {"a": 1}}
    assert client.exchange.echo(("x", "y")) == {"value": ("x", "y")}

    # Context manager closes underlying client
    with client as c:
        assert isinstance(c, TradingClient)
    assert client._client.closed is True
