from typing import Any

from neural.trading.client import TradingClient


class ModelLike:
    def __init__(self, data: Any) -> None:
        self._data = data

    def model_dump(self) -> Any:
        return self._data


class DummyApi:
    def __init__(self) -> None:
        self.answer = 42

    def get_model(self) -> Any:
        return ModelLike({"x": 1, "y": [1, 2, {"z": (3, 4)}]})

    def echo(self, v: Any) -> Any:
        return v


class DummyClient:
    def __init__(self, **kwargs: Any) -> None:  # noqa: ARG002
        self.portfolio = DummyApi()
        self.markets = DummyApi()
        self.exchange = DummyApi()


import base64

import pytest


def _fake_creds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KALSHI_API_KEY_ID", "abc123")
    monkeypatch.setenv("KALSHI_PRIVATE_KEY_BASE64", base64.b64encode(b"KEY").decode())


def test_serializes_pydantic_like_models(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_creds(monkeypatch)
    client = TradingClient(client_factory=lambda **_: DummyClient())

    data = client.portfolio.get_model()
    assert data == {"x": 1, "y": [1, 2, {"z": (3, 4)}]}


def test_serializes_nested_structures(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_creds(monkeypatch)
    client = TradingClient(client_factory=lambda **_: DummyClient())

    out = client.markets.echo([{"a": (1, 2)}, [3, 4]])
    assert out == [{"a": (1, 2)}, [3, 4]]


def test_non_callable_attribute_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_creds(monkeypatch)
    client = TradingClient(client_factory=lambda **_: DummyClient())

    assert client.exchange.answer == 42
