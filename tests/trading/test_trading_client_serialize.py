import base64
from typing import Any

import pytest

import neural.trading.client as trading_client_module
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


def test_compat_wrappers_do_not_reserialize_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    class _CompatClient:
        def list_markets(self, **kwargs: Any) -> list[dict[str, Any]]:
            del kwargs
            return [{"market_id": "KX-1"}]

        def get_positions(self) -> list[dict[str, Any]]:
            return [{"market_id": "KX-1", "quantity": 2}]

    def _unexpected_serialize(_: Any) -> Any:
        raise AssertionError("compat wrappers should not call serialize_value")

    monkeypatch.setattr(trading_client_module, "serialize_value", _unexpected_serialize)
    compat_client = _CompatClient()

    markets = trading_client_module._CompatMarkets(compat_client).get_markets()
    positions = trading_client_module._CompatPortfolio(compat_client).get_positions()

    assert markets == {"markets": [{"market_id": "KX-1"}]}
    assert positions == {"positions": [{"market_id": "KX-1", "quantity": 2}]}
