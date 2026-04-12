import base64

import pytest

from neural.trading.client import TradingClient


def test_missing_dependency_raises_importerror(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force default factory import path to raise ImportError
    import neural.trading.client as tc

    def bad_default_factory():  # type: ignore[return-type]
        raise ImportError("kalshi_python missing")

    monkeypatch.setattr(tc, "_default_client_factory", bad_default_factory)

    monkeypatch.setenv("KALSHI_API_KEY_ID", "abc123")
    monkeypatch.setenv("KALSHI_PRIVATE_KEY_BASE64", base64.b64encode(b"KEY").decode())

    with pytest.raises(ImportError) as ei:
        TradingClient()  # will call bad_default_factory
    assert "kalshi_python" in str(ei.value)


def test_factory_exception_bubbles_with_context(monkeypatch: pytest.MonkeyPatch) -> None:
    def factory(**kwargs):  # type: ignore[no-untyped-def]
        raise TypeError("unexpected kwarg")

    monkeypatch.setenv("KALSHI_API_KEY_ID", "abc123")
    monkeypatch.setenv("KALSHI_PRIVATE_KEY_BASE64", base64.b64encode(b"KEY").decode())

    with pytest.raises(TypeError):
        TradingClient(client_factory=factory)


def test_invalid_kwargs_filtered(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def factory(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)

        class C:  # minimal client
            def __init__(self):
                self.portfolio = object()
                self.markets = object()
                self.exchange = object()

        return C()

    monkeypatch.setenv("KALSHI_API_KEY_ID", "abc123")
    monkeypatch.setenv("KALSHI_PRIVATE_KEY_BASE64", base64.b64encode(b"KEY").decode())

    TradingClient(client_factory=factory)

    assert set(captured.keys()) == {"base_url", "api_key_id", "private_key_pem", "timeout"}
    assert captured["api_key_id"] == "abc123"
    assert isinstance(captured["private_key_pem"], (bytes, bytearray))


def test_unknown_exchange_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported exchange"):
        TradingClient(exchange="unknown")  # type: ignore[arg-type]


def test_place_order_with_risk_warns_when_risk_registration_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _DummyApi:
        pass

    class _DummyClient:
        def __init__(self, **kwargs):  # type: ignore[no-untyped-def,unused-argument]
            self.portfolio = _DummyApi()
            self.markets = _DummyApi()
            self.exchange = _DummyApi()

    class _BrokenRiskManager:
        def add_position(self, position):  # type: ignore[no-untyped-def,unused-argument]
            raise RuntimeError("risk store unavailable")

    monkeypatch.setenv("KALSHI_API_KEY_ID", "abc123")
    monkeypatch.setenv("KALSHI_PRIVATE_KEY_BASE64", base64.b64encode(b"KEY").decode())

    client = TradingClient(
        client_factory=lambda **_: _DummyClient(),
        risk_manager=_BrokenRiskManager(),
    )

    with pytest.warns(RuntimeWarning, match="Risk manager position registration failed"):
        order = client.place_order_with_risk(
            market_id="KX-1",
            side="yes",
            quantity=1,
            stop_loss_config={"kind": "fixed", "value": 0.1},
            price=0.55,
            paper=True,
        )

    assert order["order_id"]
