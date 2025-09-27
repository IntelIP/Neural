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
