from __future__ import annotations

import pytest

from neural.trading.client import TradingClient


class _DummySession:
    def close(self) -> None:
        return None


def _new_client() -> TradingClient:
    return TradingClient(
        exchange="polymarket_us",
        polymarket_us_api_key="test-key",
        polymarket_us_api_secret=bytes(range(32)),
        polymarket_us_passphrase="test-passphrase",
        polymarket_us_base_url="https://api.polymarket.us",
        polymarket_us_session=_DummySession(),
    )


def test_polymarket_capabilities_are_read_only_beta() -> None:
    client = _new_client()

    assert client.capabilities() == {
        "read": True,
        "paper": False,
        "live": False,
        "streaming": True,
    }

    client.close()


def test_polymarket_live_order_methods_are_not_supported_yet() -> None:
    client = _new_client()

    with pytest.raises(NotImplementedError, match="introduced in a later PR"):
        client.place_order(
            market_id="MKT-1",
            side="yes",
            count=1,
            order_type="limit",
            price=0.55,
        )

    with pytest.raises(NotImplementedError, match="introduced in a later PR"):
        client.cancel_order("order-1")

    with pytest.raises(NotImplementedError, match="introduced in a later PR"):
        client.get_order_status("order-1")

    client.close()
