import pandas as pd
import pytest

from neural.trading.rest_streaming import RESTStreamingClient


class _BrokenFetchClient:
    async def fetch(self):  # type: ignore[no-untyped-def]
        raise RuntimeError("service unavailable")


class _GoodFetchClient:
    async def fetch(self):  # type: ignore[no-untyped-def]
        return pd.DataFrame(
            [
                {
                    "ticker": "KX-1",
                    "title": "Test market",
                    "yes_bid": 55,
                    "yes_ask": 57,
                    "no_bid": 43,
                    "no_ask": 45,
                    "volume": 123,
                    "open_interest": 456,
                    "last_price": 56,
                }
            ]
        )


class _MalformedFetchClient:
    async def fetch(self):  # type: ignore[no-untyped-def]
        return pd.DataFrame([{"symbol": "KX-1"}])


async def test_fetch_market_reports_fetch_errors_without_swallowing() -> None:
    errors: list[str] = []
    client = RESTStreamingClient(on_error=errors.append)
    client.client = _BrokenFetchClient()

    await client._fetch_market("KX-1")

    assert errors == ["Error fetching KX-1: service unavailable"]
    assert client.get_snapshot("KX-1") is None


async def test_fetch_market_reports_parse_errors_without_crashing() -> None:
    errors: list[str] = []
    client = RESTStreamingClient(on_error=errors.append)
    client.client = _MalformedFetchClient()

    await client._fetch_market("KX-1")

    assert len(errors) == 1
    assert errors[0].startswith("Error parsing KX-1:")
    assert client.get_snapshot("KX-1") is None


async def test_fetch_market_propagates_callback_errors() -> None:
    client = RESTStreamingClient(
        on_market_update=lambda snapshot: (_ for _ in ()).throw(ValueError("bad callback"))
    )
    client.client = _GoodFetchClient()

    with pytest.raises(ValueError, match="bad callback"):
        await client._fetch_market("KX-1")

    snapshot = client.get_snapshot("KX-1")
    assert snapshot is not None
    assert snapshot.ticker == "KX-1"
