from __future__ import annotations

from dataclasses import dataclass

import pytest

import neural.data_collection.polymarket_us as source_module
from neural.data_collection.polymarket_us import PolymarketUSMarketsSource


@dataclass
class _AdapterStub:
    closed: bool = False

    def close(self) -> None:
        self.closed = True

    def list_markets(
        self,
        *,
        sport: str | None = None,
        limit: int = 100,
        sports_only: bool = True,
    ) -> list[object]:
        del sport, limit, sports_only
        return []

    def get_candles(
        self,
        market_id: str,
        *,
        interval: str = "1h",
        limit: int = 200,
    ) -> list[dict[str, object]]:
        del market_id, interval, limit
        return [{"timestamp": 1700000000000, "open": 0.5, "high": 0.6, "low": 0.4, "close": 0.55}]


def test_source_ignores_unknown_config_keys() -> None:
    source = PolymarketUSMarketsSource(
        config={
            "sport": "nba",
            "limit": 50,
            "sports_only": False,
            "poll_interval": 1.5,
            "unexpected": "ignored",
        },
        adapter=_AdapterStub(),
    )

    assert source._source_cfg.sport == "nba"  # noqa: SLF001
    assert source._source_cfg.limit == 50  # noqa: SLF001
    assert source._source_cfg.sports_only is False  # noqa: SLF001
    assert source._source_cfg.poll_interval == 1.5  # noqa: SLF001


def test_source_wraps_adapter_initialization_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FailingAdapter:
        def __init__(self) -> None:
            raise ValueError("bad credentials")

    monkeypatch.setattr(source_module, "PolymarketUSAdapter", _FailingAdapter)
    with pytest.raises(RuntimeError, match="Failed to initialize PolymarketUSMarketsSource adapter"):
        PolymarketUSMarketsSource()


def test_market_history_uses_adapter_public_candles_api() -> None:
    source = PolymarketUSMarketsSource(adapter=_AdapterStub())
    history = source.get_market_history("MKT-1")
    assert list(history.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert history.iloc[0]["open"] == 0.5
