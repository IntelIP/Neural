from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

import neural.data_collection.polymarket_us as source_module
from neural.data_collection.polymarket_us import PolymarketUSMarketsSource
from neural.exchanges.types import NormalizedMarket


@dataclass
class FakeAdapter:
    closed: bool = False

    def list_markets(
        self,
        *,
        sport: str | None = None,
        limit: int = 100,
        sports_only: bool = True,
    ) -> list[NormalizedMarket]:
        del limit, sports_only
        return [
            NormalizedMarket(
                market_id="MKT-1",
                ticker="MKT-1",
                title="Example",
                status="open",
                yes_price=0.6,
                no_price=0.4,
                category="sports",
                sport=sport or "nfl",
            )
        ]

    def get_candles(
        self,
        market_id: str,
        *,
        interval: str = "1h",
        limit: int = 200,
        start_ts_ms: int | None = None,
        end_ts_ms: int | None = None,
    ) -> list[dict[str, Any]]:
        del market_id, interval, limit, start_ts_ms, end_ts_ms
        return [{"timestamp": 1700000000000, "open": 0.5, "high": 0.6, "low": 0.45, "close": 0.55}]

    def get_trade_replay(
        self,
        market_id: str,
        *,
        limit: int = 500,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        del market_id, limit
        if cursor is None:
            return {
                "items": [{"id": "t1", "timestamp": 1700000000000, "price": 0.55, "size": 10}],
                "next_cursor": "cursor-2",
            }
        return {
            "items": [{"id": "t2", "timestamp": 1700000060000, "price": 0.56, "size": 7}],
            "next_cursor": None,
        }

    def get_market_events(
        self,
        market_id: str,
        *,
        limit: int = 500,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        del market_id, limit, cursor
        return {
            "items": [{"id": "e1", "timestamp": 1700000000000, "type": "fill", "sequence": 1}],
            "next_cursor": None,
        }

    def close(self) -> None:
        self.closed = True


def test_source_ignores_unknown_config_keys() -> None:
    source = PolymarketUSMarketsSource(
        config={
            "sport": "nba",
            "limit": 50,
            "sports_only": False,
            "poll_interval": 1.5,
            "unexpected": "ignored",
        },
        adapter=FakeAdapter(),
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
    with pytest.raises(
        RuntimeError, match="Failed to initialize PolymarketUSMarketsSource adapter"
    ):
        PolymarketUSMarketsSource()


def test_market_history_uses_adapter_public_candles_api() -> None:
    source = PolymarketUSMarketsSource(adapter=FakeAdapter())
    history = source.get_market_history("MKT-1")
    assert list(history.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert history.iloc[0]["open"] == 0.5


def test_market_history_and_replay_frames() -> None:
    source = PolymarketUSMarketsSource(adapter=FakeAdapter())

    history = source.get_market_history("MKT-1")
    replay, next_cursor = source.get_market_replay_df("MKT-1")
    events, event_cursor = source.get_market_events_df("MKT-1")

    assert not history.empty
    assert not replay.empty
    assert not events.empty
    assert next_cursor == "cursor-2"
    assert event_cursor is None
    assert "event_type" in events.columns


def test_replay_iterator_paginates_with_max_pages() -> None:
    source = PolymarketUSMarketsSource(adapter=FakeAdapter())
    pages = list(source.iter_market_replay("MKT-1", max_pages=2))

    assert len(pages) == 2
    assert pages[0].iloc[0]["trade_id"] == "t1"
    assert pages[1].iloc[0]["trade_id"] == "t2"
