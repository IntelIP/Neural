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
