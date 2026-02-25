from __future__ import annotations

from datetime import timedelta

import pytest

from neural.exchanges.base import BaseExchangeAdapter
from neural.exchanges.types import (
    ExchangeCapabilities,
    NormalizedMarket,
    NormalizedOrderRequest,
    NormalizedOrderResult,
    NormalizedPosition,
    NormalizedQuote,
    TradingPolicy,
)


class _PolicyAdapter(BaseExchangeAdapter):
    name = "kalshi"

    def capabilities(self) -> ExchangeCapabilities:
        return ExchangeCapabilities(read=True, paper=True, live=True, streaming=True)

    def list_markets(
        self, *, sport: str | None = None, limit: int = 100, sports_only: bool = True
    ) -> list[NormalizedMarket]:
        del sport, limit, sports_only
        return []

    def get_quote(self, market_id: str) -> NormalizedQuote:
        del market_id
        raise NotImplementedError

    def place_order(
        self,
        order: NormalizedOrderRequest,
        *,
        policy: TradingPolicy | None = None,
    ) -> NormalizedOrderResult:
        del order, policy
        raise NotImplementedError

    def cancel_order(self, order_id: str) -> NormalizedOrderResult:
        del order_id
        raise NotImplementedError

    def get_order_status(self, order_id: str) -> NormalizedOrderResult:
        del order_id
        raise NotImplementedError

    def get_positions(self) -> list[NormalizedPosition]:
        return []

    def close(self) -> None:
        return None


def test_daily_notional_resets_after_day_rollover() -> None:
    adapter = _PolicyAdapter()
    policy = TradingPolicy(live_enabled=True, max_daily_notional=5.0)

    first = NormalizedOrderRequest(market_id="MKT-1", side="buy_yes", quantity=4, price=1.0)
    adapter._enforce_policy(first, policy)

    adapter._daily_notional_date = adapter._daily_notional_date - timedelta(days=1)
    second = NormalizedOrderRequest(market_id="MKT-2", side="buy_yes", quantity=4, price=1.0)

    # Should pass after rollover rather than failing against yesterday's usage.
    adapter._enforce_policy(second, policy)


def test_daily_notional_still_enforced_within_same_day() -> None:
    adapter = _PolicyAdapter()
    policy = TradingPolicy(live_enabled=True, max_daily_notional=5.0)

    adapter._enforce_policy(
        NormalizedOrderRequest(market_id="MKT-1", side="buy_yes", quantity=4, price=1.0),
        policy,
    )
    with pytest.raises(ValueError):
        adapter._enforce_policy(
            NormalizedOrderRequest(market_id="MKT-2", side="buy_yes", quantity=2, price=1.0),
            policy,
        )
