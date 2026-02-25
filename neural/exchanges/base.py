from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

from .types import (
    ExchangeCapabilities,
    ExchangeName,
    NormalizedMarket,
    NormalizedOrderRequest,
    NormalizedOrderResult,
    NormalizedPosition,
    NormalizedQuote,
    TradingPolicy,
)


class ExchangeAdapter(ABC):
    """Contract for exchange integrations used by TradingClient."""

    name: ExchangeName

    @abstractmethod
    def capabilities(self) -> ExchangeCapabilities:
        """Return adapter capabilities."""

    @abstractmethod
    def list_markets(
        self, *, sport: str | None = None, limit: int = 100, sports_only: bool = True
    ) -> list[NormalizedMarket]:
        """List markets for discovery and strategy selection."""

    @abstractmethod
    def get_quote(self, market_id: str) -> NormalizedQuote:
        """Fetch best bid/ask quote for a market."""

    @abstractmethod
    def place_order(
        self,
        order: NormalizedOrderRequest,
        *,
        policy: TradingPolicy | None = None,
    ) -> NormalizedOrderResult:
        """Place an order on the exchange."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> NormalizedOrderResult:
        """Cancel an existing order."""

    @abstractmethod
    def get_order_status(self, order_id: str) -> NormalizedOrderResult:
        """Fetch order status."""

    @abstractmethod
    def get_positions(self) -> list[NormalizedPosition]:
        """Fetch open positions."""

    @abstractmethod
    def close(self) -> None:
        """Close network resources."""


class BaseExchangeAdapter(ExchangeAdapter):
    """Shared policy guards for live trading adapters."""

    _daily_notional_used: float
    _daily_notional_date: date

    def __init__(self) -> None:
        self._daily_notional_used = 0.0
        self._daily_notional_date = self._current_day()

    def _current_day(self) -> date:
        return date.today()

    def _rollover_daily_notional_if_needed(self) -> None:
        current_day = self._current_day()
        if self._daily_notional_date != current_day:
            self._daily_notional_date = current_day
            self._daily_notional_used = 0.0

    def _enforce_policy(self, order: NormalizedOrderRequest, policy: TradingPolicy | None) -> float:
        if policy is None:
            return 0.0
        if not policy.live_enabled:
            raise PermissionError("Live trading is disabled by TradingPolicy.")
        if not policy.market_allowed(order.market_id):
            raise PermissionError(f"Market {order.market_id} is not allowlisted.")

        notional = order.quantity * (order.price if order.price is not None else 1.0)

        if policy.max_notional_per_order is not None and notional > policy.max_notional_per_order:
            raise ValueError(
                f"Order notional {notional:.4f} exceeds per-order cap "
                f"{policy.max_notional_per_order:.4f}."
            )

        if policy.max_daily_notional is not None:
            self._rollover_daily_notional_if_needed()
            projected = self._daily_notional_used + notional
            if projected > policy.max_daily_notional:
                raise ValueError(
                    f"Projected daily notional {projected:.4f} exceeds cap "
                    f"{policy.max_daily_notional:.4f}."
                )

        return notional

    def _record_daily_notional(self, notional: float) -> None:
        if notional <= 0:
            return
        self._rollover_daily_notional_if_needed()
        self._daily_notional_used += notional
