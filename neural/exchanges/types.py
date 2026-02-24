from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ExchangeName = Literal["kalshi", "polymarket_us"]
OrderSide = Literal["buy_yes", "buy_no", "sell_yes", "sell_no"]
OrderType = Literal["market", "limit"]


@dataclass(slots=True)
class NormalizedMarket:
    market_id: str
    ticker: str
    title: str
    status: str = "open"
    yes_price: float | None = None
    no_price: float | None = None
    category: str | None = None
    sport: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class NormalizedQuote:
    market_id: str
    yes_bid: float | None = None
    yes_ask: float | None = None
    no_bid: float | None = None
    no_ask: float | None = None
    last_price: float | None = None
    volume: float | None = None
    timestamp: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class NormalizedOrderRequest:
    market_id: str
    side: OrderSide
    quantity: int
    order_type: OrderType = "limit"
    price: float | None = None
    idempotency_key: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class NormalizedOrderResult:
    success: bool
    status: str
    order_id: str | None = None
    message: str | None = None
    filled_price: float | None = None
    filled_quantity: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class NormalizedPosition:
    market_id: str
    side: Literal["yes", "no"]
    quantity: int
    entry_price: float | None = None
    current_price: float | None = None
    unrealized_pnl: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TradingPolicy:
    live_enabled: bool = False
    allowed_markets: list[str] | None = None
    max_notional_per_order: float | None = None
    max_daily_notional: float | None = None

    def market_allowed(self, market_id: str) -> bool:
        if not self.allowed_markets:
            return True
        return market_id in self.allowed_markets


@dataclass(slots=True)
class ExchangeCapabilities:
    read: bool
    paper: bool
    live: bool
    streaming: bool

    def as_dict(self) -> dict[str, bool]:
        return {
            "read": self.read,
            "paper": self.paper,
            "live": self.live,
            "streaming": self.streaming,
        }
