"""High-level trading utilities for the Neural prediction-market SDK."""

from __future__ import annotations

import importlib

from neural.exchanges.types import (
    ExchangeCapabilities,
    NormalizedMarket,
    NormalizedOrderRequest,
    NormalizedOrderResult,
    NormalizedPosition,
    NormalizedQuote,
    TradingPolicy,
)

_ATTRIBUTE_EXPORTS = {
    "TradingClient": (".client", "TradingClient"),
    "KalshiAdapter": (".kalshi_adapter", "KalshiAdapter"),
    "PolymarketUSAdapter": (".polymarket_us_adapter", "PolymarketUSAdapter"),
    "PolymarketUSMarketWebSocketClient": (
        ".polymarket_us_adapter",
        "PolymarketUSMarketWebSocketClient",
    ),
    "PolymarketUSUserWebSocketClient": (
        ".polymarket_us_adapter",
        "PolymarketUSUserWebSocketClient",
    ),
    "KalshiWebSocketClient": (".websocket", "KalshiWebSocketClient"),
    "KalshiFIXClient": (".fix", "KalshiFIXClient"),
    "FIXConnectionConfig": (".fix", "FIXConnectionConfig"),
    "PaperTradingClient": (".paper_client", "PaperTradingClient"),
    "create_paper_trading_client": (".paper_client", "create_paper_trading_client"),
    "PaperPortfolio": (".paper_portfolio", "PaperPortfolio"),
    "Position": (".paper_portfolio", "Position"),
    "Trade": (".paper_portfolio", "Trade"),
    "PaperTradingReporter": (".paper_report", "PaperTradingReporter"),
    "create_report": (".paper_report", "create_report"),
}


def __getattr__(name: str) -> object:
    target = _ATTRIBUTE_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = target
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_ATTRIBUTE_EXPORTS))


__all__ = [
    "TradingClient",
    "KalshiAdapter",
    "PolymarketUSAdapter",
    "PolymarketUSMarketWebSocketClient",
    "PolymarketUSUserWebSocketClient",
    "KalshiWebSocketClient",
    "KalshiFIXClient",
    "FIXConnectionConfig",
    "PaperTradingClient",
    "create_paper_trading_client",
    "PaperPortfolio",
    "Position",
    "Trade",
    "PaperTradingReporter",
    "create_report",
    "TradingPolicy",
    "ExchangeCapabilities",
    "NormalizedMarket",
    "NormalizedQuote",
    "NormalizedOrderRequest",
    "NormalizedOrderResult",
    "NormalizedPosition",
]
