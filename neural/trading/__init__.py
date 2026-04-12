"""High-level trading utilities for the Neural prediction-market SDK."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from neural.exchanges.types import (
    ExchangeCapabilities,
    NormalizedMarket,
    NormalizedOrderRequest,
    NormalizedOrderResult,
    NormalizedPosition,
    NormalizedQuote,
    TradingPolicy,
)

_EXPORTS = {
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
    "PaperTradingClient": (".paper_client", "PaperTradingClient"),
    "create_paper_trading_client": (".paper_client", "create_paper_trading_client"),
    "PaperPortfolio": (".paper_portfolio", "PaperPortfolio"),
    "Position": (".paper_portfolio", "Position"),
    "Trade": (".paper_portfolio", "Trade"),
    "PaperTradingReporter": (".paper_report", "PaperTradingReporter"),
    "create_report": (".paper_report", "create_report"),
}


def _placeholder(name: str, message: str, exc: ImportError) -> type[Any]:
    class _MissingDependency:  # type: ignore[too-many-ancestors]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(message) from exc

    _MissingDependency.__name__ = name
    return _MissingDependency


def _load_fix_export(name: str) -> Any:
    try:
        module = import_module(".fix", __name__)
        value = getattr(module, name)
    except ImportError as exc:
        if "simplefix" not in str(exc):
            raise
        value = _placeholder(
            name,
            "Kalshi FIX support requires optional trading extras. "
            "Install with: pip install 'neural-sdk[trading]'",
            exc,
        )
    globals()[name] = value
    return value


def _load_websocket_export(name: str) -> Any:
    try:
        module = import_module(".websocket", __name__)
        value = getattr(module, name)
    except ImportError as exc:
        if not any(dep in str(exc) for dep in ("websockets", "certifi")):
            raise
        value = _placeholder(
            name,
            "Kalshi WebSocket support requires optional trading extras. "
            "Install with: pip install 'neural-sdk[trading]'",
            exc,
        )
    globals()[name] = value
    return value


def __getattr__(name: str) -> Any:
    if name in {"FIXConnectionConfig", "KalshiFIXClient"}:
        return _load_fix_export(name)

    if name == "KalshiWebSocketClient":
        return _load_websocket_export(name)

    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        value = getattr(import_module(module_name, __name__), attr_name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(
        set(globals())
        | set(_EXPORTS)
        | {"FIXConnectionConfig", "KalshiFIXClient", "KalshiWebSocketClient"}
    )


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
