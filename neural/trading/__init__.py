"""High-level trading utilities for the Neural prediction-market SDK."""

from neural.exchanges.types import (
    ExchangeCapabilities,
    NormalizedMarket,
    NormalizedOrderRequest,
    NormalizedOrderResult,
    NormalizedPosition,
    NormalizedQuote,
    TradingPolicy,
)

from .client import TradingClient
from .fix import FIXConnectionConfig, KalshiFIXClient
from .kalshi_adapter import KalshiAdapter
from .paper_client import PaperTradingClient, create_paper_trading_client
from .paper_portfolio import PaperPortfolio, Position, Trade
from .paper_report import PaperTradingReporter, create_report
from .polymarket_us_adapter import (
    PolymarketUSAdapter,
    PolymarketUSMarketWebSocketClient,
    PolymarketUSUserWebSocketClient,
)
from .websocket import KalshiWebSocketClient

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
