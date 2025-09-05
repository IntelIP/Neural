"""
Neural SDK - Open-source SDK for algorithmic prediction market trading

This SDK provides a comprehensive framework for building algorithmic trading
systems for prediction markets. It includes:

- Real-time data streaming from multiple sources
- Strategy development framework with backtesting
- Risk management and portfolio monitoring
- Multiple data source integrations
- Comprehensive performance analytics

Example:
    ```python
    from neural_sdk import NeuralSDK
    from neural_sdk.backtesting import BacktestEngine, BacktestConfig

    # Initialize SDK
    sdk = NeuralSDK.from_env()

    # Create a trading strategy
    def my_strategy(market_data):
        for symbol, price in market_data.prices.items():
            if price < 0.3:
                return {'action': 'BUY', 'symbol': symbol, 'size': 100}

    # Backtest the strategy
    config = BacktestConfig(start_date="2024-01-01", end_date="2024-12-31")
    engine = BacktestEngine(config)
    engine.add_strategy(my_strategy)
    engine.load_data("csv", path="historical_data.csv")
    results = engine.run()

    # Deploy live (optional)
    await sdk.start_trading()
    ```
"""

__version__ = "1.3.0"
__author__ = "Neural SDK Team"
__email__ = "sdk@neural.dev"

from .core.client import MarketData, NeuralSDK, TradeResult, TradingSignal, Position, Order, Portfolio
from .core.config import SDKConfig
from .core.exceptions import (
    ConfigurationError,
    ConnectionError,
    SDKError,
    TradingError,
    ValidationError,
)

# Import strategies and utilities
from .strategies import BaseStrategy, StrategySignal
from .utils import setup_logging

# Import streaming functionality
from .streaming import NeuralWebSocket, NFLMarketStream, MarketStream

# Convenience imports for common use cases
__all__ = [
    "NeuralSDK",
    "SDKConfig",
    "TradingSignal",
    "MarketData",
    "TradeResult",
    # Portfolio management
    "Position",
    "Order", 
    "Portfolio",
    # Exceptions
    "SDKError",
    "ConfigurationError",
    "ConnectionError",
    "TradingError",
    "ValidationError",
    # Strategies
    "BaseStrategy",
    "StrategySignal",
    # Utils
    "setup_logging",
    # Streaming functionality
    "NeuralWebSocket",
    "NFLMarketStream", 
    "MarketStream",
]
