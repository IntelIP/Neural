"""
Neural SDK - Backtesting Module

Comprehensive backtesting framework for prediction market trading strategies.

Features:
- Event-driven backtesting engine
- Multiple data source support
- Portfolio simulation
- Performance analytics
- Risk metrics
- Strategy comparison tools

Example:
    ```python
    from neural_sdk.backtesting import BacktestEngine, BacktestConfig

    config = BacktestConfig(
        start_date="2024-01-01",
        end_date="2024-12-31",
        initial_capital=10000
    )

    engine = BacktestEngine(config)
    engine.add_strategy(my_strategy)
    engine.load_data("file", path="data/nfl_2024.parquet")

    results = engine.run()
    metrics = engine.analyze()
    ```
"""

from .data_loader import DataLoader
from .engine import BacktestConfig, BacktestEngine, BacktestResults
from .metrics import PerformanceMetrics
from .portfolio import Portfolio, Position, Trade
from .providers.base import DataProvider

__all__ = [
    "BacktestEngine",
    "BacktestConfig",
    "BacktestResults",
    "Portfolio",
    "Position",
    "Trade",
    "PerformanceMetrics",
    "DataLoader",
    "DataProvider",
]
