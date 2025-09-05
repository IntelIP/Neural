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
- Advanced portfolio optimization
- Kelly Criterion allocation
- Risk parity and concentration constraints
- Monte Carlo simulation
- Multi-asset optimization

Example - Basic Backtesting:
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

Example - Portfolio Optimization:
    ```python
    from neural_sdk.backtesting import PortfolioOptimizer, OptimizationConfig, Asset

    config = OptimizationConfig(
        total_budget=1000,
        max_concentration=0.25,
        strategies=['kelly', 'equal_weight', 'risk_parity']
    )

    optimizer = PortfolioOptimizer(config)
    
    # Add assets
    assets = [
        Asset("GAME1", "Team A vs Team B", 0.88, 0.12, 0.85),
        Asset("GAME2", "Team C vs Team D", 0.61, 0.39, 0.65),
    ]
    optimizer.add_assets(assets)
    
    # Optimize portfolio
    results = optimizer.optimize()
    best_allocation = results.best_allocation
    ```
"""

from .data_loader import DataLoader
from .engine import BacktestConfig, BacktestEngine, BacktestResults
from .metrics import PerformanceMetrics
from .portfolio import Portfolio, Position, Trade
from .portfolio_optimization import (
    PortfolioOptimizer,
    OptimizationConfig,
    OptimizationResults,
    AllocationResult,
    Asset,
    AllocationStrategy
)
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
    # Portfolio Optimization
    "PortfolioOptimizer",
    "OptimizationConfig",
    "OptimizationResults", 
    "AllocationResult",
    "Asset",
    "AllocationStrategy",
]
