# Backtesting Guide

Comprehensive guide to backtesting trading strategies with the Neural SDK.

## Overview

The backtesting module provides:
- Event-driven simulation engine
- Multiple data source support  
- Realistic portfolio simulation
- Performance analytics and reporting

## Basic Usage

```python
from neural_sdk.backtesting import BacktestEngine, BacktestConfig

config = BacktestConfig(
    start_date="2024-01-01",
    end_date="2024-12-31",
    initial_capital=10000
)

engine = BacktestEngine(config)
engine.add_strategy(my_strategy)
engine.load_data("csv", path="historical_data.csv")
results = engine.run()
```

## Data Sources

- CSV files
- Parquet files 
- S3 buckets (with `pip install neural-sdk[s3]`)
- SQL databases (with `pip install neural-sdk[database]`)

## Performance Metrics

The engine calculates comprehensive metrics including:
- Sharpe ratio, Sortino ratio
- Maximum drawdown
- Win rate, profit factor
- Value at Risk (VaR)

## Examples

See `examples/backtest_strategy.py` for detailed examples.