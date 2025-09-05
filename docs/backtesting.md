# Backtesting Guide

Comprehensive guide to backtesting trading strategies with the Neural SDK.

## Overview

The backtesting module provides:
- Event-driven simulation engine
- Multiple data source support  
- Realistic portfolio simulation
- Performance analytics and reporting
- **Advanced portfolio optimization**
- **Kelly Criterion allocation**
- **Risk management and concentration limits**
- **Monte Carlo simulation**
- **Multi-asset optimization strategies**

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

---

# Portfolio Optimization

Advanced portfolio optimization for prediction market trading using Kelly Criterion, risk management, and multi-asset strategies.

## Quick Start

```python
from neural_sdk.backtesting import PortfolioOptimizer, OptimizationConfig, Asset

# Configure optimization
config = OptimizationConfig(
    total_budget=1000,
    max_concentration=0.25,  # Max 25% per asset
    strategies=['kelly', 'equal_weight', 'risk_parity'],
    monte_carlo_runs=10000
)

# Create optimizer
optimizer = PortfolioOptimizer(config)

# Add prediction market assets
assets = [
    Asset("GAME1", "Chiefs vs Chargers", 0.61, 0.39, 0.65),
    Asset("GAME2", "Northwestern vs Western Illinois", 0.99, 0.01, 0.98),
    Asset("GAME3", "Maryland vs Northern Illinois", 0.88, 0.12, 0.85),
]
optimizer.add_assets(assets)

# Optimize portfolio
results = optimizer.optimize()
best_allocation = results.best_allocation

print(f"Best strategy: {best_allocation.strategy}")
print(f"Expected return: ${best_allocation.expected_return:.2f}")
print(f"Sharpe ratio: {best_allocation.sharpe_ratio:.4f}")
print(f"Allocation: {best_allocation.allocation}")
```

## Asset Definition

Define tradeable assets with market pricing and probability estimates:

```python
asset = Asset(
    symbol="GAME1",                    # Unique identifier
    name="Kansas City vs Los Angeles", # Display name
    favorite_price=0.61,              # Price to bet on favorite ($0.61 for $1)
    underdog_price=0.39,              # Price to bet on underdog ($0.39 for $1)
    implied_probability=0.65,         # Your probability estimate (65%)
    volume=853104                     # Trading volume (optional)
)
```

## Allocation Strategies

### Kelly Criterion
Mathematically optimal bet sizing for maximizing long-term growth:

```python
config = OptimizationConfig(
    strategies=['kelly'],
    total_budget=1000
)
```

**Pros:** Maximizes expected logarithmic growth  
**Cons:** Can be aggressive, high concentration risk

### Equal Weight
Distributes capital equally across selected assets:

```python
config = OptimizationConfig(
    strategies=['equal_weight'],
    total_budget=1000
)
```

**Pros:** Simple, well-diversified  
**Cons:** Ignores asset quality differences

### Risk Parity
Allocates capital based on inverse volatility (equal risk contribution):

```python
config = OptimizationConfig(
    strategies=['risk_parity'],
    total_budget=1000
)
```

**Pros:** Balances risk across assets  
**Cons:** May underweight high-return opportunities

### Constrained Kelly
Kelly Criterion with concentration limits:

```python
config = OptimizationConfig(
    strategies=['constrained_kelly'],
    max_concentration=0.25,  # Max 25% per asset
    total_budget=1000
)
```

**Pros:** Optimal growth with risk management  
**Cons:** May sacrifice some returns for diversification

## Risk Management

### Concentration Limits
Prevent over-concentration in single assets:

```python
config = OptimizationConfig(
    max_concentration=0.30,  # Maximum 30% per asset
    min_assets=3,           # Minimum 3 assets required
    max_assets=5            # Maximum 5 assets
)
```

### Performance Filters
Filter allocations by performance criteria:

```python
config = OptimizationConfig(
    min_win_probability=0.40,  # Minimum 40% win rate
    min_sharpe_ratio=0.0,      # Minimum Sharpe ratio
    max_drawdown_tolerance=0.20 # Maximum 20% drawdown
)
```

## Asset Selection

### Best N Assets
Automatically select the best assets by value score:

```python
# Add 10 assets, optimize with best 3
config = OptimizationConfig(
    min_assets=3,
    max_assets=3
)

optimizer.add_assets(many_assets)
results = optimizer.optimize()  # Uses top 3 by Kelly * Expected Return
```

### Manual Selection
Compare strategies on specific assets:

```python
asset_symbols = ["GAME1", "GAME3", "GAME5"]
comparison_df = optimizer.compare_strategies(asset_symbols)
print(comparison_df)
```

## Performance Analysis

### Results Structure
```python
results = optimizer.optimize()

# Best allocation by Sharpe ratio
best = results.best_allocation
print(f"Strategy: {best.strategy}")
print(f"Expected Return: ${best.expected_return:.2f}")
print(f"Standard Deviation: ${best.std_return:.2f}")  
print(f"Sharpe Ratio: {best.sharpe_ratio:.4f}")
print(f"Win Probability: {best.win_probability:.1%}")
print(f"Max Concentration: {best.max_concentration:.1%}")
print(f"VaR (95%): ${best.var_95:.2f}")

# Allocation breakdown
for asset, amount in best.allocation.items():
    if amount > 0:
        pct = amount / best.total_invested * 100
        print(f"  {asset}: ${amount:.2f} ({pct:.1f}%)")
```

### Alternative Views
```python
# Highest expected return (regardless of risk)
highest_return = results.highest_return

# Most diversified portfolio
most_diversified = results.most_diversified

# All results sorted by Sharpe ratio
for result in results.results[:5]:  # Top 5
    print(f"{result.strategy}: Sharpe {result.sharpe_ratio:.4f}")
```

## Advanced Features

### Monte Carlo Analysis
Access full Monte Carlo simulation results:

```python
best = results.best_allocation
mc_results = best.monte_carlo_results  # numpy array of 10,000 outcomes

# Custom analysis
import numpy as np
percentiles = np.percentile(mc_results, [5, 25, 50, 75, 95])
print(f"Outcome percentiles: {percentiles}")
```

### Portfolio Metrics
Comprehensive risk and diversification metrics:

```python
result = results.best_allocation

# Concentration analysis
print(f"Max Concentration: {result.max_concentration:.1%}")
print(f"Herfindahl Index: {result.herfindahl_index:.3f}")
print(f"Diversification Ratio: {result.diversification_ratio:.2f}")

# Risk analysis  
print(f"Expected Return: ${result.expected_return:+.2f}")
print(f"Standard Deviation: ${result.std_return:.2f}")
print(f"Value at Risk (95%): ${result.var_95:.2f}")
print(f"Risk-Adjusted Return: {result.risk_adjusted_return:.4f}")
```

## Real-World Example

Here's a complete example using September 5th college football and NFL games:

```python
from neural_sdk.backtesting import PortfolioOptimizer, OptimizationConfig, Asset

# Define the games available for betting
games = [
    Asset("BOISE", "Eastern Washington at Boise State", 0.98, 0.02, 0.98),
    Asset("NW", "Western Illinois at Northwestern", 0.99, 0.01, 0.99), 
    Asset("MD", "Northern Illinois at Maryland", 0.88, 0.12, 0.85),
    Asset("LOU", "James Madison at Louisville", 0.86, 0.14, 0.86),
    Asset("KC", "Chiefs at Chargers", 0.61, 0.39, 0.65)
]

# Configure for $50 investment with diversification
config = OptimizationConfig(
    total_budget=50,
    max_concentration=0.35,  # Max 35% per game
    strategies=['kelly', 'equal_weight', 'constrained_kelly'],
    min_assets=3,           # At least 3 games
    monte_carlo_runs=10000
)

# Optimize
optimizer = PortfolioOptimizer(config)
optimizer.add_assets(games)
results = optimizer.optimize()

# Results
best = results.best_allocation
print(f"\n🏆 OPTIMAL $50 ALLOCATION:")
print(f"Strategy: {best.strategy}")
print(f"Expected Return: ${best.expected_return:+.2f} ({best.expected_return/50*100:+.1f}%)")
print(f"Win Probability: {best.win_probability:.1%}")
print(f"Sharpe Ratio: {best.sharpe_ratio:.4f}")

print(f"\n💰 ALLOCATION:")
for asset, amount in best.allocation.items():
    if amount > 0:
        pct = amount / 50 * 100
        print(f"  {asset}: ${amount:.2f} ({pct:.1f}%)")
```

Key insight from our analysis: **Focusing on 3 profitable games (avoiding high-price, low-return favorites) delivered 42% better returns than diversifying across all 5 games.**

## Best Practices

### 1. Asset Quality Matters
- Avoid games with prices >$0.95 (minimal profit potential)  
- Focus on games with reasonable odds and good expected value
- Consider trading volume for liquidity

### 2. Diversification vs Concentration
- Use `max_concentration` to limit single-asset risk
- Consider 3-5 assets for good balance
- More assets ≠ always better (can dilute returns)

### 3. Strategy Selection
- **Kelly Criterion**: Best for long-term growth, can be aggressive
- **Constrained Kelly**: Good balance of growth and risk management  
- **Equal Weight**: Simple, well-diversified baseline
- **Risk Parity**: Conservative, good for risk-averse investors

### 4. Risk Management
- Set realistic `max_concentration` limits (20-35%)
- Use `min_win_probability` filters to avoid poor strategies
- Monitor VaR for downside risk assessment

### 5. Backtesting Integration
Combine with historical backtesting for validation:

```python
# Optimize current portfolio
optimizer_results = optimizer.optimize()
best_allocation = optimizer_results.best_allocation

# Validate with historical backtest
backtest_config = BacktestConfig(
    start_date="2024-01-01", 
    end_date="2024-12-31",
    initial_capital=best_allocation.total_invested
)

# Apply allocation to historical data
# ... backtest implementation
```

## Examples

See `examples/backtest_strategy.py` for detailed examples.