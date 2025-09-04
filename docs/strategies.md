# Strategy Development Guide

Learn how to build effective trading strategies with the Neural SDK.

## Strategy Basics

A strategy is a function that receives market data and returns trading signals:

```python
def my_strategy(market_data):
    prices = market_data['prices']
    portfolio = market_data['portfolio']
    
    for symbol, price in prices.items():
        if price < 0.3:  # Undervalued
            return {
                'action': 'BUY',
                'symbol': symbol, 
                'size': 100
            }
    
    return None  # No action
```

## Strategy Patterns

### Momentum Strategy
Buy when price is trending up:

```python
def momentum_strategy(market_data):
    # Implementation in examples/backtest_strategy.py
```

### Mean Reversion  
Buy undervalued markets:

```python
def mean_reversion_strategy(market_data):
    # Implementation in examples/backtest_strategy.py
```

### Multi-Signal Strategy
Combine multiple indicators:

```python
def multi_signal_strategy(market_data):
    # Access additional data fields
    data_points = market_data.get('data', [])
    
    for point in data_points:
        volume = point.get('volume', 0)
        price = point.get('price', 0)
        
        # Use volume + price signals
        if volume > 100 and price < 0.4:
            return signal
```

## Risk Management

### Position Sizing
Use Kelly Criterion with safety factors:

```python
# Built into BacktestEngine
config = BacktestConfig(
    position_size_limit=0.05,  # Max 5% per position
    daily_loss_limit=0.20      # Stop at 20% daily loss
)
```

### Stop Loss
```python
def strategy_with_stops(market_data):
    portfolio = market_data['portfolio']
    
    # Check existing positions for stop loss
    for symbol, position in portfolio.positions.items():
        if position.unrealized_pnl_pct < -10:  # 10% stop loss
            return {'action': 'SELL', 'symbol': symbol, 'size': position.quantity}
```

## Testing Strategies

1. **Backtest** with historical data
2. **Analyze** performance metrics
3. **Optimize** parameters  
4. **Paper trade** before going live

See `examples/backtest_strategy.py` for complete examples.