# Neural SDK

[![PyPI version](https://badge.fury.io/py/neural-sdk.svg)](https://badge.fury.io/py/neural-sdk)
[![Python Versions](https://img.shields.io/pypi/pyversions/neural-sdk.svg)](https://pypi.org/project/neural-sdk/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Professional-grade SDK for algorithmic trading on prediction markets.

[Documentation](https://neural-sdk.mintlify.app) • [Examples](./examples) • [Contributing](./CONTRIBUTING.md)

## Overview

Neural SDK is a Python framework for building algorithmic trading strategies on prediction markets. It provides data collection, strategy development, backtesting, and trade execution with production-grade reliability.

All market data comes from Kalshi's live production API via RSA-authenticated requests, using the same infrastructure that powers their trading platform.

## Features

- **Authentication**: RSA signature implementation for Kalshi API
- **Historical Data**: Collect and analyze real trade data with cursor-based pagination
- **Real-time Streaming**: REST API and FIX protocol support for live market data
- **Strategy Framework**: Pre-built strategies (mean reversion, momentum, arbitrage)
- **Risk Management**: Kelly Criterion, position sizing, stop-loss automation
- **Backtesting Engine**: Test strategies on historical data before going live
- **Order Execution**: Ultra-low latency FIX protocol integration (5-10ms)

## Quick Start

### Installation

```bash
pip install neural-sdk
pip install "neural-sdk[trading]"  # with trading extras
```

### Credentials Setup

Create a `.env` file with your Kalshi credentials:

```bash
KALSHI_API_KEY_ID=your_api_key_id
KALSHI_PRIVATE_KEY_BASE64=base64_encoded_private_key
KALSHI_ENV=prod
```

The SDK automatically loads credentials from the `.env` file.

## Usage

### Authentication

```python
from neural.auth.http_client import KalshiHTTPClient

client = KalshiHTTPClient()
markets = client.get('/markets')
print(f"Connected! Found {len(markets['markets'])} markets")
```

### Historical Data Collection

```python
from neural.data_collection.kalshi_historical import KalshiHistoricalDataSource
from neural.data_collection.base import DataSourceConfig

config = DataSourceConfig(
    source_type="kalshi_historical",
    ticker="NFLSUP-25-KCSF",
    start_time="2024-01-01",
    end_time="2024-12-31"
)

source = KalshiHistoricalDataSource(config)
trades_data = []

async def collect_trades():
    async for trade in source.collect():
        trades_data.append(trade)
        if len(trades_data) >= 1000:
            break

import asyncio
asyncio.run(collect_trades())
print(f"Collected {len(trades_data)} trades")
```

### Strategy Development

```python
from neural.analysis.strategies import MeanReversionStrategy
from neural.analysis.backtesting import BacktestEngine

strategy = MeanReversionStrategy(lookback_period=20, z_score_threshold=2.0)
engine = BacktestEngine(strategy, initial_capital=10000)
results = engine.run(historical_data)

print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
```

### Trading

```python
from neural.trading.client import TradingClient

trader = TradingClient()
order = trader.place_order(
    ticker="NFLSUP-25-KCSF",
    side="yes",
    count=10,
    price=52
)
print(f"Order placed: {order['order_id']}")
```

## Modules

| Module | Description |
|--------|-------------|
| `neural.auth` | RSA authentication for Kalshi API |
| `neural.data_collection` | Historical and real-time market data |
| `neural.analysis.strategies` | Pre-built trading strategies |
| `neural.analysis.backtesting` | Strategy testing framework |
| `neural.analysis.risk` | Position sizing and risk management |
| `neural.trading` | Order execution (REST + FIX) |

## Examples

See the [`examples/`](./examples) directory for working code samples:

- `01_init_user.py` - Authentication setup
- `stream_prices.py` - Real-time price streaming
- `test_historical_sync.py` - Historical data collection
- `05_mean_reversion_strategy.py` - Strategy implementation
- `07_live_trading_bot.py` - Automated trading bot

## Testing

```bash
pytest
pytest --cov=neural tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make changes and add tests
4. Run tests: `pytest`
5. Commit: `git commit -m "Add amazing feature"`
6. Push: `git push origin feature/amazing-feature`
7. Open a Pull Request

See [CONTRIBUTING.md](./CONTRIBUTING.md) for detailed guidelines.

## Development Setup

```bash
git clone https://github.com/IntelIP/Neural.git
cd neural
pip install -e ".[dev]"
pytest
ruff check .
black --check .
```

## Resources

- **Documentation**: [neural-sdk.mintlify.app](https://neural-sdk.mintlify.app)
- **Examples**: [examples/](./examples)
- **Issues**: [GitHub Issues](https://github.com/IntelIP/Neural/issues)
- **Discussions**: [GitHub Discussions](https://github.com/IntelIP/Neural/discussions)

## License

This project is licensed under the MIT License - see [LICENSE](./LICENSE) file for details.