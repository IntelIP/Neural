# Neural SDK

[![PyPI version](https://badge.fury.io/py/neural-sdk.svg)](https://badge.fury.io/py/neural-sdk)
[![Python Versions](https://img.shields.io/pypi/pyversions/neural-sdk.svg)](https://pypi.org/project/neural-sdk/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Kalshi-first open-core SDK for algorithmic trading on prediction markets.

[Documentation](https://neural-sdk.mintlify.app) • [Examples](./examples) • [Contributing](./CONTRIBUTING.md)

## Overview

Neural SDK is a Python framework for building prediction-market workflows. The supported public beta is intentionally narrow: Kalshi auth, market data, paper trading, selected live trading flows via `TradingClient`, and the CLI.

Neural also includes read/stream-only Polymarket US support, plus experimental research, FIX, sentiment, and deployment modules that are still being hardened.

## Beta Scope

The current public beta is intentionally narrow so the supported path is clear.

The base install is enough for importing the SDK, using the CLI, and working with the supported beta surface. Add extras only when you need them:

| Extra | Installs | Use when |
|-------|----------|----------|
| `trading` | `kalshi-python`, `websockets`, `simplefix` | You need Kalshi live trading, market-data streaming, or FIX-adjacent helpers |
| `sentiment` | `textblob`, `vaderSentiment`, `aiohttp`, `transformers`, `torch`, `scikit-learn` | You want the research-only sentiment stack |
| `deployment` | `pydantic`, `jinja2`, `docker`, `sqlalchemy`, `psycopg2-binary`, `fastapi`, `uvicorn` | You are experimenting with the Docker deployment helpers |

## Module Status

| Surface | Status | Notes |
|--------|--------|-------|
| `neural.auth` | Supported beta | Kalshi credential loading, signing, and HTTP auth helpers |
| `neural.data_collection` Kalshi sources | Supported beta | Live market reads, historical trade collection, REST helpers |
| `neural.trading.TradingClient` with `exchange="kalshi"` | Supported beta | Paper trading and selected live trading flows |
| `neural.cli` | Supported beta | `--version`, `doctor`, and `doctor --json` |
| `TradingClient(exchange="polymarket_us")` read + streaming | Read-only beta | Market reads, quotes, replay, positions, streaming clients |
| Polymarket US live order placement | Not yet supported | `place_order`, `cancel_order`, and `get_order_status` are not part of the supported beta |
| `neural.analysis.backtesting` and strategy helpers | Experimental | Useful for research, not part of the stable beta contract |
| Sentiment, ESPN/Twitter aggregation, FIX streaming, deployment | Experimental | Keep behind operator review before using in production |

## Features

- **Kalshi auth + market data**: RSA signing, REST helpers, historical trade collection
- **Paper trading**: Rehearse order flow and portfolio changes without risking capital
- **Live trading facade**: `TradingClient` with a stable Kalshi-first interface
- **Minimal operator CLI**: version and environment readiness checks
- **Experimental research toolkit**: backtesting, strategy scaffolds, and risk helpers
- **Read-only Polymarket US support**: read and streaming workflows where implemented

## Quick Start

### Installation

```bash
pip install neural-sdk
pip install "neural-sdk[trading]"     # Kalshi live trading, streaming, and FIX helpers
pip install "neural-sdk[sentiment]"    # research-only sentiment tooling
pip install "neural-sdk[deployment]"   # optional experimental deployment extras
```

If you are only reading docs, running the CLI, or importing the supported beta surface, the base install is sufficient. Add `trading` when you need Kalshi execution or streaming, `sentiment` when you are doing research work, and `deployment` only if you are intentionally testing the experimental deployment helpers.

### CLI

The package includes a minimal supported CLI:

```bash
neural --version
neural doctor
neural doctor --json
```

`neural doctor` reports local environment details, optional dependency availability, and whether credential inputs are present for Kalshi and Polymarket US.

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
from neural.analysis import Backtester

strategy = MeanReversionStrategy(lookback_period=20, z_score_threshold=2.0)
engine = Backtester(initial_capital=10000)
results = engine.backtest(strategy, start_date="2024-01-01", end_date="2024-12-31")

print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
```

### Trading

```python
from neural.trading import TradingClient

trader = TradingClient(exchange="kalshi")
order = trader.place_order(
    market_id="KXNFLGAME-25SEP25SEAARI-SEA",
    side="yes",
    quantity=10,
    order_type="limit",
    price=55,
)
print(order)
```

## Modules

| Module | Description | Status |
|--------|-------------|--------|
| `neural.auth` | Kalshi and Polymarket US credential helpers | Supported beta |
| `neural.data_collection` | Historical and real-time market data | Supported beta |
| `neural.analysis.strategies` | Strategy building blocks | Experimental |
| `neural.analysis.backtesting` | Strategy testing framework | Experimental |
| `neural.analysis.risk` | Position sizing and risk management | Experimental |
| `neural.trading` | Paper trading plus Kalshi-first live execution | Supported beta |
| `neural.deployment` | Docker-based deployment helpers | Experimental and not part of the supported beta contract |

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
cd Neural
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
