# Neural SDK

<div align="center">

[![PyPI version](https://badge.fury.io/py/neural-sdk.svg)](https://badge.fury.io/py/neural-sdk)
[![Python Versions](https://img.shields.io/pypi/pyversions/neural-sdk.svg)](https://pypi.org/project/neural-sdk/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Stars](https://img.shields.io/github/stars/IntelIP/Neural)](https://github.com/IntelIP/Neural)

**Professional-grade SDK for algorithmic trading on prediction markets**

[Documentation](https://neural-sdk.mintlify.app) • [Quick Start](#quick-start) • [Examples](./examples) • [Contributing](./CONTRIBUTING.md)

</div>

---

## ⚡ What is Neural?

Neural SDK is a comprehensive Python framework for building algorithmic trading strategies on prediction markets. It provides everything you need to collect data, develop strategies, backtest performance, and execute trades—all with production-grade reliability.

### 🔐 Real Data Guarantee

All market data comes from **Kalshi's live production API** via RSA-authenticated requests. This is the same infrastructure that powers a $100M+ trading platform—no simulations, no mocks, just real markets on real events.

### ⭐ Key Features

- **🔑 Authentication**: Battle-tested RSA signature implementation for Kalshi API
- **📊 Historical Data**: Collect and analyze real trade data with cursor-based pagination
- **🚀 Real-time Streaming**: REST API and FIX protocol support for live market data
- **🧠 Strategy Framework**: Pre-built strategies (mean reversion, momentum, arbitrage)
- **⚖️ Risk Management**: Kelly Criterion, position sizing, stop-loss automation
- **🔬 Backtesting Engine**: Test strategies on historical data before going live
- **⚡ Order Execution**: Ultra-low latency FIX protocol integration (5-10ms)

---

## 🚀 Quick Start

### Installation

```bash
# Basic installation
pip install neural-sdk

# With trading extras (recommended for live trading)
pip install "neural-sdk[trading]"

# Via uv (recommended)
uv pip install neural-sdk
uv pip install "neural-sdk[trading]"  # with trading extras
```

### Credentials Setup

Neural SDK connects to Kalshi's live API using RSA authentication. You'll need valid Kalshi credentials:

#### Environment Variables

```bash
# Option 1: Set environment variables
export KALSHI_EMAIL="your-email@example.com"
export KALSHI_PASSWORD="your-password"
export KALSHI_API_BASE="https://trading-api.kalshi.com/trade-api/v2"
```

#### .env File (Recommended)

```bash
# Option 2: Create .env file in your project root
echo "KALSHI_EMAIL=your-email@example.com" > .env
echo "KALSHI_PASSWORD=your-password" >> .env
echo "KALSHI_API_BASE=https://trading-api.kalshi.com/trade-api/v2" >> .env
```

The SDK will automatically load credentials from your .env file using python-dotenv.

### Basic Usage

#### 1. Authentication

```python
from neural.auth.http_client import KalshiHTTPClient

# Initialize with credentials
client = KalshiHTTPClient()

# Verify connection
markets = client.get('/markets')
print(f"Connected! Found {len(markets['markets'])} markets")
```

#### 2. Collect Historical Data

```python
from datetime import datetime, timedelta
import pandas as pd

# Set time range
end_ts = int(datetime.now().timestamp())
start_ts = end_ts - (7 * 24 * 3600)  # Last 7 days

# Collect trades with pagination
all_trades = []
cursor = None

while True:
    response = client.get_trades(
        ticker="KXNFLGAME-25SEP25SEAARI-SEA",
        min_ts=start_ts,
        max_ts=end_ts,
        limit=1000,
        cursor=cursor
    )

    trades = response.get("trades", [])
    if not trades:
        break

    all_trades.extend(trades)
    cursor = response.get("cursor")
    if not cursor:
        break

# Analyze
df = pd.DataFrame(all_trades)
print(f"Collected {len(df)} real trades from Kalshi")
```

#### 3. Build a Trading Strategy

```python
from neural.analysis.strategies import MeanReversionStrategy
from neural.analysis.backtesting import BacktestEngine

# Create strategy
strategy = MeanReversionStrategy(
    lookback_period=20,
    z_score_threshold=2.0
)

# Backtest
engine = BacktestEngine(strategy, initial_capital=10000)
results = engine.run(historical_data)

print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
```

#### 4. Live Trading

```python
from neural.trading.client import TradingClient

# Initialize trading client
trader = TradingClient()

# Place order
order = trader.place_order(
    ticker="KXNFLGAME-25SEP25SEAARI-SEA",
    side="yes",
    count=100,
    price=55
)

print(f"Order placed: {order['order_id']}")
```

---

## 📚 Documentation

### Core Modules

| Module | Description |
|--------|-------------|
| `neural.auth` | RSA authentication for Kalshi API |
| `neural.data_collection` | Historical and real-time market data |
| `neural.analysis.strategies` | Pre-built trading strategies |
| `neural.analysis.backtesting` | Strategy testing framework |
| `neural.analysis.risk` | Position sizing and risk management |
| `neural.trading` | Order execution (REST + FIX) |

### SDK Module Quickstart

#### Authentication Module

```python
from neural.auth.http_client import KalshiHTTPClient

# Initialize client with credentials from environment
client = KalshiHTTPClient()

# Test connection
response = client.get('/markets')
print(f"Connected! Found {len(response['markets'])} markets")

# Get specific market
market = client.get('/markets/NFLSUP-25-KCSF')
print(f"Market: {market['title']}")
```

#### Data Collection Module

```python
from neural.data_collection.kalshi_historical import KalshiHistoricalDataSource
from neural.data_collection.base import DataSourceConfig
import pandas as pd

# Configure historical data collection
config = DataSourceConfig(
    source_type="kalshi_historical",
    ticker="NFLSUP-25-KCSF",
    start_time="2024-01-01",
    end_time="2024-12-31"
)

# Collect historical trades
source = KalshiHistoricalDataSource(config)
trades_data = []

async def collect_trades():
    async for trade in source.collect():
        trades_data.append(trade)
        if len(trades_data) >= 1000:  # Limit for example
            break

# Run collection and analyze
import asyncio
asyncio.run(collect_trades())

df = pd.DataFrame(trades_data)
print(f"Collected {len(df)} trades")
print(f"Price range: {df['price'].min():.2f} - {df['price'].max():.2f}")
```

#### Trading Module

```python
from neural.trading.client import TradingClient

# Initialize trading client
trader = TradingClient()

# Check account balance
balance = trader.get_balance()
print(f"Available balance: ${balance:.2f}")

# Place a buy order
order = trader.place_order(
    ticker="NFLSUP-25-KCSF",
    side="yes",       # or "no"
    count=10,         # number of contracts
    price=52          # price in cents
)

print(f"Order placed: {order['order_id']}")

# Check order status
status = trader.get_order(order['order_id'])
print(f"Order status: {status['status']}")
```

### Examples

Explore working examples in the [`examples/`](./examples) directory:

- `01_init_user.py` - Authentication setup
- `stream_prices.py` - Real-time price streaming
- `test_historical_sync.py` - Historical data collection
- `05_mean_reversion_strategy.py` - Strategy implementation
- `07_live_trading_bot.py` - Automated trading bot

### Authentication Setup

1. Get API credentials from [Kalshi](https://kalshi.com)
2. Save credentials:
   ```bash
   # Create secrets directory
   mkdir secrets

   # Add your API key ID
   echo "your-api-key-id" > secrets/kalshi_api_key_id.txt

   # Add your private key
   cp ~/Downloads/kalshi_private_key.pem secrets/
   chmod 600 secrets/kalshi_private_key.pem
   ```

3. Set environment variables (optional):
   ```bash
   export KALSHI_API_KEY_ID="your-api-key-id"
   export KALSHI_PRIVATE_KEY_PATH="./secrets/kalshi_private_key.pem"
   ```

---

## 🧪 Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=neural tests/

# Run specific test
pytest tests/test_auth.py -v
```

---

## 🤝 Contributing

We welcome contributions! Neural SDK is open source and community-driven.

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** and add tests
4. **Run tests**: `pytest`
5. **Commit**: `git commit -m "Add amazing feature"`
6. **Push**: `git push origin feature/amazing-feature`
7. **Open a Pull Request**

See [CONTRIBUTING.md](./CONTRIBUTING.md) for detailed guidelines.

### Development Setup

```bash
# Clone repository
git clone https://github.com/IntelIP/Neural.git
cd neural

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check .
black --check .
```

---

## 📖 Resources

- **Documentation**: [neural-sdk.mintlify.app](https://neural-sdk.mintlify.app)
- **Examples**: [examples/](./examples)
- **API Reference**: [docs/api/](./docs/api)
- **Issues**: [GitHub Issues](https://github.com/IntelIP/Neural/issues)
- **Discussions**: [GitHub Discussions](https://github.com/IntelIP/Neural/discussions)

---

## 🗺️ Roadmap

### Version 0.1.0 (Beta) - Current

- ✅ Core authentication
- ✅ Historical data collection
- ✅ Strategy framework
- ✅ Backtesting engine
- ⚠️ REST streaming (stable)
- ⚠️ WebSocket streaming (experimental)

### Version 0.2.0 (Planned)

- 🔄 Enhanced WebSocket support
- 🔄 Real-time strategy execution
- 🔄 Portfolio optimization
- 🔄 Multi-market strategies

### Version 1.0.0 (Future)

- 🚀 Deployment stack (AWS/GCP integration)
- 🚀 Production monitoring & alerting
- 🚀 Advanced risk analytics
- 🚀 Machine learning strategies

---

## ⚖️ License

This project is licensed under the MIT License - see [LICENSE](./LICENSE) file for details.

### What This Means

✅ **You CAN**:
- Use commercially
- Modify the code
- Distribute
- Use privately

❌ **You CANNOT**:
- Hold us liable
- Use our trademarks

📋 **You MUST**:
- Include the original license
- Include copyright notice

---

## 🙏 Acknowledgments

- Built for the [Kalshi](https://kalshi.com) prediction market platform
- Inspired by the quantitative trading community
- Special thanks to all [contributors](https://github.com/IntelIP/Neural/graphs/contributors)

---

## 📞 Support

- **Documentation**: [neural-sdk.mintlify.app](https://neural-sdk.mintlify.app)
- **Issues**: [GitHub Issues](https://github.com/IntelIP/Neural/issues)
- **Discussions**: [GitHub Discussions](https://github.com/IntelIP/Neural/discussions)
- **Email**: support@neural-sdk.dev

---

<div align="center">

**Built with ❤️ by the Neural community**

[⭐ Star us on GitHub](https://github.com/IntelIP/Neural) • [📖 Read the Docs](https://neural-sdk.mintlify.app)

</div>