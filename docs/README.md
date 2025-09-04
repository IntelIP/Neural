# 📚 Neural SDK Documentation

**Complete documentation for the Neural SDK v1.1.0**

---

## 🚀 Getting Started

- **[Quick Start Guide](../neural_sdk/README.md)** - Installation and basic usage
- **[Configuration Guide](CONFIGURATION.md)** - Environment setup and API keys
- **[Examples Directory](../examples/)** - Working code examples

## 🔥 NEW: WebSocket Streaming (v1.1.0)

- **[WebSocket Streaming Guide](WEBSOCKET_STREAMING_GUIDE.md)** - Complete guide to real-time market data
- **[NFL Market Streaming](WEBSOCKET_STREAMING_GUIDE.md#nfl-specific-streaming)** - Sports-specific functionality
- **[WebSocket Examples](../examples/nfl_websocket_streaming.py)** - Working implementations

## 📖 Core Documentation

### Architecture & Design
- **[System Architecture](ARCHITECTURE.md)** - Overall system design
- **[SDK Structure](../neural_sdk/README.md#🏗️-sdk-structure)** - Module organization
- **[Plugin Architecture](PLUGIN_ARCHITECTURE.md)** - Extensibility framework

### Trading & Strategies
- **[Trading Logic](TRADING_LOGIC.md)** - Strategy development
- **[Backtesting Guide](backtesting.md)** - Historical testing framework
- **[Risk Management](../neural_sdk/README.md#risk-controls)** - Position and risk controls

### Data & Streaming
- **[Data Sources Guide](DATA_SOURCES_GUIDE.md)** - Available data sources
- **[WebSocket Streaming](WEBSOCKET_STREAMING_GUIDE.md)** - Real-time data streaming
- **[Market Discovery](WEBSOCKET_STREAMING_GUIDE.md#market-discovery)** - Automatic market detection

### Agents & Automation
- **[Agents Overview](AGENTS.md)** - Agent-based architecture
- **[Game Configuration](GAME_CONFIGURATION_GUIDE.md)** - Sports event configuration

## 🔧 Development

### Setup & Configuration
- **[Getting Started](GETTING_STARTED.md)** - Development environment setup
- **[Contributing Guide](../CONTRIBUTING.md)** - How to contribute
- **[Testing Guide](../TEST_RESULTS.md)** - Test results and coverage

### API Reference
- **[SDK API Reference](api_reference.md)** - Complete API documentation
- **[WebSocket API](WEBSOCKET_STREAMING_GUIDE.md#complete-api-reference)** - WebSocket-specific APIs
- **[CLI Reference](../neural_sdk/README.md#cli-usage)** - Command-line interface

## 📊 Performance & Monitoring

- **[Performance Benchmarks](WEBSOCKET_STREAMING_GUIDE.md#📊-performance-benchmarks)** - WebSocket performance metrics
- **[System Overview](SYSTEM_OVERVIEW.md)** - Monitoring and observability
- **[Troubleshooting](WEBSOCKET_STREAMING_GUIDE.md#🐛-troubleshooting)** - Common issues and solutions

## 🎯 Use Cases & Examples

### WebSocket Streaming Examples
- **[Real-time Arbitrage](WEBSOCKET_STREAMING_GUIDE.md#1-real-time-arbitrage-detection)** - Arbitrage opportunity detection
- **[NFL Momentum Trading](WEBSOCKET_STREAMING_GUIDE.md#2-nfl-game-momentum-trading)** - Sports momentum strategies
- **[Portfolio Monitoring](WEBSOCKET_STREAMING_GUIDE.md#3-multi-game-portfolio-monitoring)** - Multi-game tracking
- **[Strategy Framework](WEBSOCKET_STREAMING_GUIDE.md#4-event-driven-strategy-framework)** - Event-driven strategies

### Traditional Examples
- **[Basic Strategy](../examples/basic_usage.py)** - Simple trading strategy
- **[Backtesting](../examples/backtest_strategy.py)** - Strategy backtesting
- **[Data Streaming](../examples/stream_markets.py)** - Market data streaming

## 🔄 Version History

### v1.1.0 (Current) ✅
- **🔥 Real-time WebSocket Streaming** - Direct Kalshi WebSocket integration
- **🏈 NFL Market Support** - Sports-specific streaming functionality
- **📊 Event-driven Architecture** - Decorator-based event handling
- **🔍 Market Discovery** - Automatic game and team detection
- **🧪 Comprehensive Testing** - 35+ unit tests covering WebSocket functionality

### v1.0.0 ✅
- Core SDK architecture
- Agent-based framework
- Backtesting engine
- Configuration management
- CLI interface

## 🆘 Support

- **[Troubleshooting Guide](WEBSOCKET_STREAMING_GUIDE.md#🐛-troubleshooting)** - Common issues
- **[GitHub Issues](https://github.com/neural/neural-sdk/issues)** - Bug reports and features
- **[GitHub Discussions](https://github.com/neural/neural-sdk/discussions)** - Community support

## 📋 Quick Reference

### WebSocket Streaming (v1.1.0)
```python
# Basic WebSocket
websocket = sdk.create_websocket()
await websocket.connect()
await websocket.subscribe_markets(['KXNFLGAME*'])

# NFL Streaming
nfl_stream = sdk.create_nfl_stream()
await nfl_stream.subscribe_to_game("25SEP04DALPHI")

# Integrated Streaming
await sdk.start_streaming(['KXNFLGAME*'])
```

### Event Handlers
```python
@websocket.on_market_data
async def handle_data(data): pass

@websocket.on_trade
async def handle_trades(trade): pass

@websocket.on_error
async def handle_errors(error): pass
```

### Core SDK
```python
# Initialize
sdk = NeuralSDK.from_env()

# Strategy
@sdk.strategy
async def my_strategy(data): pass

# Backtesting
engine = BacktestEngine(config)
results = await engine.run(strategy)
```

---

**Ready to start building? Check out the [WebSocket Streaming Guide](WEBSOCKET_STREAMING_GUIDE.md) for the latest v1.1.0 features!** 🚀