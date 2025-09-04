# Changelog

All notable changes to the Neural SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2024-01-XX

### 🚀 **Major Features Added**

#### **WebSocket Streaming Integration**
- **NEW**: `NeuralWebSocket` class for real-time market data streaming
- **NEW**: `NFLMarketStream` specialized class for NFL market streaming  
- **NEW**: `MarketStream` base class for market-specific streaming
- **NEW**: Event handler decorators for WebSocket events (`@websocket.on_market_data`, `@websocket.on_trade`)

#### **Enhanced SDK Client**
- **NEW**: `sdk.create_websocket()` - Create WebSocket clients directly from SDK
- **NEW**: `sdk.create_nfl_stream()` - Create NFL-specific streaming clients
- **NEW**: `sdk.start_streaming(markets)` - Convenience method for quick streaming setup
- **NEW**: `sdk.stop_streaming()` - Stop WebSocket streaming
- **NEW**: `sdk.on_market_update()` - Alias for market data event handling

#### **Market Discovery Integration**
- **NEW**: Automatic NFL market discovery and subscription
- **NEW**: Game-specific market subscription (`websocket.subscribe_nfl_game()`)
- **NEW**: Team-specific market subscription (`websocket.subscribe_nfl_team()`)
- **NEW**: Pattern-based market subscription with wildcards

### ✨ **New Capabilities**

#### **Real-time Data Streaming**
```python
from neural_sdk import NeuralSDK

sdk = NeuralSDK.from_env()
websocket = sdk.create_websocket()

@websocket.on_market_data
async def handle_updates(market_data):
    print(f"Price: {market_data['yes_price']}")

await websocket.connect()
await websocket.subscribe_markets(['NFL-*'])
```

#### **NFL-Specific Streaming**
```python
nfl_stream = sdk.create_nfl_stream()
await nfl_stream.connect()
await nfl_stream.subscribe_to_game("25SEP04DALPHI")

summary = nfl_stream.get_game_summary("25SEP04DALPHI")
print(f"Win probability: {summary['win_probability']}")
```

#### **Integrated SDK Streaming**
```python
@sdk.on_market_data
async def handle_updates(market_data):
    # Process market updates
    pass

await sdk.start_streaming(['NFL-*'])
```

### 🛠️ **Technical Improvements**

#### **Architecture**
- **Enhanced**: Layered architecture with clean separation between data pipeline and SDK
- **Added**: Streaming module (`neural_sdk.streaming`) with specialized classes
- **Improved**: Event handling system with decorator-based registration
- **Added**: Connection management with automatic reconnection and error handling

#### **Error Handling**
- **NEW**: `ConnectionError` for WebSocket connection issues
- **Enhanced**: Comprehensive error handling in streaming operations
- **Added**: Circuit breaker protection for reliable connections
- **Improved**: Graceful degradation and recovery mechanisms

#### **Configuration**
- **Enhanced**: Unified configuration system for SDK and data pipeline layers
- **Added**: WebSocket-specific configuration options
- **Improved**: Environment-based configuration management

### 📚 **Documentation & Examples**

#### **New Examples**
- **NEW**: `examples/nfl_websocket_streaming.py` - Comprehensive WebSocket streaming examples
- **NEW**: `examples/simple_websocket_demo.py` - Simple getting-started demo
- **Enhanced**: Existing examples updated with WebSocket functionality

#### **API Documentation**
- **NEW**: Complete WebSocket API documentation
- **NEW**: NFL streaming guide and best practices
- **Enhanced**: SDK client documentation with streaming methods
- **Added**: Event handler documentation and patterns

### 🔧 **Breaking Changes**
- **NONE**: This release maintains full backward compatibility with v1.0.0
- **Note**: New streaming functionality requires data pipeline dependencies

### 🐛 **Bug Fixes**
- **Fixed**: Import paths for streaming modules
- **Improved**: Error messages for missing dependencies
- **Enhanced**: Connection stability and reconnection logic

### 📦 **Dependencies**
- **Added**: Integration with existing data pipeline WebSocket infrastructure
- **Enhanced**: Dependency management for streaming functionality
- **Maintained**: All existing dependencies remain unchanged

### 🚀 **Performance**
- **Improved**: Real-time data processing with minimal latency
- **Enhanced**: Memory management for streaming operations  
- **Added**: Flow control and backpressure management
- **Optimized**: Event dispatch and handler execution

### 💡 **Migration Guide**

#### **From v1.0.0 to v1.1.0**

**No breaking changes** - existing code continues to work unchanged.

**To use new WebSocket functionality:**

1. **Basic WebSocket streaming:**
```python
# OLD: Not available in v1.0.0
# NEW: Available in v1.1.0
websocket = sdk.create_websocket()
await websocket.connect()
```

2. **NFL market streaming:**
```python
# OLD: Manual data pipeline integration required
# NEW: Built-in NFL streaming
nfl_stream = sdk.create_nfl_stream()
await nfl_stream.subscribe_to_team("PHI")
```

3. **Event handling:**
```python
# OLD: Manual event setup
# NEW: Decorator-based events
@websocket.on_market_data
async def handle_updates(data):
    # Process updates
    pass
```

### 🎯 **What's Next**

#### **Planned for v1.2.0**
- **Enhanced**: Multi-sport streaming (NBA, MLB, etc.)
- **NEW**: Advanced analytics and indicators
- **NEW**: Portfolio integration with streaming data
- **Enhanced**: Machine learning integration for streaming data

---

## [1.0.0] - 2024-01-01

### 🎉 **Initial Release**

#### **Core Features**
- **NEW**: Strategy development framework with decorators
- **NEW**: Comprehensive backtesting engine
- **NEW**: Portfolio management and risk controls
- **NEW**: Configuration management system
- **NEW**: CLI interface for SDK management

#### **Trading Framework**
- **NEW**: `@sdk.strategy` decorator for strategy development
- **NEW**: `TradingSignal` and `MarketData` data structures
- **NEW**: Event-driven architecture for trading logic

#### **Backtesting**
- **NEW**: Historical data loading from multiple sources
- **NEW**: Performance metrics and analytics
- **NEW**: Realistic order execution simulation
- **NEW**: Portfolio simulation with fees and slippage

#### **Configuration**
- **NEW**: Environment-based configuration
- **NEW**: YAML configuration file support
- **NEW**: Multi-environment support (dev/staging/prod)

#### **CLI Tools**
- **NEW**: `neural-sdk init` - Initialize configuration
- **NEW**: `neural-sdk validate` - Validate configuration
- **NEW**: `neural-sdk status` - Check system status

---

## Legend

- 🚀 **Major Features**: Significant new functionality
- ✨ **New Capabilities**: New features and enhancements  
- 🛠️ **Technical Improvements**: Architecture and performance improvements
- 📚 **Documentation**: Documentation and examples
- 🔧 **Breaking Changes**: Changes that may require code updates
- 🐛 **Bug Fixes**: Bug fixes and stability improvements
- 📦 **Dependencies**: Dependency changes
- 💡 **Migration Guide**: How to upgrade from previous versions
