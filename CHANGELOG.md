# Changelog

All notable changes to the Neural SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.0] - 2025-09-05

### 🏈 **CRITICAL SPORTS MARKET DISCOVERY FIX**

#### **Root Cause Resolution**
- **FIXED**: Resolved "No markets found" errors plaguing sports traders (Cowboys vs Eagles, etc.)
- **ISSUE**: Wrong Kalshi API tag filtering was preventing sports market discovery
- **IMPACT**: All sports traders were failing due to incorrect API usage

#### **Corrected Kalshi API Integration**
- **BEFORE**: Used incorrect tags (`tags=NFL`, `tags=NBA`, `tags=MLB`) → 0 series found
- **AFTER**: Use proper Kalshi tags → 198+ sports series discovered
  - NFL → `tags=Football` (124+ series including KXNFLGAME)
  - NBA → `tags=Basketball` (74+ series)
  - MLB → `tags=Baseball` (active series)
  - Soccer → `tags=Soccer` (La Liga, etc.)

#### **Updated Sports Market Discovery**
- **NEW**: `KalshiClient.get_nfl_series()` using corrected `tags=Football`
- **NEW**: `KalshiClient.get_nba_series()` using corrected `tags=Basketball`
- **NEW**: `KalshiClient.get_mlb_series()` using corrected `tags=Baseball`
- **NEW**: `KalshiClient.get_soccer_series()` for soccer markets
- **ENHANCED**: `SportsMarketDiscovery` class with proper series-first workflow

#### **Cowboys vs Eagles Trader Restoration**
- ✅ **DISCOVERED**: `KXNFLGAME` series (Professional Football Game)
- ✅ **FOUND**: 10+ current NFL markets in system
- ✅ **LOCATED**: Cowboys (`KXNFLWINS-DAL`) and Eagles (`KXNFLWINS-PHI`) series
- ✅ **RESOLVED**: "No markets found" error - trader now functional

#### **Neural SDK Integration**
- **ENHANCED**: `sdk.discover_sports_markets()` with corrected discovery
- **ENHANCED**: `sdk.find_nfl_markets()` returns actual NFL markets
- **NEW**: `sdk.get_working_market_ticker()` for testing purposes
- **UPDATED**: WebSocket subscriptions use discovered series instead of hardcoded patterns

#### **Breaking Changes**
- Sports market discovery now uses correct Kalshi tag mapping
- Hardcoded series patterns replaced with dynamic discovery
- WebSocket NFL subscriptions updated to use proper series discovery

### 📊 **Verification Results**
```
✅ NFL Series Found: 124 (was 0)
✅ NBA Series Found: 74 (was 0)  
✅ Current NFL Markets: 10+ available
✅ Key Series Discovered: KXNFLGAME, KXNFLCOMBO, KXNFLFIRSTTD
✅ Cowboys vs Eagles: FUNCTIONAL
```

## [1.2.0] - 2024-09-05

### 🚀 **Major Portfolio Management Features**

#### **Complete Portfolio Management Integration**
- **NEW**: `Position` data class for clean position representation with market name parsing
- **NEW**: `Order` data class for trading orders with fill tracking and status management  
- **NEW**: `Portfolio` data class for comprehensive portfolio summaries
- **NEW**: `sdk.get_balance()` - Get current account balance in dollars
- **NEW**: `sdk.get_positions()` - Get all current positions with clean market names
- **NEW**: `sdk.get_orders()` - Get order history with filtering options
- **NEW**: `sdk.place_order()` - Execute trades directly through SDK
- **NEW**: `sdk.get_portfolio_summary()` - Complete portfolio overview

#### **User Experience Revolution**
- **BREAKING THE PIPELINE**: Users no longer need to understand data pipeline internals
- **CLEAN API**: Simple, intuitive methods replace complex `KalshiClient` calls
- **RICH DATA CLASSES**: Smart properties like `position.market_name`, `order.fill_percentage`
- **PRODUCTION READY**: Real money trading with proper production endpoint handling

### ✨ **New Portfolio Capabilities**

#### **Simple Balance & Positions**
```python
from neural_sdk import NeuralSDK

sdk = NeuralSDK.from_env()

# Get account balance
balance = await sdk.get_balance()
print(f"Balance: ${balance:.2f}")

# Get all positions with clean names  
positions = await sdk.get_positions()
for pos in positions:
    print(f"{pos.market_name}: {pos.position} shares @ ${pos.avg_price:.3f}")
```

#### **Complete Portfolio Overview**
```python
# Get comprehensive portfolio summary
portfolio = await sdk.get_portfolio_summary()
print(f"Total Value: ${portfolio.total_value:.2f}")
print(f"Active Positions: {portfolio.position_count}")
print(f"Largest Position: {portfolio.largest_position.market_name}")
```

#### **Trade Execution**
```python  
# Place orders directly through SDK
order = await sdk.place_order(
    ticker="KXNFLGAME-25SEP04DALPHI-PHI",
    side="YES",
    quantity=10, 
    price=0.65
)
print(f"Order placed: {order.order_id}")
```

### 🛠️ **Technical Improvements**

#### **Data Classes**
- **Enhanced**: Rich data classes with computed properties and human-readable names
- **Added**: Market name parsing for NFL markets (e.g., "A.J. Brown First TD (PHI)")
- **Improved**: Automatic unit conversion (cents to dollars) throughout
- **Added**: Fill tracking, average price calculation, and portfolio metrics

#### **Error Handling**
- **NEW**: Comprehensive validation for order parameters
- **Enhanced**: Graceful handling of missing API endpoints  
- **Added**: Detailed error messages with context
- **Improved**: Connection management with automatic cleanup

#### **Production Fixes**
- **FIXED**: Environment variable mismatch causing demo API usage
- **RESOLVED**: Production endpoint connectivity issues
- **Added**: Forced production endpoints for consistent behavior
- **Enhanced**: Endpoint validation and logging

### 📚 **Documentation & Examples**

#### **New Examples** 
- **NEW**: `portfolio_example.py` - Complete portfolio management demo
- **NEW**: `test_portfolio_sdk.py` - Comprehensive test suite
- **Added**: Usage examples for all new portfolio methods

#### **API Documentation**
- **NEW**: Complete portfolio management API reference
- **Enhanced**: Data class documentation with property descriptions
- **Added**: Trading examples and best practices

### 🔧 **Breaking Changes**
- **NONE**: Full backward compatibility maintained
- **Enhanced**: Existing `get_portfolio_status()` method now uses new portfolio system
- **Added**: New data classes exported in main SDK namespace

### 🐛 **Critical Fixes**
- **FIXED**: Environment mismatch (.env had "production" but code expected "prod")
- **RESOLVED**: Demo API endpoint usage when production credentials configured
- **Fixed**: Missing dateutil import handling
- **Enhanced**: Client connection management and cleanup

### 📦 **Dependencies**
- **Maintained**: All existing dependencies unchanged
- **Enhanced**: Better integration with data pipeline KalshiClient
- **Added**: Support for python-dateutil for timestamp parsing

### 🚀 **Performance**
- **Optimized**: Concurrent portfolio data fetching with asyncio.gather()
- **Enhanced**: Connection pooling and reuse in KalshiClient
- **Improved**: Error handling performance with graceful degradation

### 💡 **Migration Guide**

#### **From v1.1.0 to v1.2.0**

**No breaking changes** - existing code continues to work unchanged.

**NEW portfolio management capabilities:**

```python
# Before v1.2.0 - Complex data pipeline usage
from neural_sdk.data_pipeline.data_sources.kalshi.client import KalshiClient
client = KalshiClient()  
data = client.get('/portfolio/positions')
# Manual parsing and error handling...

# v1.2.0+ - Clean SDK interface
from neural_sdk import NeuralSDK
sdk = NeuralSDK.from_env()
portfolio = await sdk.get_portfolio_summary()
print(f"Balance: ${portfolio.balance:.2f}")
```

**Access to rich data classes:**
```python
# Get positions with clean market names
positions = await sdk.get_positions()
for pos in positions:
    print(f"{pos.market_name}: ${pos.market_exposure:.2f}")

# Track orders with fill percentages
orders = await sdk.get_orders()
for order in orders:
    print(f"Fill: {order.fill_percentage:.1f}%")
```

### 🎯 **What's Next**

#### **Planned for v1.3.0**
- **Enhanced**: Real-time portfolio updates via WebSocket
- **NEW**: Advanced portfolio analytics and performance metrics
- **NEW**: Risk management integration with portfolio data
- **Enhanced**: Machine learning features with portfolio context

---

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
