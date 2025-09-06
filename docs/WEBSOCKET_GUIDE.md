# 🌐 WebSocket Infrastructure Guide

> **Complete guide to Neural SDK's real-time WebSocket infrastructure**

## 📚 Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Components](#components)
- [Usage Examples](#usage-examples)
- [API Reference](#api-reference)
- [Best Practices](#best-practices)

## Overview

The Neural SDK v1.4.0 introduces a production-grade WebSocket infrastructure for real-time prediction market trading. This system provides:

- **Real-time market data** streaming from Kalshi
- **Multi-source data correlation** between Kalshi and sportsbooks
- **Automatic arbitrage detection** with configurable thresholds
- **Risk-managed trading** with circuit breakers
- **Event-driven architecture** for sub-second response times

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     UnifiedStreamManager                     │
│  ┌─────────────────────┐      ┌────────────────────────┐   │
│  │ KalshiWebSocket     │      │ OddsAPIPoller          │   │
│  │ - Ticker updates    │      │ - 70+ sportsbooks      │   │
│  │ - Orderbook changes │      │ - Live odds            │   │
│  │ - Trade executions  │      │ - Historical data      │   │
│  └─────────────────────┘      └────────────────────────┘   │
│                     ▼                    ▼                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │            Data Correlation & Analysis               │  │
│  │ - Arbitrage detection                                │  │
│  │ - Divergence monitoring                              │  │
│  │ - Volatility calculation                             │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  RealTimeTradingEngine                       │
│  ┌─────────────────────┐      ┌────────────────────────┐   │
│  │ Strategy Manager    │      │ Risk Manager           │   │
│  │ - Multi-strategy    │      │ - Position limits      │   │
│  │ - Signal generation │      │ - Stop-loss/take-profit│   │
│  │ - Backtesting       │      │ - Circuit breakers     │   │
│  └─────────────────────┘      └────────────────────────┘   │
│                     ▼                    ▼                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │               Order Management System                │  │
│  │ - Order execution                                    │  │
│  │ - Position tracking                                  │  │
│  │ - P&L calculation                                    │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Basic WebSocket Connection

```python
from neural_sdk.data_sources.kalshi.websocket_adapter import KalshiWebSocketAdapter

# Create WebSocket adapter
ws = KalshiWebSocketAdapter(api_key="your_api_key")

# Connect and subscribe
await ws.connect()
await ws.subscribe_market("KXNFLGAME-25SEP04DALPHI-PHI")

# Handle events
@ws.on("ticker_update")
async def handle_ticker(data):
    print(f"Price: ${data['yes_price']}")
```

### Real-Time Trading Engine

```python
from neural_sdk.trading.real_time_engine import RealTimeTradingEngine, RiskLimits
from neural_sdk.data_sources.unified.stream_manager import UnifiedStreamManager

# Setup stream manager
stream_manager = UnifiedStreamManager()

# Configure risk limits
risk_limits = RiskLimits(
    max_position_size=1000,
    max_daily_loss=500,
    stop_loss_percentage=0.10
)

# Create trading engine
engine = RealTimeTradingEngine(stream_manager, risk_limits)

# Add trading strategy
async def my_strategy(market_data, engine):
    if market_data.kalshi_yes_price < 0.30:
        return TradingSignal(
            signal_type=SignalType.BUY,
            market_ticker=market_data.ticker,
            confidence=0.8,
            size=100
        )

engine.add_strategy(my_strategy)
await engine.start()
```

## Components

### 1. WebSocketDataSource (Base Class)

Base class providing core WebSocket functionality:

```python
from neural_sdk.data_sources.base.websocket_source import WebSocketDataSource, ConnectionConfig

config = ConnectionConfig(
    url="wss://api.kalshi.com/v2/ws",
    api_key="your_api_key",
    heartbeat_interval=30,
    max_reconnect_attempts=10
)
```

**Features:**
- Automatic reconnection with exponential backoff
- Heartbeat/keepalive mechanism
- Message queuing during disconnections
- Event-driven architecture

### 2. KalshiWebSocketAdapter

Kalshi-specific WebSocket implementation:

```python
from neural_sdk.data_sources.kalshi.websocket_adapter import (
    KalshiWebSocketAdapter, KalshiChannel
)

ws = KalshiWebSocketAdapter(api_key="your_api_key")

# Available channels
channels = [
    KalshiChannel.TICKER,           # Price updates
    KalshiChannel.ORDERBOOK_DELTA,  # Order book changes
    KalshiChannel.TRADE,            # Executed trades
    KalshiChannel.FILL,             # Your order fills
    KalshiChannel.MARKET_POSITIONS, # Your positions
    KalshiChannel.MARKET_LIFECYCLE  # Market status
]

await ws.subscribe_market("KXNFLGAME-25SEP04DALPHI-PHI", channels)
```

### 3. UnifiedStreamManager

Coordinates multiple data sources:

```python
from neural_sdk.data_sources.unified.stream_manager import (
    UnifiedStreamManager, StreamConfig, EventType
)

config = StreamConfig(
    enable_kalshi=True,
    enable_odds_polling=True,
    odds_poll_interval=30,
    correlation_window=5,
    divergence_threshold=0.05
)

manager = UnifiedStreamManager(config)

# Register event handlers
manager.on(EventType.ARBITRAGE_OPPORTUNITY, handle_arbitrage)
manager.on(EventType.DIVERGENCE_DETECTED, handle_divergence)
manager.on(EventType.PRICE_UPDATE, handle_price)

await manager.start()
```

### 4. RealTimeTradingEngine

Complete trading engine with risk management:

```python
from neural_sdk.trading.real_time_engine import (
    RealTimeTradingEngine, RiskLimits, TradingSignal, SignalType
)

# Risk configuration
risk_limits = RiskLimits(
    max_position_size=1000,      # Max shares per position
    max_order_size=100,           # Max shares per order
    max_daily_loss=500.0,         # Daily loss limit
    max_daily_trades=50,          # Max trades per day
    max_open_positions=10,        # Max concurrent positions
    stop_loss_percentage=0.10,    # 10% stop-loss
    take_profit_percentage=0.20   # 20% take-profit
)

engine = RealTimeTradingEngine(stream_manager, risk_limits)
```

## Usage Examples

### Example 1: Arbitrage Scanner

```python
from neural_sdk.data_sources.unified.stream_manager import UnifiedStreamManager

stream_manager = UnifiedStreamManager()

# Track arbitrage opportunities
arbitrage_opportunities = []

@stream_manager.on(EventType.ARBITRAGE_OPPORTUNITY)
async def on_arbitrage(event):
    ticker = event["ticker"]
    data = event["data"]
    
    divergence = abs(data.kalshi_yes_price - data.odds_implied_prob_home)
    
    if divergence > 0.08:  # 8% arbitrage
        opportunity = {
            "ticker": ticker,
            "kalshi_price": data.kalshi_yes_price,
            "odds_price": data.odds_implied_prob_home,
            "divergence": divergence,
            "timestamp": datetime.utcnow()
        }
        arbitrage_opportunities.append(opportunity)
        print(f"💰 ARBITRAGE: {ticker} - {divergence:.1%} profit potential")

await stream_manager.start()
```

### Example 2: Multi-Strategy Trading

```python
# Define multiple strategies
async def momentum_strategy(data, engine):
    history = engine.stream_manager.get_market_history(data.ticker, limit=10)
    if len(history) >= 5:
        recent_prices = [h.kalshi_yes_price for h in history[-5:]]
        if recent_prices[-1] > recent_prices[0] * 1.02:  # 2% increase
            return TradingSignal(
                signal_type=SignalType.BUY,
                market_ticker=data.ticker,
                confidence=0.7,
                size=50
            )

async def mean_reversion_strategy(data, engine):
    volatility = engine.stream_manager.calculate_volatility(data.ticker)
    if volatility and volatility > 0.05:  # High volatility
        history = engine.stream_manager.get_market_history(data.ticker, limit=20)
        avg_price = sum(h.kalshi_yes_price for h in history) / len(history)
        
        if data.kalshi_yes_price < avg_price * 0.95:  # 5% below average
            return TradingSignal(
                signal_type=SignalType.BUY,
                market_ticker=data.ticker,
                confidence=0.6,
                size=30
            )

# Add all strategies
engine.add_strategy(momentum_strategy)
engine.add_strategy(mean_reversion_strategy)
```

### Example 3: Connection Monitoring

```python
from neural_sdk.data_sources.base.websocket_source import ConnectionState

# Monitor connection health
async def monitor_connection(ws_adapter):
    while True:
        if ws_adapter.state == ConnectionState.CONNECTED:
            stats = ws_adapter.stats
            print(f"✅ Connected - Messages: {stats.messages_received}")
            print(f"   Uptime: {stats.uptime}")
            print(f"   Error rate: {stats.error_rate:.2%}")
        else:
            print(f"⚠️ Connection state: {ws_adapter.state.value}")
        
        await asyncio.sleep(10)

# Start monitoring
asyncio.create_task(monitor_connection(ws_adapter))
```

## API Reference

### ConnectionConfig

```python
@dataclass
class ConnectionConfig:
    url: str                          # WebSocket URL
    api_key: Optional[str] = None     # API key for authentication
    heartbeat_interval: int = 30      # Seconds between heartbeats
    reconnect_interval: int = 5       # Initial reconnect delay
    max_reconnect_attempts: int = 10  # Max reconnection attempts
    connection_timeout: int = 30      # Connection timeout in seconds
    message_queue_size: int = 10000   # Max queued messages
```

### UnifiedMarketData

```python
@dataclass
class UnifiedMarketData:
    ticker: str                              # Market ticker
    timestamp: datetime                      # Data timestamp
    
    # Kalshi data
    kalshi_yes_price: Optional[float]       # YES contract price
    kalshi_no_price: Optional[float]        # NO contract price
    kalshi_volume: Optional[float]          # Trading volume
    kalshi_open_interest: Optional[float]   # Open interest
    
    # Odds data
    odds_consensus_home: Optional[float]    # Consensus home odds
    odds_consensus_away: Optional[float]    # Consensus away odds
    
    # Computed metrics
    arbitrage_exists: bool                  # Arbitrage opportunity
    divergence_score: Optional[float]       # Price divergence
```

### TradingSignal

```python
@dataclass
class TradingSignal:
    signal_id: str                    # Unique signal ID
    timestamp: datetime                # Signal generation time
    market_ticker: str                 # Market to trade
    signal_type: SignalType           # BUY/SELL/HOLD/CLOSE
    confidence: float                  # Signal confidence (0-1)
    size: Optional[int]                # Position size
    reason: Optional[str]              # Signal reason
    metadata: Optional[Dict]           # Additional data
```

### RiskLimits

```python
@dataclass
class RiskLimits:
    max_position_size: int = 1000        # Max shares per position
    max_order_size: int = 100            # Max shares per order
    max_daily_loss: float = 1000.0       # Daily loss limit
    max_daily_trades: int = 100          # Max trades per day
    max_open_positions: int = 10         # Max concurrent positions
    stop_loss_percentage: float = 0.10   # Stop-loss threshold
    take_profit_percentage: float = 0.20 # Take-profit threshold
```

## Best Practices

### 1. Connection Management

```python
# Always use context managers or proper cleanup
async def safe_websocket_usage():
    ws = KalshiWebSocketAdapter(api_key="your_key")
    try:
        await ws.connect()
        # Do work
    finally:
        await ws.disconnect()

# Or use async context manager (if implemented)
async with KalshiWebSocketAdapter(api_key="your_key") as ws:
    await ws.subscribe_market("KXNFLGAME-25SEP04DALPHI-PHI")
```

### 2. Error Handling

```python
# Handle connection errors gracefully
@ws.on("error")
async def handle_error(error):
    logger.error(f"WebSocket error: {error}")
    
    if error.get("type") == "CONNECTION_LOST":
        # Will auto-reconnect, but log for monitoring
        await alert_monitoring_system(error)
    elif error.get("type") == "AUTHENTICATION_FAILED":
        # Critical error, needs intervention
        await stop_trading()
```

### 3. Resource Management

```python
# Limit subscriptions to avoid overwhelming the system
MAX_SUBSCRIPTIONS = 50
active_subscriptions = set()

async def subscribe_market(ticker):
    if len(active_subscriptions) >= MAX_SUBSCRIPTIONS:
        # Unsubscribe from least active market
        least_active = find_least_active_market()
        await ws.unsubscribe_market(least_active)
        active_subscriptions.remove(least_active)
    
    await ws.subscribe_market(ticker)
    active_subscriptions.add(ticker)
```

### 4. Data Validation

```python
# Always validate incoming data
async def process_market_data(data):
    # Validate required fields
    if not data.kalshi_yes_price or data.kalshi_yes_price < 0:
        logger.warning(f"Invalid price data: {data}")
        return
    
    # Sanity checks
    if data.kalshi_yes_price + data.kalshi_no_price > 1.01:
        logger.error(f"Price sum exceeds 1: {data}")
        return
    
    # Process valid data
    await execute_strategy(data)
```

### 5. Performance Optimization

```python
# Batch operations when possible
pending_orders = []

async def batch_order_processor():
    while True:
        await asyncio.sleep(0.1)  # 100ms batching window
        
        if pending_orders:
            # Process all pending orders together
            batch = pending_orders[:10]  # Max 10 per batch
            pending_orders[:10] = []
            
            await execute_batch_orders(batch)
```

## Troubleshooting

### Common Issues

1. **Connection Drops**
   - Check network stability
   - Verify API key is valid
   - Monitor rate limits

2. **High Latency**
   - Reduce number of subscriptions
   - Optimize event handlers
   - Check network proximity to servers

3. **Memory Usage**
   - Limit history window size
   - Clear old data periodically
   - Monitor message queue size

### Debug Mode

```python
# Enable debug logging
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("neural_sdk.websocket")

# Add debug handler
@ws.on("debug")
async def debug_handler(msg):
    logger.debug(f"WebSocket debug: {msg}")
```

## Performance Metrics

The WebSocket infrastructure is designed for high performance:

- **Latency**: < 50ms for market data updates
- **Throughput**: 10,000+ messages/second
- **Reliability**: 99.9% uptime with auto-reconnection
- **Scalability**: Support for 1,000+ concurrent market subscriptions

## Security Considerations

1. **API Key Protection**: Never expose API keys in code
2. **TLS/SSL**: All connections use secure WebSocket (wss://)
3. **Rate Limiting**: Automatic rate limit handling
4. **Authentication**: Token-based authentication with refresh

---

*For more information, see the [API Documentation](./API_REFERENCE.md) or [Examples](../examples/)*