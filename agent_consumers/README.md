# Agent Consumers

## Purpose
Redis consumers that process real-time data streams from the data pipeline. These consumers subscribe to specific Redis channels and trigger trading decisions based on incoming events.

## Architecture

```
Redis Pub/Sub Channels
       ↓
Agent Consumers (this directory)
       ↓
Trading Decisions & Signals
```

## Components

### base_consumer.py
Abstract base class providing common functionality for all agent consumers:
- Automatic reconnection with exponential backoff
- Health monitoring and statistics
- Error handling and recovery
- Message processing pipeline

### DataCoordinator Consumer
**File:** `DataCoordinator/data_coordinator.py`  
**Purpose:** Central hub for all incoming data, correlates events and generates trading signals  
**Subscribes to:**
- `kalshi:markets` - Market price updates
- `kalshi:trades` - Trade confirmations
- `espn:games` - Game events
- `twitter:sentiment` - Social sentiment

**Key Responsibilities:**
- Detect extreme price movements
- Correlate ESPN events with market changes
- Generate high-priority signals for other agents

### MarketEngineer Consumer
**File:** `MarketEngineer/market_engineer.py`  
**Purpose:** Identifies market inefficiencies and arbitrage opportunities  
**Subscribes to:**
- `kalshi:markets` - For arbitrage detection
- `espn:games` - For event-driven opportunities

**Key Features:**
- Arbitrage detection (YES + NO < 0.98)
- Market inefficiency identification
- Opportunity publication for TradeExecutor

### TradeExecutor Consumer
**File:** `TradeExecutor/trade_executor.py`  
**Purpose:** Executes trades based on signals from other agents  
**Subscribes to:**
- `kalshi:signals` - High-priority trading signals

**Key Features:**
- Kelly Criterion position sizing
- Order placement logic
- Trade confirmation handling

### RiskManager Consumer
**File:** `RiskManager/risk_manager.py`  
**Purpose:** Monitors portfolio risk and enforces limits  
**Subscribes to:**
- `kalshi:trades` - Monitor executed trades
- `kalshi:positions` - Track position changes

**Key Features:**
- Position limit enforcement
- Stop-loss monitoring
- Portfolio exposure management

### StrategyAnalyst Consumer
**File:** `StrategyAnalyst/strategy_analyst.py`  
**Purpose:** Analyzes market conditions and generates trading strategies  
**Subscribes to:**
- `kalshi:markets` - Market analysis
- `twitter:sentiment` - Sentiment analysis

## Configuration

### Environment Variables
```bash
REDIS_URL=redis://localhost:6379  # Redis connection URL
REDIS_HOST=localhost              # Alternative: specify host
REDIS_PORT=6379                   # Alternative: specify port
```

### Consumer Configuration
Each consumer can be configured with:
- `max_retries`: Maximum reconnection attempts (default: 5)
- `initial_retry_delay`: Initial backoff delay in seconds (default: 1.0)
- `max_retry_delay`: Maximum backoff delay (default: 60.0)
- `health_check_interval`: Health check frequency (default: 30.0)

## Usage

### Running Individual Consumers
```python
from agent_consumers.DataCoordinator.data_coordinator import DataCoordinatorRedisConsumer

async def main():
    consumer = DataCoordinatorRedisConsumer()
    await consumer.connect()
    await consumer.subscribe_to_channels([
        "kalshi:markets",
        "kalshi:signals"
    ])
    await consumer.start_consuming()
```

### Running All Consumers
See `examples/agent_redis_consumer.py` for running multiple consumers concurrently.

## Message Format

All Redis messages follow this structure:
```json
{
    "timestamp": "2024-01-01T00:00:00Z",
    "type": "market_update|trade|signal|game_event",
    "data": {
        "market_ticker": "NFL-WINNER-2024",
        "yes_price": 0.55,
        "no_price": 0.45,
        "volume": 10000,
        ...
    }
}
```

## Error Handling

### Connection Failures
- Automatic reconnection with exponential backoff
- Maximum 5 retry attempts by default
- Graceful degradation if connection cannot be established

### Message Processing Errors
- Errors logged but don't stop consumer
- Failed messages tracked in statistics
- Optional dead letter queue for investigation

## Performance Considerations

- Each consumer processes messages asynchronously
- Redis pub/sub can handle 10K+ messages/second
- Consumers implement backpressure control
- Health checks ensure system stability

## Testing

Run consumer tests:
```bash
pytest tests/test_agent_redis_integration.py
```

## Troubleshooting

### Consumer Not Receiving Messages
1. Check Redis connection: `redis-cli ping`
2. Verify channel subscriptions match publisher
3. Check logs for connection errors
4. Ensure Redis publisher is running

### High Memory Usage
1. Check message processing rate
2. Verify backpressure control is working
3. Consider increasing consumer instances
4. Review message retention policies

## Development Guidelines

When creating new consumers:
1. Inherit from `base_consumer.py`
2. Implement `process_message()` method
3. Define subscription channels
4. Add error handling and logging
5. Include health check logic
6. Write integration tests