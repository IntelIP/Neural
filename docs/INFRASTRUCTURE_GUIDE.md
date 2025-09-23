# Neural SDK Infrastructure Guide

## Overview
This guide documents the complete trading infrastructure for the Neural SDK, including market data collection, order execution, and real-time streaming capabilities.

## Infrastructure Components

### 1. REST API (Market Data) ✅ Working
- **Purpose**: Fetch market data, prices, and trading information
- **Latency**: 1 second polling intervals
- **Authentication**: Standard API key with RSA signature
- **Status**: Fully operational

### 2. FIX API (Order Execution) ✅ Working
- **Purpose**: Ultra-fast order placement and execution
- **Latency**: 5-10ms
- **Protocol**: FIX 5.0 SP2
- **Connection**: `fix.elections.kalshi.com:8228`
- **Status**: Fully operational

### 3. WebSocket API ❌ Requires Special Permissions
- **Purpose**: Real-time market data streaming
- **Issue**: Returns 403 Forbidden - requires additional permissions
- **Workaround**: Use REST API polling instead

## Quick Start

### Prerequisites
1. Kalshi API credentials:
   ```bash
   # In .env or environment variables
   KALSHI_API_KEY_ID=your-api-key-id
   KALSHI_PRIVATE_KEY_PATH=/path/to/private_key.pem
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Testing Infrastructure

Run the complete infrastructure test:
```bash
python test_infrastructure_final.py
```

Expected output:
```
✅ REST API (Market Data)    - Primary data source
✅ FIX API (Order Execution) - Ultra-fast trading
❌ WebSocket (Streaming)     - Optional - needs permissions
```

## Market Data Collection

### REST API Polling
The REST API provides reliable market data through polling:

```python
from neural.trading.rest_streaming import RESTStreamingClient

# Create streaming client
client = RESTStreamingClient(
    poll_interval=1.0,  # Poll every second
    on_market_update=handle_update
)

# Subscribe to markets
async with client:
    await client.subscribe([
        "KXNFLGAME-25SEP25SEAARI-SEA",
        "KXNFLGAME-25SEP25SEAARI-ARI"
    ])
    await asyncio.sleep(30)  # Stream for 30 seconds
```

### Direct Market Fetching
For simple market queries:

```python
from neural.data_collection import get_game_markets

# Fetch markets for a game
markets = await get_game_markets("KXNFLGAME-25SEP25SEAARI")
```

## Order Execution

### FIX API Setup
The FIX API provides millisecond-latency order execution:

```python
from neural.trading.fix import KalshiFIXClient, FIXConnectionConfig

# Configure FIX connection
config = FIXConnectionConfig(
    heartbeat_interval=30,
    reset_seq_num=True
)

# Create client
client = KalshiFIXClient(config=config)

# Connect and place order
await client.connect()
await client.new_order_single(
    cl_order_id="ORDER_001",
    symbol="KXNFLGAME-25SEP25SEAARI-SEA",
    side="buy",
    quantity=1,
    price=50,  # 50 cents
    order_type="limit",
    time_in_force="ioc"
)
```

## Complete Trading Pipeline

### Hybrid Infrastructure (REST + FIX)
Combines REST polling for data with FIX for execution:

```python
# See test_rest_fix_infrastructure.py for complete example
# This provides:
# - Market data every second via REST
# - Order execution in 5-10ms via FIX
# - Signal generation from price movements
# - Complete pipeline from data to execution
```

## Sports Markets

### Ticker Format
Sports markets follow this pattern:
- **NFL**: `KXNFLGAME-[DATE][TEAMS]-[TEAM]`
- **NBA**: `KXNBA-[DATE][TEAMS]-[TEAM]`
- **MLB**: `KXMLB-[DATE][TEAMS]-[TEAM]`

Example: `KXNFLGAME-25SEP25SEAARI-SEA`
- Sport: NFL
- Date: September 25, 2025
- Teams: Seattle at Arizona
- Market: Seattle to win

### Finding Games
```python
from neural.data_collection import KalshiMarketsSource

source = KalshiMarketsSource()
markets = source.get_sports_markets("NFL")  # Get all NFL markets
```

## Troubleshooting

### WebSocket 403 Forbidden
- **Issue**: WebSocket requires special permissions
- **Solution**: Use REST API polling (1-second updates are sufficient for most strategies)
- **To Enable**: Contact Kalshi support to request WebSocket access for your API key

### Authentication Errors
- **Check**: API key and private key are correctly set
- **Verify**: REST API works with `test_auth_verify.py`
- **Fix**: Ensure private key has correct permissions (600)

### Market Not Found
- **Check**: Event ticker format is correct
- **Verify**: Game hasn't concluded
- **Fix**: Don't filter by status="open" (sports use "active")

## Performance Benchmarks

| Component | Latency | Update Rate | Reliability |
|-----------|---------|-------------|-------------|
| REST API | 100-200ms | 1/second | 99.9% |
| FIX API | 5-10ms | Real-time | 99.9% |
| WebSocket | 50-100ms | Real-time | N/A (requires permissions) |

## Best Practices

1. **Use REST for market data** - Reliable and no special permissions needed
2. **Use FIX for execution** - Ultra-fast order placement
3. **Poll efficiently** - 1-second intervals are optimal
4. **Handle errors gracefully** - Implement retry logic
5. **Monitor spreads** - Look for tight spreads for better fills

## Example Trading Strategy

See `neural/analysis/strategies/` for complete strategy implementations:
- Mean reversion
- Momentum trading
- Arbitrage detection
- News-based trading

## Support

For issues or questions:
1. Check this documentation
2. Run `test_infrastructure_final.py` to verify setup
3. Review test files in `tests/` directory
4. Contact Kalshi support for API-specific issues