# Data Pipeline

## Purpose
Real-time data ingestion and distribution system that collects market data from multiple sources (Kalshi, ESPN, Twitter) and publishes to Redis for agent consumption.

## Architecture

```
External APIs
     ↓
WebSocket Connections
     ↓
Stream Manager (Orchestration)
     ↓
Redis Publisher
     ↓
Redis Pub/Sub Channels
     ↓
Agent Consumers
```

## Directory Structure

```
data_pipeline/
├── orchestration/              # Stream coordination
│   ├── unified_stream_manager.py    # Central orchestrator
│   └── redis_event_publisher.py     # Redis publishing
├── streaming/                  # WebSocket clients
│   ├── websocket.py           # Kalshi WebSocket
│   └── handlers.py            # Message handlers
├── data_sources/              # API integrations
│   ├── kalshi/               # Kalshi API client
│   ├── espn/                # ESPN data client
│   └── twitter/             # Twitter stream
├── reliability/              # Resilience components
│   ├── circuit_breaker.py  # Failure protection
│   ├── backpressure_manager.py  # Flow control
│   ├── rate_limiter.py     # API rate limiting
│   ├── health_monitor.py   # Health checks
│   └── resilience_coordinator.py  # Coordination
├── config/                  # Configuration
│   └── settings.py         # Environment config
└── utils/                  # Utilities
    ├── logger.py          # Logging setup
    └── helpers.py         # Helper functions
```

## Core Components

### Unified Stream Manager
**File:** `orchestration/unified_stream_manager.py`  
**Purpose:** Coordinates all data streams and manages lifecycle

**Responsibilities:**
- Initialize WebSocket connections
- Handle reconnection logic
- Correlate events from multiple sources
- Publish to Redis channels
- Manage backpressure

**Key Features:**
- Automatic reconnection with exponential backoff
- Event buffering during disconnections
- Priority-based event processing
- Health monitoring
- Graceful degradation

### Redis Event Publisher
**File:** `orchestration/redis_event_publisher.py`  
**Purpose:** Bridge between WebSocket streams and Redis pub/sub

**Channels:**
```python
CHANNELS = {
    "kalshi:markets": "Market price updates",
    "kalshi:trades": "Executed trades",
    "kalshi:signals": "Trading signals",
    "espn:games": "Game events",
    "twitter:sentiment": "Social sentiment"
}
```

**Message Format:**
```json
{
    "timestamp": "2024-01-01T00:00:00Z",
    "source": "kalshi|espn|twitter",
    "type": "market_update|trade|game_event",
    "data": {
        // Source-specific data
    },
    "metadata": {
        "correlation_id": "uuid",
        "priority": "high|medium|low"
    }
}
```

### WebSocket Clients

#### Kalshi WebSocket
**File:** `streaming/websocket.py`  
**Features:**
- RSA-PSS authentication
- Market price streaming
- Trade execution updates
- Order book depth

**Subscriptions:**
```python
await ws.subscribe({
    "type": "subscribe",
    "channels": ["ticker", "trade", "orderbook"],
    "markets": ["NFL-*", "NBA-*"]
})
```

#### ESPN Stream
**File:** `data_sources/espn/stream.py`  
**Data Types:**
- Live scores
- Play-by-play events
- Team statistics
- Win probability

#### Twitter Stream
**File:** `data_sources/twitter/stream.py`  
**Features:**
- Keyword filtering
- Sentiment analysis
- Influencer detection
- Trend identification

### Reliability Components

#### Circuit Breaker
**File:** `reliability/circuit_breaker.py`  
**States:**
- **Closed:** Normal operation
- **Open:** Failing, reject requests
- **Half-Open:** Testing recovery

**Configuration:**
```python
CIRCUIT_BREAKER_CONFIG = {
    "failure_threshold": 5,
    "recovery_timeout": 60,
    "expected_exception": ConnectionError
}
```

#### Backpressure Manager
**File:** `reliability/backpressure_manager.py`  
**Pressure Levels:**
- **LOW:** Normal processing
- **MEDIUM:** Slow down non-critical
- **HIGH:** Process only critical
- **CRITICAL:** Emergency mode

#### Rate Limiter
**File:** `reliability/rate_limiter.py`  
**Features:**
- Hierarchical limits
- Token bucket algorithm
- Per-endpoint tracking
- Automatic throttling

#### Health Monitor
**File:** `reliability/health_monitor.py`  
**Checks:**
- WebSocket connection status
- Redis connectivity
- Message processing rate
- Error rates
- Memory usage

## Configuration

### Environment Variables
```bash
# Kalshi Configuration
KALSHI_API_KEY_ID=your_key_id
KALSHI_PRIVATE_KEY=your_private_key
KALSHI_ENVIRONMENT=demo  # or prod
KALSHI_WS_URL=wss://demo-api.kalshi.co/trade-api/ws/v2

# Redis Configuration  
REDIS_URL=redis://localhost:6379
REDIS_MAX_CONNECTIONS=10

# ESPN Configuration
ESPN_API_KEY=your_key  # Optional

# Twitter Configuration
TWITTER_BEARER_TOKEN=your_token  # Optional

# Reliability Settings
MAX_RECONNECT_ATTEMPTS=5
RECONNECT_DELAY_MS=1000
CIRCUIT_BREAKER_THRESHOLD=5
RATE_LIMIT_PER_SECOND=10
```

### Stream Configuration
```python
STREAM_CONFIG = {
    "kalshi": {
        "enabled": True,
        "markets": ["NFL-*", "NBA-*"],
        "channels": ["ticker", "trade"],
        "buffer_size": 1000
    },
    "espn": {
        "enabled": True,
        "sports": ["football", "basketball"],
        "update_interval": 5
    },
    "twitter": {
        "enabled": True,
        "keywords": ["NFL", "NBA", "betting"],
        "languages": ["en"],
        "sample_rate": 0.1
    }
}
```

## Data Flow

### 1. Data Ingestion
```python
# WebSocket receives market update
{
    "type": "ticker",
    "market": "NFL-CHAMPIONSHIP",
    "yes_price": 0.55,
    "no_price": 0.45,
    "volume": 10000
}
```

### 2. Event Processing
```python
# Stream manager enriches event
event = {
    "source": "kalshi",
    "timestamp": datetime.utcnow(),
    "data": market_update,
    "correlation_id": generate_id()
}

# Apply transformations
event = transform_event(event)
```

### 3. Redis Publishing
```python
# Publish to appropriate channel
channel = determine_channel(event)
await redis_publisher.publish(channel, event)
```

### 4. Consumer Reception
```python
# Agents receive via Redis subscription
async for message in redis.listen():
    await process_message(message)
```

## Performance Optimization

### Buffering Strategy
- Use event buffer for temporary disconnections
- Prioritize recent events
- Discard stale data
- Batch publish when possible

### Connection Management
- Connection pooling for Redis
- Persistent WebSocket connections
- Automatic reconnection
- Connection health monitoring

### Memory Management
- Limited buffer sizes
- Event expiration
- Garbage collection tuning
- Memory usage monitoring

## Error Handling

### WebSocket Errors
```python
try:
    await websocket.connect()
except WebSocketError as e:
    logger.error(f"WebSocket error: {e}")
    await exponential_backoff_retry()
```

### Redis Failures
```python
try:
    await redis.publish(channel, message)
except RedisError as e:
    logger.error(f"Redis error: {e}")
    await buffer_message(message)
    trigger_circuit_breaker()
```

### Data Source Errors
- Fallback to cached data
- Graceful degradation
- Alert on critical failures
- Automatic recovery attempts

## Monitoring

### Metrics
```python
METRICS = {
    "websocket_connections": gauge,
    "messages_received": counter,
    "messages_published": counter,
    "processing_latency": histogram,
    "error_rate": rate,
    "buffer_size": gauge
}
```

### Logging
```python
# Structured logging
logger.info(
    "Market update received",
    extra={
        "market": market_ticker,
        "price": yes_price,
        "source": "kalshi",
        "latency_ms": latency
    }
)
```

### Alerts
- WebSocket disconnection > 30 seconds
- Redis publish failures > 5%
- Buffer size > 80% capacity
- Processing latency > 1 second

## Testing

### Unit Tests
```bash
# Test individual components
pytest tests/test_stream_manager.py
pytest tests/test_redis_publisher.py
pytest tests/test_circuit_breaker.py
```

### Integration Tests
```bash
# Test full pipeline
pytest tests/test_data_pipeline_integration.py
```

### Load Testing
```python
# Simulate high message volume
async def load_test():
    for _ in range(10000):
        await publish_test_message()
```

## Troubleshooting

### WebSocket Not Connecting
1. Check API credentials
2. Verify network connectivity
3. Check WebSocket URL
4. Review authentication logs

### No Data in Redis
1. Verify Redis is running: `redis-cli ping`
2. Check publisher is connected
3. Review channel names
4. Check for backpressure

### High Latency
1. Check buffer sizes
2. Review processing logic
3. Monitor Redis performance
4. Check network latency

### Memory Issues
1. Review buffer configurations
2. Check for memory leaks
3. Monitor event expiration
4. Tune garbage collection

## Development

### Adding New Data Source
1. Create client in `data_sources/`
2. Implement stream adapter
3. Add to stream manager
4. Define Redis channels
5. Update configuration
6. Add error handling
7. Write tests

### Extending Pipeline
1. Identify extension point
2. Implement new processor
3. Update data flow
4. Add monitoring
5. Document changes
6. Test thoroughly

## Production Considerations

### Deployment
- Use environment-specific configs
- Implement health checks
- Set up monitoring
- Configure alerts
- Plan for scaling

### Scaling
- Horizontal scaling of consumers
- Redis cluster for pub/sub
- Load balancing WebSockets
- Partitioned data streams
- Caching strategies

### Security
- Secure API credentials
- Encrypt sensitive data
- Audit logging
- Access control
- Rate limiting