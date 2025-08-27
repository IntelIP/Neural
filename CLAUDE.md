# CLAUDE.md

This file provides comprehensive guidance to Claude Code (claude.ai/code) when working with the Kalshi Trading Agent System.

## Project Overview

Autonomous multi-agent trading system for Kalshi sports event contracts using the Agentuity framework. The system uses real-time WebSocket streams, Redis pub/sub for data distribution, and sophisticated agents with LLM analysis to execute trades using the Kelly Criterion for optimal position sizing.

## System Architecture

### Data Flow Pipeline
```
[External APIs] → [WebSocket Streams] → [Redis Pub/Sub] → [Agent Consumers] → [Trading Decisions]
                           ↓                    ↓                 ↓
                    [Stream Manager]    [Event Publisher]   [Agentuity Agents]
                           ↓                    ↓                 ↓
                    [Reliability Layer]  [Redis Channels]   [Kalshi API]
```

### Directory Structure (Production Names)
```
.
├── agents/                    # Agentuity platform agents (high-level logic)
├── agent_consumers/           # Redis consumers (data processing)
├── data_pipeline/            # Data ingestion and streaming
│   ├── orchestration/        # Stream coordination
│   ├── streaming/           # WebSocket clients
│   ├── data_sources/        # API integrations
│   └── reliability/         # Resilience components
├── trading_logic/           # Trading algorithms and tools
├── tests/                   # Test suite
├── examples/               # Usage examples
└── docs/                   # Documentation
```

## Key Design Decisions

### 1. Redis Pub/Sub Architecture
**Decision:** Use Redis as central message broker  
**Rationale:** 
- Decouples data sources from consumers
- Enables horizontal scaling
- Provides buffering and persistence
- Supports multiple subscribers per channel

### 2. Separation of Concerns
**Decision:** Split agents into consumers and platform agents  
**Rationale:**
- `agent_consumers/`: Handle real-time data, no LLM calls
- `agents/`: Complex decisions with LLM analysis
- Clear performance boundaries
- Easier testing and debugging

### 3. Kelly Criterion with Safety Factor
**Decision:** Use 25% of Kelly for position sizing  
**Rationale:**
- Reduces risk of ruin
- Accounts for estimation errors
- Provides smoother equity curve
- Industry standard practice

## Common Development Tasks

### Adding a New Data Source

1. **Create WebSocket Client**
```python
# data_pipeline/streaming/new_source.py
class NewSourceWebSocket:
    async def connect(self):
        # Implementation
    
    async def subscribe(self, channels):
        # Implementation
```

2. **Add to Stream Manager**
```python
# data_pipeline/orchestration/unified_stream_manager.py
self.new_source_client = NewSourceWebSocket(config)
await self.new_source_client.connect()
```

3. **Define Redis Channel**
```python
# data_pipeline/orchestration/redis_event_publisher.py
await self.publish("newsource:events", event_data)
```

4. **Create Consumer**
```python
# agent_consumers/NewConsumer/consumer.py
class NewConsumer(BaseConsumer):
    async def process_message(self, channel, data):
        # Process events
```

### Implementing a Trading Strategy

1. **Define Strategy in Agent**
```python
# agents/StrategyAnalyst/agent.py
@tool
def analyze_opportunity(market_data, sentiment):
    # Strategy logic
    return signal
```

2. **Process Signals in Consumer**
```python
# agent_consumers/DataCoordinator/data_coordinator.py
if signal.strength > THRESHOLD:
    await self.publish_signal(signal)
```

3. **Execute Trade**
```python
# agents/TradeExecutor/agent.py
@tool
def execute_trade(signal):
    position_size = calculate_kelly_position(signal)
    order = place_order(market, side, position_size)
    return order
```

### Testing Workflows

1. **Test Redis Integration**
```bash
# Start Redis
redis-server

# Run integration tests
pytest tests/test_redis_integration.py
```

2. **Test Individual Agent**
```bash
# Start in dev mode
agentuity dev

# Test specific agent
agentuity agent test DataCoordinator
```

3. **Full System Test**
```bash
# Run all components
python examples/agent_redis_consumer.py all
```

## Code Style & Conventions

### Python Standards
- Use Python 3.10+ features
- Type hints for all functions
- Async/await for I/O operations
- Dataclasses for data structures

### Naming Conventions
```python
# Files and modules: snake_case
unified_stream_manager.py

# Classes: PascalCase
class DataCoordinator:
    pass

# Functions and variables: snake_case
def calculate_position_size():
    pass

# Constants: UPPER_SNAKE_CASE
MAX_POSITION_SIZE = 100
```

### Import Organization
```python
# Standard library
import os
import asyncio
from datetime import datetime

# Third-party
import redis
import websockets
from agentuity import Agent, tool

# Local application
from data_pipeline.orchestration import StreamManager
from agent_consumers.base_consumer import BaseConsumer
from trading_logic.kelly_tools import calculate_kelly
```

### Error Handling Pattern
```python
async def robust_operation():
    """Standard error handling pattern."""
    max_retries = 3
    retry_delay = 1.0
    
    for attempt in range(max_retries):
        try:
            result = await risky_operation()
            return result
        except TemporaryError as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                raise
        except PermanentError as e:
            logger.error(f"Permanent failure: {e}")
            raise
```

## Trading Logic Implementation

### Position Sizing
Always use Kelly Criterion with safety factors:
```python
# Never use full Kelly
position = kelly_fraction * 0.25  # 25% of Kelly

# Apply hard limits
position = min(position, MAX_POSITION_SIZE)
position = max(position, MIN_POSITION_SIZE)

# Check portfolio exposure
if total_exposure + position > MAX_PORTFOLIO_EXPOSURE:
    position = MAX_PORTFOLIO_EXPOSURE - total_exposure
```

### Risk Management Rules
1. **Stop Loss**: Always set at order time
2. **Position Limits**: Max 5% per position
3. **Correlation**: Reduce size for correlated bets
4. **Drawdown**: Stop at 20% daily loss

### Market Analysis
```python
# Check for arbitrage
if yes_price + no_price < 0.98:
    # Arbitrage opportunity exists
    
# Check liquidity
if volume < MIN_LIQUIDITY:
    # Skip illiquid markets
    
# Check spread
if abs(yes_price - no_price) > MAX_SPREAD:
    # Market too wide
```

## Performance Considerations

### WebSocket Management
- Maintain persistent connections
- Implement heartbeat/ping
- Buffer during disconnections
- Automatic reconnection

### Redis Optimization
- Use connection pooling
- Batch publish when possible
- Set appropriate TTLs
- Monitor memory usage

### Agent Performance
- Cache LLM responses
- Batch similar requests
- Use appropriate models
- Monitor token usage

## Security Practices

### API Keys
```bash
# Never hardcode keys
KALSHI_API_KEY_ID=xxx  # In .env file

# Use environment variables
key_id = os.getenv("KALSHI_API_KEY_ID")
```

### Data Privacy
- Don't log sensitive data
- Sanitize user inputs
- Encrypt stored credentials
- Audit access logs

## Troubleshooting Guide

### Common Issues

#### WebSocket Disconnections
```python
# Check: Connection status
logger.info(f"WebSocket state: {ws.state}")

# Solution: Automatic reconnection
await exponential_backoff_retry()
```

#### Redis Pub/Sub Issues
```bash
# Check: Redis connectivity
redis-cli ping

# Check: Active subscriptions
redis-cli PUBSUB CHANNELS

# Monitor: Message flow
redis-cli MONITOR
```

#### No Trading Signals
```python
# Check: Data pipeline
assert stream_manager.is_connected()

# Check: Agent consumers
assert len(active_consumers) > 0

# Check: Signal thresholds
logger.info(f"Signal threshold: {SIGNAL_THRESHOLD}")
```

#### High Latency
```bash
# Profile: Message processing
python -m cProfile -s cumulative server.py

# Check: Redis performance
redis-cli --latency

# Monitor: System resources
htop
```

## Deployment Checklist

### Pre-Deployment
- [ ] All tests passing
- [ ] Environment variables set
- [ ] Redis configured
- [ ] API credentials valid
- [ ] Risk limits configured

### Deployment Steps
```bash
# 1. Set production environment
export ENVIRONMENT=production

# 2. Verify configuration
agentuity config verify

# 3. Deploy agents
agentuity deploy

# 4. Monitor logs
agentuity logs --follow
```

### Post-Deployment
- [ ] Verify WebSocket connections
- [ ] Check Redis pub/sub
- [ ] Confirm agent health
- [ ] Monitor first trades
- [ ] Review error logs

## Performance Benchmarks

### Expected Metrics
- WebSocket latency: < 100ms
- Redis publish: < 10ms
- Agent processing: < 500ms
- End-to-end: < 1 second

### Capacity Planning
- Redis: 10K messages/second
- WebSocket: 1K updates/second
- Agents: 100 decisions/minute
- Trades: 20 per day maximum

## Emergency Procedures

### Market Halt
```python
# Immediate stop
await risk_manager.emergency_stop()

# Cancel all orders
await trade_executor.cancel_all_orders()

# Notify team
await send_alert("Trading halted")
```

### Data Loss
```python
# Switch to cached data
await use_fallback_data()

# Reduce position sizes
KELLY_FRACTION *= 0.5

# Log incident
logger.critical("Data loss detected")
```

## Testing Strategy

### Unit Tests
Test individual components in isolation:
```bash
pytest tests/unit/ -v
```

### Integration Tests
Test component interactions:
```bash
pytest tests/integration/ -v
```

### End-to-End Tests
Test full system flow:
```bash
pytest tests/e2e/ -v
```

### Performance Tests
```bash
# Load testing
locust -f tests/load/locustfile.py

# Stress testing
python tests/stress/stress_test.py
```

## Important Notes

### Do's
- ✅ Always use try/except for external calls
- ✅ Log all trading decisions
- ✅ Validate data before processing
- ✅ Use type hints
- ✅ Write tests for new features

### Don'ts
- ❌ Never commit secrets
- ❌ Don't bypass risk checks
- ❌ Avoid synchronous I/O
- ❌ Don't ignore error logs
- ❌ Never use full Kelly

## Quick Commands Reference

```bash
# Development
agentuity dev                    # Start dev server
uv run server.py                 # Run directly
uv run pytest tests/            # Run tests

# Redis
redis-cli ping                   # Check Redis
redis-cli MONITOR               # Monitor messages
redis-cli FLUSHALL              # Clear all data

# Deployment
agentuity deploy                # Deploy to cloud
agentuity logs DataCoordinator  # View agent logs
agentuity status                # Check deployment

# Monitoring
python examples/agent_redis_consumer.py all  # Run all consumers
python examples/test_kalshi_websocket.py    # Test WebSocket
```

## Contact & Support

For questions about:
- **Architecture**: Review docs/ARCHITECTURE.md
- **Agents**: Check agents/README.md
- **Trading Logic**: See trading_logic/README.md
- **Data Pipeline**: Read data_pipeline/README.md

When debugging:
1. Check logs first
2. Verify configuration
3. Test components individually
4. Review recent changes
5. Check system resources