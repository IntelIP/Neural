# MVP Plan - Kalshi Trading System

## Core Objective
WebSockets → Redis → Agents → Kalshi Trading on Agentuity

## What We Have Built
- ✅ WebSocket infrastructure for Kalshi market data
- ✅ ESPN data streaming
- ✅ Circuit breaker protection
- ✅ Rate limiting & backpressure
- ✅ Stream orchestration (StreamManager)

## What We Need (Minimal Path to Trading)

### 1. Redis Integration (TODAY)
- Connect StreamManager to Redis pub/sub
- Publish market events to channels:
  - `kalshi:markets` - Market updates
  - `kalshi:trades` - Trade events
  - `espn:games` - Game updates

### 2. Agent Data Consumption (NEXT)
- Agents subscribe to Redis channels
- Process real-time events
- Make trading decisions

### 3. Trade Execution (FINAL)
- TradeExecutor places orders via Kalshi API
- RiskManager monitors positions
- Basic P&L tracking

## NO BLOAT - What We're NOT Building
- ❌ Complex order management systems
- ❌ Sophisticated backtesting
- ❌ Multiple strategy frameworks
- ❌ Advanced ML models
- ❌ Custom monitoring dashboards
- ❌ Mock/simulation systems

## Implementation Steps

### Step 1: Redis Publisher (Today)
```python
# StreamManager publishes to Redis
await redis.publish('kalshi:markets', market_data)
```

### Step 2: Agent Subscribers (Tomorrow)
```python
# Agents consume from Redis
async for message in redis.subscribe('kalshi:markets'):
    await process_market_data(message)
```

### Step 3: Execute Trades (Day After)
```python
# Simple trade execution
if signal > threshold:
    await kalshi_api.place_order(market, side, quantity)
```

## Success Criteria
1. WebSockets stream live data ✅
2. Data flows through Redis to agents
3. Agents analyze and identify opportunities
4. Trades execute on Kalshi
5. Positions are monitored
6. System runs on Agentuity

## Timeline
- Day 1-7: Infrastructure ✅
- Day 8-9: Redis integration
- Day 10-11: Agent consumption
- Day 12-13: Trade execution
- Day 14: Deploy on Agentuity

**Target: Trading within 7 days**

## Remember
- Keep it simple
- No mock systems
- Real data only
- Focus on execution
- Deploy and iterate