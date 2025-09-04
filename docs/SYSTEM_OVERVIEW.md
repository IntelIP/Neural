# System Overview - Neural Trading Platform

## Purpose

This platform is an automated trading system for Neural sports prediction markets. It ingests real-time data from multiple sources (game stats, odds, sentiment, weather), correlates events across sources to identify mispricing, and executes trades faster than human reaction time.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION LAYER                         │
│                                                                      │
│  NeuralWebSocket    ESPN GameCast    DraftKings    Reddit    Weather│
│       ↓                  ↓              ↓           ↓          ↓    │
│  [Market Prices]   [Game Events]   [Pro Odds]  [Sentiment] [Conditions]│
└─────────────────┬───────────────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                     STREAM PROCESSING LAYER                         │
│                                                                      │
│  UnifiedStreamManager                                               │
│  ├── Event Correlation (10-second window)                          │
│  ├── Time Synchronization                                          │
│  └── State Management                                              │
└─────────────────┬───────────────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      DISTRIBUTION LAYER                             │
│                                                                      │
│  Redis Pub/Sub Hub                                                  │
│  ├── Channel: neural:markets                                       │
│  ├── Channel: espn:games                                           │
│  ├── Channel: signals:opportunities                                │
│  └── 10,000 msg/sec throughput                                     │
└─────────────────┬───────────────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      DECISION LAYER                                 │
│                                                                      │
│  Always-On Agents              On-Demand Agents                    │
│  ├── DataCoordinator          ├── StrategyAnalyst (GPT-4)         │
│  ├── PortfolioMonitor         ├── TradeExecutor                   │
│  └── RiskManager              └── GameAnalyst                     │
└─────────────────┬───────────────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      EXECUTION LAYER                                │
│                                                                      │
│  Neural Trading API                                                 │
│  ├── Order Management                                              │
│  ├── Position Tracking                                             │
│  └── Settlement                                                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Component Responsibilities

### **Data Ingestion Layer**

Each data source provides unique information:

| Source | Data Type | Latency | Purpose |
|--------|-----------|---------|---------|
| **Neural WebSocket** | Market prices, trades | ~50ms | Current betting prices |
| **ESPN GameCast** | Play-by-play, scores | ~1-2s | Game events |
| **DraftKings API** | Professional odds | ~500ms | Sharp money movements |
| **Reddit Streams** | Game thread comments | ~2-5s | Crowd sentiment |
| **Weather API** | Conditions at stadium | 5min | Environmental factors |

### **Stream Processing Layer**

The `UnifiedStreamManager` (`src/data_pipeline/orchestration/unified_stream_manager.py`) handles:

1. **Event Correlation**: Links related events across sources within a time window
2. **State Management**: Maintains current state of all tracked games
3. **Buffering**: Handles backpressure when downstream can't keep up
4. **Transformation**: Converts source-specific formats to `StandardizedEvent`

### **Distribution Layer**

Redis Pub/Sub acts as our message bus:

```python
# Publishing example
redis_client.publish("neural:markets", {
    "market": "NFL-KC-BUF",
    "yes_price": 0.65,
    "timestamp": "2024-01-21T18:30:45Z"
})

# Subscribing example
pubsub.subscribe("neural:markets")
for message in pubsub.listen():
    process_market_update(message)
```

### **Decision Layer**

Two types of agents:

**Always-On Agents** (No LLM calls, pure logic):
- `DataCoordinator`: Correlates events, maintains market context
- `PortfolioMonitor`: Tracks positions, P&L, risk metrics
- `RiskManager`: Enforces limits, stops losses

**On-Demand Agents** (LLM-powered analysis):
- `StrategyAnalyst`: Complex pattern recognition
- `TradeExecutor`: Execution optimization
- `GameAnalyst`: Deep game situation analysis

### **Execution Layer**

Interfaces with Neural API for trading:
- Places market/limit orders
- Manages existing positions
- Handles order fills and cancellations

---

## Data Flow Example

Let's trace a touchdown event through the system:

```
1. ESPN GameCast (T+0.0s)
   └── "Touchdown Chiefs! Mahomes 45-yard pass"
   
2. UnifiedStreamManager (T+0.1s)
   └── Creates StandardizedEvent(type=GAME_EVENT, impact=HIGH)
   
3. Redis Pub/Sub (T+0.2s)
   └── Publishes to "espn:games" channel
   
4. DataCoordinator Agent (T+0.3s)
   ├── Correlates with current Neural price (0.65)
   ├── Detects price hasn't moved yet
   └── Publishes opportunity signal
   
5. StrategyAnalyst Agent (T+0.5s)
   ├── Calculates expected price movement (+0.05)
   ├── Confidence: 85%
   └── Generates BUY signal
   
6. TradeExecutor Agent (T+0.7s)
   ├── Kelly position sizing: $1,250
   ├── Places order: BUY 1923 shares @ 0.65
   └── Order filled (T+1.2s)
   
7. Market Adjustment (T+15s)
   └── Neural price moves to 0.70 (+7.7% profit)
```

---

## Performance Characteristics

### **Latency Budget**

Total time from event to trade execution:

| Stage | Latency | Cumulative |
|-------|---------|------------|
| Data ingestion | 50-2000ms | 2000ms |
| Stream processing | 10ms | 2010ms |
| Redis pub/sub | 5ms | 2015ms |
| Agent decision | 100-500ms | 2515ms |
| Order execution | 200ms | **2715ms** |

**Target**: Sub-3 second total latency

### **Throughput**

- **Events/second**: 10,000 (burst)
- **Decisions/minute**: 100
- **Concurrent markets**: 500+
- **Orders/day**: 1,000 (regulatory limit)

### **Reliability**

- WebSocket auto-reconnection
- Redis persistence
- Agent health monitoring
- Circuit breakers for risk

---

## Why This Architecture?

### **Why Redis Pub/Sub?**

1. **Decoupling**: Data sources don't need to know about consumers
2. **Fan-out**: Multiple agents can subscribe to same data
3. **Buffering**: Handles temporary slowdowns
4. **Persistence**: Can replay missed messages

### **Why Separate Always-On vs On-Demand Agents?**

1. **Performance**: Always-on agents have no LLM latency
2. **Cost**: Reduce LLM API calls
3. **Reliability**: Core functions work without external APIs
4. **Specialization**: Different tools for different jobs

### **Why Multiple Data Sources?**

Each source provides unique alpha:
- **Neural**: Current market consensus
- **ESPN**: Ground truth of game events  
- **DraftKings**: Sharp money movements
- **Reddit**: Retail sentiment extremes
- **Weather**: Undervalued environmental factors

---

## System States

The platform operates in several modes:

### **1. Market Hours (Active Trading)**
- All data sources active
- Agents making decisions
- Orders being placed

### **2. Pre-Game (Preparation)**
- Loading historical data
- Calibrating thresholds
- Pre-computing correlations

### **3. Post-Game (Settlement)**
- Positions settling
- Performance analysis
- Strategy adjustment

### **4. Maintenance Mode**
- Data sources connected
- No trading allowed
- System updates

---

## Key Metrics

Monitor these to ensure system health:

```python
# System Health
websocket_connections: int  # Should equal number of sources
redis_queue_depth: int      # Should be <1000
agent_response_time: float  # Should be <500ms

# Trading Performance  
win_rate: float            # Target >65%
sharpe_ratio: float        # Target >2.0
average_edge: float        # Target >3%

# Risk Metrics
max_position_size: float   # Never exceed 5% of capital
daily_drawdown: float      # Stop at -20%
correlation_risk: float    # Reduce when >0.7
```

---

## Extension Points

The system is designed to be extended:

### **Adding Data Sources**

Implement the `DataSourceAdapter` interface:

```python
class MyDataSource(DataSourceAdapter):
    async def connect(self) -> bool
    async def stream(self) -> AsyncGenerator[StandardizedEvent, None]
    def transform(self, raw_data) -> StandardizedEvent
```

### **Adding Trading Strategies**

Create new agents or modify existing ones:

```python
class MyStrategy(BaseAgent):
    def analyze(self, market_data) -> Signal
    def calculate_position(self, signal) -> float
```

### **Adding Risk Rules**

Extend the risk manager:

```python
risk_manager.add_rule(
    name="correlation_limit",
    condition=lambda: portfolio.correlation > 0.8,
    action=lambda: reduce_all_positions(0.5)
)
```

---

## Common Patterns

### **Event Correlation Pattern**

```python
# Correlate events within time window
window = []
async for event in stream:
    window.append(event)
    window = [e for e in window if e.timestamp > now - timedelta(seconds=10)]
    
    if detect_opportunity(window):
        yield trading_signal(window)
```

### **Backpressure Handling**

```python
# Prevent overwhelming downstream
if redis_queue.size() > MAX_QUEUE:
    await rate_limiter.throttle()
```

### **State Management**

```python
# Maintain game state across events
game_states = {}
async for event in stream:
    game_id = event.game_id
    game_states[game_id] = update_state(game_states.get(game_id), event)
```

---

## Next Steps

1. Read [GETTING_STARTED.md](GETTING_STARTED.md) to set up the system
2. Configure data sources in [DATA_SOURCES_GUIDE.md](DATA_SOURCES_GUIDE.md)
3. Set up game monitoring with [GAME_CONFIGURATION_GUIDE.md](GAME_CONFIGURATION_GUIDE.md)
4. Understand trading logic in [TRADING_LOGIC.md](TRADING_LOGIC.md)