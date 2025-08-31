# Getting Started - Neural Trading Platform

## Prerequisites

Before you begin, ensure you have:

- Python 3.10 or higher
- Redis server installed
- Git for version control
- At least one API key (Kalshi, DraftKings, Reddit, or OpenWeatherMap)

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/IntelIP/Neural-Trading-Platform.git
cd Neural-Trading-Platform
```

### 2. Install Dependencies

```bash
# Using pip
pip install -r requirements.txt

# Or using uv (recommended for faster installs)
uv pip install -r requirements.txt
```

### 3. Set Up Redis

```bash
# macOS
brew install redis
brew services start redis

# Ubuntu/Debian
sudo apt-get install redis-server
sudo systemctl start redis

# Verify Redis is running
redis-cli ping
# Should return: PONG
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# Kalshi Trading (Required)
KALSHI_API_KEY_ID=your_key_id_here
KALSHI_API_KEY=your_api_key_here

# Data Sources (Optional - add what you have)
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_secret
OPENWEATHER_API_KEY=your_weather_api_key
```

---

## Quick Start: Monitor Your First Game

### Step 1: Configure a Game to Monitor

Edit `config/data_sources.yaml` to enable your available data sources:

```yaml
sources:
  # If you have a weather API key, enable this
  - name: weather
    enabled: true  # Change from false to true
    config:
      api_key: ${OPENWEATHER_API_KEY}
      
  # DraftKings is free, enable this
  - name: draftkings
    enabled: true
    config:
      sports:
        - NFL  # Pick the sport you want
```

### Step 2: Run the SDK Demo

This will show you data flowing through the system:

```bash
python scripts/demo_sdk.py
```

You'll see output like:
```
📡 Initializing data sources...
✅ Loaded 2 adapters:
  • DraftKings v1.0.0
    Type: sportsbook
    Latency: 500ms
    Reliability: 95.0%
  • Weather v1.0.0
    Type: environmental
    Latency: 2000ms
    Reliability: 99.0%

🚀 Starting data streams...
📊 Processing events (30 seconds)...

📊 Odds Change: Chiefs vs Bills
   Market: spread
   Change: 0.650 → 0.675
   Direction: up

🌤️ Weather Alert: Arrowhead Stadium
   Condition: high_wind
   Impact: ['passing_game', 'field_goals', 'punts']
```

### Step 3: Connect to Kalshi Markets

Once you have Kalshi API credentials, start the full platform:

```bash
# Start the unified stream manager
python -m src.data_pipeline.orchestration.unified_stream_manager

# In another terminal, start agent consumers
python examples/agent_redis_consumer.py all
```

---

## Understanding the Data Flow

Here's what happens when you run the platform:

```
1. Data Sources Connect
   ↓
2. Events Stream In (odds changes, weather updates, etc.)
   ↓
3. Stream Manager Standardizes Events
   ↓
4. Redis Distributes to Subscribers
   ↓
5. Agents Analyze for Opportunities
   ↓
6. Trading Signals Generated
   ↓
7. Orders Placed on Kalshi
```

### Real Example: Touchdown Scored

```
ESPN GameCast → "Touchdown Chiefs!"
     ↓ (100ms)
Stream Manager → StandardizedEvent(type=GAME_EVENT, impact=HIGH)
     ↓ (10ms)
Redis Pub/Sub → Channel: "espn:games"
     ↓ (5ms)
DataCoordinator → Detects Kalshi price hasn't moved
     ↓ (200ms)
StrategyAnalyst → Calculates expected +5% price move
     ↓ (100ms)
TradeExecutor → Places BUY order on Kalshi
     ↓ (200ms)
Total Time: ~615ms (before other traders react!)
```

---

## Basic Operations

### Starting Individual Components

```bash
# Just weather monitoring
python -m src.sdk.adapters.weather

# Just DraftKings odds
python -m src.sdk.adapters.draftkings

# Just Reddit sentiment
python -m src.sdk.adapters.reddit
```

### Monitoring System Health

```bash
# Check Redis messages
redis-cli MONITOR

# See active channels
redis-cli PUBSUB CHANNELS

# Count messages in a channel
redis-cli PUBSUB NUMSUB kalshi:markets
```

### Running Tests

```bash
# Test SDK functionality
python scripts/test_sdk.py

# Test specific adapter
python -c "
from src.sdk import SDKManager
import asyncio

async def test():
    sdk = SDKManager()
    await sdk.initialize()
    results = await sdk.test_adapter('draftkings', duration=10)
    print(f'Events received: {results[\"events_received\"]}')
    
asyncio.run(test())
"
```

---

## Common Configurations

### Focus on Specific Games

To monitor only specific games, configure your sources:

```yaml
# config/data_sources.yaml
sources:
  - name: draftkings
    config:
      sports:
        - NFL
      teams:  # Optional: focus on specific teams
        - "Kansas City Chiefs"
        - "Buffalo Bills"
```

### Adjust Update Frequencies

```yaml
sources:
  - name: draftkings
    config:
      poll_interval: 2  # Check every 2 seconds (was 5)
      
  - name: weather
    config:
      update_interval: 60  # Check every minute (was 5 minutes)
```

### Set Trading Thresholds

```yaml
# config/trading_config.yaml
thresholds:
  min_edge: 0.03  # 3% minimum advantage
  min_confidence: 0.75  # 75% confidence required
  max_position: 0.05  # 5% of capital max per trade
```

---

## Troubleshooting

### No Events Showing Up?

1. **Check Redis is running:**
```bash
redis-cli ping
# Should return PONG
```

2. **Verify API credentials:**
```bash
# Test weather API
curl "https://api.openweathermap.org/data/2.5/weather?q=London&appid=YOUR_KEY"
```

3. **Enable debug logging:**
```python
# Add to your script
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Events But No Trading?

1. **Check Kalshi connection:**
```bash
python examples/test_kalshi_websocket.py
```

2. **Verify market is open:**
- Sports markets only trade during games
- Check Kalshi website for active markets

3. **Review thresholds:**
- Your edge threshold might be too high
- Lower confidence requirement for testing

### High Latency?

1. **Check network:**
```bash
ping api.kalshi.com
```

2. **Reduce data sources:**
- Start with just one source
- Add more gradually

3. **Optimize Redis:**
```bash
# Check Redis latency
redis-cli --latency
```

---

## Next Steps

### 1. Add More Data Sources

Create your own adapter in under 50 lines:

```python
from src.sdk import DataSourceAdapter, StandardizedEvent, EventType

class MyAdapter(DataSourceAdapter):
    async def connect(self):
        # Your connection logic
        return True
    
    async def stream(self):
        while self.is_connected:
            # Your data fetching
            data = await self.fetch()
            yield StandardizedEvent(
                source="MySource",
                event_type=EventType.CUSTOM,
                data=data
            )
```

### 2. Customize Trading Logic

Modify agents to implement your strategy:

```python
# agents/StrategyAnalyst/agent.py
@tool
def my_custom_strategy(market_data, weather, sentiment):
    # Your alpha generation logic
    if weather['wind_speed'] > 20 and market_data['spread'] > 3:
        return {"action": "buy", "confidence": 0.85}
```

### 3. Set Up Production Monitoring

```bash
# Run with full logging
python examples/production_monitor.py

# Set up alerts
python scripts/setup_alerts.py --email your@email.com
```

---

## Essential Commands

```bash
# Development
python scripts/demo_sdk.py              # Test SDK
python examples/agent_redis_consumer.py # Run consumers
redis-cli MONITOR                       # Watch Redis

# Testing
pytest tests/                            # Run all tests
pytest tests/test_sdk.py -v            # Test SDK only

# Production
python -m src.data_pipeline.orchestration.unified_stream_manager  # Start streams
python agents/launch_all.py            # Start all agents
python scripts/monitor_health.py       # Health dashboard
```

---

## Getting Help

If you run into issues:

1. Check the logs in `logs/` directory
2. Review configuration in `config/` 
3. Run diagnostic script: `python scripts/diagnose.py`
4. See troubleshooting guide in docs

Remember: Start simple with one data source, verify it works, then add complexity!

---

## Ready to Trade?

You're now ready to:
- ✅ Stream real-time data
- ✅ Process events through the pipeline  
- ✅ Generate trading signals
- ✅ Execute on Kalshi markets

Next: Read [DATA_SOURCES_GUIDE.md](DATA_SOURCES_GUIDE.md) to understand each data source in detail.