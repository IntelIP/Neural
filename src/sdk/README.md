# Data Source SDK

The Data Source SDK provides a plug-and-play framework for integrating real-time data feeds into the Neural Trading Platform.

## Architecture

```
Your Data Source → Adapter → StandardizedEvent → Redis Pub/Sub → Trading Agents
```

## Quick Example

```python
from src.sdk import SDKManager

# Initialize and start streaming
sdk = SDKManager(config_path="config/data_sources.yaml")
await sdk.initialize()
await sdk.start()

# Process events
async for event in sdk.get_events():
    if event.confidence > 0.8 and event.impact == "high":
        # High confidence event detected
        await process_trading_signal(event)
```

## Core Components

### 1. DataSourceAdapter (Base Class)

All adapters inherit from this abstract base class:

```python
class DataSourceAdapter(ABC):
    @abstractmethod
    async def connect() -> bool
    
    @abstractmethod
    async def stream() -> AsyncGenerator[StandardizedEvent, None]
    
    @abstractmethod
    def transform(raw_data) -> StandardizedEvent
```

### 2. StandardizedEvent

Unified event format across all sources:

```python
@dataclass
class StandardizedEvent:
    source: str              # "DraftKings", "Weather", etc.
    event_type: EventType    # ODDS_CHANGE, WEATHER_UPDATE, etc.
    timestamp: datetime
    game_id: Optional[str]
    data: Dict[str, Any]
    confidence: float        # 0.0 to 1.0
    impact: str             # "low", "medium", "high", "critical"
    metadata: Dict
```

### 3. SDKManager

Orchestrates multiple adapters:

```python
sdk = SDKManager()
sdk.add_adapter("draftkings", DraftKingsAdapter(config))
sdk.add_adapter("weather", WeatherAdapter(config))
await sdk.start()
```

## Built-in Adapters

### DraftKings (`adapters/draftkings.py`)
- **Purpose**: Track professional sportsbook odds
- **Events**: ODDS_CHANGE when lines move >3%
- **Config**: `poll_interval`, `min_odds_change`

### Weather (`adapters/weather.py`)
- **Purpose**: Monitor weather conditions at venues
- **Events**: WEATHER_UPDATE for wind/rain/visibility
- **Config**: `api_key`, `stadiums`, `thresholds`
- **API**: OpenWeatherMap (free tier: 60 calls/min)

### Reddit (`adapters/reddit.py`)
- **Purpose**: Analyze game thread sentiment
- **Events**: SENTIMENT_SHIFT, VOLUME_SPIKE
- **Config**: `client_id`, `client_secret`, `subreddits`

## Creating Custom Adapters

### Step 1: Create Adapter Class

```python
# src/sdk/adapters/my_source.py
from ..core.base_adapter import DataSourceAdapter, StandardizedEvent

class MySourceAdapter(DataSourceAdapter):
    def get_metadata(self):
        return DataSourceMetadata(
            name="MySource",
            version="1.0.0",
            source_type="api",
            latency_ms=100
        )
    
    async def connect(self):
        self.client = MyAPIClient(self.config['api_key'])
        return await self.client.connect()
    
    async def stream(self):
        while self.is_connected:
            data = await self.client.fetch()
            if self.is_significant(data):
                yield self.transform(data)
            await asyncio.sleep(self.config['interval'])
    
    def transform(self, raw_data):
        return StandardizedEvent(
            source="MySource",
            event_type=EventType.CUSTOM,
            data=raw_data,
            confidence=0.85
        )
```

### Step 2: Add to Configuration

```yaml
# config/data_sources.yaml
sources:
  - name: my_source
    enabled: true
    class: MySourceAdapter
    module: src.sdk.adapters.my_source
    config:
      api_key: ${MY_API_KEY}
      interval: 5
```

### Step 3: Test

```python
# Test your adapter
adapter = MySourceAdapter(config)
await adapter.connect()

async for event in adapter.stream():
    print(f"Event: {event.data}")
```

## Event Types

```python
class EventType(Enum):
    # Market events
    ODDS_CHANGE = "odds_change"
    MARKET_MOVEMENT = "market_movement"
    
    # Game events
    GAME_EVENT = "game_event"
    SCORE_UPDATE = "score_update"
    
    # Environmental
    WEATHER_UPDATE = "weather_update"
    
    # Social
    SENTIMENT_SHIFT = "sentiment_shift"
    VOLUME_SPIKE = "volume_spike"
    
    # News
    NEWS_ALERT = "news_alert"
    INJURY_UPDATE = "injury_update"
```

## Configuration

### Environment Variables

```bash
# .env file
OPENWEATHER_API_KEY=your_key
REDDIT_CLIENT_ID=your_id
REDDIT_CLIENT_SECRET=your_secret
```

### YAML Configuration

```yaml
sources:
  - name: weather
    enabled: true
    priority: 1  # Higher = processed first
    config:
      api_key: ${OPENWEATHER_API_KEY}
      update_interval: 300
      thresholds:
        wind_speed: 15
        precipitation: 0.1
```

## Testing

### Unit Test Template

```python
import pytest
from src.sdk.adapters.my_source import MySourceAdapter

@pytest.mark.asyncio
async def test_adapter():
    adapter = MySourceAdapter({"api_key": "test"})
    assert await adapter.connect()
    
    events = []
    async for event in adapter.stream():
        events.append(event)
        if len(events) >= 5:
            break
    
    assert len(events) == 5
    assert all(e.confidence > 0 for e in events)
```

### Performance Testing

```python
sdk = SDKManager()
results = await sdk.test_adapter("my_source", duration=60)

print(f"Events/sec: {results['events_per_second']}")
print(f"Avg latency: {results['avg_latency_ms']}ms")
print(f"Errors: {results['errors']}")
```

## Best Practices

1. **Error Handling**: Always wrap external calls in try/except
2. **Rate Limiting**: Respect API limits with built-in rate limiter
3. **Caching**: Cache responses when appropriate (30s default)
4. **Validation**: Validate data before creating events
5. **Logging**: Use adapter's logger for debugging

## Performance Metrics

- **Target Latency**: <500ms for critical sources
- **Event Rate**: Support 1000+ events/second
- **Reliability**: 99%+ uptime with auto-reconnect
- **Memory**: <100MB per adapter

## Support

- Full documentation: [docs/SDK_DOCUMENTATION.md](../../docs/SDK_DOCUMENTATION.md)
- Examples: [scripts/demo_sdk.py](../../scripts/demo_sdk.py)
- Tests: [scripts/test_weather_adapter.py](../../scripts/test_weather_adapter.py)