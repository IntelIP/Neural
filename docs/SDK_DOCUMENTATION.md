# 📚 Neural Trading Platform SDK Documentation

## Overview

The Neural Trading Platform SDK enables easy integration of custom data sources for algorithmic trading. With just a few lines of code, you can connect any websocket, API, or database to the platform.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Concepts](#core-concepts)
3. [Creating Adapters](#creating-adapters)
4. [Configuration](#configuration)
5. [Built-in Adapters](#built-in-adapters)
6. [Testing](#testing)
7. [Best Practices](#best-practices)
8. [API Reference](#api-reference)

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/IntelIP/Neural-Trading-Platform.git
cd Neural-Trading-Platform

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from src.sdk import SDKManager

# Initialize SDK
sdk = SDKManager(config_path="config/data_sources.yaml")

# Load adapters
await sdk.initialize()

# Start streaming
await sdk.start()

# Process events
async for event in sdk.get_events():
    print(f"Event: {event.source} - {event.event_type}")
    # Your trading logic here
```

### Running the Demo

```bash
python scripts/demo_sdk.py
```

---

## Core Concepts

### StandardizedEvent

All data sources produce `StandardizedEvent` objects with a consistent structure:

```python
@dataclass
class StandardizedEvent:
    source: str              # Data source name
    event_type: EventType    # Type of event
    timestamp: datetime      # When event occurred
    game_id: Optional[str]   # Game/market identifier
    data: Dict[str, Any]     # Event data
    confidence: float        # 0.0 to 1.0
    impact: str             # 'low', 'medium', 'high', 'critical'
    metadata: Dict          # Additional context
```

### Event Types

```python
class EventType(Enum):
    ODDS_CHANGE = "odds_change"           # Sportsbook line movement
    SENTIMENT_SHIFT = "sentiment_shift"   # Social sentiment change
    GAME_EVENT = "game_event"            # In-game occurrence
    WEATHER_UPDATE = "weather_update"    # Weather conditions
    VOLUME_SPIKE = "volume_spike"        # Unusual activity
    NEWS_ALERT = "news_alert"            # Breaking news
```

---

## Creating Adapters

### Basic Adapter Structure

```python
from src.sdk import DataSourceAdapter, DataSourceMetadata, StandardizedEvent

class MyCustomAdapter(DataSourceAdapter):
    """Your custom data source adapter"""
    
    def get_metadata(self) -> DataSourceMetadata:
        """Describe your adapter"""
        return DataSourceMetadata(
            name="MySource",
            version="1.0.0",
            author="Your Name",
            description="Description of data source",
            source_type="api",  # 'api', 'websocket', 'database'
            latency_ms=100,
            reliability=0.99,
            requires_auth=True
        )
    
    async def connect(self) -> bool:
        """Establish connection"""
        # Your connection logic
        return True
    
    async def disconnect(self) -> None:
        """Close connection"""
        # Cleanup logic
        pass
    
    async def validate_connection(self) -> bool:
        """Check if still connected"""
        return True
    
    async def stream(self) -> AsyncGenerator[StandardizedEvent, None]:
        """Stream events"""
        while self.is_connected:
            # Fetch data
            data = await self.fetch_data()
            
            # Transform to event
            event = self.transform(data)
            if event:
                yield event
    
    def transform(self, raw_data: Any) -> Optional[StandardizedEvent]:
        """Convert raw data to standardized event"""
        return StandardizedEvent(
            source=self.metadata.name,
            event_type=EventType.MARKET_MOVEMENT,
            timestamp=datetime.now(),
            data={"price": raw_data['price']},
            confidence=0.95,
            impact="medium"
        )
```

### WebSocket Example

```python
import websockets

class WebSocketAdapter(DataSourceAdapter):
    async def connect(self):
        self.ws = await websockets.connect(self.config['url'])
        return True
    
    async def stream(self):
        async for message in self.ws:
            data = json.loads(message)
            yield self.transform(data)
```

### REST API Example

```python
import aiohttp

class APIAdapter(DataSourceAdapter):
    async def connect(self):
        self.session = aiohttp.ClientSession()
        return True
    
    async def stream(self):
        while self.is_connected:
            async with self.session.get(self.config['endpoint']) as resp:
                data = await resp.json()
                yield self.transform(data)
            await asyncio.sleep(self.config['poll_interval'])
```

---

## Configuration

### YAML Configuration

Create `config/data_sources.yaml`:

```yaml
sources:
  # Your custom adapter
  - name: my_source
    enabled: true
    class: MyCustomAdapter
    module: adapters.my_adapter
    priority: 1
    config:
      url: "wss://api.example.com/stream"
      api_key: ${MY_API_KEY}  # Environment variable
      symbols: ["AAPL", "GOOGL"]
  
  # Built-in DraftKings adapter
  - name: draftkings
    enabled: true
    class: DraftKingsAdapter
    module: src.sdk.adapters.draftkings
    config:
      sports: ["NFL", "NBA"]
      poll_interval: 5
```

### Environment Variables

```bash
# .env file
MY_API_KEY=your_api_key_here
REDDIT_CLIENT_ID=reddit_client_id
REDDIT_CLIENT_SECRET=reddit_secret
OPENWEATHER_API_KEY=weather_api_key
```

---

## Built-in Adapters

### DraftKings Adapter

**Purpose**: Monitor professional sportsbook odds for arbitrage opportunities

```yaml
config:
  sports: ["NFL", "NBA", "MLB"]
  poll_interval: 5  # seconds
  min_odds_change: 0.03  # 3% minimum
```

**Events Generated**:
- `ODDS_CHANGE`: When lines move significantly
- Impact: Correlate with Kalshi prices

### Reddit Adapter

**Purpose**: Analyze game thread sentiment and reactions

```yaml
config:
  client_id: ${REDDIT_CLIENT_ID}
  client_secret: ${REDDIT_CLIENT_SECRET}
  subreddits: ["nfl", "nba"]
  min_comment_karma: 5
```

**Events Generated**:
- `SENTIMENT_SHIFT`: Major sentiment changes
- `VOLUME_SPIKE`: Unusual comment activity
- `SOCIAL_MENTION`: Keyword detection (injury, touchdown, etc.)

### Weather Adapter

**Purpose**: Track weather conditions affecting games

```yaml
config:
  api_key: ${OPENWEATHER_API_KEY}
  update_interval: 300  # 5 minutes
  thresholds:
    wind_speed: 15  # mph
    precipitation: 0.1  # inches/hour
```

**Events Generated**:
- `WEATHER_UPDATE`: Significant weather changes
- Impact: Wind affects passing, rain affects scoring

---

## Testing

### Unit Testing Your Adapter

```python
import pytest
from your_adapter import YourAdapter

@pytest.mark.asyncio
async def test_adapter_connection():
    adapter = YourAdapter({"api_key": "test"})
    assert await adapter.connect() == True

@pytest.mark.asyncio
async def test_event_generation():
    adapter = YourAdapter({})
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
# Use built-in testing
sdk = SDKManager()
results = await sdk.test_adapter("your_adapter", duration=60)

print(f"Events/sec: {results['events_per_second']}")
print(f"Latency: {results['avg_latency_ms']}ms")
print(f"Errors: {results['errors']}")
```

---

## Best Practices

### 1. Error Handling

```python
async def stream(self):
    while self.is_connected:
        try:
            data = await self.fetch_data()
            yield self.transform(data)
        except Exception as e:
            self.logger.error(f"Stream error: {e}")
            self._increment_error_count()
            await asyncio.sleep(5)  # Back off on error
```

### 2. Rate Limiting

```python
from src.sdk.core.base_adapter import RateLimiter

class MyAdapter(DataSourceAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.rate_limiter = RateLimiter(calls_per_second=10)
    
    async def fetch_data(self):
        await self.rate_limiter.acquire()
        # Make API call
```

### 3. Data Validation

```python
def transform(self, raw_data):
    # Validate required fields
    if not all(k in raw_data for k in ['price', 'timestamp']):
        self.logger.warning("Missing required fields")
        return None
    
    # Validate data types
    try:
        price = float(raw_data['price'])
        timestamp = datetime.fromisoformat(raw_data['timestamp'])
    except (ValueError, TypeError) as e:
        self.logger.error(f"Invalid data format: {e}")
        return None
    
    return StandardizedEvent(...)
```

### 4. Connection Resilience

```python
async def stream(self):
    while self.is_connected:
        try:
            async for data in self.websocket:
                yield self.transform(data)
        except ConnectionError:
            self.logger.warning("Connection lost, reconnecting...")
            await self.connect()
```

---

## API Reference

### SDKManager

```python
class SDKManager:
    async def initialize() -> None
        """Load and initialize all configured adapters"""
    
    async def start() -> None
        """Start streaming from all adapters"""
    
    async def stop() -> None
        """Stop all adapters and cleanup"""
    
    async def get_events() -> AsyncGenerator[StandardizedEvent, None]
        """Get unified event stream from all sources"""
    
    def get_adapter(name: str) -> Optional[DataSourceAdapter]
        """Get specific adapter by name"""
    
    async def health_check() -> Dict[str, Any]
        """Check health of all adapters"""
    
    async def test_adapter(name: str, duration: int) -> Dict
        """Test specific adapter performance"""
```

### DataSourceAdapter

```python
class DataSourceAdapter(ABC):
    @abstractmethod
    def get_metadata() -> DataSourceMetadata
        """Return adapter metadata"""
    
    @abstractmethod
    async def connect() -> bool
        """Establish connection"""
    
    @abstractmethod
    async def stream() -> AsyncGenerator[StandardizedEvent, None]
        """Stream events"""
    
    @abstractmethod
    def transform(raw_data: Any) -> Optional[StandardizedEvent]
        """Transform raw data to event"""
    
    async def health_check() -> Dict[str, Any]
        """Perform health check"""
    
    @property
    def statistics() -> Dict[str, Any]
        """Get adapter statistics"""
```

---

## Examples

### Correlating Multiple Sources

```python
from collections import defaultdict

class MultiSourceCorrelator:
    def __init__(self, sdk: SDKManager):
        self.sdk = sdk
        self.events_by_game = defaultdict(list)
    
    async def correlate(self):
        async for event in self.sdk.get_events():
            game_id = event.game_id
            self.events_by_game[game_id].append(event)
            
            # Check for correlated signals
            recent = self.events_by_game[game_id][-10:]
            if self.detect_opportunity(recent):
                await self.execute_trade(game_id, recent)
    
    def detect_opportunity(self, events):
        # Check if DraftKings moved but Kalshi hasn't
        dk_events = [e for e in events if e.source == "DraftKings"]
        kalshi_events = [e for e in events if e.source == "Kalshi"]
        
        if dk_events and not kalshi_events:
            return True  # Opportunity!
```

### Custom Aggregation

```python
class WindowedAggregator:
    def __init__(self, window_seconds=60):
        self.window = window_seconds
        self.events = deque()
    
    async def aggregate(self, sdk):
        async for event in sdk.get_events():
            # Add to window
            self.events.append(event)
            
            # Remove old events
            cutoff = datetime.now() - timedelta(seconds=self.window)
            while self.events and self.events[0].timestamp < cutoff:
                self.events.popleft()
            
            # Calculate aggregates
            if len(self.events) > 10:
                sentiment = self.calculate_sentiment()
                momentum = self.calculate_momentum()
                yield {
                    "sentiment": sentiment,
                    "momentum": momentum,
                    "event_rate": len(self.events) / self.window
                }
```

---

## Troubleshooting

### Common Issues

**Adapter not loading**:
- Check module path in config
- Verify class name matches
- Check for import errors

**No events received**:
- Verify API credentials
- Check rate limits
- Enable debug logging

**High latency**:
- Reduce poll interval
- Use WebSocket instead of polling
- Check network connectivity

### Debug Logging

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Or for specific adapter
logging.getLogger("src.sdk.adapters.draftkings").setLevel(logging.DEBUG)
```

---

## Support

For questions or issues:
1. Check the [examples](../scripts/demo_sdk.py)
2. Review [adapter implementations](../src/sdk/adapters/)
3. Open an issue on GitHub

---

*The Neural Trading Platform SDK - Bringing any data source to algorithmic trading*