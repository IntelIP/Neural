# Data Sources Guide - Neural Trading Platform

## Overview

Each data source provides unique alpha for trading decisions. This guide explains what each source offers, how to configure it, and how to interpret its signals.

---

## Core Data Sources

### 1. DraftKings Sportsbook

**Purpose**: Professional odds movements often lead Neural markets by 5-30 seconds

**What It Provides**:
- Real-time odds for all major sports
- Line movements that indicate sharp money
- Market efficiency indicators
- Professional betting consensus

**Configuration**:
```yaml
- name: draftkings
  enabled: true
  config:
    sports: ["NFL", "NBA", "MLB", "NHL"]
    poll_interval: 3  # seconds - lower for faster games
    min_odds_change: 0.03  # 3% minimum to trigger event
    markets_to_track:
      - spread
      - total
      - moneyline
```

**Event Types Generated**:
```python
ODDS_CHANGE: {
    "event_name": "Chiefs vs Bills",
    "market_type": "spread",
    "previous_odds": 0.650,
    "current_odds": 0.675,
    "change": 0.025,
    "direction": "up",
    "sharp_money_indicator": true
}
```

**Trading Signal Interpretation**:
- **Large sudden moves** (>5%): Sharp money detected, follow quickly
- **Gradual drift** (<2%): Public money, potential fade opportunity
- **Line freeze** despite volume: Books have information, be cautious

**Example Strategy**:
```python
if event.data['change'] > 0.05 and event.data['sharp_money_indicator']:
    # Sharp money detected, high confidence signal
    signal = {"action": "follow", "confidence": 0.85}
elif event.data['change'] < 0.02 and volume_high:
    # Public money pushing line, potential fade
    signal = {"action": "fade", "confidence": 0.65}
```

---

### 2. Reddit Game Threads

**Purpose**: Capture retail sentiment extremes and breaking news

**What It Provides**:
- Real-time sentiment analysis
- Volume spikes indicating major events
- Keyword detection (injury, touchdown, controversy)
- Crowd psychology indicators

**Configuration**:
```yaml
- name: reddit
  enabled: true
  config:
    client_id: ${REDDIT_CLIENT_ID}
    client_secret: ${REDDIT_CLIENT_SECRET}
    subreddits:
      - nfl        # Main league subreddit
      - KansasCityChiefs  # Team specific
      - buffalobills
    game_thread_patterns:
      - "Game Thread"
      - "GameThread"
      - "[Game Thread]"
    min_comment_karma: 5  # Filter noise
    keywords:
      injury: ["injury", "injured", "hurt", "limping", "down"]
      touchdown: ["touchdown", "td", "score", "scored"]
      momentum: ["momentum", "turning point", "comeback"]
      controversy: ["ref", "rigged", "bullshit", "robbed"]
```

**Event Types Generated**:
```python
SENTIMENT_SHIFT: {
    "thread_id": "abc123",
    "sentiment_before": 0.65,
    "sentiment_after": 0.45,
    "shift": -0.20,
    "trigger": "injury_mention",
    "comment_rate": 145,  # comments per minute
    "sample_comments": ["Oh no, Mahomes is limping!", "This doesn't look good"]
}

VOLUME_SPIKE: {
    "normal_rate": 50,
    "current_rate": 200,
    "spike_ratio": 4.0,
    "likely_cause": "big_play"
}
```

**Sentiment Calculation**:
```python
# Simple sentiment scoring
positive_keywords = ["let's go", "touchdown", "amazing", "yes"]
negative_keywords = ["fuck", "terrible", "injury", "over"]

sentiment = (positive_count - negative_count) / total_comments
```

**Trading Signal Interpretation**:
- **Sentiment crashes** (-0.3+ shift): Something bad happened, sell quickly
- **Volume spike** (3x+ normal): Major event, check other sources
- **Controversy keywords**: Emotional betting incoming, wait for overreaction

---

### 3. Weather Conditions

**Purpose**: Weather significantly impacts scoring and play style

**What It Provides**:
- Real-time conditions at stadium
- Wind speed and direction
- Precipitation type and intensity
- Temperature changes
- Visibility conditions

**Configuration**:
```yaml
- name: weather
  enabled: true
  config:
    api_key: ${OPENWEATHER_API_KEY}
    update_interval: 300  # 5 minutes
    thresholds:
      wind_speed: 15      # mph - affects passing
      precipitation: 0.1   # inches/hour - affects ball handling
      temperature_change: 10  # degrees F - rapid change
      visibility: 1       # miles - fog/heavy rain
    stadiums:
      - name: "Arrowhead Stadium"
        team: "KC"
        lat: 39.0489
        lon: -94.4839
        outdoor: true  # Only monitor outdoor stadiums
```

**Event Types Generated**:
```python
WEATHER_UPDATE: {
    "stadium": "Lambeau Field",
    "condition": "high_wind",
    "wind_speed": 22,
    "wind_gust": 35,
    "impact": {
        "passing_yards_reduction": "33%",
        "field_goal_accuracy_reduction": "44%",
        "recommended_strategy": "establish_run"
    }
}
```

**Impact on Different Sports**:

| Condition | NFL Impact | MLB Impact | Soccer Impact |
|-----------|------------|------------|---------------|
| Wind >15mph | -30% passing, -40% FG accuracy | +20% home runs (with wind) | -20% long passes |
| Rain | -15% scoring, +fumbles | Game delay likely | -10% scoring |
| Snow | -25% scoring, +running game | Game cancelled | Advantage defenders |
| Cold <32°F | -10% passing | N/A (dome/cancelled) | -5% scoring |

**Trading Applications**:
```python
# Wind strategy
if wind_speed > 20:
    # Bet UNDER on total points
    signal = {"market": "total", "position": "under", "confidence": 0.75}
    
# Precipitation strategy  
if precipitation > 0.2:  # Heavy rain
    # Reduced scoring, more turnovers
    signal = {"market": "total", "position": "under", "confidence": 0.80}
```

---

## Advanced Data Sources

### 4. ESPN GameCast (Coming Soon)

**Purpose**: Official play-by-play data with minimal latency

**What It Provides**:
- Real-time game events
- Player statistics
- Drive summaries
- Injury reports

**Why It Matters**:
- Ground truth for game state
- Fastest non-venue source
- Detailed play outcomes

---

### 5. Twitter/X Streams (In Development)

**Purpose**: Breaking news and insider information

**What It Provides**:
- Beat reporter updates
- Injury news
- Pre-game inactive lists
- Coaching decisions

**Key Accounts to Monitor**:
```python
NFL_INSIDERS = [
    "@AdamSchefter",  # ESPN
    "@RapSheet",      # NFL Network
    "@JayGlazer",     # FOX
]
```

---

## Data Correlation Strategies

### Multi-Source Confirmation

The real edge comes from correlating multiple sources:

```python
class OpportunityDetector:
    def analyze(self, events_window):
        # Get events from last 10 seconds
        dk_event = find_event(events_window, source="DraftKings")
        reddit_event = find_event(events_window, source="Reddit")
        weather_event = find_event(events_window, source="Weather")
        
        # Scenario 1: Sharp move + sentiment shift
        if dk_event and reddit_event:
            if dk_event.data['sharp_money'] and reddit_event.data['sentiment_shift'] < -0.2:
                return Signal(
                    action="SELL",
                    confidence=0.90,
                    reason="Sharp money and crowd panic aligned"
                )
        
        # Scenario 2: Weather advantage not priced in
        if weather_event and not dk_event:
            if weather_event.data['wind_speed'] > 25:
                return Signal(
                    action="BET_UNDER",
                    confidence=0.75,
                    reason="Severe wind not reflected in odds yet"
                )
```

### Latency Advantages

Different sources have different latencies:

| Source | Latency | Reliability | Best For |
|--------|---------|-------------|----------|
| ESPN GameCast | 1-2s | 99% | Ground truth |
| DraftKings | 0.5s | 95% | Sharp money |
| Reddit | 2-5s | 85% | Sentiment |
| Weather | 5min | 99% | Conditions |
| Twitter | 1-3s | 90% | Breaking news |

**Strategy**: Use fastest sources for immediate action, slower for confirmation

---

## Adding Custom Data Sources

### Template for New Source

```python
from src.sdk import DataSourceAdapter, StandardizedEvent, EventType

class CustomSourceAdapter(DataSourceAdapter):
    """Your custom data source."""
    
    def get_metadata(self):
        return DataSourceMetadata(
            name="CustomSource",
            version="1.0.0",
            source_type="api",
            latency_ms=1000,
            reliability=0.95
        )
    
    async def connect(self):
        """Establish connection to your source."""
        self.client = YourAPIClient(self.config['api_key'])
        return await self.client.connect()
    
    async def stream(self):
        """Generate events from your source."""
        while self.is_connected:
            data = await self.client.fetch_data()
            
            # Detect interesting events
            if self.is_significant(data):
                yield StandardizedEvent(
                    source=self.metadata.name,
                    event_type=EventType.CUSTOM,
                    timestamp=datetime.now(),
                    data=data,
                    confidence=self.calculate_confidence(data),
                    impact=self.assess_impact(data)
                )
            
            await asyncio.sleep(self.config['poll_interval'])
    
    def is_significant(self, data):
        """Determine if data represents a trading opportunity."""
        # Your logic here
        return data['change'] > self.config['threshold']
```

### Integration Steps

1. **Create adapter file**: `src/sdk/adapters/your_source.py`
2. **Add to config**: `config/data_sources.yaml`
3. **Test adapter**: `python scripts/test_adapter.py your_source`
4. **Monitor events**: `redis-cli SUBSCRIBE "your_source:*"`

---

## Optimizing Data Source Performance

### 1. Tune Poll Intervals

```yaml
# Fast games need faster polling
NBA:
  poll_interval: 2  # Quick possessions
  
NFL:
  poll_interval: 5  # Slower pace
  
MLB:
  poll_interval: 10  # Even slower
```

### 2. Filter Noise

```yaml
reddit:
  min_comment_karma: 10  # Higher = less noise
  min_account_age: 30    # Days
  
draftkings:
  min_odds_change: 0.05  # Only significant moves
```

### 3. Prioritize Sources

```yaml
# Process most important first
sources:
  - name: draftkings
    priority: 1  # Highest
  - name: weather
    priority: 5  # Lowest
```

---

## Debugging Data Sources

### Check Connection

```python
# Test individual adapter
from src.sdk.adapters.draftkings import DraftKingsAdapter

adapter = DraftKingsAdapter(config)
connected = await adapter.connect()
print(f"Connected: {connected}")
```

### Monitor Events

```bash
# Watch all events from a source
redis-cli PSUBSCRIBE "draftkings:*"

# Count events
redis-cli PUBSUB NUMSUB "reddit:sentiment"
```

### Debug Output

```python
# Enable debug logging for specific adapter
import logging
logging.getLogger("src.sdk.adapters.reddit").setLevel(logging.DEBUG)
```

---

## Best Practices

### 1. Handle Source Failures Gracefully

```python
async def stream_with_fallback(self):
    try:
        async for event in self.primary_source.stream():
            yield event
    except SourceError:
        self.logger.warning("Primary failed, using backup")
        async for event in self.backup_source.stream():
            yield event
```

### 2. Validate Data Quality

```python
def validate_event(self, event):
    # Check data completeness
    required_fields = ['price', 'timestamp', 'market']
    if not all(f in event.data for f in required_fields):
        return False
    
    # Check data freshness
    age = datetime.now() - event.timestamp
    if age.total_seconds() > 60:  # Too old
        return False
    
    return True
```

### 3. Cache When Appropriate

```python
class CachedAdapter(DataSourceAdapter):
    def __init__(self):
        self.cache = {}
        self.cache_duration = 30  # seconds
    
    async def fetch_with_cache(self, key):
        if key in self.cache:
            cached_time, data = self.cache[key]
            if time.time() - cached_time < self.cache_duration:
                return data
        
        data = await self.fetch_fresh(key)
        self.cache[key] = (time.time(), data)
        return data
```

---

## Summary

Each data source provides a different edge:

- **DraftKings**: Follow smart money
- **Reddit**: Fade emotional extremes
- **Weather**: Exploit unpriced conditions
- **ESPN**: React to events first
- **Twitter**: Get insider information

The key to success is:
1. Start with 1-2 reliable sources
2. Understand what each source tells you
3. Combine signals for higher confidence
4. React quickly when sources align

Next: See [GAME_CONFIGURATION_GUIDE.md](GAME_CONFIGURATION_GUIDE.md) to configure monitoring for specific games.