# Neural SDK API Reference

## Overview

The Neural SDK is an institutional-grade framework for sports trading on prediction markets, providing comprehensive data collection, analysis, and trading infrastructure.

## Architecture

The SDK is organized into four main infrastructure layers:

1. **Data Collection Infrastructure** - Real-time data ingestion from multiple sources
2. **Analysis Infrastructure** - Data processing and signal generation (coming soon)
3. **Trading Infrastructure** - Order execution and risk management (coming soon)
4. **Deployment Infrastructure** - Production deployment and monitoring (coming soon)

## Data Collection Infrastructure

### Base Classes

#### `BaseDataSource`
Abstract base class for all data sources.

```python
from neural.data_collection import BaseDataSource, DataSourceConfig

class MyDataSource(BaseDataSource):
    async def _connect_impl(self) -> bool:
        # Implementation
        return True
```

**Key Methods:**
- `connect()` - Establish connection with retry logic
- `disconnect()` - Graceful disconnection
- `register_callback(event, callback)` - Register event handlers
- `get_metrics()` - Get performance metrics

#### `RestDataSource`
Base class for REST API data sources.

```python
from neural.data_collection import RestDataSource, RestConfig

config = RestConfig(
    name="my_api",
    base_url="https://api.example.com",
    rate_limit_requests=10.0,
    cache_enabled=True
)
client = RestDataSource(config)
```

**Features:**
- Token bucket rate limiting
- Response caching with configurable TTL
- Automatic retry with exponential backoff
- Request/response interceptors

#### `WebSocketDataSource`
Base class for WebSocket data sources.

```python
from neural.data_collection import WebSocketDataSource, WebSocketConfig

config = WebSocketConfig(
    name="ws_feed",
    url="wss://stream.example.com",
    heartbeat_interval=30.0,
    auto_reconnect=True
)
ws = WebSocketDataSource(config)
```

**Features:**
- Automatic reconnection
- Heartbeat/ping-pong support
- Message buffering
- Event-driven architecture

### Sports Data Sources

#### ESPN Sports API

##### `ESPNNFL`
NFL data from ESPN API.

```python
from neural.sports import ESPNNFL

nfl = ESPNNFL()

# Get scoreboard
scores = await nfl.get_scoreboard(week=10, seasontype=2)

# Get team roster
roster = await nfl.get_team_roster("GB", include_stats=True)

# Get injuries
injuries = await nfl.get_injuries("GB")
```

**Available Methods:**
- `get_scoreboard(dates, week, seasontype, year)`
- `get_team_roster(team_id, include_stats, include_projections)`
- `get_depth_chart(team_id, season)`
- `get_injuries(team_id)`
- `get_team_statistics(team_id, season, seasontype)`
- `get_game_summary(game_id)`
- `get_team_schedule(team_id, season)`
- `get_all_teams()`
- `get_news(team_id, limit)`
- `get_standings(season, seasontype)`
- `get_playoff_picture(season)`

##### `ESPNCFB`
College Football data from ESPN API.

```python
from neural.sports import ESPNCFB

cfb = ESPNCFB()

# Get scoreboard for specific date
scores = await cfb.get_scoreboard(dates="20250913")

# Get team info
team = await cfb.get_team("michigan")
```

**Available Methods:**
- `get_scoreboard(dates, week, seasontype, year, groups, conference)`
- `get_rankings(year, week, seasontype)`
- `get_conferences()`
- `get_teams_by_conference(conference_id)`
- `get_bowl_games(year)`

##### `ESPNNBA`
NBA data from ESPN API.

```python
from neural.sports import ESPNNBA

nba = ESPNNBA()

# Get today's games
scores = await nba.get_scoreboard()

# Get team roster
roster = await nba.get_team_roster("LAL")
```

**Available Methods:**
- `get_scoreboard(dates)`
- `get_team_roster(team_id, include_stats)`
- `get_player_stats(player_id, season, seasontype)`
- `get_team_statistics(team_id, season, seasontype)`
- `get_standings(season, seasontype)`
- `get_playoffs_bracket(year)`

### Social Media Data Sources

#### Twitter API (twitterapi.io)

```python
from neural.social import TwitterClient, TwitterConfig

# Configuration (auto-loads TWITTERAPI_IO_KEY from env)
config = TwitterConfig(
    cache_tweets_ttl=60,
    rate_limit_requests=100.0,
    track_costs=True
)

twitter = TwitterClient(config)

# Search tweets
tweets = await twitter.search_tweets("NFL playoffs", limit=20)

# Get user info
user = await twitter.get_user_info("ESPN_NFL")

# Get user tweets
timeline = await twitter.get_user_tweets("NFL", limit=50)

# Get trends
trends = await twitter.get_trends(woeid=23424977)  # USA

# Pagination
all_tweets = await twitter.collect_paginated_data(
    method="search_tweets",
    max_pages=5,
    query="#SuperBowl"
)

# Check API costs
costs = twitter.get_api_costs()
print(f"Total cost: ${costs['total_cost']:.6f}")
```

**Available Methods:**
- `get_user_info(username)` - Get user profile
- `get_user_tweets(username, limit, cursor)` - Get user timeline
- `search_tweets(query, limit, cursor, search_type)` - Search tweets
- `get_tweet(tweet_id)` - Get single tweet
- `get_tweets_bulk(tweet_ids)` - Get multiple tweets
- `get_replies(tweet_id, cursor)` - Get tweet replies
- `get_quotes(tweet_id, cursor)` - Get quote tweets
- `get_followers(username, limit, cursor)` - Get followers
- `get_following(username, limit, cursor)` - Get following
- `get_trends(woeid)` - Get trending topics
- `search_users(query, limit, cursor)` - Search users
- `collect_paginated_data(method, max_pages, **kwargs)` - Auto-pagination

**Cost Tracking:**
- $0.15 per 1,000 tweets
- $0.18 per 1,000 user profiles
- $0.15 per 1,000 followers
- Minimum $0.00015 per request

### Data Pipeline

#### Pipeline Stages

```python
from neural.data_collection import DataPipeline, TransformStage

# Create custom transform stage
class MyTransform(TransformStage):
    def __init__(self):
        super().__init__("my_transform")
    
    async def process(self, data):
        # Transform data
        return transformed_data

# Build pipeline
pipeline = DataPipeline()
pipeline.add_stage(MyTransform())
pipeline.add_stage(AnotherTransform())

# Process data
result = await pipeline.process(input_data)
```

#### Buffer Management

```python
from neural.data_collection import CircularBuffer, OverflowStrategy

# Create buffer with overflow strategy
buffer = CircularBuffer(
    capacity=10000,
    overflow_strategy=OverflowStrategy.DROP_OLDEST
)

# Add data
buffer.append(data)

# Get statistics
stats = buffer.get_statistics()
```

**Overflow Strategies:**
- `DROP_OLDEST` - Drop oldest items when full
- `DROP_NEWEST` - Drop new items when full
- `BLOCK` - Block until space available
- `EXPAND` - Dynamically expand capacity

## Configuration

### Environment Variables

Create a `.env` file in your project root:

```bash
# Kalshi API
KALSHI_API_KEY_ID=your-key-id
KALSHI_PRIVATE_KEY_FILE=/path/to/key.txt
KALSHI_ENVIRONMENT=production

# Twitter API
TWITTERAPI_IO_KEY=your-api-key

# Odds API
ODDS_API_KEY=your-api-key
```

### YAML Configuration

```yaml
# config.yaml
data_sources:
  - name: espn_nfl
    type: rest
    config:
      base_url: https://site.api.espn.com/apis/site/v2/sports/
      sport: football
      league: nfl
      rate_limit_requests: 2.0
      cache_enabled: true
      cache_ttl: 60

  - name: twitter_feed
    type: rest
    config:
      base_url: https://api.twitterapi.io
      rate_limit_requests: 100.0
      track_costs: true
```

Load configuration:

```python
from neural.data_collection import ConfigManager

config_manager = ConfigManager()
config = config_manager.load_config("config.yaml")
```

## Error Handling

The SDK defines custom exceptions for different error scenarios:

```python
from neural.data_collection.exceptions import (
    DataSourceError,      # Base exception
    ConnectionError,      # Connection failures
    ConfigurationError,   # Invalid configuration
    RateLimitError,      # Rate limit exceeded
    AuthenticationError, # Auth failures
    ValidationError,     # Data validation errors
    BufferOverflowError, # Buffer capacity exceeded
    TimeoutError        # Operation timeouts
)

try:
    await client.connect()
except ConnectionError as e:
    print(f"Connection failed: {e}")
    print(f"Details: {e.details}")
```

## Examples

### Monitoring NFL Games

```python
import asyncio
from neural.sports import ESPNNFL
from neural.social import TwitterClient

async def monitor_game():
    nfl = ESPNNFL()
    twitter = TwitterClient()
    
    # Get current games
    scores = await nfl.get_scoreboard()
    
    for event in scores.get("events", []):
        game_name = event["name"]
        
        # Search for game tweets
        tweets = await twitter.search_tweets(game_name, limit=10)
        
        print(f"Game: {game_name}")
        print(f"Tweet count: {len(tweets['tweets'])}")
    
    await nfl.disconnect()
    await twitter.disconnect()

asyncio.run(monitor_game())
```

### Collecting Historical Data

```python
from neural.sports import ESPNCFB
from datetime import datetime, timedelta

async def collect_season_data():
    cfb = ESPNCFB()
    
    all_games = []
    start_date = datetime(2025, 9, 1)
    
    for week in range(1, 16):
        date = start_date + timedelta(weeks=week-1)
        date_str = date.strftime("%Y%m%d")
        
        scores = await cfb.get_scoreboard(dates=date_str)
        all_games.extend(scores.get("events", []))
    
    print(f"Collected {len(all_games)} games")
    await cfb.disconnect()
```

## Testing

Run the test suite:

```bash
# All tests
python -m pytest tests/ -v

# Specific module
python -m pytest tests/data_collection/ -v
python -m pytest tests/social/ -v
python -m pytest tests/sports/ -v

# With coverage
python -m pytest tests/ --cov=neural --cov-report=html
```

## Performance Metrics

The SDK tracks various performance metrics:

```python
# Get data source metrics
metrics = client.get_metrics()
print(f"Messages received: {metrics['messages_received']}")
print(f"Errors: {metrics['errors_count']}")
print(f"Uptime: {metrics['total_uptime']}s")

# Get buffer statistics
stats = buffer.get_statistics()
print(f"Current size: {stats['current_size']}")
print(f"Overflow events: {stats['overflow_events']}")

# Get API costs (Twitter)
costs = twitter.get_api_costs()
print(f"Total cost: ${costs['total_cost']:.6f}")
print(f"Requests: {costs['requests_count']}")
```

## Best Practices

1. **Connection Management**
   - Always use context managers or try/finally for cleanup
   - Implement proper disconnect in shutdown handlers
   - Use connection pooling for REST APIs

2. **Rate Limiting**
   - Set conservative rate limits below API maximums
   - Monitor rate limit headers in responses
   - Implement backoff strategies for 429 errors

3. **Error Handling**
   - Always handle specific exceptions
   - Log errors with full context
   - Implement circuit breakers for failing services

4. **Performance**
   - Use caching for frequently accessed data
   - Batch requests when possible
   - Use pagination for large datasets

5. **Cost Management (Twitter)**
   - Track API costs regularly
   - Set budget alerts
   - Cache responses to minimize requests
   - Use pagination efficiently

## Support

- GitHub Issues: [Report bugs or request features](https://github.com/your-org/neural-sdk/issues)
- Documentation: [Full documentation](https://neural-sdk.readthedocs.io)
- Examples: See the `examples/` directory for more use cases