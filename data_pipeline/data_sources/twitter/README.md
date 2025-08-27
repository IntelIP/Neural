# Twitter Sentiment Integration

Real-time Twitter sentiment analysis for market correlation using TwitterAPI.io WebSocket.

## Setup

### 1. Get TwitterAPI.io API Key

1. Visit [TwitterAPI.io](https://twitterapi.io)
2. Sign up for an account
3. Get your API key from the dashboard
4. Note: Billing starts when you activate filter rules

### 2. Add API Key to Environment

Add to your `.env` file:

```bash
# TwitterAPI.io API Key
TWITTERAPI_KEY=your_api_key_here
```

Or set as environment variable:

```bash
export TWITTERAPI_KEY="your_api_key_here"
```

### 3. Install Dependencies

```bash
pip install websocket-client httpx
```

## Features

### WebSocket Streaming
- Real-time tweet streaming via TwitterAPI.io WebSocket
- Automatic reconnection with 90-second backoff
- Ping/pong heartbeat mechanism
- Handles connection errors gracefully

### Filter Management
- Dynamic filter creation for games and players
- Monitors insider accounts (Schefter, Woj, etc.)
- Team-specific account tracking
- Keyword-based filtering (injuries, status updates)
- Automatic cleanup of old filters

### Sentiment Analysis
- Multi-level sentiment scoring (-1 to 1)
- Context-aware analysis for sports content
- Market keyword detection (injury, trade, etc.)
- Author credibility weighting
- Engagement-based impact scoring

### Market Impact Classification
- **CRITICAL**: Breaking news from verified insiders
- **HIGH**: Verified accounts with high engagement
- **MEDIUM**: Moderate credibility or engagement
- **LOW**: Regular users with some engagement
- **NOISE**: Likely irrelevant content

## Usage Examples

### Basic Sentiment Streaming

```python
from kalshi_web_infra.twitter import TwitterStreamAdapter

stream = TwitterStreamAdapter()
await stream.start()

# Monitor a game
await stream.monitor_game(
    game_id="GAME123",
    home_team="Chiefs",
    away_team="Bills",
    sport="nfl",
    players=["Patrick Mahomes", "Josh Allen"]
)
```

### Filter Rule Management

```python
from kalshi_web_infra.twitter import FilterManager

manager = FilterManager()

# Create game filter
rule = await manager.create_game_filter(
    home_team="Lakers",
    away_team="Celtics",
    sport="nba"
)

# Monitor specific player
await manager.create_player_filter(
    player_name="LeBron James",
    keywords=["injury", "status", "update"]
)
```

### Sentiment Analysis

```python
from kalshi_web_infra.twitter import SentimentAnalyzer

analyzer = SentimentAnalyzer()

# Analyze single tweet
sentiment = analyzer.analyze_tweet(tweet)
print(f"Score: {sentiment.score:+.2f}")
print(f"Confidence: {sentiment.confidence:.2f}")

# Aggregate over time window
event = analyzer.aggregate_sentiment(
    tweets,
    window_minutes=5
)
print(f"Avg sentiment: {event.avg_sentiment_score:+.2f}")
print(f"Market impact: {event.market_impact.value}")
```

## Filter Rule Syntax

TwitterAPI.io uses Twitter's PowerTrack syntax:

### Basic Operators
- `OR`: Match any condition
- `AND`: Match all conditions
- `from:username`: Tweets from specific user
- `#hashtag`: Tweets with hashtag
- `"exact phrase"`: Exact phrase match

### Example Filters

```python
# Game monitoring
"(Chiefs OR Bills) AND (#NFL OR #ChiefsVsBills)"

# Player injury monitoring
'"Patrick Mahomes" AND (injury OR injured OR questionable)'

# Insider accounts
"from:AdamSchefter OR from:RapSheet OR from:wojespn"
```

## Polling Intervals

- **0.1 seconds**: Live games (fastest)
- **0.5 seconds**: Active monitoring
- **1.0 seconds**: Player updates
- **2.0 seconds**: General events
- **5.0+ seconds**: Background monitoring

Note: Lower intervals increase API costs.

## Message Types

### Connected Event
```json
{
  "event_type": "connected",
  "timestamp": 1642789123456
}
```

### Tweet Event
```json
{
  "event_type": "tweet",
  "rule_id": "rule_12345",
  "rule_tag": "game_Chiefs_Bills",
  "tweets": [...],
  "timestamp": 1642789123456
}
```

### Ping Event
```json
{
  "event_type": "ping",
  "timestamp": 1642789123456
}
```

## Best Practices

1. **Single Connection**: Maintain only one WebSocket connection per API key
2. **Filter Optimization**: Combine related filters to reduce API calls
3. **Cleanup**: Delete old filter rules after games end
4. **Error Handling**: Implement exponential backoff for reconnections
5. **Rate Limiting**: Monitor tweet velocity to avoid overwhelming system

## Troubleshooting

### Connection Issues
- Verify API key is correct
- Check network connectivity
- Ensure only one connection is active
- Wait 90 seconds before reconnecting

### No Data Received
- Verify filter rules are active
- Check filter syntax is correct
- Ensure monitored accounts/keywords have activity
- Verify polling interval is appropriate

### High Latency
- Reduce polling interval for critical events
- Optimize filter rules to reduce noise
- Consider geographic proximity to API servers

## Cost Considerations

- Billing starts when filter rules are activated
- Cost based on polling interval and data volume
- Delete unused rules promptly
- Use longer polling intervals for non-critical monitoring

## Support

- [TwitterAPI.io Documentation](https://docs.twitterapi.io)
- [API Status](https://status.twitterapi.io)
- Support: support@twitterapi.io