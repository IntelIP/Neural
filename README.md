# Neural SDK

**Institutional-Grade Sports Trading Infrastructure for Kalshi**

Neural is a comprehensive Python SDK designed for institutional trading of sports event contracts on Kalshi's prediction market platform. It provides a robust, scalable infrastructure for data collection, analysis, trading, and deployment.

## Features

### Data Collection Infrastructure ✅
- **Multi-source data ingestion** - WebSocket and REST API support
- **Real-time streaming** - Low-latency data processing pipelines
- **Smart buffering** - Configurable overflow strategies
- **Auto-reconnection** - Resilient connection management
- **Rate limiting** - Token bucket algorithm implementation
- **Configuration management** - YAML/JSON with environment variables

### Coming Soon
- **Analysis Infrastructure** - Statistical models and ML pipelines
- **Trading Infrastructure** - Order management and execution
- **Deployment Infrastructure** - Production-ready deployment tools

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/neural.git
cd neural

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Quick Start

### Sports Data (ESPN)

```python
from neural.sports import ESPNNFL, ESPNCFB

# NFL Data
nfl = ESPNNFL()
scores = await nfl.get_scoreboard()
roster = await nfl.get_team_roster("GB", include_stats=True)
injuries = await nfl.get_injuries("GB")

# College Football Data
cfb = ESPNCFB()
games = await cfb.get_scoreboard(dates="20250913")
rankings = await cfb.get_rankings()
```

### Social Media Data (Twitter)

```python
from neural.social import TwitterClient

# Auto-loads TWITTERAPI_IO_KEY from environment
twitter = TwitterClient()

# Search tweets
tweets = await twitter.search_tweets("NFL playoffs", limit=20)

# Get user timeline
timeline = await twitter.get_user_tweets("ESPN", limit=50)

# Monitor API costs
costs = twitter.get_api_costs()
print(f"Total cost: ${costs['total_cost']:.6f}")
```

### WebSocket Data Source

```python
from neural.data_collection import WebSocketDataSource, WebSocketConfig

# Configure WebSocket connection
config = WebSocketConfig(
    url="wss://api.example.com/stream",
    headers={"Authorization": "Bearer ${API_KEY}"},
    reconnect=True,
    heartbeat_interval=30
)

# Create and connect
ws = WebSocketDataSource(config)

# Register event handlers
ws.register_callback("data", lambda msg: print(f"Received: {msg}"))
ws.register_callback("error", lambda err: print(f"Error: {err}"))

# Connect and stream
await ws.connect()
```

### Data Pipeline

```python
from neural.data_collection import DataPipeline, TransformStage

# Create pipeline
pipeline = DataPipeline()

# Add data sources
await pipeline.add_source("websocket", ws)
await pipeline.add_source("rest", rest)

# Add transformation stages
class EnrichmentStage(TransformStage):
    async def process(self, data):
        # Add metadata, validate, transform
        return enriched_data

pipeline.add_stage(EnrichmentStage())

# Add consumers
async def process_data(data):
    print(f"Processing: {data}")

await pipeline.add_consumer(process_data)

# Start pipeline
await pipeline.start()
```

### Configuration Management

```python
from neural.data_collection import ConfigManager

# Load configuration
config = ConfigManager(
    config_file="config.yaml",
    env_prefix="NEURAL_"
)
config.load_config()

# Access nested values
api_key = config.get("kalshi.api_key")
timeout = config.get("connections.timeout", default=30)
```

## Architecture

Neural follows a modular, layered architecture:

```
neural/
├── data_collection/     # Data ingestion layer
│   ├── base.py         # Abstract base classes
│   ├── websocket.py    # WebSocket implementation
│   ├── rest.py         # REST API implementation
│   ├── pipeline.py     # Data pipeline orchestration
│   ├── buffer.py       # Buffer management
│   └── config.py       # Configuration management
├── analysis/           # Analysis layer (coming soon)
├── trading/            # Trading layer (coming soon)
└── deployment/         # Deployment layer (coming soon)
```

## Configuration

Neural supports multiple configuration sources with the following priority:
1. Default values
2. Configuration files (YAML/JSON)
3. Environment variables

### Example Configuration File

```yaml
# config.yaml
kalshi:
  api_key: ${KALSHI_API_KEY}
  api_secret: ${KALSHI_API_SECRET}
  base_url: https://api.kalshi.com

data_sources:
  websocket:
    url: wss://api.kalshi.com/v2/stream
    reconnect: true
    max_reconnect_attempts: 5
    heartbeat_interval: 30
  
  rest:
    rate_limit: 100
    cache_ttl: 60
    timeout: 30

pipeline:
  buffer_size: 10000
  overflow_strategy: drop_oldest
```

### Environment Variables

```bash
export KALSHI_API_KEY="your-api-key"
export KALSHI_API_SECRET="your-api-secret"
export NEURAL_LOG_LEVEL="INFO"
```

## Testing

```bash
# Run all tests
pytest

# Run specific test module
pytest tests/data_collection/

# Run with coverage
pytest --cov=neural --cov-report=term-missing

# Run only fast tests
pytest -m "not slow"
```

## Development

### Project Structure

```
Neural/
├── neural/              # Main package
├── tests/              # Test suite
├── examples/           # Usage examples
├── docs/              # Documentation
├── requirements.txt    # Dependencies
├── pytest.ini         # Test configuration
└── README.md          # This file
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run tests and ensure they pass
6. Submit a pull request

### Code Style

- Follow PEP 8
- Use type hints
- Add docstrings to all public functions
- Keep functions focused and small
- Write tests for new features

## Roadmap

### Phase 1: Data Collection (Complete ✅)
- WebSocket and REST handlers
- Configuration management
- Data pipeline orchestration
- Buffer management
- ESPN Sports APIs (NFL, CFB, NBA)
- Twitter API integration (twitterapi.io)
- Comprehensive testing

### Phase 2: Analysis Infrastructure (In Progress)
- Statistical models
- ML pipelines
- Backtesting framework
- Performance metrics

### Phase 3: Trading Infrastructure
- Order management system
- Risk management
- Position tracking
- Execution algorithms

### Phase 4: Deployment Infrastructure
- Docker containers
- Kubernetes manifests
- Monitoring and alerting
- CI/CD pipelines

## License

MIT License - see LICENSE file for details

## Support

For questions and support, please open an issue on GitHub.

## Acknowledgments

Built for institutional trading on Kalshi's prediction market platform.