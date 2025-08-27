<div align="center">
    <img src="https://raw.githubusercontent.com/agentuity/cli/refs/heads/main/.github/Agentuity.png" alt="Agentuity" width="100"/> <br/>
    <strong>Build Agents, Not Infrastructure</strong> <br/>
    <br/>
        <a target="_blank" href="https://app.agentuity.com/deploy" alt="Agentuity">
            <img src="https://app.agentuity.com/img/deploy.svg" /> 
        </a>
    <br />
</div>

# 🤖 Kalshi Trading Agent System

Autonomous multi-agent trading system for Kalshi sports event contracts. Uses real-time data streams from Kalshi markets, ESPN games, and Twitter sentiment to execute trades with Kelly Criterion position sizing.

## 🏗️ Architecture

### Core Pipeline
```
WebSockets → Redis Pub/Sub → Agents → Kalshi Trading
```

The system uses Redis pub/sub to distribute real-time data from WebSocket streams to specialized agents:

### Data Flow
1. **WebSocket Streams** collect real-time data:
   - Kalshi market prices and trades
   - ESPN game events and scores
   - Twitter sentiment analysis

2. **Redis Publisher** distributes data across channels:
   - `kalshi:markets` - Market price updates
   - `kalshi:trades` - Trade execution confirmations
   - `kalshi:signals` - Trading signals and alerts
   - `espn:games` - Game events and scores

3. **Agent Consumers** process specific data streams:
   - **DataCoordinator** - Orchestrates data flow and generates signals
   - **MarketEngineer** - Detects arbitrage and market inefficiencies
   - **TradeExecutor** - Executes trades using Kelly Criterion
   - **RiskManager** - Monitors positions and enforces risk limits

## 📋 Prerequisites

- **Python**: Version 3.10 or higher
- **UV**: Version 0.5.25 or higher ([Documentation](https://docs.astral.sh/uv/))
- **Redis**: Running locally or accessible remote instance
- **Agentuity CLI**: For deployment and development

## 🚀 Getting Started

### 1. Environment Setup

Copy the example environment file:
```bash
cp .env.example .env
```

Edit `.env` with your credentials:
```bash
# Kalshi API (Demo or Production)
KALSHI_ENVIRONMENT=demo  # or prod
KALSHI_API_KEY_ID=your_key_id
KALSHI_PRIVATE_KEY=your_private_key

# Redis
REDIS_URL=redis://localhost:6379

# OpenRouter (for LLM analysis)
OPENROUTER_API_KEY=your_key

# Agentuity Platform
AGENTUITY_SDK_KEY=your_sdk_key
AGENTUITY_PROJECT_KEY=your_project_key
```

### 2. Install Dependencies

```bash
# UV will automatically install dependencies
uv sync
```

### 3. Start Redis

```bash
# Local Redis
redis-server

# Or use Docker
docker run -d -p 6379:6379 redis:alpine
```

### 4. Development Mode

```bash
# Start with Agentuity console
agentuity dev

# Or run directly
uv run server.py
```

## 🌐 Deployment

Deploy to Agentuity Cloud:

```bash
agentuity deploy
```

## 📚 Project Structure

```
├── agents/                    # Agent implementations
│   ├── base_redis_consumer.py # Base class for Redis consumers
│   ├── DataCoordinator/       # Data orchestration agent
│   ├── MarketEngineer/        # Market analysis agent
│   ├── TradeExecutor/         # Trade execution agent
│   └── RiskManager/           # Risk management agent
├── kalshi_web_infra/          # WebSocket infrastructure
│   ├── stream_manager.py      # Unified stream management
│   ├── redis_publisher.py     # Redis publishing layer
│   └── websockets/            # WebSocket clients
├── tools/                     # Utility tools
│   ├── kalshi_tools.py        # Kalshi API integration
│   ├── kelly_tools.py         # Kelly Criterion calculator
│   └── risk_tools.py          # Risk management utilities
├── tests/                     # Test files
├── docs/                      # Documentation
├── examples/                  # Example implementations
└── server.py                  # Main server entry point
```

## 🔧 Configuration

### Environment Files
- `.env.development` - Development configuration
- `.env.production` - Production configuration  
- `.env.example` - Template with all options

### Trading Parameters
Configure in `.env`:
```bash
KELLY_FRACTION=0.25          # Kelly safety factor
MAX_POSITION_SIZE=100        # Max contracts per trade
STOP_LOSS_PERCENTAGE=0.10    # 10% stop loss
TAKE_PROFIT_PERCENTAGE=0.30  # 30% take profit
```

## 🧪 Testing

Run the test suite:
```bash
# All tests
uv run pytest tests/

# Specific test file
uv run pytest tests/test_redis_integration.py

# With coverage
uv run pytest tests/ --cov=agents --cov=kalshi_web_infra
```

## 📖 Examples

### Running Individual Agents

```bash
# Data Coordinator
uv run python examples/agent_redis_consumer.py data

# Market Engineer
uv run python examples/agent_redis_consumer.py market

# All agents concurrently
uv run python examples/agent_redis_consumer.py all
```

### WebSocket Testing

```bash
# Test Kalshi WebSocket
uv run python examples/test_kalshi_websocket.py

# Test ESPN WebSocket
uv run python examples/test_espn_websocket.py
```

## 🛠️ Development

### Creating New Agents

1. Inherit from `BaseAgentRedisConsumer`:
```python
from agents.base_redis_consumer import BaseAgentRedisConsumer

class MyAgentConsumer(BaseAgentRedisConsumer):
    async def process_message(self, channel: str, data: Dict[str, Any]):
        # Process incoming messages
        pass
```

2. Subscribe to relevant channels:
```python
await consumer.subscribe_to_channels([
    "kalshi:markets",
    "kalshi:signals"
])
```

### Adding WebSocket Streams

1. Implement stream handler in `StreamManager`
2. Publish to appropriate Redis channel
3. Update agent consumers to process new data

## 📊 Monitoring

The system provides real-time monitoring through:
- Redis pub/sub statistics
- WebSocket connection health
- Agent processing metrics
- Position and P&L tracking

## 🆘 Troubleshooting

### Common Issues

1. **Redis Connection Failed**
   - Ensure Redis is running: `redis-cli ping`
   - Check REDIS_URL in `.env`

2. **WebSocket Authentication Error**
   - Verify Kalshi API credentials
   - Check if using correct environment (demo/prod)

3. **Agent Not Receiving Messages**
   - Verify Redis subscriptions
   - Check channel names match publisher

### Debug Mode

Enable detailed logging:
```bash
LOG_LEVEL=DEBUG uv run server.py
```

## 📝 License

This project is licensed under the terms specified in the LICENSE file.

## 🤝 Support

- [Agentuity Documentation](https://agentuity.dev/SDKs/python)
- [Discord Community](https://discord.gg/agentuity)
- [GitHub Issues](https://github.com/your-repo/issues)