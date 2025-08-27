# Agentuity Agents

## Purpose
High-level trading agents that run on the Agentuity platform. These agents use LLMs and sophisticated decision-making to analyze data and execute trading strategies.

## Architecture

```
Agent Consumers → Agentuity Agents (this directory) → Kalshi API
                           ↓
                    LLM Analysis (Gemini/GPT)
                           ↓
                    Trading Decisions
```

## Agent Hierarchy

### DataCoordinator
**Location:** `DataCoordinator/agent.py`  
**Role:** Master orchestrator  
**Responsibilities:**
- Collects data from all sources (Kalshi, ESPN, Twitter)
- Routes events to appropriate agents
- Maintains system state
- Triggers analysis workflows

**Tools:**
- `fetch_kalshi_markets` - Get current market data
- `analyze_market_sentiment` - Sentiment analysis
- `correlate_events` - Event correlation

### StrategyAnalyst
**Location:** `StrategyAnalyst/agent.py`  
**Role:** Strategy generation  
**Responsibilities:**
- Analyzes market conditions
- Generates trading strategies
- Calculates win probabilities
- Provides recommendations to TradeExecutor

**Tools:**
- `calculate_kelly_position` - Kelly Criterion sizing
- `analyze_sentiment` - Social sentiment analysis
- `evaluate_opportunity` - Opportunity scoring

### MarketEngineer
**Location:** `MarketEngineer/agent.py`  
**Role:** Market analysis  
**Responsibilities:**
- Identifies market inefficiencies
- Detects arbitrage opportunities
- Analyzes market microstructure
- Monitors liquidity

**Tools:**
- `detect_arbitrage` - Find pricing inefficiencies
- `analyze_liquidity` - Liquidity assessment
- `calculate_spread` - Bid-ask analysis

### TradeExecutor
**Location:** `TradeExecutor/agent.py`  
**Role:** Order execution  
**Responsibilities:**
- Places orders on Kalshi
- Manages order lifecycle
- Handles partial fills
- Reports execution status

**Tools:**
- `place_order` - Submit orders to Kalshi
- `cancel_order` - Cancel pending orders
- `get_order_status` - Check execution status

### RiskManager
**Location:** `RiskManager/agent.py`  
**Role:** Risk control  
**Responsibilities:**
- Enforces position limits
- Monitors portfolio exposure
- Triggers stop-losses
- Manages drawdown limits

**Tools:**
- `check_position_limits` - Validate position sizes
- `calculate_portfolio_risk` - Risk metrics
- `trigger_stop_loss` - Emergency exits

## Agentuity Integration

### Agent Definition
Each agent inherits from Agentuity's base agent:
```python
from agentuity import Agent, tool, message_handler

class DataCoordinatorAgent(Agent):
    def __init__(self):
        super().__init__(
            name="DataCoordinator",
            description="Orchestrates data flow",
            llm_config={"model": "gpt-4"}
        )
    
    @tool
    def fetch_markets(self):
        """Tool to fetch market data"""
        pass
    
    @message_handler
    def handle_message(self, message):
        """Process incoming messages"""
        pass
```

### Tool Definitions
Tools are Python functions decorated with `@tool`:
```python
@tool
def calculate_position(
    confidence: float,
    odds: float,
    bankroll: float
) -> dict:
    """
    Calculate optimal position size using Kelly Criterion.
    
    Args:
        confidence: Win probability (0-1)
        odds: Decimal odds
        bankroll: Available capital
    
    Returns:
        Position size and risk metrics
    """
    # Implementation
```

### Message Handling
Agents communicate via message passing:
```python
@message_handler
async def handle_market_update(self, message):
    """Process market update messages"""
    if message.type == "MARKET_UPDATE":
        analysis = await self.analyze_market(message.data)
        if analysis.signal_strength > 0.7:
            await self.send_message(
                to="TradeExecutor",
                type="TRADE_SIGNAL",
                data=analysis
            )
```

## Configuration

### Environment Variables
```bash
# LLM Configuration
LLM_MODEL=google/gemini-2.0-flash
LLM_TEMPERATURE=0.2
LLM_SEED=42
OPENROUTER_API_KEY=your_key

# Agentuity Platform
AGENTUITY_SDK_KEY=your_sdk_key
AGENTUITY_PROJECT_KEY=your_project_key
```

### Agent Configuration
In `agentuity.yaml`:
```yaml
agents:
  - name: DataCoordinator
    entry: agents/DataCoordinator/agent.py
    description: Data orchestration agent
    
  - name: StrategyAnalyst
    entry: agents/StrategyAnalyst/agent.py
    description: Strategy generation agent
    
  - name: MarketEngineer
    entry: agents/MarketEngineer/agent.py
    description: Market analysis agent
    
  - name: TradeExecutor
    entry: agents/TradeExecutor/agent.py
    description: Trade execution agent
    
  - name: RiskManager
    entry: agents/RiskManager/agent.py
    description: Risk management agent
```

## Development

### Creating New Agents
1. Create directory: `agents/YourAgent/`
2. Create `agent.py` with agent class
3. Define tools using `@tool` decorator
4. Implement message handlers
5. Add to `agentuity.yaml`
6. Test locally with `agentuity dev`

### Testing Agents
```bash
# Run in development mode
agentuity dev

# Test specific agent
agentuity agent test DataCoordinator

# Deploy to cloud
agentuity deploy
```

### Local Development
```python
# server.py
from agentuity import create_app
from agents.DataCoordinator.agent import DataCoordinatorAgent

app = create_app()
app.register_agent(DataCoordinatorAgent())

if __name__ == "__main__":
    app.run(port=3500)
```

## Communication Flow

1. **Data Ingestion**
   - DataCoordinator receives real-time data
   - Processes and enriches events
   - Routes to relevant agents

2. **Analysis Phase**
   - StrategyAnalyst evaluates opportunities
   - MarketEngineer checks for inefficiencies
   - Both send signals to TradeExecutor

3. **Execution Phase**
   - TradeExecutor validates signals
   - RiskManager approves position size
   - Order placed on Kalshi

4. **Monitoring**
   - RiskManager tracks positions
   - DataCoordinator monitors results
   - Feedback loop to StrategyAnalyst

## Performance Optimization

### LLM Usage
- Use structured outputs for consistency
- Cache frequent analyses
- Batch similar requests
- Use appropriate models for task complexity

### Message Processing
- Implement priority queues
- Use async/await for I/O operations
- Batch database operations
- Monitor message latency

## Error Handling

### Agent Failures
- Automatic restart with state recovery
- Message retry with exponential backoff
- Dead letter queue for failed messages
- Health checks and monitoring

### LLM Errors
- Fallback to simpler models
- Retry with adjusted prompts
- Log and alert on repeated failures
- Manual intervention triggers

## Deployment

### Local Testing
```bash
# Start development server
agentuity dev

# View logs
agentuity logs DataCoordinator
```

### Production Deployment
```bash
# Deploy all agents
agentuity deploy

# Deploy specific agent
agentuity deploy DataCoordinator

# View production logs
agentuity logs --prod DataCoordinator
```

## Monitoring

### Metrics
- Message processing rate
- LLM token usage
- Trading performance
- Error rates

### Alerts
- Agent downtime
- High error rates
- Risk limit breaches
- Unusual trading patterns

## Troubleshooting

### Agent Not Starting
1. Check `agentuity.yaml` configuration
2. Verify environment variables
3. Check import paths
4. Review agent logs

### No Trading Signals
1. Verify data pipeline is running
2. Check Redis connections
3. Review signal thresholds
4. Analyze LLM responses

### High Latency
1. Check LLM response times
2. Review message queue depth
3. Optimize tool implementations
4. Consider scaling agents