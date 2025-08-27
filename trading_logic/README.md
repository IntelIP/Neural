# Trading Logic

## Purpose
Core trading algorithms, risk management tools, and integration clients. This directory contains the business logic for position sizing, order execution, and market analysis.

## Structure

```
trading_logic/
├── execution/           # Trade execution logic
│   ├── kelly_criterion.py    # Position sizing
│   ├── order_manager.py      # Order lifecycle
│   └── stop_loss_monitor.py  # Risk controls
├── analysis/            # Market analysis tools
│   ├── sentiment_analyzer.py # Sentiment scoring
│   ├── probability_calc.py   # Win probability
│   └── data_aggregator.py    # Data correlation
└── integrations/        # External API clients
    ├── kalshi_client.py      # Kalshi API
    ├── espn_client.py        # ESPN API
    └── llm_client.py         # LLM integration
```

## Core Components

### Kelly Criterion Calculator
**File:** `kelly_criterion.py` (to be reorganized)  
**Purpose:** Optimal position sizing based on edge and odds  
**Mathematical Foundation:**
```
f* = (p * b - q) / b

Where:
- f* = Fraction of bankroll to bet
- p = Probability of winning
- q = Probability of losing (1 - p)
- b = Odds received on the bet
```

**Safety Features:**
- Kelly fraction scaling (default 0.25)
- Maximum position limits
- Minimum edge requirements

**Usage:**
```python
from trading_logic.kelly_tools import calculate_kelly_position

position = calculate_kelly_position(
    confidence=0.65,      # 65% win probability
    yes_price=0.45,      # Current YES price
    no_price=0.55,       # Current NO price
    bankroll=10000,      # Available capital
    kelly_fraction=0.25  # Safety factor
)
```

### Sentiment Analysis
**File:** `sentiment_probability.py`  
**Purpose:** Converts social sentiment to trading signals  
**Features:**
- Twitter sentiment scoring
- Reddit discussion analysis
- News impact assessment
- Sentiment momentum tracking

**Algorithm:**
1. Collect sentiment data from multiple sources
2. Weight by source reliability and recency
3. Apply Bayesian updates to base probability
4. Generate directional bias score

### Stop Loss Manager
**File:** `stop_loss.py`  
**Purpose:** Automated position exit on adverse moves  
**Configuration:**
```python
STOP_LOSS_PERCENTAGE = 0.10  # 10% stop loss
TRAILING_STOP = True         # Use trailing stops
STOP_ADJUSTMENT_FACTOR = 0.5 # Tighten stops over time
```

**Features:**
- Fixed and trailing stops
- Time-based stop adjustments
- Correlated position management
- Emergency market exit

### Data Aggregator
**File:** `data_aggregator.py`  
**Purpose:** Correlates data from multiple sources  
**Responsibilities:**
- Synchronize ESPN game events with market moves
- Correlate Twitter sentiment with price changes
- Detect leading indicators
- Generate composite signals

### E2B Code Executor
**File:** `e2b_executor.py`  
**Purpose:** Secure code execution for complex calculations  
**Use Cases:**
- Running statistical models
- Backtesting strategies
- Complex probability calculations
- Data transformations

### LLM Client
**File:** `llm_client.py`  
**Purpose:** Interface with language models for analysis  
**Features:**
- Structured prompt generation
- Response parsing
- Error handling and retries
- Cost optimization

## Integration Clients

### Kalshi Tools
**Current Location:** `espn_tools.py` (to be reorganized)  
**Capabilities:**
- Market data fetching
- Order placement
- Position management
- Account information

**Authentication:**
```python
# RSA-PSS signing for Kalshi API
auth = KalshiAuth(
    key_id=os.getenv("KALSHI_API_KEY_ID"),
    private_key=os.getenv("KALSHI_PRIVATE_KEY")
)
```

### ESPN Integration
**File:** `espn_tools.py`  
**Data Available:**
- Live game scores
- Play-by-play events
- Team statistics
- Player injuries
- Game probabilities

## Risk Management

### Position Sizing Rules
1. **Kelly Criterion Base**
   - Never exceed full Kelly
   - Default to 25% of Kelly
   - Adjust for correlation

2. **Hard Limits**
   - Maximum 5% of portfolio per position
   - Maximum 20% total market exposure
   - Minimum position size: $10

3. **Dynamic Adjustments**
   - Reduce size in volatile markets
   - Scale with confidence level
   - Account for liquidity

### Risk Metrics
```python
# Portfolio risk calculation
total_risk = sum([
    position.size * position.volatility 
    for position in active_positions
])

# Correlation adjustment
correlated_risk = total_risk * correlation_factor

# Maximum drawdown check
if current_drawdown > MAX_DRAWDOWN_LIMIT:
    halt_trading()
```

## Configuration

### Environment Variables
```bash
# Trading Parameters
KELLY_FRACTION=0.25
MAX_POSITION_SIZE=100
RISK_PERCENTAGE=0.02

# Stop Loss Settings
STOP_LOSS_PERCENTAGE=0.10
TAKE_PROFIT_PERCENTAGE=0.30

# API Keys
E2B_API_KEY=your_key
EXA_API_KEY=your_key
```

### Trading Rules Configuration
```python
TRADING_RULES = {
    "min_edge": 0.05,           # Minimum 5% edge
    "min_liquidity": 1000,       # $1000 minimum volume
    "max_correlation": 0.7,      # Position correlation limit
    "cooldown_period": 300,      # 5 minute cooldown after loss
    "max_daily_trades": 20,      # Daily trade limit
}
```

## Usage Examples

### Calculate Optimal Position
```python
from trading_logic import calculate_kelly_position, check_risk_limits

# Calculate position size
position_size = calculate_kelly_position(
    confidence=0.60,
    current_price=0.45,
    bankroll=10000
)

# Apply risk limits
safe_size = check_risk_limits(
    position_size,
    current_exposure,
    portfolio_value
)
```

### Execute Trade with Stop Loss
```python
from trading_logic import place_order, set_stop_loss

# Place the order
order = place_order(
    market="NFL-CHAMPIONSHIP",
    side="YES",
    quantity=safe_size,
    price=0.45
)

# Set stop loss
stop_loss = set_stop_loss(
    order_id=order.id,
    stop_price=0.40,
    trailing=True
)
```

## Testing

### Unit Tests
```bash
# Test Kelly calculations
pytest tests/test_kelly_criterion.py

# Test risk management
pytest tests/test_risk_management.py

# Integration tests
pytest tests/test_trading_integration.py
```

### Backtesting
```python
from trading_logic.backtest import run_backtest

results = run_backtest(
    strategy="mean_reversion",
    start_date="2024-01-01",
    end_date="2024-12-31",
    initial_capital=10000
)
```

## Performance Optimization

### Calculation Caching
- Cache Kelly calculations for same inputs
- Store sentiment scores with TTL
- Reuse ESPN data within time windows

### Batch Operations
- Group order submissions
- Batch position updates
- Aggregate risk calculations

## Error Handling

### API Failures
- Exponential backoff with retry
- Fallback to cached data
- Circuit breaker for repeated failures
- Alert on critical failures

### Calculation Errors
- Validate all inputs
- Bounds checking on outputs
- Fallback to conservative estimates
- Log anomalies for review

## Troubleshooting

### Position Size Too Small
1. Check Kelly calculation inputs
2. Verify confidence levels
3. Review risk limits
4. Check minimum position requirements

### Orders Not Executing
1. Verify API credentials
2. Check market liquidity
3. Review price limits
4. Confirm account balance

### Stop Losses Not Triggering
1. Check stop loss configuration
2. Verify price feed connection
3. Review trigger conditions
4. Check execution logs

## Development Guidelines

### Adding New Strategies
1. Create strategy class in `analysis/`
2. Implement position sizing logic
3. Add risk management rules
4. Include backtesting support
5. Write comprehensive tests

### Extending Integrations
1. Create client in `integrations/`
2. Implement authentication
3. Add rate limiting
4. Include error handling
5. Document API endpoints