# Neural SDK Analysis Infrastructure Implementation Plan

## Overview
Build a comprehensive analysis layer on top of the data collection infrastructure that transforms raw market data into actionable trading insights, validated strategies, and performance metrics for Kalshi sports trading.

## Core Architecture

### 1. Module Structure
```
neural/
├── analysis/
│   ├── __init__.py
│   ├── base.py              # Abstract base classes for analysis
│   ├── market_data.py       # Historical data management with SQLite
│   ├── edge_detection.py    # Market inefficiency identification
│   ├── probability.py       # Probability calculation engines
│   └── metrics.py           # Performance metrics calculations
├── strategy/
│   ├── __init__.py
│   ├── base.py              # Strategy abstract base class
│   ├── builder.py           # Strategy composition framework
│   ├── library/             # Pre-built strategy implementations
│   │   ├── mean_reversion.py
│   │   ├── arbitrage.py
│   │   ├── news_based.py
│   │   └── line_movement.py
│   └── signals.py           # Signal generation system
├── backtesting/
│   ├── __init__.py
│   ├── engine.py            # Core backtesting engine
│   ├── simulator.py         # Market simulation with realistic fills
│   ├── validator.py         # Walk-forward & out-of-sample validation
│   └── optimizer.py         # Parameter optimization (grid/random search)
├── risk/
│   ├── __init__.py
│   ├── position_sizing.py   # Kelly Criterion & fixed/proportional sizing
│   ├── portfolio.py         # Portfolio optimization & correlation analysis
│   ├── limits.py            # Risk limits & constraint enforcement
│   └── monitor.py           # Real-time risk monitoring & alerts
├── visualization/
│   ├── __init__.py
│   ├── charts.py            # Plotly interactive charts
│   ├── dashboard.py         # Real-time Dash dashboard
│   ├── reports.py           # PDF/HTML performance reports
│   └── export.py            # Export to CSV/Excel/JSON
└── kalshi/
    ├── __init__.py
    ├── client.py            # Kalshi API client wrapper
    ├── markets.py           # Market data interface
    ├── orders.py            # Order management system
    └── fees.py              # Fee calculation (0.07 × P × (1-P))
```

## Database Architecture (SQLite)

### Schema Design
```sql
-- Core market data table
CREATE TABLE market_prices (
    market_id TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    bid REAL,
    ask REAL,
    last REAL,
    volume INTEGER,
    open_interest INTEGER,
    PRIMARY KEY (market_id, timestamp)
);

-- Market metadata
CREATE TABLE markets (
    market_id TEXT PRIMARY KEY,
    event_ticker TEXT,
    event_name TEXT,
    sport TEXT,
    close_time INTEGER,
    resolution_time INTEGER,
    outcome INTEGER,  -- 1 for YES, 0 for NO, NULL if pending
    metadata JSON
);

-- Strategy performance tracking
CREATE TABLE trades (
    trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id TEXT,
    market_id TEXT,
    side TEXT,  -- 'YES' or 'NO'
    entry_price REAL,
    exit_price REAL,
    quantity INTEGER,
    entry_time INTEGER,
    exit_time INTEGER,
    pnl REAL,
    fees REAL,
    edge_estimate REAL,
    metadata JSON
);

-- Backtest results
CREATE TABLE backtest_runs (
    run_id TEXT PRIMARY KEY,
    strategy_id TEXT,
    start_date INTEGER,
    end_date INTEGER,
    initial_capital REAL,
    final_capital REAL,
    total_trades INTEGER,
    win_rate REAL,
    sharpe_ratio REAL,
    max_drawdown REAL,
    parameters JSON,
    created_at INTEGER
);

-- Create indexes for performance
CREATE INDEX idx_market_timestamp ON market_prices(timestamp);
CREATE INDEX idx_market_id ON market_prices(market_id);
CREATE INDEX idx_trades_strategy ON trades(strategy_id);
```

## Implementation Phases

### Phase 1: Foundation & Data Layer (Week 1)

**Core Components:**
- SQLite database setup with optimized schema
- Market data ingestion from Kalshi API
- Historical data storage and retrieval
- Data cleaning and normalization

**Key Classes:**
```python
class MarketDataStore:
    """SQLite-based market data storage"""
    def __init__(self, db_path='kalshi_data.db'):
        self.conn = sqlite3.connect(db_path)
        self._optimize_performance()
    
    def store_price_update(self, market_id, price_data)
    def get_price_history(self, market_id, start, end)
    def get_markets_by_sport(self, sport, active_only=True)

class KalshiMarket:
    """Market data representation"""
    market_id: str
    current_price: float
    volume: int
    calculate_implied_probability()
    calculate_fees(quantity)
```

**New Dependencies:**
```yaml
# Add to requirements.txt
pandas>=2.0.0          # Data manipulation
numpy>=1.24.0          # Numerical computing
sqlalchemy>=2.0.0      # Advanced SQL toolkit
python-dateutil>=2.8.0 # Date parsing
```

### Phase 2: Edge Detection & Analysis (Week 1-2)

**Core Components:**
- Probability calculation engines
- Edge detection algorithms
- Sportsbook consensus aggregation
- Market inefficiency detection

**Key Features:**
```python
class EdgeCalculator:
    def calculate_edge(self, kalshi_price, true_probability, confidence):
        """Calculate trading edge with confidence adjustment"""
        raw_edge = true_probability - kalshi_price
        return raw_edge * confidence
    
    def calculate_ev(self, probability, kalshi_price):
        """Expected value calculation including fees"""
        fees = self.calculate_fees(kalshi_price)
        profit_if_win = 1.0 - kalshi_price - fees
        loss_if_lose = kalshi_price + fees
        return (probability * profit_if_win) - ((1-probability) * loss_if_lose)

class ProbabilityEngine:
    def aggregate_sportsbook_odds(self, odds_list)
    def adjust_for_market_conditions(self, base_prob, factors)
    def calculate_confidence_interval(self, probability, sample_size)
```

### Phase 3: Strategy Framework (Week 2)

**Core Components:**
- Abstract strategy base class
- Signal generation system
- Pre-built strategy library
- Strategy composition and combination

**Strategy Implementations:**
```python
class BaseStrategy(ABC):
    @abstractmethod
    async def analyze(self, market_data) -> Signal
    
    @abstractmethod
    def calculate_position_size(self, signal, portfolio)

class MeanReversionStrategy(BaseStrategy):
    """Trade divergences between Kalshi and sportsbook consensus"""
    
class ArbitrageStrategy(BaseStrategy):
    """Detect risk-free profit opportunities"""
    
class NewsBasedStrategy(BaseStrategy):
    """Trade on news sentiment before market reaction"""
    
class LineMovementStrategy(BaseStrategy):
    """Fade or follow significant line movements"""
```

### Phase 4: Backtesting Engine (Week 2-3)

**Core Components:**
- Event-driven backtesting engine
- Realistic market simulation
- Walk-forward validation
- Parameter optimization

**Implementation:**
```python
class BacktestEngine:
    def __init__(self, initial_capital=1000):
        self.portfolio = Portfolio(initial_capital)
        self.market_simulator = MarketSimulator()
        
    async def run(self, strategy, data, start_date, end_date):
        """Run backtest with realistic fills and fees"""
        for market_update in data:
            signal = await strategy.analyze(market_update)
            if signal.action != 'HOLD':
                fill_price = self.market_simulator.simulate_fill(
                    signal, market_update
                )
                self.portfolio.execute_trade(signal, fill_price)
        
        return self.calculate_metrics()
    
    def calculate_metrics(self):
        """Calculate Sharpe, max drawdown, win rate, etc."""
```

### Phase 5: Risk Management (Week 3)

**Core Components:**
- Position sizing algorithms (Kelly, fixed, proportional)
- Portfolio optimization
- Risk limit enforcement
- Correlation analysis

**Implementation:**
```python
class PositionSizer:
    def kelly_criterion(self, bankroll, edge, odds, max_fraction=0.25):
        """Conservative Kelly sizing"""
        if edge <= 0:
            return 0
        kelly = edge / odds
        return min(kelly * bankroll, max_fraction * bankroll)
    
    def fixed_percentage(self, bankroll, risk_pct=0.02):
        """Fixed percentage of bankroll"""
        return bankroll * risk_pct

class RiskManager:
    def check_position_limit(self, position, bankroll, max_pct=0.10)
    def check_correlation_limit(self, new_position, portfolio, max_corr=0.30)
    def calculate_var(self, portfolio, confidence=0.95)
```

### Phase 6: Visualization & Reporting (Week 4)

**Core Components:**
- Interactive Plotly charts
- Real-time Dash dashboard
- Performance reports
- Strategy comparison tools

**Dashboard Features:**
- Live P&L tracking
- Win rate and edge analysis
- Drawdown visualization
- Position monitoring
- Risk metrics display

**New Dependencies:**
```yaml
plotly>=5.0.0         # Interactive charts
dash>=2.0.0           # Web dashboard
dash-bootstrap-components>=1.0.0
matplotlib>=3.5.0     # Static charts
seaborn>=0.12.0      # Statistical plots
reportlab>=4.0.0     # PDF generation
```

## Configuration System

```yaml
# config/analysis.yaml
database:
  type: "sqlite"
  path: "data/kalshi_trading.db"
  
analysis:
  historical_days: 90
  min_edge_threshold: 0.03
  confidence_level: 0.95
  
strategy:
  max_positions: 10
  rebalance_frequency: "daily"
  
risk:
  max_position_pct: 0.10
  daily_loss_limit: 0.20
  kelly_fraction: 0.25
  stop_loss_pct: 0.50
  
backtesting:
  initial_capital: 1000
  fee_calculation: "kalshi_standard"
  slippage_pct: 0.01
  
visualization:
  dashboard_port: 8050
  update_interval: 5  # seconds
```

## Database Decision: SQLite vs Alternatives

### Why SQLite for V1:
- **Zero configuration** - No server setup required
- **Single file database** - Easy backup and distribution
- **Built into Python** - No additional dependencies
- **Fast enough** - Handles millions of rows with sub-second queries
- **Perfect for local analysis** - Embedded database ideal for desktop apps

### Performance Optimizations:
```python
# Configure SQLite for maximum performance
conn.execute("PRAGMA journal_mode = WAL")      # Write-ahead logging
conn.execute("PRAGMA synchronous = NORMAL")    # Faster writes
conn.execute("PRAGMA cache_size = -64000")     # 64MB cache
conn.execute("PRAGMA temp_store = MEMORY")     # In-memory temp tables
conn.execute("PRAGMA mmap_size = 30000000000") # Memory-mapped I/O
```

### When to Consider TimescaleDB:
- Processing 100M+ rows (3+ years of tick data)
- Need continuous aggregates for real-time calculations
- Running 10+ concurrent backtests
- Require advanced time-series functions

### Why Not Convex:
- Cloud-based (not local)
- Network latency kills backtesting performance
- Unnecessary costs for analytical workloads
- Built for real-time collaborative apps, not quant analysis

## Success Metrics

- ✅ Backtest 1000+ markets in < 60 seconds
- ✅ Strategy creation in < 100 lines of code  
- ✅ 95%+ test coverage
- ✅ Sub-second signal generation
- ✅ Real-time dashboard with < 5s latency
- ✅ Support for 10+ concurrent strategies

## Testing Strategy

```
tests/
├── analysis/
│   ├── test_edge_detection.py
│   ├── test_probability.py
│   └── test_market_data.py
├── strategy/
│   ├── test_strategies.py
│   └── test_signals.py
├── backtesting/
│   ├── test_engine.py
│   └── test_simulator.py
└── integration/
    └── test_full_pipeline.py
```

## Key Algorithms from Kalshi Trading Manual

### Expected Value Calculation
```python
def calculate_ev(your_probability, kalshi_price):
    """Calculate expected value including Kalshi fees"""
    fee = 0.07 * kalshi_price * (1 - kalshi_price)
    profit_if_win = 1.0 - kalshi_price - fee
    loss_if_lose = kalshi_price + fee
    ev = (your_probability * profit_if_win) - ((1 - your_probability) * loss_if_lose)
    return ev
```

### Kelly Criterion Position Sizing
```python
def kelly_position_size(bankroll, edge, odds, max_kelly_fraction=0.25):
    """Conservative Kelly sizing for Kalshi trades"""
    if edge <= 0:
        return 0
    kelly_fraction = edge / odds
    capped_kelly = min(kelly_fraction, max_kelly_fraction)
    return bankroll * capped_kelly
```

### Mean Reversion Signal
```python
def mean_reversion_signal(kalshi_price, sportsbook_consensus, threshold=0.05):
    """Signal when Kalshi diverges from sportsbook consensus"""
    price_diff = abs(kalshi_price - sportsbook_consensus)
    
    if price_diff > threshold:
        if kalshi_price < sportsbook_consensus:
            return "BUY_YES"  # Kalshi underpricing
        else:
            return "BUY_NO"   # Kalshi overpricing
    
    return "HOLD"
```

## Next Steps After Implementation

1. **Week 5+**: Integrate with live Kalshi API for paper trading
2. **Week 6+**: Deploy automated trading bots with monitoring
3. **Future**: Add ML-based probability models
4. **Future**: Expand to other prediction markets (Polymarket, Manifold)

This plan builds a production-ready analysis infrastructure that enables rapid strategy development, robust backtesting, and automated trading on Kalshi markets.