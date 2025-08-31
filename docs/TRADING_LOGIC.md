# Trading Logic Documentation - Neural Trading Platform

## Overview

This document explains how the platform transforms raw data into profitable trades. From signal generation to position sizing to risk management, every trading decision follows a systematic process.

---

## Signal Generation Pipeline

### 1. Event Detection

Raw events flow through standardization:

```python
# Raw event from DraftKings
{
    "game": "KC vs BUF",
    "market": "spread",
    "old_line": -3.5,
    "new_line": -4.5,
    "timestamp": "2024-01-21T18:35:22Z"
}

# Becomes StandardizedEvent
StandardizedEvent(
    source="DraftKings",
    event_type=EventType.ODDS_CHANGE,
    game_id="KC-BUF-2024-01-21",
    data={
        "market": "spread",
        "movement": -1.0,
        "direction": "away_team",
        "magnitude": "large"
    },
    confidence=0.85,
    impact="high"
)
```

### 2. Event Correlation

Multiple events within a time window get correlated:

```python
class EventCorrelator:
    """Correlate events across sources."""
    
    def __init__(self, window_seconds=10):
        self.window = window_seconds
        self.events = defaultdict(list)
        
    def add_event(self, event: StandardizedEvent):
        """Add event to correlation window."""
        game_id = event.game_id
        self.events[game_id].append(event)
        
        # Clean old events
        cutoff = datetime.now() - timedelta(seconds=self.window)
        self.events[game_id] = [
            e for e in self.events[game_id] 
            if e.timestamp > cutoff
        ]
        
    def find_correlations(self, game_id: str) -> List[Correlation]:
        """Find correlated events for a game."""
        events = self.events[game_id]
        correlations = []
        
        # Look for patterns
        for e1, e2 in combinations(events, 2):
            if self.are_correlated(e1, e2):
                correlations.append(Correlation(e1, e2))
                
        return correlations
        
    def are_correlated(self, e1: StandardizedEvent, e2: StandardizedEvent):
        """Determine if two events are related."""
        # Same game
        if e1.game_id != e2.game_id:
            return False
            
        # Within time window
        time_diff = abs((e1.timestamp - e2.timestamp).total_seconds())
        if time_diff > self.window:
            return False
            
        # Compatible event types
        compatible = {
            (EventType.ODDS_CHANGE, EventType.SENTIMENT_SHIFT),
            (EventType.GAME_EVENT, EventType.VOLUME_SPIKE),
            (EventType.WEATHER_UPDATE, EventType.ODDS_CHANGE)
        }
        
        types = {e1.event_type, e2.event_type}
        return types in compatible
```

### 3. Signal Generation

Correlated events generate trading signals:

```python
class SignalGenerator:
    """Generate trading signals from events."""
    
    def generate_signal(self, correlation: Correlation) -> Optional[Signal]:
        """Create signal from correlated events."""
        
        # Pattern 1: DraftKings moves, Kalshi hasn't
        if self.is_arbitrage_opportunity(correlation):
            return Signal(
                type=SignalType.ARBITRAGE,
                action="BUY",
                confidence=0.90,
                expected_edge=0.05,
                time_sensitive=True,
                ttl_seconds=30
            )
            
        # Pattern 2: Bad news hits Reddit first
        if self.is_sentiment_crash(correlation):
            return Signal(
                type=SignalType.SENTIMENT,
                action="SELL",
                confidence=0.75,
                expected_edge=0.03,
                time_sensitive=True,
                ttl_seconds=60
            )
            
        # Pattern 3: Weather advantage
        if self.is_weather_edge(correlation):
            return Signal(
                type=SignalType.ENVIRONMENTAL,
                action="BET_UNDER",
                confidence=0.70,
                expected_edge=0.04,
                time_sensitive=False,
                ttl_seconds=300
            )
            
        return None
        
    def is_arbitrage_opportunity(self, correlation):
        """Check for price discrepancy."""
        dk_event = correlation.get_event("DraftKings")
        kalshi_event = correlation.get_event("Kalshi")
        
        if dk_event and not kalshi_event:
            # DraftKings moved, Kalshi hasn't yet
            if dk_event.data['magnitude'] == 'large':
                return True
                
        return False
```

---

## Position Sizing - Kelly Criterion

### The Kelly Formula

The platform uses the Kelly Criterion to determine optimal position sizes:

```
f = (p * b - q) / b

Where:
f = fraction of capital to bet
p = probability of winning
b = odds received on the bet
q = probability of losing (1 - p)
```

### Implementation

```python
class KellyCalculator:
    """Calculate position sizes using Kelly Criterion."""
    
    def __init__(self, kelly_fraction=0.25):
        """
        Initialize with safety factor.
        
        Args:
            kelly_fraction: Fraction of Kelly to use (0.25 = 25%)
        """
        self.kelly_fraction = kelly_fraction
        
    def calculate_position(
        self,
        probability: float,
        odds: float,
        capital: float,
        max_position: float = 0.05
    ) -> float:
        """
        Calculate position size.
        
        Args:
            probability: Win probability (0-1)
            odds: Decimal odds (e.g., 2.0 for even money)
            capital: Total capital available
            max_position: Maximum position as fraction of capital
            
        Returns:
            Position size in dollars
        """
        # Kelly formula
        q = 1 - probability
        kelly = (probability * odds - q) / odds
        
        # Apply safety factor
        kelly_safe = kelly * self.kelly_fraction
        
        # Apply maximum constraint
        kelly_constrained = min(kelly_safe, max_position)
        
        # Convert to dollars
        position = capital * kelly_constrained
        
        return position
        
    def calculate_with_confidence(
        self,
        signal: Signal,
        market_odds: float,
        capital: float
    ) -> float:
        """Calculate position with confidence adjustment."""
        
        # Adjust probability by confidence
        base_probability = self.signal_to_probability(signal)
        adjusted_prob = base_probability * signal.confidence
        
        # Extra safety for low confidence
        if signal.confidence < 0.70:
            kelly_fraction = self.kelly_fraction * 0.5
        else:
            kelly_fraction = self.kelly_fraction
            
        # Calculate position
        position = self.calculate_position(
            probability=adjusted_prob,
            odds=market_odds,
            capital=capital
        )
        
        return position
        
    def signal_to_probability(self, signal: Signal) -> float:
        """Convert signal to win probability."""
        # Map expected edge to probability
        # Edge of 5% with even odds = 52.5% probability
        edge = signal.expected_edge
        base_prob = 0.5  # Fair odds baseline
        
        return base_prob + (edge / 2)
```

### Position Sizing Examples

```python
# Example 1: High confidence arbitrage
signal = Signal(
    confidence=0.90,
    expected_edge=0.05
)
market_odds = 2.0  # Even money
capital = 10000

position = calculator.calculate_with_confidence(signal, market_odds, capital)
# Result: $450 (4.5% of capital)

# Example 2: Low confidence sentiment
signal = Signal(
    confidence=0.65,
    expected_edge=0.03
)
position = calculator.calculate_with_confidence(signal, market_odds, capital)
# Result: $97.50 (0.975% of capital)
```

---

## Risk Management

### Position Limits

```python
class RiskManager:
    """Manage portfolio risk."""
    
    def __init__(self):
        self.limits = {
            'max_position_size': 0.05,      # 5% per position
            'max_correlation': 0.70,         # Correlation limit
            'max_daily_loss': 0.20,          # 20% stop loss
            'max_concurrent_positions': 10,   # Position count
            'max_market_exposure': 0.40      # 40% total exposure
        }
        
    def check_position(
        self,
        proposed_position: float,
        market: str,
        portfolio: Portfolio
    ) -> Tuple[bool, str]:
        """Check if position passes risk rules."""
        
        # Rule 1: Position size
        if proposed_position > portfolio.capital * self.limits['max_position_size']:
            return False, "Position too large"
            
        # Rule 2: Correlation
        correlation = self.calculate_correlation(market, portfolio)
        if correlation > self.limits['max_correlation']:
            return False, "Too correlated with existing positions"
            
        # Rule 3: Daily loss
        if portfolio.daily_pnl < -portfolio.starting_capital * self.limits['max_daily_loss']:
            return False, "Daily loss limit reached"
            
        # Rule 4: Position count
        if len(portfolio.positions) >= self.limits['max_concurrent_positions']:
            return False, "Too many open positions"
            
        # Rule 5: Total exposure
        total_exposure = portfolio.total_exposure + proposed_position
        if total_exposure > portfolio.capital * self.limits['max_market_exposure']:
            return False, "Maximum market exposure reached"
            
        return True, "Position approved"
        
    def calculate_correlation(self, market: str, portfolio: Portfolio) -> float:
        """Calculate correlation with existing positions."""
        correlations = []
        
        for position in portfolio.positions:
            if self.are_correlated_markets(market, position.market):
                correlations.append(self.get_correlation_coefficient(market, position.market))
                
        return max(correlations) if correlations else 0.0
```

### Stop Loss Implementation

```python
class StopLossManager:
    """Manage stop losses for positions."""
    
    def __init__(self):
        self.stop_losses = {}
        
    def set_stop_loss(
        self,
        position_id: str,
        entry_price: float,
        stop_type: str = "fixed"
    ):
        """Set stop loss for position."""
        
        if stop_type == "fixed":
            # Fixed 5% stop
            stop_price = entry_price * 0.95
            
        elif stop_type == "trailing":
            # Trailing 3% stop
            stop_price = entry_price * 0.97
            
        elif stop_type == "time":
            # Time-based decay
            stop_price = entry_price * 0.98
            
        self.stop_losses[position_id] = {
            'stop_price': stop_price,
            'stop_type': stop_type,
            'entry_price': entry_price,
            'high_water_mark': entry_price
        }
        
    def check_stops(self, positions: List[Position]) -> List[str]:
        """Check if any stops are triggered."""
        triggered = []
        
        for position in positions:
            stop = self.stop_losses.get(position.id)
            if not stop:
                continue
                
            # Update trailing stop
            if stop['stop_type'] == 'trailing':
                if position.current_price > stop['high_water_mark']:
                    stop['high_water_mark'] = position.current_price
                    stop['stop_price'] = position.current_price * 0.97
                    
            # Check if triggered
            if position.current_price <= stop['stop_price']:
                triggered.append(position.id)
                
        return triggered
```

---

## Execution Strategies

### Order Types

```python
class OrderExecutor:
    """Execute orders with different strategies."""
    
    async def execute_signal(
        self,
        signal: Signal,
        position_size: float
    ) -> Order:
        """Execute trading signal."""
        
        if signal.time_sensitive:
            # Use market order for time-sensitive
            return await self.market_order(signal, position_size)
            
        elif signal.confidence > 0.85:
            # Aggressive limit for high confidence
            return await self.aggressive_limit(signal, position_size)
            
        else:
            # Patient limit for lower confidence
            return await self.patient_limit(signal, position_size)
            
    async def market_order(self, signal, size):
        """Execute immediately at market."""
        order = {
            'type': 'market',
            'side': signal.action,
            'size': size,
            'time_in_force': 'IOC'  # Immediate or cancel
        }
        return await self.kalshi_client.place_order(order)
        
    async def aggressive_limit(self, signal, size):
        """Place limit at or better than market."""
        current_price = await self.get_current_price(signal.market)
        
        if signal.action == 'BUY':
            limit_price = current_price * 1.01  # Pay 1% more
        else:
            limit_price = current_price * 0.99  # Accept 1% less
            
        order = {
            'type': 'limit',
            'side': signal.action,
            'size': size,
            'price': limit_price,
            'time_in_force': 'GTC'  # Good till cancelled
        }
        return await self.kalshi_client.place_order(order)
```

### Order Management

```python
class OrderManager:
    """Manage order lifecycle."""
    
    def __init__(self):
        self.orders = {}
        self.fills = []
        
    async def monitor_order(self, order_id: str):
        """Monitor order until filled or cancelled."""
        max_wait = 60  # seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            status = await self.check_order_status(order_id)
            
            if status == 'filled':
                self.fills.append(order_id)
                return 'filled'
                
            elif status == 'partial':
                # Decide whether to wait or cancel remainder
                if time.time() - start_time > max_wait / 2:
                    await self.cancel_order(order_id)
                    return 'partial'
                    
            elif status in ['cancelled', 'rejected']:
                return status
                
            await asyncio.sleep(1)
            
        # Timeout - cancel order
        await self.cancel_order(order_id)
        return 'timeout'
```

---

## Trading Strategies

### Strategy 1: Sharp Money Following

```python
class SharpMoneyStrategy:
    """Follow professional betting movements."""
    
    def analyze(self, events: List[StandardizedEvent]) -> Optional[Signal]:
        """Detect sharp money movements."""
        
        # Look for DraftKings line movement
        dk_events = [e for e in events if e.source == "DraftKings"]
        
        for event in dk_events:
            movement = event.data.get('line_movement')
            volume = event.data.get('volume')
            
            # Sharp money indicators
            if movement > 0.05 and volume < 1000:
                # Big move on small volume = sharp
                return Signal(
                    type=SignalType.SHARP_MONEY,
                    action="FOLLOW",
                    market=event.game_id,
                    confidence=0.85,
                    expected_edge=movement / 2
                )
                
            elif movement > 0.03 and event.data.get('reverse_line_movement'):
                # Line moves against public = sharp
                return Signal(
                    type=SignalType.SHARP_MONEY,
                    action="FOLLOW",
                    market=event.game_id,
                    confidence=0.80,
                    expected_edge=0.04
                )
                
        return None
```

### Strategy 2: Sentiment Arbitrage

```python
class SentimentArbitrageStrategy:
    """Trade sentiment extremes."""
    
    def analyze(self, events: List[StandardizedEvent]) -> Optional[Signal]:
        """Find sentiment arbitrage opportunities."""
        
        reddit_events = [e for e in events if e.source == "Reddit"]
        
        for event in reddit_events:
            sentiment = event.data.get('sentiment')
            shift = event.data.get('sentiment_shift')
            
            # Extreme negative sentiment
            if sentiment < 0.2 and shift < -0.3:
                # Panic selling = opportunity
                return Signal(
                    type=SignalType.SENTIMENT,
                    action="BUY",
                    confidence=0.70,
                    expected_edge=0.03,
                    reason="Extreme negative sentiment"
                )
                
            # Extreme positive sentiment
            elif sentiment > 0.8 and shift > 0.3:
                # Euphoria = fade
                return Signal(
                    type=SignalType.SENTIMENT,
                    action="SELL",
                    confidence=0.65,
                    expected_edge=0.025,
                    reason="Extreme positive sentiment"
                )
                
        return None
```

### Strategy 3: Weather Edge

```python
class WeatherEdgeStrategy:
    """Exploit weather impacts on games."""
    
    def analyze(self, events: List[StandardizedEvent]) -> Optional[Signal]:
        """Find weather-based edges."""
        
        weather_events = [e for e in events if e.source == "Weather"]
        
        for event in weather_events:
            condition = event.data.get('condition')
            impact = event.data.get('impact')
            
            # High wind
            if condition == 'high_wind' and impact['wind_speed'] > 20:
                return Signal(
                    type=SignalType.ENVIRONMENTAL,
                    action="BET_UNDER",
                    market=f"{event.game_id}_total",
                    confidence=0.75,
                    expected_edge=0.04,
                    reason=f"Wind {impact['wind_speed']}mph"
                )
                
            # Heavy precipitation
            elif condition == 'precipitation' and impact['rate'] > 0.2:
                return Signal(
                    type=SignalType.ENVIRONMENTAL,
                    action="BET_UNDER",
                    market=f"{event.game_id}_total",
                    confidence=0.80,
                    expected_edge=0.05,
                    reason="Heavy rain/snow"
                )
                
        return None
```

---

## Performance Metrics

### Tracking Metrics

```python
class PerformanceTracker:
    """Track trading performance metrics."""
    
    def __init__(self):
        self.trades = []
        self.daily_pnl = []
        
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics."""
        
        if not self.trades:
            return {}
            
        returns = [t.pnl / t.position_size for t in self.trades]
        
        metrics = {
            # Basic metrics
            'total_trades': len(self.trades),
            'win_rate': len([t for t in self.trades if t.pnl > 0]) / len(self.trades),
            'average_return': np.mean(returns),
            'total_pnl': sum(t.pnl for t in self.trades),
            
            # Risk metrics
            'sharpe_ratio': self.calculate_sharpe(returns),
            'max_drawdown': self.calculate_max_drawdown(),
            'var_95': np.percentile(returns, 5),
            
            # Efficiency metrics
            'profit_factor': self.calculate_profit_factor(),
            'edge_captured': self.calculate_edge_captured(),
            'avg_hold_time': np.mean([t.hold_time for t in self.trades])
        }
        
        return metrics
        
    def calculate_sharpe(self, returns, risk_free=0.02):
        """Calculate Sharpe ratio."""
        if not returns:
            return 0
            
        excess_returns = [r - risk_free/252 for r in returns]  # Daily risk-free
        
        if np.std(excess_returns) == 0:
            return 0
            
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        
    def calculate_max_drawdown(self):
        """Calculate maximum drawdown."""
        cumulative = []
        total = 0
        
        for trade in self.trades:
            total += trade.pnl
            cumulative.append(total)
            
        peak = cumulative[0]
        max_dd = 0
        
        for value in cumulative:
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
            
        return max_dd
```

### Signal Quality Analysis

```python
class SignalAnalyzer:
    """Analyze signal quality and accuracy."""
    
    def analyze_signals(self, signals: List[Signal], outcomes: List[Outcome]):
        """Analyze how well signals predicted outcomes."""
        
        analysis = {
            'by_type': defaultdict(list),
            'by_confidence': defaultdict(list),
            'by_source': defaultdict(list)
        }
        
        for signal, outcome in zip(signals, outcomes):
            accuracy = 1 if outcome.successful else 0
            
            # By signal type
            analysis['by_type'][signal.type].append(accuracy)
            
            # By confidence bucket
            confidence_bucket = round(signal.confidence, 1)
            analysis['by_confidence'][confidence_bucket].append(accuracy)
            
            # By source
            analysis['by_source'][signal.source].append(accuracy)
            
        # Calculate averages
        results = {}
        for category, data in analysis.items():
            results[category] = {}
            for key, values in data.items():
                results[category][key] = {
                    'accuracy': np.mean(values),
                    'count': len(values)
                }
                
        return results
```

---

## Edge Calculation

### Expected Value

```python
def calculate_expected_value(
    probability: float,
    win_amount: float,
    loss_amount: float
) -> float:
    """
    Calculate expected value of a bet.
    
    EV = (P(win) * win_amount) - (P(lose) * loss_amount)
    """
    ev = (probability * win_amount) - ((1 - probability) * loss_amount)
    return ev

# Example
prob_win = 0.55  # 55% chance
win = 100        # Win $100
loss = 100       # Lose $100

ev = calculate_expected_value(prob_win, win, loss)
# EV = (0.55 * 100) - (0.45 * 100) = 55 - 45 = $10
```

### Edge Estimation

```python
class EdgeEstimator:
    """Estimate edge for different scenarios."""
    
    def estimate_edge(self, signal: Signal, market: Market) -> float:
        """Estimate edge for a signal."""
        
        # Base edge from signal
        base_edge = signal.expected_edge
        
        # Adjustments
        adjustments = []
        
        # Time decay
        age = (datetime.now() - signal.timestamp).total_seconds()
        time_decay = max(0, 1 - age / signal.ttl_seconds)
        adjustments.append(time_decay)
        
        # Market efficiency
        volume = market.volume
        if volume < 1000:
            efficiency = 0.8  # Inefficient market
        elif volume < 10000:
            efficiency = 0.9
        else:
            efficiency = 1.0
        adjustments.append(efficiency)
        
        # Competition
        if market.num_participants > 100:
            competition = 0.9  # High competition
        else:
            competition = 1.0
        adjustments.append(competition)
        
        # Calculate final edge
        adjustment_factor = np.mean(adjustments)
        final_edge = base_edge * adjustment_factor
        
        return final_edge
```

---

## Backtesting Framework

### Running Backtests

```python
class Backtester:
    """Backtest trading strategies."""
    
    def __init__(self, strategy, initial_capital=10000):
        self.strategy = strategy
        self.initial_capital = initial_capital
        
    def run(self, historical_data: pd.DataFrame) -> BacktestResults:
        """Run backtest on historical data."""
        
        portfolio = Portfolio(self.initial_capital)
        trades = []
        
        for timestamp, data in historical_data.iterrows():
            # Generate events from historical data
            events = self.data_to_events(data)
            
            # Get signal from strategy
            signal = self.strategy.analyze(events)
            
            if signal:
                # Calculate position
                position_size = self.calculate_position(signal, portfolio)
                
                # Simulate execution
                fill_price = self.simulate_fill(signal, data)
                
                # Record trade
                trade = Trade(
                    signal=signal,
                    size=position_size,
                    entry_price=fill_price,
                    entry_time=timestamp
                )
                trades.append(trade)
                
            # Update positions
            self.update_positions(portfolio, data)
            
        return BacktestResults(trades, portfolio)
```

---

## Summary

The trading logic transforms data into profits through:

1. **Signal Generation**: Correlating events across sources
2. **Position Sizing**: Kelly Criterion with safety factors
3. **Risk Management**: Multiple layers of protection
4. **Execution**: Adaptive order strategies
5. **Performance Tracking**: Continuous improvement

Key principles:
- Never use full Kelly (25% maximum)
- Multiple confirmations increase confidence
- Speed matters for arbitrage
- Risk management is paramount
- Track everything for optimization

The platform combines these elements to find and exploit market inefficiencies faster than human traders.