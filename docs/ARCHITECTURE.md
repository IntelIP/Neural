# Neural Trading Agent Architecture

## Overview
A practical multi-agent system for automated sports betting on Neural, with clear separation between always-on monitoring agents and on-demand analysis agents.

## Agent Types

### 1. Always-On Agents (24/7 Operation)

#### **Data Collection Agent**
**Purpose:** Continuously gather and process data from all sources
**Location:** `agent_consumers/DataCollector/`
**Responsibilities:**
- Monitor ESPN for live games and scores
- Track Twitter sentiment in real-time
- Watch Neural market prices and volumes
- Store historical data for analysis
- Detect anomalies and significant events

**Trigger Conditions:**
- Price movements > 5% in 1 minute
- Volume spikes > 3x average
- Major game events (touchdowns, injuries)
- Sentiment shifts > 0.3 points
- Breaking news from verified sources

**Implementation:**
```python
class DataCollectionAgent:
    async def run_forever(self):
        """Runs continuously, consuming from Redis"""
        while True:
            # Monitor ESPN
            espn_data = await self.consume_espn_stream()
            
            # Monitor Twitter
            sentiment = await self.consume_twitter_stream()
            
            # Monitor Neural
            markets = await self.consume_neural_stream()
            
            # Detect triggers
            if self.detect_opportunity(espn_data, sentiment, markets):
                await self.trigger_analysis()
```

#### **Portfolio Monitor Agent**
**Purpose:** Continuously monitor positions and enforce risk limits
**Location:** `agent_consumers/PortfolioMonitor/`
**Responsibilities:**
- Track all open positions
- Monitor P&L in real-time
- Enforce position limits (max 5% per position)
- Trigger stop-losses (10% loss)
- Alert on margin requirements
- Calculate portfolio Greeks (Delta, Gamma)

**Risk Rules:**
- Max position size: $100 or 5% of bankroll
- Max daily loss: 20% of bankroll
- Stop-loss: 10% per position
- Max correlation: 0.7 between positions
- Margin requirement: 2x Kelly fraction

**Implementation:**
```python
class PortfolioMonitorAgent:
    async def monitor_positions(self):
        """Check positions every 10 seconds"""
        while True:
            positions = await self.get_open_positions()
            
            for position in positions:
                # Check stop-loss
                if position.unrealized_pnl < -0.10 * position.cost:
                    await self.execute_stop_loss(position)
                
                # Check take-profit
                if position.unrealized_pnl > 0.30 * position.cost:
                    await self.execute_take_profit(position)
            
            # Check portfolio limits
            if self.portfolio.daily_loss > 0.20 * self.bankroll:
                await self.halt_trading()
            
            await asyncio.sleep(10)
```

### 2. On-Demand Agents (Triggered)

#### **Game Analyst Agent**
**Purpose:** Deep analysis of specific games when requested
**Trigger Conditions:**
- User requests analysis of specific game
- Data Collection Agent detects high-opportunity game
- Pre-game analysis (1 hour before kickoff)

**Analysis Workflow:**
```python
class GameAnalystAgent:
    async def analyze_game(self, game_id: str, teams: tuple):
        """Comprehensive game analysis"""
        
        # 1. Historical Analysis
        historical = await self.get_team_history(teams)
        h2h_record = await self.get_head_to_head(teams)
        
        # 2. Current Form
        recent_games = await self.get_recent_performance(teams, last_n=5)
        injuries = await self.get_injury_report(teams)
        
        # 3. Market Analysis
        market_data = await self.get_neural_markets(game_id)
        implied_prob = self.calculate_implied_probability(market_data)
        
        # 4. Sentiment Analysis
        twitter_sentiment = await self.get_game_sentiment(teams)
        news_sentiment = await self.analyze_news(teams)
        
        # 5. Weather & Venue
        weather = await self.get_weather_conditions(game_id)
        venue_advantage = self.calculate_home_advantage(teams[0])
        
        # 6. Generate Prediction
        prediction = await self.llm_analyze({
            "historical": historical,
            "form": recent_games,
            "injuries": injuries,
            "market": market_data,
            "sentiment": twitter_sentiment,
            "weather": weather
        })
        
        return {
            "recommended_position": prediction.position,
            "confidence": prediction.confidence,
            "kelly_fraction": prediction.kelly_size,
            "key_factors": prediction.factors,
            "risk_warnings": prediction.risks
        }
```

#### **Arbitrage Hunter Agent**
**Purpose:** Find and execute arbitrage opportunities
**Trigger Conditions:**
- YES + NO prices < 0.98 (2% arbitrage)
- Cross-market price differences > 3%
- Correlated markets with price divergence

**Implementation:**
```python
class ArbitrageHunterAgent:
    async def hunt_arbitrage(self):
        """Called when opportunity detected"""
        
        markets = await self.get_all_markets()
        
        for market in markets:
            # Direct arbitrage
            if market.yes_price + market.no_price < 0.98:
                profit = 0.98 - (market.yes_price + market.no_price)
                if profit > 0.02:  # 2% minimum
                    await self.execute_arbitrage(market, profit)
            
            # Cross-market arbitrage
            related = self.find_related_markets(market)
            for related_market in related:
                if self.calculate_divergence(market, related_market) > 0.03:
                    await self.execute_cross_arbitrage(market, related_market)
```

#### **Strategy Optimizer Agent**
**Purpose:** Optimize trading strategies based on performance
**Trigger Conditions:**
- End of day analysis
- After 20 trades completed
- Significant drawdown detected
- User requests optimization

**Optimization Process:**
```python
class StrategyOptimizerAgent:
    async def optimize_strategy(self):
        """Analyze and optimize trading strategy"""
        
        # 1. Performance Analysis
        trades = await self.get_recent_trades(days=7)
        metrics = self.calculate_metrics(trades)
        
        # 2. Identify Patterns
        winning_patterns = self.analyze_winners(trades)
        losing_patterns = self.analyze_losers(trades)
        
        # 3. Backtest Adjustments
        adjustments = {
            "kelly_fraction": self.optimize_kelly(trades),
            "confidence_threshold": self.optimize_threshold(trades),
            "stop_loss": self.optimize_stop_loss(trades),
            "position_size": self.optimize_position_size(trades)
        }
        
        # 4. Recommend Changes
        recommendations = await self.llm_analyze({
            "current_performance": metrics,
            "winning_patterns": winning_patterns,
            "losing_patterns": losing_patterns,
            "proposed_adjustments": adjustments
        })
        
        return recommendations
```

## Communication Architecture

### Agent Communication Pattern
```
                    Redis Pub/Sub
                         |
        +----------------+----------------+
        |                |                |
   Always-On         Triggers        On-Demand
   Agents            Service         Agents
        |                |                |
   Data Collector    Evaluates      Game Analyst
   Portfolio Mon.    Conditions     Arbitrage Hunter
                          |          Strategy Opt.
                     Activates
                     On-Demand
```

### Trigger Service
```python
class TriggerService:
    """Coordinates between always-on and on-demand agents"""
    
    def __init__(self):
        self.triggers = {
            'price_spike': {
                'condition': lambda d: d['price_change'] > 0.05,
                'agent': 'ArbitrageHunter',
                'priority': 'HIGH'
            },
            'game_start': {
                'condition': lambda d: d['time_to_game'] < 3600,
                'agent': 'GameAnalyst',
                'priority': 'MEDIUM'
            },
            'sentiment_shift': {
                'condition': lambda d: abs(d['sentiment_change']) > 0.3,
                'agent': 'GameAnalyst',
                'priority': 'HIGH'
            },
            'daily_review': {
                'condition': lambda d: d['time'] == '23:00',
                'agent': 'StrategyOptimizer',
                'priority': 'LOW'
            }
        }
    
    async def evaluate(self, event):
        """Evaluate if event should trigger an agent"""
        for trigger_name, trigger in self.triggers.items():
            if trigger['condition'](event):
                await self.activate_agent(
                    trigger['agent'],
                    event,
                    trigger['priority']
                )
```

## Practical Scenarios

### Scenario 1: User Requests Game Analysis
```python
async def handle_user_request(game: str):
    """User wants analysis of Chiefs vs Bills"""
    
    # 1. Data Collector provides current data
    current_data = await data_collector.get_game_data(game)
    
    # 2. Trigger Game Analyst
    analysis = await game_analyst.analyze_game(
        game_id=current_data['game_id'],
        teams=(current_data['home'], current_data['away'])
    )
    
    # 3. Check with Portfolio Monitor
    risk_check = await portfolio_monitor.check_capacity(
        proposed_size=analysis['recommended_position']
    )
    
    # 4. Execute if approved
    if risk_check['approved']:
        await trade_executor.place_order(analysis)
    
    return analysis
```

### Scenario 2: Automatic Opportunity Detection
```python
async def automatic_trading_flow():
    """Automated detection and execution"""
    
    # Data Collector detects opportunity
    # (Running continuously)
    if data_collector.detect_opportunity():
        opportunity = {
            'type': 'ARBITRAGE',
            'market': 'NFL-CHIEFS-WIN',
            'profit': 0.03
        }
        
        # Trigger Arbitrage Hunter
        await trigger_service.activate('ArbitrageHunter', opportunity)
        
        # Arbitrage Hunter analyzes
        strategy = await arbitrage_hunter.analyze(opportunity)
        
        # Portfolio Monitor checks risk
        if portfolio_monitor.check_limits(strategy):
            # Execute trade
            await trade_executor.execute(strategy)
```

### Scenario 3: Live Game Monitoring
```python
async def monitor_live_game(game_id: str):
    """Monitor game in progress"""
    
    # Data Collector streams live data
    async for event in data_collector.stream_game(game_id):
        
        # Check for significant events
        if event.type == 'TOUCHDOWN':
            # Quick market analysis
            market_impact = await game_analyst.quick_analysis(event)
            
            # Check if we have positions
            positions = await portfolio_monitor.get_game_positions(game_id)
            
            if positions:
                # Adjust positions based on game flow
                adjustments = await portfolio_monitor.calculate_adjustments(
                    positions, 
                    market_impact
                )
                await trade_executor.adjust_positions(adjustments)
        
        elif event.type == 'INJURY':
            # Immediate risk assessment
            await portfolio_monitor.emergency_check(event)
```

### Scenario 4: End-of-Day Optimization
```python
async def daily_review():
    """Runs at 11 PM EST every day"""
    
    # Portfolio Monitor provides day's data
    daily_performance = await portfolio_monitor.get_daily_summary()
    
    # Trigger Strategy Optimizer
    optimization = await strategy_optimizer.analyze_performance(
        trades=daily_performance['trades'],
        pnl=daily_performance['pnl'],
        metrics=daily_performance['metrics']
    )
    
    # Apply recommended changes
    if optimization['recommendations']:
        await apply_strategy_updates(optimization['recommendations'])
    
    # Generate report
    report = await generate_daily_report(daily_performance, optimization)
    await send_report(report)
```

## Implementation Priority

### Phase 1: Core Infrastructure (Week 1)
1. ✅ Data Collection Agent (always-on)
2. ✅ Portfolio Monitor Agent (always-on)
3. ✅ Trigger Service
4. ✅ Redis communication layer

### Phase 2: Analysis Agents (Week 2)
1. Game Analyst Agent
2. Arbitrage Hunter Agent
3. Basic trade execution

### Phase 3: Optimization (Week 3)
1. Strategy Optimizer Agent
2. Performance tracking
3. Backtesting framework

### Phase 4: Advanced Features (Week 4)
1. Multi-game correlation analysis
2. Cross-market arbitrage
3. Advanced risk metrics
4. Machine learning predictions

## Configuration

### Agent Settings
```yaml
# config/agents.yaml
data_collector:
  enabled: true
  mode: always_on
  redis_channels:
    - neural:markets
    - espn:games
    - twitter:sentiment
  
portfolio_monitor:
  enabled: true
  mode: always_on
  check_interval: 10  # seconds
  risk_limits:
    max_position_pct: 0.05
    max_daily_loss_pct: 0.20
    stop_loss_pct: 0.10
    take_profit_pct: 0.30
    
game_analyst:
  enabled: true
  mode: on_demand
  llm_model: gpt-4
  confidence_threshold: 0.70
  
arbitrage_hunter:
  enabled: true
  mode: on_demand
  min_profit_pct: 0.02
  max_position_size: 100
  
strategy_optimizer:
  enabled: true
  mode: scheduled
  schedule: "23:00"  # Daily at 11 PM
  lookback_days: 7
```

## Monitoring Dashboard

### Real-Time Metrics
- Active positions and P&L
- Current market opportunities
- Agent status (running/idle)
- Risk metrics (exposure, drawdown)
- Recent trades and performance

### Historical Analytics
- Win rate by market type
- Profit factor
- Sharpe ratio
- Maximum drawdown
- Best/worst trades

## Next Steps

1. **Simplify Agentuity integration** - Only use for complex LLM analysis
2. **Implement always-on agents** as simple Python services
3. **Create trigger service** for coordination
4. **Build monitoring dashboard** for visibility
5. **Test with paper trading** before going live