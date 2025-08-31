# Game Configuration Guide - Neural Trading Platform

## Overview

This guide shows you how to configure the platform to monitor and trade specific games. Whether you're tracking a single high-stakes game or multiple games across different sports, proper configuration is key to success.

---

## Basic Game Configuration

### Single Game Focus

To monitor one specific game (e.g., Chiefs vs Bills playoff game):

```yaml
# config/game_config.yaml
game_monitoring:
  mode: "single_game"
  game:
    sport: "NFL"
    home_team: "Kansas City Chiefs"
    away_team: "Buffalo Bills"
    game_id: "KC-BUF-2024-01-21"
    start_time: "2024-01-21T18:30:00Z"
    venue: "Arrowhead Stadium"
    
  # Kalshi markets to trade
  kalshi_markets:
    - "NFL-KC-BUF-WINNER"
    - "NFL-KC-BUF-SPREAD"
    - "NFL-KC-BUF-TOTAL"
    
  # Data sources specific to this game
  data_sources:
    draftkings:
      event_id: "dk_12345"
      markets: ["spread", "total", "props"]
    reddit:
      game_thread: "https://reddit.com/r/nfl/comments/abc123"
      team_subs: ["KansasCityChiefs", "buffalobills"]
    weather:
      stadium_coords: [39.0489, -94.4839]
      update_frequency: 300  # 5 minutes
```

### Multiple Games

To monitor several games simultaneously:

```yaml
# config/game_config.yaml
game_monitoring:
  mode: "multi_game"
  max_concurrent: 3  # Resource limit
  
  games:
    - game_id: "KC-BUF-2024-01-21"
      sport: "NFL"
      priority: 1  # Highest priority
      allocation: 0.5  # 50% of capital
      
    - game_id: "LAL-BOS-2024-01-21"
      sport: "NBA"
      priority: 2
      allocation: 0.3  # 30% of capital
      
    - game_id: "NYY-LAD-2024-01-21"
      sport: "MLB"
      priority: 3
      allocation: 0.2  # 20% of capital
```

---

## Sport-Specific Configurations

### NFL Configuration

```yaml
# config/sports/nfl.yaml
nfl:
  # Game phases to track
  phases:
    pregame:
      duration: 60  # minutes before kickoff
      data_sources: ["draftkings", "reddit", "weather"]
      focus: "line_movements"
      
    first_half:
      quarters: [1, 2]
      data_sources: ["all"]
      focus: "game_events"
      
    halftime:
      duration: 12  # minutes
      data_sources: ["draftkings", "reddit"]
      focus: "adjustment_analysis"
      
    second_half:
      quarters: [3, 4]
      data_sources: ["all"]
      focus: "momentum_shifts"
      
    overtime:
      data_sources: ["all"]
      focus: "sudden_death"
  
  # Key events to monitor
  events:
    high_impact:
      - touchdown
      - field_goal
      - turnover
      - injury_star_player
    medium_impact:
      - first_down
      - punt
      - penalty_major
    low_impact:
      - penalty_minor
      - timeout
      
  # Weather thresholds
  weather_impact:
    wind:
      low: 10  # mph
      medium: 15
      high: 20
      extreme: 25
    precipitation:
      light: 0.05  # inches/hour
      moderate: 0.1
      heavy: 0.2
```

### NBA Configuration

```yaml
# config/sports/nba.yaml
nba:
  # Faster-paced game settings
  update_frequency: 2  # seconds
  
  phases:
    pregame:
      duration: 30  # minutes
      focus: "injury_reports"
      
    quarters:
      duration: 12  # minutes each
      overtime: 5   # minutes
      
  events:
    high_impact:
      - "player_ejection"
      - "injury_star"
      - "technical_foul"
    momentum:
      - "10_point_run"
      - "timeout_called"
      - "and_one"
      
  # No weather but venue matters
  venue_factors:
    altitude:  # Denver advantage
      high: 5280  # feet
    home_court:
      advantage: 0.03  # 3% edge
```

### MLB Configuration

```yaml
# config/sports/mlb.yaml
mlb:
  # Slower pace, longer game
  update_frequency: 10  # seconds
  
  phases:
    early_innings: [1, 2, 3]
    middle_innings: [4, 5, 6]
    late_innings: [7, 8, 9]
    extra_innings: [10, 11, 12]  # etc
    
  events:
    high_impact:
      - "home_run"
      - "grand_slam"
      - "pitcher_change"
      - "injury_pitcher"
    medium_impact:
      - "double"
      - "triple"
      - "stolen_base"
      - "double_play"
      
  weather_critical: true  # Very weather dependent
  wind_direction_matters: true  # For home runs
```

---

## Advanced Configuration Patterns

### Dynamic Game Discovery

Automatically find and monitor games:

```yaml
# config/auto_discovery.yaml
auto_discovery:
  enabled: true
  
  # Find games with these criteria
  criteria:
    min_kalshi_volume: 10000  # Minimum liquidity
    min_edge_required: 0.03   # 3% edge
    sports: ["NFL", "NBA"]
    
  # Scanning schedule
  schedule:
    scan_interval: 3600  # Check hourly
    lookahead: 24  # hours
    
  # Auto-subscribe to found games
  auto_subscribe:
    enabled: true
    max_games: 5
    capital_per_game: 0.2  # 20% max
```

### Conditional Configuration

Different settings based on game state:

```yaml
# config/conditional.yaml
conditional_rules:
  # Close game = more conservative
  - condition: "score_difference < 7"
    adjustments:
      position_size_multiplier: 0.5
      confidence_threshold: 0.85
      
  # Blowout = look for garbage time value
  - condition: "score_difference > 21"
    adjustments:
      focus: "player_props"
      reduce_main_markets: true
      
  # Bad weather = bet unders
  - condition: "wind_speed > 20"
    adjustments:
      prefer_position: "under"
      total_confidence_boost: 0.1
      
  # Prime time = more liquidity
  - condition: "time >= 20:00"
    adjustments:
      position_size_multiplier: 1.5
      aggression: "high"
```

### Team-Specific Configuration

Track teams differently based on style:

```yaml
# config/teams/chiefs.yaml
team_config:
  name: "Kansas City Chiefs"
  
  # Team tendencies
  tendencies:
    passing_heavy: true
    comeback_ability: "elite"
    weather_resilience: "moderate"
    
  # Key players to monitor
  key_players:
    - name: "Patrick Mahomes"
      position: "QB"
      injury_impact: "critical"
      
    - name: "Travis Kelce"
      position: "TE"
      injury_impact: "high"
      
  # Historical patterns
  patterns:
    slow_starts: true  # Often behind early
    strong_finish: true  # Dominant 4th quarter
    playoff_experience: "excellent"
    
  # Betting adjustments
  adjustments:
    spread:
      home: -0.5  # Tends to cover at home
      away: +0.5   # Struggles on road
    total:
      dome: +3     # Higher scoring indoors
      outdoor: -2  # Lower in weather
```

---

## Configuration Templates

### High-Frequency Trading Template

For maximum speed on liquid markets:

```yaml
# config/templates/hft.yaml
template: "high_frequency"

settings:
  latency_target: 100  # milliseconds
  
  data_sources:
    priorities:
      draftkings: 1  # Fastest
      espn: 2
      reddit: 5  # Slowest
      
  execution:
    order_type: "market"  # No limit orders
    size_limits:
      min: 100
      max: 5000
      
  risk:
    stop_loss: 0.02  # 2% tight stop
    position_time: 300  # 5 min max hold
    
  optimization:
    cache_ttl: 1  # 1 second cache
    batch_orders: false  # Send immediately
```

### Value Hunting Template

For finding mispriced markets:

```yaml
# config/templates/value.yaml
template: "value_hunting"

settings:
  scan_frequency: 60  # Check every minute
  
  criteria:
    min_edge: 0.05  # 5% minimum
    min_confidence: 0.75
    
  data_correlation:
    required_sources: 2  # Need 2+ sources
    agreement_threshold: 0.8
    
  execution:
    order_type: "limit"
    price_improvement: 0.01
    
  hold_strategy:
    target_profit: 0.10  # 10%
    max_hold_time: 3600  # 1 hour
```

### Safe Mode Template

For conservative trading:

```yaml
# config/templates/safe.yaml
template: "conservative"

settings:
  risk_limits:
    max_position: 0.02  # 2% of capital
    max_daily_loss: 0.05  # 5% stop
    max_concurrent: 2  # Max 2 positions
    
  requirements:
    min_confidence: 0.85
    min_edge: 0.07
    sources_required: 3
    
  execution:
    order_type: "limit"
    partial_fills: false
    
  monitoring:
    alert_on_loss: true
    require_confirmation: true
```

---

## Environment-Specific Settings

### Development

```yaml
# config/environments/development.yaml
environment: "development"

settings:
  trading:
    enabled: false  # No real trades
    paper_trading: true
    
  logging:
    level: "DEBUG"
    verbose: true
    
  data_sources:
    use_mock: true  # Simulated data
    replay_historical: true
    
  limits:
    max_requests_per_minute: 10
    cache_everything: true
```

### Production

```yaml
# config/environments/production.yaml
environment: "production"

settings:
  trading:
    enabled: true
    paper_trading: false
    
  logging:
    level: "INFO"
    verbose: false
    
  monitoring:
    alerts: true
    health_checks: 60  # seconds
    
  failover:
    backup_redis: "redis://backup:6379"
    emergency_stop: true
```

---

## YAML Configuration Basics

### Structure

```yaml
# Comments start with #

# Key-value pairs
simple_key: "value"
number: 42
boolean: true

# Nested objects
parent:
  child: "value"
  another_child: 123
  
# Lists
items:
  - first
  - second
  - third
  
# List of objects
games:
  - id: "game1"
    sport: "NFL"
  - id: "game2"
    sport: "NBA"
```

### Environment Variables

```yaml
# Use ${} for environment variables
api_key: ${KALSHI_API_KEY}

# With defaults
redis_url: ${REDIS_URL:-redis://localhost:6379}

# Nested usage
credentials:
  username: ${API_USER}
  password: ${API_PASS}
```

### Including Other Files

```yaml
# Include another YAML file
base_config: !include base.yaml

# Merge configurations
settings:
  <<: !include defaults.yaml
  custom_value: 42  # Override
```

---

## Loading Configurations in Code

### Basic Loading

```python
import yaml
import os

def load_config(path):
    """Load YAML configuration with env var substitution."""
    with open(path, 'r') as f:
        content = f.read()
    
    # Substitute environment variables
    for match in re.findall(r'\${(\w+)}', content):
        value = os.getenv(match, '')
        content = content.replace(f'${{{match}}}', value)
    
    return yaml.safe_load(content)

# Usage
config = load_config('config/game_config.yaml')
game_id = config['game_monitoring']['game']['game_id']
```

### Advanced Configuration Manager

```python
class ConfigManager:
    """Manage all platform configurations."""
    
    def __init__(self, base_path='config'):
        self.base_path = base_path
        self.configs = {}
        
    def load_all(self):
        """Load all configuration files."""
        for file in os.listdir(self.base_path):
            if file.endswith('.yaml'):
                name = file[:-5]  # Remove .yaml
                self.configs[name] = self.load_file(file)
                
    def get_game_config(self, game_id):
        """Get configuration for specific game."""
        base = self.configs.get('game_config', {})
        overrides = self.configs.get(f'games/{game_id}', {})
        
        # Merge configurations
        return {**base, **overrides}
        
    def get_active_games(self):
        """Get list of games to monitor."""
        config = self.configs.get('game_config', {})
        
        if config.get('mode') == 'single_game':
            return [config['game']]
        elif config.get('mode') == 'multi_game':
            return config['games']
        else:
            return []
```

---

## Testing Your Configuration

### Validation Script

```python
# scripts/validate_config.py
import yaml
import sys

def validate_game_config(config):
    """Validate game configuration."""
    errors = []
    
    # Check required fields
    required = ['game_monitoring', 'kalshi_markets', 'data_sources']
    for field in required:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    # Validate game monitoring
    game = config.get('game_monitoring', {}).get('game', {})
    if not game.get('game_id'):
        errors.append("Game ID is required")
    
    # Validate data sources
    sources = config.get('data_sources', {})
    if not sources:
        errors.append("At least one data source required")
    
    return errors

# Run validation
config = yaml.safe_load(open('config/game_config.yaml'))
errors = validate_game_config(config)

if errors:
    print("Configuration errors:")
    for error in errors:
        print(f"  - {error}")
    sys.exit(1)
else:
    print("Configuration valid!")
```

### Dry Run

Test configuration without trading:

```bash
# Test configuration loading
python -c "
from src.config import ConfigManager
cm = ConfigManager()
cm.load_all()
print(f'Loaded {len(cm.configs)} configurations')
print(f'Active games: {cm.get_active_games()}')
"

# Test with specific game
python scripts/test_game_config.py --game-id KC-BUF-2024-01-21 --dry-run
```

---

## Common Configuration Patterns

### 1. Start Conservative, Scale Up

```yaml
# Week 1: Learning
week_1:
  max_position: 0.01  # 1% only
  confidence_required: 0.90
  
# Week 2: Gaining confidence
week_2:
  max_position: 0.02
  confidence_required: 0.85
  
# Week 3: Full deployment
week_3:
  max_position: 0.05
  confidence_required: 0.75
```

### 2. Time-Based Adjustments

```yaml
time_adjustments:
  # Pre-game: Look for value
  - time: "-60 to 0"  # 60 min before
    strategy: "value_hunting"
    
  # Early game: React to events
  - time: "0 to 30"  # First 30 min
    strategy: "event_driven"
    
  # Late game: Arbitrage
  - time: "90 to end"
    strategy: "arbitrage"
```

### 3. Liquidity-Based Sizing

```yaml
position_sizing:
  rules:
    - volume: [0, 10000]
      max_position: 100  # Small position
      
    - volume: [10000, 100000]
      max_position: 1000  # Medium
      
    - volume: [100000, null]
      max_position: 5000  # Large
```

---

## Troubleshooting Configuration Issues

### Config Not Loading?

1. Check YAML syntax:
```bash
python -c "import yaml; yaml.safe_load(open('config/game_config.yaml'))"
```

2. Verify environment variables:
```bash
env | grep KALSHI
```

3. Check file permissions:
```bash
ls -la config/
```

### Wrong Game Being Monitored?

1. Verify game ID matches:
```python
print(config['game_monitoring']['game']['game_id'])
```

2. Check timezone:
```yaml
start_time: "2024-01-21T18:30:00Z"  # Always use UTC
```

3. Confirm data source IDs:
```yaml
draftkings:
  event_id: "dk_12345"  # Must match DraftKings
```

---

## Best Practices

1. **Version Control Your Configs**
   - Track changes
   - Easy rollback
   - Team collaboration

2. **Use Templates**
   - Consistency across games
   - Faster setup
   - Fewer errors

3. **Separate Secrets**
   ```yaml
   # config/game.yaml
   api_key: ${KALSHI_KEY}  # Don't hardcode
   
   # .env (not in git)
   KALSHI_KEY=actual_key_here
   ```

4. **Test Before Trading**
   - Dry run first
   - Paper trade
   - Small positions

5. **Monitor and Adjust**
   - Track performance
   - Adjust thresholds
   - Learn from each game

---

## Summary

Proper configuration is critical for successful trading. Start with simple single-game configs, test thoroughly, then expand to multiple games and advanced patterns. The YAML format makes it easy to adjust settings without changing code.

Next: Read [TRADING_LOGIC.md](TRADING_LOGIC.md) to understand how the platform makes trading decisions.