# 📚 Documentation Index

Complete documentation for the Neural Trading Platform - organized by topic and experience level.

## 🚀 Start Here

### For New Users
1. **[Getting Started](GETTING_STARTED.md)** - Installation, setup, and your first trade
2. **[System Overview](SYSTEM_OVERVIEW.md)** - How the platform works end-to-end
3. **[Quick Weather Demo](../scripts/quick_weather_demo.py)** - See real-time data in action

### For Developers
1. **[SDK Documentation](SDK_DOCUMENTATION.md)** - Build custom data adapters
2. **[Data Sources Guide](DATA_SOURCES_GUIDE.md)** - Configure and optimize data feeds
3. **[Trading Logic](TRADING_LOGIC.md)** - Signal generation and execution

### For Traders
1. **[Game Configuration Guide](GAME_CONFIGURATION_GUIDE.md)** - Monitor specific games
2. **[Trading Logic](TRADING_LOGIC.md)** - Understand trading decisions
3. **[Risk Management](TRADING_LOGIC.md#risk-management)** - Safety features

## 📖 Documentation by Category

### Architecture & Design
- **[System Overview](SYSTEM_OVERVIEW.md)** - Complete architecture with diagrams
- **[Architecture](ARCHITECTURE.md)** - Technical deep dive
- **[Plugin Architecture](PLUGIN_ARCHITECTURE.md)** - Extension system design
- **[Simplified Architecture Summary](SIMPLIFIED_ARCHITECTURE_SUMMARY.md)** - Quick overview

### Data & SDK
- **[SDK Documentation](SDK_DOCUMENTATION.md)** - Complete SDK reference
- **[Data Sources Guide](DATA_SOURCES_GUIDE.md)** - All data source details
- **[src/sdk/README.md](../src/sdk/README.md)** - SDK quick reference

### Trading & Strategy
- **[Trading Logic](TRADING_LOGIC.md)** - How trades are made
- **[Game Configuration Guide](GAME_CONFIGURATION_GUIDE.md)** - Game monitoring setup
- **[Backtesting](../scripts/run_backtest.py)** - Historical testing

### Development & Deployment
- **[Git Workflow](GIT_WORKFLOW.md)** - Version control practices
- **[GitHub Push Guide](GITHUB_PUSH_GUIDE.md)** - Deployment process
- **[Contributing](../CONTRIBUTING.md)** - Contribution guidelines

### Platform & Vision
- **[Platform Roadmap](../PLATFORM_ROADMAP.md)** - Future development plans
- **[Investor Demo](../INVESTOR_DEMO.md)** - Platform capabilities
- **[MVP Plan](MVP_PLAN.md)** - Minimum viable product

### Agents & Components
- **[Agents](AGENTS.md)** - Trading agent descriptions
- **[CLAUDE.md](../CLAUDE.md)** - AI assistant guidelines

## 🔑 Key Concepts

### Data Flow
```
Data Sources → SDK Adapters → Redis → Agents → Trading
```

### Event Types
- **ODDS_CHANGE** - Sportsbook line movements
- **WEATHER_UPDATE** - Venue weather changes
- **SENTIMENT_SHIFT** - Social sentiment swings
- **GAME_EVENT** - In-game occurrences

### Trading Signals
- **Confidence**: 0.0 to 1.0 probability
- **Impact**: low, medium, high, critical
- **Edge**: Expected profit percentage
- **Kelly Sizing**: 25% fractional Kelly

## 🧪 Examples & Demos

### Scripts
- **[demo_sdk.py](../scripts/demo_sdk.py)** - Full SDK demonstration
- **[test_weather_adapter.py](../scripts/test_weather_adapter.py)** - Weather API testing
- **[quick_weather_demo.py](../scripts/quick_weather_demo.py)** - Quick weather check
- **[demo_investor.py](../scripts/demo_investor.py)** - Investor presentation

### Configuration Files
- **[data_sources.yaml](../config/data_sources.yaml)** - Data source setup
- **[.env.example](../.env.example)** - Environment variables

## 📊 Current Implementation Status

### ✅ Completed
- Data Source SDK framework
- Weather monitoring (OpenWeatherMap)
- DraftKings odds adapter
- Redis pub/sub distribution
- Kelly Criterion position sizing
- Risk management system
- Documentation suite

### 🔄 In Progress
- Reddit sentiment analysis
- ESPN GameCast integration
- Kalshi production trading

### 📅 Planned
- Machine learning models
- Multi-sport expansion
- Mobile monitoring app

## 🔍 Quick Reference

### API Keys Required
```bash
KALSHI_API_KEY_ID=xxx        # Trading
KALSHI_API_KEY=xxx           # Trading
OPENWEATHER_API_KEY=xxx      # Weather (included)
REDDIT_CLIENT_ID=xxx         # Sentiment (optional)
REDDIT_CLIENT_SECRET=xxx     # Sentiment (optional)
```

### Key Commands
```bash
# Test SDK
python scripts/demo_sdk.py

# Start trading
python scripts/run_agents.py

# Run backtest
python scripts/run_backtest.py

# Monitor weather
python scripts/quick_weather_demo.py
```

### Performance Targets
- **Latency**: <3 seconds event-to-trade
- **Win Rate**: >65%
- **Sharpe Ratio**: >2.0
- **Max Drawdown**: <20%

## 📞 Getting Help

1. Check relevant documentation above
2. Review [scripts/README.md](../scripts/README.md) for examples
3. See [CONTRIBUTING.md](../CONTRIBUTING.md) for development
4. Open an issue on GitHub

---

*Last updated: August 31, 2025*
*Platform version: 1.0.0*
*SDK version: 1.0.0*