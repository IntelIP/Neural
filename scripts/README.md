# Scripts Directory

This directory contains utility scripts for testing and demonstrating the Neural Trading Platform.

## Production Scripts

### `run_agents.py`
Launches all trading agents in production mode with Redis pub/sub.
```bash
python scripts/run_agents.py
```

### `run_backtest.py`
Runs historical backtests on trading strategies.
```bash
python scripts/run_backtest.py --strategy sharp_money --days 30
```

## Demo Scripts

### `demo_sdk.py`
Complete demonstration of the Data Source SDK capabilities.
- Shows how to use multiple data sources
- Demonstrates event processing
- Includes custom adapter example
```bash
python scripts/demo_sdk.py
```

### `demo_investor.py`
Interactive demo for potential investors showing platform capabilities.
```bash
python scripts/demo_investor.py
```

## Testing Scripts

### `test_weather_adapter.py`
Comprehensive test suite for OpenWeatherMap integration.
- Tests API connection
- Fetches real weather data
- Simulates extreme weather scenarios
```bash
export OPENWEATHER_API_KEY="your_key"
python scripts/test_weather_adapter.py
```

### `quick_weather_demo.py`
Quick demonstration of weather monitoring at NFL stadiums.
```bash
python scripts/quick_weather_demo.py
```

## Environment Setup

Before running scripts, ensure you have:

1. **API Keys configured** in `.env` or environment variables:
   - `KALSHI_API_KEY_ID` and `KALSHI_API_KEY`
   - `OPENWEATHER_API_KEY` (currently: 78596505b0f5fea89e98ebcbf3bd6e21)
   - `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET` (optional)

2. **Redis running**:
   ```bash
   redis-server
   ```

3. **Dependencies installed**:
   ```bash
   pip install -r requirements.txt
   ```