# Flexible Sports WebSocket Client

A clean, user-driven approach to streaming sports market data from Kalshi using the existing Neural SDK infrastructure.

## Overview

This implementation leverages the existing SDK market discovery functions to create a flexible WebSocket client that:

1. **Discovers markets by sport** using existing SDK functions
2. **Lets users select specific markets** (no hardcoded filtering)
3. **Streams only user-selected markets** with real-time updates

## Key Components

### 1. Market Discovery (`sports_market_browser.py`)
- Uses existing `KalshiMarketDiscovery.discover_sport_markets()`
- Extended sports config with Pro Football, College Football, Basketball, etc.
- Interactive browsing and search functionality

### 2. Flexible WebSocket Client (`flexible_sports_websocket.py`)
- Clean WebSocket streaming with user-selected markets
- No hardcoded team/game filtering logic
- Real-time market data and trade updates
- Session summaries and statistics

### 3. Auto-Select Demo (`auto_select_demo.py`)
- Demonstrates programmatic market selection
- Examples: Chiefs vs Chargers, Game markets only, Search-based selection

## Quick Start

### Browse Available Markets
```bash
# Interactive browser for all sports
python sports_market_browser.py

# Quick discovery for specific sport
python sports_market_browser.py nfl
python sports_market_browser.py college_football
```

### Stream Selected Markets
```bash
# Interactive market selection and streaming
python flexible_sports_websocket.py

# Quick mode for specific sport
python flexible_sports_websocket.py nfl
```

### Run Demos
```bash
# All demos
python auto_select_demo.py

# Specific demo (Chiefs vs Chargers)
python auto_select_demo.py 1
```

## Supported Sports

| Sport | Display Name | Status | Markets Found |
|-------|--------------|--------|---------------|
| NFL | Pro Football | 🟢 Active | 62 markets |
| COLLEGE_FOOTBALL | College Football | 🟢 Active | TBD |
| NBA | Pro Basketball (M) | 🟢 Active | 74 series |
| WNBA | Pro Basketball (W) | 🟢 Active | TBD |
| CFP | College Football Playoff | 🔴 Off-season | 0 markets |

## Example Usage

### 1. Chiefs vs Chargers Auto-Selection
```python
from neural_sdk.data_pipeline.sports_config import Sport
from flexible_sports_websocket import FlexibleSportsWebSocket

client = FlexibleSportsWebSocket()
await client.initialize()

# Discover NFL markets
markets = await client.browser.discover_markets_by_sport(Sport.NFL)

# Auto-select Chiefs/Chargers markets
chiefs_chargers = [m['ticker'] for m in markets 
                   if any(team in m['ticker'] for team in ['KC', 'LAC'])]

# Stream selected markets
client.selected_markets = chiefs_chargers
await client.stream_selected_markets(duration_seconds=60)
```

### 2. Interactive Market Selection
```python
client = FlexibleSportsWebSocket()
await client.initialize()

# Interactive discovery and selection
selected_tickers = await client.discover_and_select_markets(Sport.NFL)

# Stream user selections
await client.stream_selected_markets()
```

## Architecture Benefits

### ✅ What We Fixed:
- **Removed hardcoded filtering** - no more client-side team name matching
- **Leveraged existing SDK functions** - uses `discover_sport_markets()` properly
- **User-driven selection** - flexibility to choose any markets
- **Efficient subscriptions** - only subscribe to selected markets
- **Extensible sports config** - easy to add new sports

### ✅ What We Improved:
- **Market discovery** - found actual NFL game markets like `KXNFLGAME-25SEP05KCLAC-KC`
- **Real-time streaming** - live trades and market data
- **Clean separation** - discovery vs streaming vs selection
- **Interactive tools** - browse, search, and select markets
- **Programmatic options** - auto-select via code logic

## Live Demo Results

The Chiefs vs Chargers demo successfully:
- 🔍 Discovered 62 NFL markets
- 🎯 Auto-selected 6 Chiefs/Chargers markets including `KXNFLGAME-25SEP05KCLAC-KC`
- 📡 Streamed live market data for 30 seconds
- 💰 Captured 17 real trades including large trades (566 contracts)

## Files

| File | Purpose |
|------|---------|
| `sports_market_browser.py` | Interactive market discovery and browsing |
| `flexible_sports_websocket.py` | Clean WebSocket client with user selection |
| `auto_select_demo.py` | Demonstration of programmatic market selection |
| `test_market_discovery.py` | Testing existing SDK discovery functions |
| `neural_sdk/data_pipeline/sports_config.py` | Extended sports configuration |

## Legacy Files (Reference)

| File | Purpose | Status |
|------|---------|--------|
| `chiefs_vs_chargers_websocket.py` | Original hardcoded approach | ⚠️ Deprecated |
| `launch_chiefs_chargers.py` | Quick launcher for old approach | ⚠️ Deprecated |

The new flexible approach is cleaner, more maintainable, and gives users full control over what markets to stream.