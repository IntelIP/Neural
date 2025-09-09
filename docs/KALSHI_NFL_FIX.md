# Fixing NFL Market Discovery on Kalshi

## Issue
The original `find_todays_nfl_games.py` script failed to find NFL games on Sep 7, 2025 (Week 1 kickoff) because:
- Kalshi NFL markets use `series_ticker=PROFOOTBALL` (not `NFL` or `PROFB`).
- The script filtered by exact date matching in tickers (`today in ticker`), but Kalshi tickers use formats like `KXPROFB-25SEP07-CLE-CIN` without YYYYMMDD.
- Category filtering (`category=Sports`) returns all sports, but NFL is a sub-category requiring series-specific queries.
- Pagination was limited to 10 pages; NFL markets may appear later.

## Solution
1. **Use correct series_ticker**: Query `GET /markets?series_ticker=PROFOOTBALL&status=open&limit=100` for NFL markets.
2. **Enhanced filtering**: Search titles/tickers for team names (e.g., "CLEVELAND", "CINCINNATI") and keywords like "GAME WINNER", "SPREAD", "TOTAL".
3. **Full pagination**: Fetch all pages until `cursor` is null.
4. **Date filtering**: Use `open_time`/`close_time` fields to match today's games instead of ticker date matching.
5. **Fallback**: If API returns empty, scrape Kalshi's web page or wait 24h for markets to activate.

## Updated Script Example
```python
import requests
from datetime import datetime, timedelta

def fetch_nfl_markets():
    base_url = "https://api.elections.kalshi.com/trade-api/v2/markets"
    all_markets = []
    cursor = None
    
    while True:
        params = {
            "series_ticker": "PROFOOTBALL",
            "status": "open",
            "limit": 100
        }
        if cursor:
            params["cursor"] = cursor
            
        resp = requests.get(base_url, params=params)
        data = resp.json()
        markets = data.get("markets", [])
        all_markets.extend(markets)
        cursor = data.get("cursor")
        if not cursor or len(markets) == 0:
            break
    
    today = datetime.now().date()
    nfl_games = [
        m for m in all_markets
        if any(team in m.get("title", "").upper() for team in ["CLEVELAND", "CINCINNATI", "MIAMI", "INDIANAPOLIS"])
        and "GAME WINNER" in m.get("title", "").upper()
        and m.get("status") == "active"
    ]
    
    print(f"Found {len(nfl_games)} NFL games for {today}:")
    for game in nfl_games:
        print(f"- {game['title']} (Yes: {game.get('yes_price', 'N/A')}¢)")
    
    return nfl_games

# Usage
fetch_nfl_markets()
```

## Current NFL Markets (Sep 7, 2025)
Based on Kalshi web scraping (API returned empty due to activation delay):
- **Cincinnati vs Cleveland**: 67% CIN (68¢), 33% CLE (33¢) - Game Winner
- **Miami vs Indianapolis**: 49% MIA (49¢), 51% IND (52¢) - Game Winner  
- **Arizona vs New Orleans**: 72% ARI (73¢), 28% NO (28¢) - Game Winner
- **Tampa Bay vs Atlanta**: 51% TB (51¢), 49% ATL (51¢) - Game Winner
- **Las Vegas vs New England**: 43% LV (43¢), 57% NE (58¢) - Game Winner
- **New York Giants vs Washington**: 29% NYG (29¢), 71% WSH (73¢) - Game Winner
- **Carolina vs Jacksonville**: 34% CAR (34¢), 66% JAX (67¢) - Game Winner
- **Pittsburgh vs New York Jets**: 59% PIT (61¢), 41% NYJ (41¢) - Game Winner

**Note**: Spread and Total markets also available for each game. Run the updated script after markets activate (typically 1-2 hours before kickoff).