#!/usr/bin/env python3
"""
Discover Today's College Football Games
========================================
Script to find college football games happening today on Kalshi.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neural_sdk.data_pipeline.data_sources.kalshi.client import KalshiClient


def discover_cfb_game_series():
    """Try to find college football game series"""
    print("\n🏈 Searching for College Football Game Series...")
    print("=" * 60)
    
    client = KalshiClient()
    
    # Try different possible series tickers
    potential_tickers = [
        'KXNCAAFGAME',    # Most likely for individual games
        'KXNCAAGAME',     # Alternative
        'KXCFBGAME',      # Alternative
        'KXCOLLEGEGAME',  # Alternative
    ]
    
    for ticker in potential_tickers:
        print(f"\nTrying series ticker: {ticker}")
        
        try:
            params = {
                'series_ticker': ticker,
                'status': 'open',
                'limit': 100,
                'with_nested_markets': True
            }
            
            response = client.get('/events', params=params)
            events = response.get('events', [])
            
            if events:
                print(f"✅ Found {len(events)} events for {ticker}!")
                return ticker, events
            else:
                print(f"   No events for {ticker}")
                
        except Exception as e:
            print(f"   Error: {e}")
    
    return None, []


def get_todays_games(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter events for games happening today"""
    today = datetime.now().date()
    tomorrow = today + timedelta(days=1)
    
    todays_games = []
    
    for event in events:
        # Check expected expiration time or close time
        exp_time_str = event.get('expected_expiration_time') or event.get('close_time')
        if exp_time_str:
            try:
                exp_time = datetime.fromisoformat(exp_time_str.replace('Z', '+00:00'))
                if today <= exp_time.date() <= tomorrow:
                    todays_games.append(event)
            except:
                pass
    
    return todays_games


def display_game_markets(event: Dict[str, Any]):
    """Display markets for a game"""
    print(f"\n🎯 {event.get('title', 'Unknown Game')}")
    print(f"   Event: {event.get('ticker', 'N/A')}")
    print(f"   Category: {event.get('category', 'N/A')}")
    
    markets = event.get('markets', [])
    if markets:
        print(f"   Markets ({len(markets)}):")
        for market in markets:
            ticker = market.get('ticker')
            yes_team = market.get('yes_sub_title', '')
            no_team = market.get('no_sub_title', '')
            
            # Get prices if available
            yes_price = market.get('yes_ask')
            if yes_price:
                price_str = f" - Current: ${yes_price/100:.2f}"
            else:
                price_str = ""
            
            print(f"     • {ticker}: {yes_team} to win{price_str}")


def main():
    """Main function to discover today's college football games"""
    print("\n" + "=" * 60)
    print("🏈 TODAY'S COLLEGE FOOTBALL GAMES ON KALSHI")
    print("=" * 60)
    print(f"📅 Date: {datetime.now().strftime('%A, %B %d, %Y')}")
    
    # Step 1: Find the college football game series
    series_ticker, events = discover_cfb_game_series()
    
    if not series_ticker:
        print("\n❌ Could not find college football game series")
        print("\nTrying to search all open events for college teams...")
        
        # Fallback: Search all events
        client = KalshiClient()
        
        # Get events with college football keywords
        college_teams = ['Alabama', 'Georgia', 'Michigan', 'Ohio State', 'Texas', 
                        'Oklahoma', 'LSU', 'Clemson', 'Florida', 'Auburn',
                        'Penn State', 'Oregon', 'USC', 'Notre Dame']
        
        found_events = []
        
        for team in college_teams[:5]:  # Check first 5 teams
            print(f"\nSearching for {team}...")
            
            try:
                # Get all open events
                params = {
                    'status': 'open',
                    'limit': 200,
                    'with_nested_markets': True
                }
                
                response = client.get('/events', params=params)
                all_events = response.get('events', [])
                
                # Filter for team
                for event in all_events:
                    title = event.get('title', '')
                    if team.upper() in title.upper():
                        found_events.append(event)
                        print(f"   ✓ Found: {title}")
                        
            except Exception as e:
                print(f"   Error: {e}")
        
        events = found_events
    
    if not events:
        print("\n❌ No college football events found")
        return
    
    # Step 2: Filter for today's games
    print(f"\n📊 Total events found: {len(events)}")
    
    todays_games = get_todays_games(events)
    
    if todays_games:
        print(f"\n🎮 Games Today/Tomorrow: {len(todays_games)}")
        print("=" * 60)
        
        for game in todays_games:
            display_game_markets(game)
    else:
        print("\n📅 No games scheduled for today")
        print("\n🔍 All available games:")
        print("=" * 60)
        
        # Show all games
        for i, event in enumerate(events[:10], 1):  # Show first 10
            print(f"\n{i}. {event.get('title', 'Unknown')}")
            
            # Parse date from expected expiration
            exp_time_str = event.get('expected_expiration_time')
            if exp_time_str:
                try:
                    exp_time = datetime.fromisoformat(exp_time_str.replace('Z', '+00:00'))
                    print(f"   Date: {exp_time.strftime('%B %d, %Y')}")
                except:
                    pass
            
            display_game_markets(event)


if __name__ == "__main__":
    main()