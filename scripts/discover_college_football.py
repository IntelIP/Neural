#!/usr/bin/env python3
"""
Discover College Football Markets
==================================
Script to discover available college football markets on Kalshi.

This script will search for college football games and markets
by trying different series tickers and search patterns.
"""

import asyncio
import sys
from pathlib import Path
from typing import List, Dict, Any
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neural_sdk.data_pipeline.data_sources.kalshi.client import KalshiClient


def discover_series():
    """Discover all available series on Kalshi"""
    print("\n🔍 Discovering all series on Kalshi...")
    print("=" * 60)
    
    client = KalshiClient()
    
    # Get all series
    try:
        response = client.get('/series')
        series_list = response.get('series', [])
        
        print(f"Found {len(series_list)} total series\n")
        
        # Filter for potential college football series
        college_keywords = ['NCAA', 'CFB', 'CFP', 'COLLEGE', 'FOOTBALL', 'BOWL']
        college_series = []
        
        for series in series_list:
            ticker = series.get('ticker', '')
            title = series.get('title', '')
            
            # Check if any keyword matches
            for keyword in college_keywords:
                if keyword in ticker.upper() or keyword in title.upper():
                    college_series.append(series)
                    break
        
        if college_series:
            print("📚 Potential College Football Series:")
            print("-" * 40)
            for series in college_series:
                print(f"Ticker: {series.get('ticker')}")
                print(f"Title: {series.get('title')}")
                print(f"Category: {series.get('category')}")
                print()
        else:
            print("No college football specific series found.")
            print("\nSearching all series for football content...")
            
            # Broader search
            football_series = []
            for series in series_list:
                ticker = series.get('ticker', '')
                title = series.get('title', '')
                if 'FOOTBALL' in title.upper() or 'GAME' in ticker.upper():
                    football_series.append(series)
            
            if football_series:
                print("\n🏈 All Football-Related Series:")
                print("-" * 40)
                for series in football_series[:10]:  # Limit to first 10
                    print(f"Ticker: {series.get('ticker')}")
                    print(f"Title: {series.get('title')}")
                    print(f"Category: {series.get('category')}")
                    print()
        
        return series_list
        
    except Exception as e:
        print(f"Error fetching series: {e}")
        return []


def discover_events_by_series(series_ticker: str):
    """Discover events for a specific series"""
    print(f"\n🎯 Discovering events for series: {series_ticker}")
    print("-" * 40)
    
    client = KalshiClient()
    
    try:
        params = {
            'series_ticker': series_ticker,
            'status': 'open',
            'limit': 100,
            'with_nested_markets': True
        }
        
        response = client.get('/events', params=params)
        events = response.get('events', [])
        
        if events:
            print(f"Found {len(events)} events:")
            for event in events[:5]:  # Show first 5
                print(f"\nEvent: {event.get('ticker')}")
                print(f"Title: {event.get('title')}")
                print(f"Category: {event.get('category')}")
                
                # Show nested markets if available
                markets = event.get('markets', [])
                if markets:
                    print(f"Markets ({len(markets)}):")
                    for market in markets:
                        print(f"  - {market.get('ticker')}: {market.get('yes_sub_title')} vs {market.get('no_sub_title')}")
        else:
            print("No events found for this series")
            
        return events
        
    except Exception as e:
        print(f"Error fetching events: {e}")
        return []


def search_markets_by_keyword(keyword: str):
    """Search markets by keyword"""
    print(f"\n🔎 Searching markets with keyword: '{keyword}'")
    print("-" * 40)
    
    client = KalshiClient()
    
    try:
        # Get all markets (paginated)
        all_markets = []
        cursor = None
        
        for _ in range(5):  # Limit pagination
            params = {
                'limit': 100,
                'status': 'open'
            }
            if cursor:
                params['cursor'] = cursor
                
            response = client.get('/markets', params=params)
            markets = response.get('markets', [])
            
            # Filter by keyword
            for market in markets:
                title = market.get('title', '')
                subtitle = market.get('subtitle', '')
                ticker = market.get('ticker', '')
                
                if keyword.upper() in title.upper() or keyword.upper() in subtitle.upper() or keyword.upper() in ticker.upper():
                    all_markets.append(market)
            
            cursor = response.get('cursor')
            if not cursor:
                break
        
        if all_markets:
            print(f"Found {len(all_markets)} markets containing '{keyword}':")
            for market in all_markets[:10]:  # Show first 10
                print(f"\nTicker: {market.get('ticker')}")
                print(f"Title: {market.get('title')}")
                print(f"Event: {market.get('event_ticker')}")
                print(f"YES: {market.get('yes_sub_title')} | NO: {market.get('no_sub_title')}")
                
                # Show current prices
                yes_ask = market.get('yes_ask')
                if yes_ask:
                    print(f"Price: ${yes_ask/100:.2f}")
        else:
            print(f"No markets found containing '{keyword}'")
            
        return all_markets
        
    except Exception as e:
        print(f"Error searching markets: {e}")
        return []


def main():
    """Main discovery function"""
    print("\n" + "=" * 60)
    print("🏈 COLLEGE FOOTBALL MARKET DISCOVERY")
    print("=" * 60)
    
    # 1. Discover all series
    series_list = discover_series()
    
    # 2. Try specific college football series tickers
    potential_tickers = [
        'KXCFP',      # College Football Playoff
        'KXCFPGAME',  # CFP Games
        'KXNCAAF',    # NCAA Football
        'KXCFB',      # College Football
        'KXBOWL',     # Bowl Games
    ]
    
    print("\n📋 Trying potential college football series tickers...")
    print("=" * 60)
    
    found_events = []
    for ticker in potential_tickers:
        events = discover_events_by_series(ticker)
        if events:
            found_events.extend(events)
    
    # 3. Search by keywords
    print("\n🔍 Searching by college football keywords...")
    print("=" * 60)
    
    keywords = ['NCAA', 'College Football', 'Bowl', 'CFP', 'Alabama', 'Georgia', 'Michigan', 'Ohio State']
    
    found_markets = []
    for keyword in keywords:
        markets = search_markets_by_keyword(keyword)
        if markets:
            found_markets.extend(markets)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 DISCOVERY SUMMARY")
    print("=" * 60)
    print(f"Total Series Found: {len(series_list)}")
    print(f"College Football Events Found: {len(found_events)}")
    print(f"College Football Markets Found: {len(found_markets)}")
    
    # Save results
    results = {
        'series_count': len(series_list),
        'events': [{'ticker': e.get('ticker'), 'title': e.get('title')} for e in found_events[:5]],
        'markets': [{'ticker': m.get('ticker'), 'title': m.get('title')} for m in found_markets[:5]]
    }
    
    output_file = Path(__file__).parent / 'college_football_discovery.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")


if __name__ == "__main__":
    main()