#!/usr/bin/env python3
"""Test if sports markets require authentication."""

import os
import asyncio
from dotenv import load_dotenv
from neural.data_collection import KalshiMarketsSource
from neural.trading import TradingClient

# Load environment variables
load_dotenv()

def test_authenticated_api():
    """Test the authenticated API directly."""
    print("Testing authenticated API...")

    try:
        # Initialize authenticated client
        client = TradingClient()

        # Try to get markets without filters first
        print("\n1. Getting ALL markets (authenticated)...")
        all_markets = client.markets.get_markets(limit=100)
        print(f"   Total markets returned: {len(all_markets.get('markets', []))}")

        # Show first few market titles
        if all_markets.get('markets'):
            print("\n   First 5 market titles:")
            for m in all_markets['markets'][:5]:
                print(f"     - {m.get('title', 'N/A')}")
                if 'event_ticker' in m:
                    print(f"       Event: {m['event_ticker']}")

        # Search for football in titles
        football_markets = [
            m for m in all_markets.get('markets', [])
            if any(word in m.get('title', '').lower()
                   for word in ['football', 'nfl', 'ravens', 'detroit', 'baltimore'])
        ]

        print(f"\n2. Football markets found: {len(football_markets)}")
        for m in football_markets[:3]:
            print(f"   - {m.get('title', 'N/A')}")
            print(f"     Ticker: {m.get('ticker', 'N/A')}")
            print(f"     Event: {m.get('event_ticker', 'N/A')}")

        # Try events endpoint
        print("\n3. Getting events (authenticated)...")
        try:
            events = client.events.get_events(limit=100)
            print(f"   Total events: {len(events.get('events', []))}")

            # Look for sports events
            sports_events = [
                e for e in events.get('events', [])
                if any(word in e.get('title', '').lower()
                       for word in ['football', 'basketball', 'baseball', 'hockey', 'sport'])
            ]
            print(f"   Sports events found: {len(sports_events)}")
            for e in sports_events[:3]:
                print(f"     - {e.get('title', 'N/A')} (series: {e.get('series_ticker', 'N/A')})")

        except Exception as e:
            print(f"   Events endpoint error: {e}")

        # Try series endpoint
        print("\n4. Getting series (authenticated)...")
        try:
            series = client.series.get_series(limit=100)
            print(f"   Total series: {len(series.get('series', []))}")

            # Look for sports series
            if series.get('series'):
                print("\n   First 10 series:")
                for s in series['series'][:10]:
                    print(f"     - {s.get('ticker', 'N/A')}: {s.get('title', 'N/A')}")

                # Search for sports
                sports_series = [
                    s for s in series.get('series', [])
                    if any(word in s.get('title', '').lower()
                           for word in ['football', 'basketball', 'baseball', 'sport'])
                ]
                print(f"\n   Sports series found: {len(sports_series)}")
                for s in sports_series[:5]:
                    print(f"     - {s.get('ticker', 'N/A')}: {s.get('title', 'N/A')}")

        except Exception as e:
            print(f"   Series endpoint error: {e}")

        client.close()

    except Exception as e:
        print(f"Authentication error: {e}")
        print("\nMake sure you have valid credentials in:")
        print("  - Environment variables: KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH")
        print("  - Or in secrets/ folder")

async def test_async_authenticated():
    """Test async implementation with auth."""
    print("\n\n5. Testing async implementation with authentication...")

    source = KalshiMarketsSource(
        series_ticker=None,  # Get all markets
        status=None,  # All statuses
        use_authenticated=True,
        interval=float('inf')
    )

    async with source:
        async for df in source.collect():
            print(f"\n   DataFrame shape: {df.shape}")

            if not df.empty and 'title' in df.columns:
                # Look for sports
                sports_mask = df['title'].str.contains(
                    'football|basketball|baseball|hockey|nfl|nba|mlb|nhl',
                    case=False, na=False
                )
                sports_df = df[sports_mask]

                print(f"   Sports markets in DataFrame: {len(sports_df)}")
                if not sports_df.empty:
                    print("\n   Sample sports markets:")
                    for _, row in sports_df.head(3).iterrows():
                        print(f"     - {row['title']}")

            break  # Just one fetch

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING AUTHENTICATED API FOR SPORTS MARKETS")
    print("=" * 60)

    # Test synchronous authenticated API
    test_authenticated_api()

    # Test async implementation
    asyncio.run(test_async_authenticated())

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)