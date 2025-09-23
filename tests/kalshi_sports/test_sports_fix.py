#!/usr/bin/env python3
"""Test the updated sports market collection with event-based approach."""

import asyncio
from dotenv import load_dotenv
from neural.data_collection import KalshiMarketsSource, get_game_markets, get_live_sports

# Load environment variables
load_dotenv()


async def test_nfl_games():
    """Test fetching NFL game markets."""
    print("\n" + "="*60)
    print("TEST 1: Fetching NFL Game Markets (KXNFLGAME)")
    print("="*60)

    # Test with the KXNFLGAME series ticker (no status filter)
    source = KalshiMarketsSource(
        series_ticker="KXNFLGAME",
        status=None,  # Get ALL statuses
        use_authenticated=True,
        interval=float('inf')
    )

    async with source:
        async for df in source.collect():
            print(f"\nTotal markets found: {len(df)}")

            if not df.empty:
                # Show all unique statuses
                if 'status' in df.columns:
                    statuses = df['status'].value_counts()
                    print("\nMarket statuses:")
                    for status, count in statuses.items():
                        print(f"  - {status}: {count}")

                # Show sample markets
                print("\nFirst 5 markets:")
                for idx, row in df.head(5).iterrows():
                    print(f"  - {row.get('title', 'N/A')}")
                    print(f"    Ticker: {row.get('ticker', 'N/A')}")
                    print(f"    Status: {row.get('status', 'N/A')}")
                    print(f"    Event: {row.get('event_ticker', 'N/A')}")

                # Look for Detroit vs Baltimore
                if 'title' in df.columns:
                    det_bal = df[
                        df['title'].str.contains('Detroit|Baltimore|DET|BAL', case=False, na=False)
                    ]
                    if not det_bal.empty:
                        print(f"\nDetroit vs Baltimore markets found: {len(det_bal)}")
                        for _, row in det_bal.iterrows():
                            print(f"  - {row['title']}")
                            print(f"    Ticker: {row['ticker']}")
                            print(f"    Status: {row['status']}")

            break  # Just one fetch


async def test_specific_game():
    """Test fetching markets for a specific game."""
    print("\n" + "="*60)
    print("TEST 2: Fetching Specific Game (Detroit vs Baltimore)")
    print("="*60)

    # Try the known event ticker format
    event_ticker = "KXNFLGAME-25SEP22DETBAL"

    print(f"\nFetching markets for event: {event_ticker}")
    df = await get_game_markets(event_ticker, use_authenticated=True)

    if not df.empty:
        print(f"Markets found: {len(df)}")
        for _, row in df.iterrows():
            print(f"  - {row.get('title', 'N/A')}")
            print(f"    Ticker: {row.get('ticker', 'N/A')}")
            print(f"    Status: {row.get('status', 'N/A')}")
            print(f"    Yes Ask: ${row.get('yes_ask', 'N/A')}")
            print(f"    No Ask: ${row.get('no_ask', 'N/A')}")
    else:
        print("No markets found for this event")


async def test_live_sports():
    """Test fetching all live/active sports markets."""
    print("\n" + "="*60)
    print("TEST 3: Fetching Live Sports Markets")
    print("="*60)

    df = await get_live_sports(sports=["NFL", "NBA"])

    if not df.empty:
        print(f"\nTotal live sports markets: {len(df)}")

        # Group by series
        if 'series_ticker' in df.columns:
            series_counts = df['series_ticker'].value_counts()
            print("\nMarkets by series:")
            for series, count in series_counts.items():
                print(f"  - {series}: {count}")

        # Show sample
        print("\nFirst 5 live markets:")
        for _, row in df.head(5).iterrows():
            print(f"  - {row.get('title', 'N/A')}")
            print(f"    Status: {row.get('status', 'N/A')}")
    else:
        print("No live sports markets found")


async def test_all_statuses():
    """Test fetching markets with all statuses to see what's available."""
    print("\n" + "="*60)
    print("TEST 4: Fetching ALL NFL Markets (Any Status)")
    print("="*60)

    from neural.data_collection import get_markets_by_sport

    # Get all NFL markets regardless of status
    df = await get_markets_by_sport(
        sport="NFL",  # Will map to KXNFLGAME
        status=None,  # No status filter
        use_authenticated=True
    )

    if not df.empty:
        print(f"\nTotal NFL markets (all statuses): {len(df)}")

        # Group by status
        if 'status' in df.columns:
            status_counts = df['status'].value_counts()
            print("\nBreakdown by status:")
            for status, count in status_counts.items():
                print(f"  - {status}: {count} markets")

        # Show markets by status
        for status in df['status'].unique() if 'status' in df.columns else []:
            status_df = df[df['status'] == status]
            print(f"\nSample {status} markets:")
            for _, row in status_df.head(2).iterrows():
                print(f"  - {row.get('title', 'N/A')[:60]}...")
    else:
        print("No NFL markets found")


async def main():
    """Run all tests."""
    print("="*60)
    print("TESTING UPDATED SPORTS MARKET COLLECTION")
    print("="*60)

    # Test 1: Fetch NFL games
    await test_nfl_games()

    # Test 2: Fetch specific game
    await test_specific_game()

    # Test 3: Fetch live sports
    await test_live_sports()

    # Test 4: Fetch all statuses
    await test_all_statuses()

    print("\n" + "="*60)
    print("ALL TESTS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())