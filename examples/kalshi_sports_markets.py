"""
Comprehensive example of Kalshi sports market data collection using the Neural SDK.

This example demonstrates:
1. Fetching sports markets with authentication
2. Using utility functions for easy market access
3. Working with the returned Pandas DataFrames
4. Filtering and analyzing market data
"""

import sys
import os
import asyncio
from typing import Optional
import pandas as pd
from dotenv import load_dotenv

# Add the neural package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neural.data_collection import (
    KalshiMarketsSource,
    get_sports_series,
    get_markets_by_sport,
    get_all_sports_markets,
    search_markets
)


def display_market_summary(df: pd.DataFrame, title: str):
    """Display a summary of market data."""
    print(f"\n=== {title} ===")

    if df.empty:
        print("No markets found")
        return

    print(f"Total markets: {len(df)}")
    print(f"Active markets: {len(df[df['status'] == 'open'])}")

    # Show top markets by volume
    print("\n📊 Top 5 Markets by 24h Volume:")
    top_markets = df.nlargest(5, 'volume_24h')[['ticker', 'title', 'volume_24h', 'mid_price']]
    for idx, row in top_markets.iterrows():
        print(f"  - {row['title'][:50]}...")
        print(f"    Ticker: {row['ticker']}")
        print(f"    Volume: ${row['volume_24h']:,.0f}")
        print(f"    Mid Price: {row['mid_price']:.1f}¢")

    # Show market statistics
    print("\n📈 Market Statistics:")
    print(f"  - Average spread: {df['spread'].mean():.2f}¢")
    print(f"  - Total volume (24h): ${df['volume_24h'].sum():,.0f}")
    print(f"  - Average liquidity score: {df['liquidity_score'].mean():.2f}")


async def example_basic_usage():
    """Basic usage example with public API."""
    print("\n🏈 Example 1: Basic NFL Markets (Public API)")

    # Create source for NFL markets
    source = KalshiMarketsSource(
        series_ticker="NFL",
        status="open",
        use_authenticated=False  # Use public API
    )

    # Collect markets once
    async with source:
        async for df in source.collect():
            display_market_summary(df, "NFL Markets")

            # Example: Filter for specific teams
            print("\n🔍 Filtering for specific teams (Cowboys):")
            cowboys_markets = df[df['title'].str.contains('Cowboys', case=False, na=False)]
            if not cowboys_markets.empty:
                print(cowboys_markets[['ticker', 'title', 'yes_ask', 'volume']].head(3))

            break  # Just one collection


async def example_authenticated_usage():
    """Example using authenticated API for more data."""
    print("\n🔐 Example 2: Authenticated API with Multiple Sports")

    # Load credentials from environment
    load_dotenv()

    # Note: Will fall back to public API if credentials not found
    source = KalshiMarketsSource(
        series_ticker="NBA",
        status="open",
        use_authenticated=True  # Try authenticated API
    )

    async with source:
        async for df in source.collect():
            display_market_summary(df, "NBA Markets (Authenticated)")
            break


async def example_utility_functions():
    """Example using utility functions for easy access."""
    print("\n🛠️ Example 3: Using Utility Functions")

    # Get available sports
    sports = await get_sports_series()
    print(f"Available sports: {list(sports.keys())}")

    # Get NFL markets
    print("\n📊 Fetching NFL markets...")
    nfl_df = await get_markets_by_sport("NFL", status="open", use_authenticated=False)
    display_market_summary(nfl_df, "NFL Markets via Utility")

    # Search for specific markets
    print("\n🔎 Searching for 'playoff' markets...")
    playoff_df = await search_markets("playoff", status="open", use_authenticated=False)
    if not playoff_df.empty:
        print(f"Found {len(playoff_df)} playoff markets")
        print(playoff_df[['ticker', 'title', 'series_ticker']].head(3))


async def example_multiple_sports():
    """Example fetching multiple sports at once."""
    print("\n🏆 Example 4: Multiple Sports Markets")

    # Get markets for specific sports
    sports_list = ["NFL", "NBA", "NHL"]
    print(f"Fetching markets for: {sports_list}")

    all_sports_df = await get_all_sports_markets(
        sports=sports_list,
        status="open",
        use_authenticated=False
    )

    if not all_sports_df.empty:
        print(f"\nTotal markets across all sports: {len(all_sports_df)}")

        # Group by series
        by_series = all_sports_df.groupby('series_ticker').size()
        print("\nMarkets by sport:")
        for series, count in by_series.items():
            print(f"  - {series}: {count} markets")

        # Find most liquid markets across all sports
        print("\n💰 Top 3 Most Liquid Markets (All Sports):")
        top_liquid = all_sports_df.nlargest(3, 'liquidity_score')[['title', 'series_ticker', 'liquidity_score', 'volume_24h']]
        for idx, row in top_liquid.iterrows():
            print(f"  - [{row['series_ticker']}] {row['title'][:40]}...")
            print(f"    Liquidity Score: {row['liquidity_score']:.0f}")


async def example_dataframe_analysis():
    """Example of DataFrame analysis and manipulation."""
    print("\n📊 Example 5: DataFrame Analysis")

    # Get NFL markets
    df = await get_markets_by_sport("NFL", use_authenticated=False)

    if df.empty:
        print("No NFL markets available")
        return

    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Analyze pricing
    print("\n💵 Pricing Analysis:")
    print(f"  - Mean yes price: {df['yes_ask'].mean():.1f}¢")
    print(f"  - Markets above 50¢: {len(df[df['yes_ask'] > 50])}")
    print(f"  - Markets below 20¢: {len(df[df['yes_ask'] < 20])}")

    # Volume analysis
    print("\n📈 Volume Analysis:")
    total_volume = df['volume_24h'].sum()
    print(f"  - Total 24h volume: ${total_volume:,.0f}")
    print(f"  - Average per market: ${df['volume_24h'].mean():,.0f}")
    print(f"  - Median per market: ${df['volume_24h'].median():,.0f}")

    # Time analysis
    if 'close_time' in df.columns and not df['close_time'].isna().all():
        print("\n⏰ Time Analysis:")
        df['close_time'] = pd.to_datetime(df['close_time'])
        df['days_until_close'] = (df['close_time'] - pd.Timestamp.now()).dt.days

        closing_soon = df[df['days_until_close'] <= 7]
        print(f"  - Markets closing within 7 days: {len(closing_soon)}")

    # Export to CSV
    output_file = "kalshi_sports_markets.csv"
    df.to_csv(output_file, index=False)
    print(f"\n💾 Data exported to {output_file}")


async def example_continuous_monitoring():
    """Example of continuous market monitoring."""
    print("\n🔄 Example 6: Continuous Market Monitoring")
    print("Monitoring NFL markets every 30 seconds (3 iterations for demo)...")

    source = KalshiMarketsSource(
        series_ticker="NFL",
        status="open",
        interval=30.0,  # Poll every 30 seconds
        use_authenticated=False
    )

    iteration = 0
    async with source:
        async for df in source.collect():
            iteration += 1
            print(f"\n[Iteration {iteration}] {pd.Timestamp.now()}")
            print(f"  - Active markets: {len(df)}")
            print(f"  - Total volume: ${df['volume_24h'].sum():,.0f}")

            # Track changes in top market
            if not df.empty:
                top_market = df.nlargest(1, 'volume_24h').iloc[0]
                print(f"  - Top market: {top_market['title'][:40]}...")
                print(f"    Price: {top_market['yes_ask']}¢ | Volume: ${top_market['volume_24h']:,.0f}")

            if iteration >= 3:
                break  # Stop after 3 iterations for demo


async def main():
    """Run all examples."""
    print("=" * 60)
    print("🏆 Kalshi Sports Markets Collection Examples")
    print("=" * 60)

    try:
        # Run examples
        await example_basic_usage()
        await example_utility_functions()
        await example_multiple_sports()
        await example_dataframe_analysis()

        # Optional: Run authenticated example if credentials available
        if os.getenv("KALSHI_API_KEY_ID"):
            await example_authenticated_usage()
        else:
            print("\n⚠️ Skipping authenticated example (no credentials found)")

        # Optional: Run continuous monitoring (commented out by default)
        # await example_continuous_monitoring()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("✅ Examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())