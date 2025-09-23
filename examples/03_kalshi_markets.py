"""
Example usage of Kalshi markets data collection.

This demonstrates fetching sports tickers from Kalshi,
returning as Pandas DataFrame.
"""

import sys
import os
from typing import Optional

# Add the neural package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neural.data_collection import KalshiMarketsSource, get_markets_by_sport
import asyncio


async def collect_kalshi_markets(series_ticker: str = "NFL"):
    """Collect Kalshi markets for a series using the new implementation."""
    # Use the proper series ticker (not name)
    source = KalshiMarketsSource(
        series_ticker=series_ticker,
        status="open",
        use_authenticated=False,  # Use public API by default
        interval=60.0
    )

    async with source:
        async for df in source.collect():
            print(f"Fetched {len(df)} markets for {series_ticker}")

            # Example filtering
            if not df.empty and 'title' in df.columns:
                ravens_lions = df[df['title'].str.contains('Ravens|Lions', case=False, na=False)]
                if not ravens_lions.empty:
                    print(f"Found {len(ravens_lions)} Ravens/Lions markets")
                    display_cols = ['ticker', 'title', 'yes_ask', 'volume_24h']
                    available_cols = [col for col in display_cols if col in df.columns]
                    print(ravens_lions[available_cols].head())
                else:
                    print("No Ravens/Lions markets found in current data")

            # Show sample of data
            print("\nSample of DataFrame:")
            display_cols = ['ticker', 'title', 'yes_ask', 'volume_24h', 'mid_price']
            available_cols = [col for col in display_cols if col in df.columns]
            print(df[available_cols].head(10))

            return df


async def main():
    """Run Kalshi markets example."""
    print("=== Kalshi Markets Data Collection ===\n")

    # Fetch NFL markets using proper ticker
    print("Fetching NFL markets...")
    df = await collect_kalshi_markets("NFL")

    # Alternative: Use utility function
    print("\n=== Using Utility Function ===")
    nba_df = await get_markets_by_sport("NBA", use_authenticated=False)
    if not nba_df.empty:
        print(f"Fetched {len(nba_df)} NBA markets")

    # Show available sports series
    print("\n=== Available Sports Series ===")
    print("NFL, NBA, MLB, NHL, NCAAF, NCAAB, SOCCER, TENNIS, GOLF, MMA, F1")

    print("\n=== Usage with Authentication ===")
    print("To use with real credentials:")
    print("source = KalshiMarketsSource(")
    print("    series_ticker='NFL',")
    print("    use_authenticated=True,")
    print("    api_key_id=your_key,")
    print("    private_key_pem=your_pem")
    print(")")
    print("async with source:")
    print("    async for df in source.collect():")
    print("        # df is comprehensive Pandas DataFrame with all market data")


if __name__ == "__main__":
    asyncio.run(main())