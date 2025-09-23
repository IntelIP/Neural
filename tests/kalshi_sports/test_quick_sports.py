#!/usr/bin/env python3
"""Quick test to see if Detroit vs Baltimore markets are accessible."""

import asyncio
from dotenv import load_dotenv
from neural.data_collection import KalshiMarketsSource

load_dotenv()


async def test_simple():
    """Simple test to fetch NFL markets without status filter."""
    print("Fetching KXNFLGAME markets (no status filter)...")

    source = KalshiMarketsSource(
        series_ticker="KXNFLGAME",
        status=None,  # No filtering by status
        use_authenticated=True,
        interval=float('inf')
    )

    async with source:
        async for df in source.collect():
            print(f"Found {len(df)} markets")

            if not df.empty:
                # Check statuses
                if 'status' in df.columns:
                    print("\nStatuses found:", df['status'].unique())

                # Look for Detroit/Baltimore
                det_bal = df[
                    df['title'].str.contains('Detroit|Baltimore', case=False, na=False)
                ]
                print(f"\nDetroit/Baltimore markets: {len(det_bal)}")
                if not det_bal.empty:
                    for _, row in det_bal.iterrows():
                        print(f"  {row['ticker']}: {row['title']} [{row['status']}]")
            break


if __name__ == "__main__":
    asyncio.run(test_simple())