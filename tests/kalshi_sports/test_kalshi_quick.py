#!/usr/bin/env python3
"""Quick test of Kalshi market collection."""

import asyncio
import sys
import os

sys.path.insert(0, '.')

from neural.data_collection import KalshiMarketsSource, get_markets_by_sport


async def test_basic():
    """Test basic functionality."""
    print("Testing Kalshi markets collection...")

    # Test with public API
    source = KalshiMarketsSource(
        series_ticker="NFL",
        status="open",
        use_authenticated=False,
        interval=float('inf')  # Single fetch only
    )

    async with source:
        count = 0
        async for df in source.collect():
            print(f"✅ Fetched {len(df)} markets")
            if not df.empty:
                print(f"✅ Columns: {df.columns.tolist()[:5]}...")
                print(f"✅ First market: {df.iloc[0]['title'] if 'title' in df.columns else 'N/A'}")
            count += 1
            if count >= 1:  # Only one iteration
                break

    # Test utility function
    print("\nTesting utility function...")
    df = await get_markets_by_sport("NBA", use_authenticated=False)
    print(f"✅ Utility function returned {len(df)} NBA markets")

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    asyncio.run(test_basic())