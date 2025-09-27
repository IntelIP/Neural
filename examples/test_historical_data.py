#!/usr/bin/env python3
"""
Test script for Kalshi historical data collection.

This script demonstrates how to collect historical trade data from Kalshi markets
and verifies the historical data integration is working correctly.
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from neural.data_collection.kalshi_historical import KalshiHistoricalDataSource
from neural.data_collection.base import DataSourceConfig


async def test_trade_collection():
    """Test historical trade data collection."""
    print("=" * 60)
    print("Testing Kalshi Historical Trade Data Collection")
    print("=" * 60)

    # Initialize the historical data source
    config = DataSourceConfig(name="test_historical")
    source = KalshiHistoricalDataSource(config)

    # Test with a known market ticker
    ticker = "KXNFLGAME-25SEP25SEAARI-SEA"

    # Get trades from last 7 days
    end_time = datetime.now()
    start_time = end_time - timedelta(days=7)

    start_ts = int(start_time.timestamp())
    end_ts = int(end_time.timestamp())

    print(f"\nCollecting trades for: {ticker}")
    print(f"Time range: {start_time} to {end_time}")
    print(f"Timestamps: {start_ts} to {end_ts}")
    print("-" * 60)

    try:
        # Collect trades
        trades_df = await source.collect_trades(
            ticker=ticker,
            start_ts=start_ts,
            end_ts=end_ts,
            limit=100
        )

        # Display results
        if not trades_df.empty:
            print(f"\n✅ SUCCESS: Collected {len(trades_df)} trades")
            print(f"\nFirst 5 trades:")
            print(trades_df.head())
            print(f"\nColumns: {list(trades_df.columns)}")
            print(f"\nData types:\n{trades_df.dtypes}")

            # Save to CSV for inspection
            output_file = "test_historical_trades.csv"
            trades_df.to_csv(output_file, index=False)
            print(f"\n💾 Saved to: {output_file}")
        else:
            print("\n⚠️  No trades found in the specified time range")
            print("This could mean:")
            print("  - The market had no trading activity")
            print("  - The time range is outside trading hours")
            print("  - The ticker may be incorrect")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


async def test_quick_trades():
    """Quick test with minimal data for debugging."""
    print("\n" + "=" * 60)
    print("Quick Trade Test (Last 24 hours, max 10 trades)")
    print("=" * 60)

    config = DataSourceConfig(name="quick_test")
    source = KalshiHistoricalDataSource(config)

    ticker = "KXNFLGAME-25SEP25SEAARI-SEA"
    end_ts = int(datetime.now().timestamp())
    start_ts = end_ts - (24 * 3600)  # Last 24 hours

    print(f"\nTicker: {ticker}")
    print(f"Time range: Last 24 hours")

    try:
        trades = await source.collect_trades(ticker, start_ts, end_ts, limit=10)

        if not trades.empty:
            print(f"✅ Found {len(trades)} trades")
            print("\nTrade details:")
            for idx, row in trades.iterrows():
                print(f"  [{row['created_time']}] "
                      f"Yes: {row.get('yes_price', 'N/A')}, "
                      f"No: {row.get('no_price', 'N/A')}, "
                      f"Count: {row.get('count', 'N/A')}")
        else:
            print("⚠️  No recent trades found")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def main():
    """Run all tests."""
    print("\n🔍 Kalshi Historical Data Integration Test\n")

    # Run quick test first
    success1 = await test_quick_trades()

    # Run full test
    success2 = await test_trade_collection()

    print("\n" + "=" * 60)
    if success1 and success2:
        print("✅ All tests completed successfully!")
    else:
        print("⚠️  Some tests failed - check output above")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())