#!/usr/bin/env python3
"""Test async historical data collection."""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from neural.data_collection.kalshi_historical import KalshiHistoricalDataSource
from neural.data_collection.base import DataSourceConfig


async def main():
    print("Testing Async Historical Data Collection")
    print("=" * 60)

    # Initialize source
    config = DataSourceConfig(name="async_test")
    source = KalshiHistoricalDataSource(config)

    # Test parameters
    ticker = "KXNFLGAME-25SEP25SEAARI-SEA"
    end_ts = int(datetime.now().timestamp())
    start_ts = end_ts - (7 * 24 * 3600)  # Last 7 days

    print(f"\nTicker: {ticker}")
    print(f"Time range: Last 7 days")
    print(f"Limit: 20 trades\n")

    try:
        # Collect trades
        trades_df = await source.collect_trades(
            ticker=ticker,
            start_ts=start_ts,
            end_ts=end_ts,
            limit=20
        )

        print(f"Result type: {type(trades_df)}")
        print(f"Number of trades: {len(trades_df)}")

        if not trades_df.empty:
            print(f"\n✅ SUCCESS - Collected {len(trades_df)} trades\n")
            print("Sample trades:")
            print(trades_df[['created_time', 'yes_price', 'no_price', 'count']].head(10))

            # Save to file
            trades_df.to_csv('historical_trades_test.csv', index=False)
            print(f"\n💾 Saved to: historical_trades_test.csv")
        else:
            print("\n⚠️  No trades collected")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())