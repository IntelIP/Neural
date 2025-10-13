#!/usr/bin/env python3
"""Direct synchronous test of historical data."""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from neural.auth.http_client import KalshiHTTPClient


def collect_trades_sync(client, ticker, start_ts, end_ts, limit=100):
    """Synchronous trade collection with pagination."""
    all_trades = []
    cursor = None

    while True:
        try:
            # Call API
            response = client.get_trades(
                ticker=ticker, min_ts=start_ts, max_ts=end_ts, limit=limit, cursor=cursor
            )

            # Parse trades
            trades = response.get("trades", [])
            if not trades:
                break

            all_trades.extend(trades)
            print(f"  Fetched {len(trades)} trades (total: {len(all_trades)})")

            # Check for next page
            cursor = response.get("cursor")
            if not cursor:
                break

            # Safety limit
            if len(all_trades) > 10000:
                print("  Reached safety limit (10k trades)")
                break

        except Exception as e:
            print(f"  Error: {e}")
            break

    return pd.DataFrame(all_trades) if all_trades else pd.DataFrame()


def main():
    print("Historical Data Collection Test (Synchronous)")
    print("=" * 60)

    # Initialize client
    client = KalshiHTTPClient()

    # Test parameters
    ticker = "KXNFLGAME-25SEP25SEAARI-SEA"
    end_ts = int(datetime.now().timestamp())
    start_ts = end_ts - (7 * 24 * 3600)

    print(f"\nTicker: {ticker}")
    print("Time range: Last 7 days")
    print(f"Start: {datetime.fromtimestamp(start_ts)}")
    print(f"End: {datetime.fromtimestamp(end_ts)}\n")

    # Collect trades
    print("Collecting trades...")
    trades_df = collect_trades_sync(client, ticker, start_ts, end_ts, limit=500)

    # Results
    if not trades_df.empty:
        print(f"\n✅ SUCCESS: Collected {len(trades_df)} trades\n")

        # Convert timestamp
        trades_df["created_time"] = pd.to_datetime(trades_df["created_time"])

        # Show sample
        print("Sample trades:")
        print(trades_df[["created_time", "yes_price", "no_price", "count", "taker_side"]].head(10))

        # Statistics
        print("\nStatistics:")
        print(
            f"  Time range: {trades_df['created_time'].min()} to {trades_df['created_time'].max()}"
        )
        print(f"  Total volume: {trades_df['count'].sum():,}")
        print(f"  Price range: {trades_df['yes_price'].min()}-{trades_df['yes_price'].max()}")

        # Save
        output_file = "historical_trades_output.csv"
        trades_df.to_csv(output_file, index=False)
        print(f"\n💾 Saved to: {output_file}")
    else:
        print("\n⚠️  No trades found")

    # Cleanup
    client.close()
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
