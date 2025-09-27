#!/usr/bin/env python3
"""
Backfill historical trade data for Ravens vs Lions game markets.

This script collects historical trade data from Kalshi and saves it to CSV files
for analysis and backtesting purposes.

Updated to use the working KalshiHTTPClient implementation.
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neural.auth.http_client import KalshiHTTPClient

RAVENS_TICKER = "KXNFLGAME-25SEP22DETBAL-BAL"
LIONS_TICKER = "KXNFLGAME-25SEP22DETBAL-DET"
OUTPUT_DIR = Path("data/historical")


def collect_market_trades(client: KalshiHTTPClient, ticker: str, start_ts: int, end_ts: int) -> pd.DataFrame:
    """Collect all trades for a market within a time range."""
    print(f"\nCollecting trades for {ticker}...")
    all_trades = []
    cursor = None

    while True:
        try:
            response = client.get_trades(
                ticker=ticker,
                min_ts=start_ts,
                max_ts=end_ts,
                limit=1000,
                cursor=cursor
            )

            trades = response.get("trades", [])
            if not trades:
                break

            all_trades.extend(trades)
            print(f"  Fetched {len(trades)} trades (total: {len(all_trades)})")

            cursor = response.get("cursor")
            if not cursor:
                break

            # Safety limit
            if len(all_trades) > 100000:
                print("  Reached safety limit (100k trades)")
                break

        except Exception as e:
            print(f"  Error: {e}")
            break

    if all_trades:
        df = pd.DataFrame(all_trades)
        df['created_time'] = pd.to_datetime(df['created_time'])
        df = df.sort_values('created_time').reset_index(drop=True)
        print(f"  ✅ Collected {len(df)} trades")
        return df
    else:
        print("  ⚠️  No trades found")
        return pd.DataFrame()


def main(start: Optional[str] = None, end: Optional[str] = None) -> None:
    """Main backfill function."""
    print("=" * 60)
    print("Ravens vs Lions Historical Data Backfill")
    print("=" * 60)

    # Parse dates
    if start and end:
        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end)
    else:
        # Default to known 2022 game window
        start_dt = datetime(2022, 9, 25, 16, 0, 0)
        end_dt = datetime(2022, 9, 25, 23, 0, 0)

    start_ts = int(start_dt.timestamp())
    end_ts = int(end_dt.timestamp())

    print(f"\nTime range:")
    print(f"  Start: {start_dt}")
    print(f"  End: {end_dt}")
    print(f"  Timestamps: {start_ts} to {end_ts}")

    # Initialize client
    client = KalshiHTTPClient()

    try:
        # Collect Ravens trades
        ravens_df = collect_market_trades(client, RAVENS_TICKER, start_ts, end_ts)

        # Collect Lions trades
        lions_df = collect_market_trades(client, LIONS_TICKER, start_ts, end_ts)

        # Save results
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        if not ravens_df.empty:
            ravens_file = OUTPUT_DIR / "ravens_trades.csv"
            ravens_df.to_csv(ravens_file, index=False)
            print(f"\n💾 Ravens data saved to: {ravens_file}")

        if not lions_df.empty:
            lions_file = OUTPUT_DIR / "lions_trades.csv"
            lions_df.to_csv(lions_file, index=False)
            print(f"💾 Lions data saved to: {lions_file}")

        # Summary
        print(f"\n" + "=" * 60)
        print("Summary:")
        print(f"  Ravens trades: {len(ravens_df):,}")
        print(f"  Lions trades: {len(lions_df):,}")
        print(f"  Total trades: {len(ravens_df) + len(lions_df):,}")
        print("=" * 60)

    finally:
        client.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Backfill Ravens vs Lions historical trade data")
    parser.add_argument("--start", help="Start datetime (ISO format)", default=None)
    parser.add_argument("--end", help="End datetime (ISO format)", default=None)
    args = parser.parse_args()

    main(args.start, args.end)


