#!/usr/bin/env python3
"""Simple test for historical data API endpoint."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime

from neural.auth.http_client import KalshiHTTPClient

# Initialize HTTP client
client = KalshiHTTPClient()

# Test parameters
ticker = "KXNFLGAME-25SEP25SEAARI-SEA"
end_ts = int(datetime.now().timestamp())
start_ts = end_ts - (7 * 24 * 3600)  # Last 7 days

print("Testing GET /markets/trades")
print(f"Ticker: {ticker}")
print(f"Time range: {datetime.fromtimestamp(start_ts)} to {datetime.fromtimestamp(end_ts)}")
print("-" * 60)

try:
    # Make direct API call
    response = client.get_trades(ticker=ticker, min_ts=start_ts, max_ts=end_ts, limit=10)

    print(f"\nResponse type: {type(response)}")
    print(f"Response keys: {list(response.keys()) if isinstance(response, dict) else 'Not a dict'}")
    print("\nFull response:")
    print(response)

    # Check for trades
    if isinstance(response, dict):
        trades = response.get("trades", [])
        print(f"\n✅ Found {len(trades)} trades")
        if trades:
            print(f"\nFirst trade: {trades[0]}")
    else:
        print("\n⚠️  Response is not a dictionary")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback

    traceback.print_exc()

finally:
    client.close()
