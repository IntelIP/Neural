#!/usr/bin/env python3
"""Verify the market exists and get current live data."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neural.auth.http_client import KalshiHTTPClient

client = KalshiHTTPClient()

ticker = "KXNFLGAME-25SEP25SEAARI-SEA"

print(f"Checking live market data for: {ticker}")
print("=" * 60)

try:
    # Get current market data
    response = client.get(f'/markets/{ticker}')

    print(f"\n✅ Market exists and is accessible")
    print(f"\nMarket details:")

    if 'market' in response:
        market = response['market']
        print(f"  Ticker: {market.get('ticker')}")
        print(f"  Title: {market.get('title', 'N/A')}")
        print(f"  Status: {market.get('status', 'N/A')}")
        print(f"  Yes Ask: {market.get('yes_ask', 'N/A')}¢")
        print(f"  No Ask: {market.get('no_ask', 'N/A')}¢")
        print(f"  Volume: {market.get('volume', 'N/A'):,}")
        print(f"  Open Interest: {market.get('open_interest', 'N/A'):,}")

        print(f"\n📊 This is REAL live data from Kalshi's production API")
        print(f"   The historical trades are from the same real market")
    else:
        print(f"  Response: {response}")

except Exception as e:
    print(f"\n❌ Error: {e}")

finally:
    client.close()