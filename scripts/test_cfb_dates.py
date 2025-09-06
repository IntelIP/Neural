#!/usr/bin/env python3
"""Test CFB date extraction"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from neural_sdk.data_pipeline.data_sources.kalshi.cfb_discovery import CFBMarketDiscovery

discovery = CFBMarketDiscovery()
events = discovery.get_all_cfb_events()

print(f"Found {len(events)} total events\n")

# Check first few events
for i, event in enumerate(events[:3]):
    print(f"Event {i+1}:")
    print(f"  Title: {event.get('title')}")
    print(f"  Ticker: {event.get('ticker', 'NO TICKER')}")
    
    # Check dates
    for field in ['expected_expiration_time', 'close_time', 'expiration_time']:
        value = event.get(field)
        if value:
            print(f"  {field}: {value}")
            try:
                dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
                print(f"    -> Date: {dt.date()}")
            except:
                pass
    
    # Check markets
    markets = event.get('markets', [])
    if markets and len(markets) > 0:
        print(f"  Markets: {len(markets)}")
        print(f"  First market ticker: {markets[0].get('ticker')}")
    
    print()