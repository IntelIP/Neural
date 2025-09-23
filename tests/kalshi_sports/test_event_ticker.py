#!/usr/bin/env python3
"""Test fetching specific event."""

import asyncio
from dotenv import load_dotenv
from neural.data_collection import get_game_markets

load_dotenv()


async def test():
    # Try the exact event ticker we found earlier
    event_ticker = "KXNFLGAME-25SEP22DETBAL"
    print(f"Fetching {event_ticker}...")

    df = await get_game_markets(event_ticker, use_authenticated=False)

    if not df.empty:
        print(f"Found {len(df)} markets")
        for _, row in df.iterrows():
            print(f"  {row.get('ticker')}: {row.get('title')} [{row.get('status')}]")
    else:
        print("No markets found")


asyncio.run(test())