#!/usr/bin/env python3
"""Get details for Seahawks vs Cardinals game."""

import asyncio
from dotenv import load_dotenv
from neural.data_collection import get_game_markets

load_dotenv()


async def test():
    event_ticker = "KXNFLGAME-25SEP25SEAARI"
    print(f"Fetching Seahawks vs Cardinals: {event_ticker}\n")

    df = await get_game_markets(event_ticker, use_authenticated=False)

    if not df.empty:
        print(f"Found {len(df)} markets for this game:\n")
        for _, row in df.iterrows():
            print(f"Market: {row.get('title', 'N/A')}")
            print(f"  Ticker: {row.get('ticker')}")
            print(f"  Status: {row.get('status')}")
            print(f"  Yes Ask: ${row.get('yes_ask', 'N/A')}")
            print(f"  No Ask: ${row.get('no_ask', 'N/A')}")
            print(f"  Volume: {row.get('volume', 'N/A')}")
            print(f"  Open Interest: {row.get('open_interest', 'N/A')}")
            print()
    else:
        print("No markets found")


asyncio.run(test())