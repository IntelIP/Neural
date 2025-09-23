#!/usr/bin/env python3
"""Test finding Seahawks vs Cardinals Thursday night game."""

import asyncio
from dotenv import load_dotenv
from neural.data_collection import KalshiMarketsSource

load_dotenv()


async def test():
    print("Searching for Seahawks vs Cardinals game...")

    # First, let's search all KXNFLGAME markets for Seahawks/Cardinals
    source = KalshiMarketsSource(
        series_ticker="KXNFLGAME",
        status=None,  # Get all statuses
        use_authenticated=False,
        interval=float('inf')
    )

    async with source:
        async for df in source.collect():
            print(f"\nTotal NFL markets: {len(df)}")

            if not df.empty and 'title' in df.columns:
                # Search for Seahawks or Cardinals
                seahawks_cards = df[
                    df['title'].str.contains(
                        'Seahawks|Cardinals|Seattle|Arizona|SEA|ARI',
                        case=False, na=False
                    )
                ]

                if not seahawks_cards.empty:
                    print(f"\nFound {len(seahawks_cards)} Seahawks/Cardinals markets:")
                    for _, row in seahawks_cards.iterrows():
                        print(f"\n  Title: {row['title']}")
                        print(f"  Ticker: {row['ticker']}")
                        print(f"  Status: {row.get('status', 'N/A')}")
                        print(f"  Event: {row.get('event_ticker', 'N/A')}")

                        # Try to identify if it's Thursday night
                        if any(day in str(row.get('title', '')).upper() for day in ['THURSDAY', 'THU']):
                            print("  >>> This appears to be a Thursday game!")
                else:
                    print("\nNo Seahawks vs Cardinals markets found")

                # Also check event_ticker patterns
                if 'ticker' in df.columns:
                    print("\nChecking ticker patterns...")
                    # Look for patterns like KXNFLGAME-[date]SEAARI or KXNFLGAME-[date]ARISEA
                    sea_ari_tickers = df[
                        df['ticker'].str.contains(
                            'SEA.*ARI|ARI.*SEA',
                            case=False, na=False
                        )
                    ]
                    if not sea_ari_tickers.empty:
                        print(f"Found by ticker pattern:")
                        for _, row in sea_ari_tickers.iterrows():
                            print(f"  {row['ticker']}: {row['title']}")
            break


asyncio.run(test())