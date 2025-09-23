#!/usr/bin/env python3
"""
Test: Find Arizona Cardinals vs Seattle Seahawks game and get pricing history
"""

import asyncio
import pandas as pd
from datetime import datetime
from neural.data_collection import get_game_markets, get_all_sports_markets, get_live_sports


async def find_cardinals_seahawks():
    """Find and analyze Cardinals vs Seahawks game"""
    print("🏈 Searching for Arizona Cardinals vs Seattle Seahawks")
    print("=" * 60)

    print("\n📊 Fetching all NFL markets...")
    # Get all NFL markets
    markets_df = await get_all_sports_markets(sports=["KXNFLGAME"], status=None)

    if markets_df.empty:
        print("❌ No NFL markets found")
        return

    print(f"✅ Found {len(markets_df)} total NFL markets")

    # Search for Cardinals vs Seahawks
    # Cardinals = ARI, Seahawks = SEA
    print("\n🔍 Searching for ARI vs SEA game...")

    # Filter for games containing both teams
    ari_sea_markets = markets_df[
        (markets_df['event_ticker'].str.contains('ARI', na=False) &
         markets_df['event_ticker'].str.contains('SEA', na=False)) |
        (markets_df['event_ticker'].str.contains('ARIZSEA', na=False)) |
        (markets_df['event_ticker'].str.contains('SEAARI', na=False))
    ]

    if ari_sea_markets.empty:
        print("❌ No Cardinals vs Seahawks markets found")
        print("\n📋 Available games:")
        # Show unique events
        unique_events = markets_df['event_ticker'].unique()
        for event in unique_events[:10]:  # Show first 10
            print(f"  - {event}")
        return

    # Get the event ticker
    event_ticker = ari_sea_markets.iloc[0]['event_ticker']
    print(f"✅ Found game: {event_ticker}")

    # Display basic info
    print(f"\n📈 Market Information:")
    print(f"  Event: {event_ticker}")
    print(f"  Title: {ari_sea_markets.iloc[0].get('title', 'N/A')}")
    print(f"  Status: {ari_sea_markets.iloc[0].get('status', 'N/A')}")
    print(f"  Close Time: {ari_sea_markets.iloc[0].get('close_time', 'N/A')}")

    # Get detailed market data including history
    print(f"\n📊 Fetching detailed market data and pricing history...")
    try:
        detailed_markets = await get_game_markets(event_ticker)

        if detailed_markets.empty:
            print("❌ No detailed market data available")
            return

        print(f"✅ Found {len(detailed_markets)} markets for this game")

        # Display market types
        print("\n📋 Available Markets:")
        for idx, row in detailed_markets.iterrows():
            print(f"\n  Market {idx + 1}:")
            print(f"    Ticker: {row.get('ticker', 'N/A')}")
            print(f"    Title: {row.get('title', 'N/A')}")
            print(f"    YES Ask: ${row.get('yes_ask', 0)/100:.2f}")
            print(f"    NO Ask: ${row.get('no_ask', 0)/100:.2f}")
            print(f"    YES Bid: ${row.get('yes_bid', 0)/100:.2f}")
            print(f"    NO Bid: ${row.get('no_bid', 0)/100:.2f}")
            print(f"    Volume: {row.get('volume', 0):,}")
            print(f"    Open Interest: {row.get('open_interest', 0):,}")
            print(f"    Status: {row.get('status', 'N/A')}")

            if idx >= 4:  # Limit display
                print(f"\n  ... and {len(detailed_markets) - 5} more markets")
                break

        # Focus on the main win market
        win_markets = detailed_markets[
            detailed_markets['title'].str.contains('win', case=False, na=False) |
            detailed_markets['title'].str.contains('winner', case=False, na=False)
        ]

        if not win_markets.empty:
            main_market = win_markets.iloc[0]
            print(f"\n🎯 Main Win Market:")
            print(f"  Ticker: {main_market['ticker']}")
            print(f"  Title: {main_market['title']}")
            print(f"  Current YES Price: ${main_market['yes_ask']/100:.2f}")
            print(f"  Current NO Price: ${main_market['no_ask']/100:.2f}")
            print(f"  Implied Probability (YES): {main_market['yes_ask']:.1f}%")
            print(f"  Volume: {main_market['volume']:,}")

            # Check for arbitrage
            total_cost = (main_market['yes_ask'] + main_market['no_ask']) / 100
            if total_cost < 1.0:
                profit = 1.0 - total_cost
                print(f"\n💰 ARBITRAGE OPPORTUNITY DETECTED!")
                print(f"  YES + NO = ${total_cost:.3f}")
                print(f"  Profit per $1: ${profit:.3f}")
                print(f"  Return: {(profit/total_cost)*100:.1f}%")

        # Create pricing DataFrame for analysis
        print("\n📊 Pricing Summary:")
        pricing_df = pd.DataFrame({
            'Market': detailed_markets['title'],
            'YES_Ask': detailed_markets['yes_ask'] / 100,
            'NO_Ask': detailed_markets['no_ask'] / 100,
            'Spread': (detailed_markets['yes_ask'] - detailed_markets['yes_bid']) / 100,
            'Volume': detailed_markets['volume'],
            'Status': detailed_markets['status']
        })

        print(pricing_df.head(10).to_string())

        # Save to CSV for further analysis
        csv_filename = f"cardinals_seahawks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        detailed_markets.to_csv(csv_filename, index=False)
        print(f"\n💾 Full data saved to: {csv_filename}")

    except Exception as e:
        print(f"❌ Error fetching detailed markets: {e}")
        return

    print("\n✅ Analysis complete!")


async def main():
    """Main function"""
    print("\n🚀 Neural SDK - Cardinals vs Seahawks Market Analysis\n")
    await find_cardinals_seahawks()


if __name__ == "__main__":
    asyncio.run(main())