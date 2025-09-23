#!/usr/bin/env python3
"""
Test: Find Arizona Cardinals vs Seattle Seahawks game and get pricing history
FIXED VERSION - Correctly extracts event ticker from ticker field
"""

import asyncio
import pandas as pd
from datetime import datetime
from neural.data_collection import KalshiMarketsSource, get_game_markets


async def find_cardinals_seahawks():
    """Find and analyze Cardinals vs Seahawks game"""
    print("🏈 Searching for Arizona Cardinals vs Seattle Seahawks")
    print("=" * 60)

    # First, try the known working ticker directly
    print("\n📊 Method 1: Using known ticker KXNFLGAME-25SEP25SEAARI")
    try:
        detailed_markets = await get_game_markets("KXNFLGAME-25SEP25SEAARI")

        if not detailed_markets.empty:
            print(f"✅ Found {len(detailed_markets)} markets for Seahawks vs Cardinals!")
            display_market_data(detailed_markets, "KXNFLGAME-25SEP25SEAARI")
            return detailed_markets
        else:
            print("⚠️ Game might be completed or not available")
    except Exception as e:
        print(f"❌ Error with direct ticker: {e}")

    # Method 2: Search all NFL markets using KalshiMarketsSource
    print("\n📊 Method 2: Searching all NFL markets...")
    source = KalshiMarketsSource(
        series_ticker="KXNFLGAME",
        status=None,  # Get all statuses
        use_authenticated=True
    )

    async with source:
        async for df in source.collect():
            print(f"\n✅ Found {len(df)} total NFL markets")

            if df.empty:
                print("❌ No NFL markets found")
                break

            # Extract event ticker from ticker field
            # Ticker format: KXNFLGAME-YYMMMDDTEAMS-[market_specific]
            if 'ticker' in df.columns:
                # Create event_ticker column by extracting the game portion
                df['extracted_event'] = df['ticker'].apply(extract_event_ticker)

                # Search for Cardinals (ARI) vs Seahawks (SEA)
                sea_ari_markets = df[
                    (df['ticker'].str.contains('SEA', na=False) &
                     df['ticker'].str.contains('ARI', na=False)) |
                    (df['ticker'].str.contains('SEAARI', na=False)) |
                    (df['ticker'].str.contains('ARISEA', na=False))
                ]

                if not sea_ari_markets.empty:
                    print(f"\n✅ Found {len(sea_ari_markets)} Cardinals vs Seahawks markets!")

                    # Get unique events
                    unique_events = sea_ari_markets['extracted_event'].unique()
                    print(f"Unique game events: {unique_events}")

                    # Process each unique event
                    for event in unique_events:
                        if pd.notna(event):
                            print(f"\n📈 Fetching details for event: {event}")
                            try:
                                detailed = await get_game_markets(event)
                                if not detailed.empty:
                                    display_market_data(detailed, event)
                                    return detailed
                            except Exception as e:
                                print(f"  Error fetching {event}: {e}")
                else:
                    print("\n❌ No Cardinals vs Seahawks markets found")

                    # Show sample of available games for debugging
                    print("\n📋 Sample of available games:")
                    unique_events = df['extracted_event'].dropna().unique()[:10]
                    for event in unique_events:
                        # Parse teams from event ticker
                        teams = extract_teams(event)
                        if teams:
                            print(f"  - {event} ({teams})")
            else:
                print("❌ No 'ticker' column in data")

            break  # Only process first batch

    print("\n❌ Could not find Cardinals vs Seahawks game")
    return pd.DataFrame()


def extract_event_ticker(ticker):
    """Extract event ticker from full ticker string"""
    if pd.isna(ticker):
        return None

    # Ticker formats we've seen:
    # 1. KXNFLGAME-YYMMMDDTEAMS-MARKETTYPE
    # 2. KXNFLGAME-YYMMMDDTEAMS

    parts = ticker.split('-')
    if len(parts) >= 2:
        # Return first two parts (series and date/teams)
        return f"{parts[0]}-{parts[1]}"
    return ticker


def extract_teams(event_ticker):
    """Extract team codes from event ticker"""
    if pd.isna(event_ticker) or '-' not in event_ticker:
        return None

    parts = event_ticker.split('-')
    if len(parts) >= 2:
        # The second part contains date and teams
        # Format: YYMMMDDTEAMS (e.g., 25SEP25SEAARI)
        date_teams = parts[1]

        # Extract last 6 characters as team codes (3 chars each)
        if len(date_teams) >= 6:
            teams = date_teams[-6:]
            team1 = teams[:3]
            team2 = teams[3:]
            return f"{team1} vs {team2}"

    return None


def display_market_data(df, event_ticker):
    """Display detailed market data and pricing"""
    print(f"\n📊 Market Data for {event_ticker}")
    print("-" * 60)

    # Show market types
    print("\n📋 Available Markets:")
    for idx, row in df.head(10).iterrows():
        print(f"\n  Market {idx + 1}:")
        print(f"    Title: {row.get('title', 'N/A')}")
        print(f"    Ticker: {row.get('ticker', 'N/A')}")
        print(f"    Status: {row.get('status', 'N/A')}")

        # Pricing information
        yes_ask = row.get('yes_ask', 0)
        no_ask = row.get('no_ask', 0)
        yes_bid = row.get('yes_bid', 0)
        no_bid = row.get('no_bid', 0)

        if yes_ask or no_ask:
            print(f"    Pricing:")
            print(f"      YES: Bid ${yes_bid/100:.2f} / Ask ${yes_ask/100:.2f}")
            print(f"      NO:  Bid ${no_bid/100:.2f} / Ask ${no_ask/100:.2f}")
            print(f"      Implied Prob: {yes_ask:.1f}%")

            # Check for arbitrage
            total_cost = (yes_ask + no_ask) / 100
            if total_cost < 1.0:
                profit = 1.0 - total_cost
                print(f"    💰 ARBITRAGE: YES+NO = ${total_cost:.3f}, Profit = ${profit:.3f}")

        print(f"    Volume: {row.get('volume', 0):,}")
        print(f"    Open Interest: {row.get('open_interest', 0):,}")

    if len(df) > 10:
        print(f"\n  ... and {len(df) - 10} more markets")

    # Focus on main winner market
    winner_markets = df[
        df['title'].str.contains('Winner', case=False, na=False) |
        df['title'].str.contains('win', case=False, na=False)
    ]

    if not winner_markets.empty:
        main = winner_markets.iloc[0]
        print(f"\n🎯 Main Winner Market:")
        print(f"  Title: {main['title']}")
        print(f"  Current Prices:")
        print(f"    YES: ${main['yes_ask']/100:.2f} ({main['yes_ask']:.1f}% implied)")
        print(f"    NO:  ${main['no_ask']/100:.2f} ({main['no_ask']:.1f}% implied)")
        print(f"  Spread: ${(main['yes_ask'] - main['yes_bid'])/100:.2f}")
        print(f"  Volume: {main['volume']:,}")

        # Pricing history summary
        print(f"\n📈 Pricing Analysis:")
        print(f"  Best YES Price: ${main['yes_bid']/100:.2f} (bid)")
        print(f"  Best NO Price: ${main['no_bid']/100:.2f} (bid)")

        total = (main['yes_ask'] + main['no_ask']) / 100
        if total < 1.0:
            print(f"  ⚠️ Arbitrage Available: Total = ${total:.3f}")

    # Save data
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"seahawks_cardinals_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"\n💾 Data saved to: {filename}")


async def main():
    """Main function"""
    print("\n🚀 Neural SDK - Cardinals vs Seahawks Market Analysis\n")

    # Run the analysis
    markets = await find_cardinals_seahawks()

    if not markets.empty:
        print("\n✅ Analysis complete! Found pricing data.")
    else:
        print("\n⚠️ Could not retrieve market data. Game may be completed.")


if __name__ == "__main__":
    asyncio.run(main())