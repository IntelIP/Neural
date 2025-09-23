#!/usr/bin/env python3
"""
Direct polling of Seahawks vs Cardinals markets
"""

import asyncio
from datetime import datetime
import pandas as pd
from neural.data_collection import get_game_markets

async def poll_markets_directly():
    """Poll markets directly using the working method"""

    event_ticker = "KXNFLGAME-25SEP25SEAARI"

    print("🏈 Polling Seahawks vs Cardinals Markets")
    print("="*60)
    print(f"Event: {event_ticker}")
    print(f"Duration: 15 seconds")
    print(f"Poll interval: 1 second\n")

    all_data = []

    for i in range(15):
        timestamp = datetime.now()

        # Get current market data
        markets = await get_game_markets(event_ticker)

        if not markets.empty:
            # Process each market
            for _, market in markets.iterrows():
                ticker = market['ticker']
                team = "Seattle" if "SEA" in ticker else "Arizona"

                # Calculate mid price and implied probability
                yes_bid = market['yes_bid'] / 100
                yes_ask = market['yes_ask'] / 100
                yes_mid = (yes_bid + yes_ask) / 2
                yes_spread = yes_ask - yes_bid
                implied_prob = yes_mid * 100

                # Store data point
                data_point = {
                    'timestamp': timestamp,
                    'ticker': ticker,
                    'team': team,
                    'yes_bid': yes_bid,
                    'yes_ask': yes_ask,
                    'yes_mid': yes_mid,
                    'yes_spread': yes_spread,
                    'implied_prob': implied_prob,
                    'volume': market['volume'],
                    'open_interest': market['open_interest']
                }

                all_data.append(data_point)

                # Print update
                print(f"[{timestamp.strftime('%H:%M:%S')}] {team}: "
                      f"${yes_mid:.3f} ({implied_prob:.1f}%) "
                      f"Spread: ${yes_spread:.3f} "
                      f"Vol: {market['volume']:,}")

        # Wait 1 second before next poll
        if i < 14:  # Don't wait after last poll
            await asyncio.sleep(1)

    return pd.DataFrame(all_data)

# Run the polling
print("\n🚀 Starting direct REST API polling...\n")
df = asyncio.run(poll_markets_directly())

print("\n" + "="*60)
print("📊 PRICING DATA COLLECTED")
print("="*60)

if not df.empty:
    print(f"\n✅ Collected {len(df)} data points over 15 seconds")

    # Analyze by team
    for team in ['Seattle', 'Arizona']:
        team_data = df[df['team'] == team]

        if not team_data.empty:
            print(f"\n🏈 {team}:")

            # Price statistics
            start_price = team_data.iloc[0]['yes_mid']
            end_price = team_data.iloc[-1]['yes_mid']
            price_change = end_price - start_price

            print(f"  Data points: {len(team_data)}")
            print(f"  Starting: ${start_price:.3f} ({team_data.iloc[0]['implied_prob']:.1f}%)")
            print(f"  Ending:   ${end_price:.3f} ({team_data.iloc[-1]['implied_prob']:.1f}%)")

            if abs(price_change) > 0.001:
                direction = "📈" if price_change > 0 else "📉"
                print(f"  Movement: {direction} ${abs(price_change):.3f} ({price_change*100:+.1f}¢)")
            else:
                print(f"  Movement: → No change")

            print(f"  Price range: ${team_data['yes_mid'].min():.3f} - ${team_data['yes_mid'].max():.3f}")
            print(f"  Avg spread: ${team_data['yes_spread'].mean():.3f}")
            print(f"  Final volume: {team_data.iloc[-1]['volume']:,} contracts")

    # Check for arbitrage
    print("\n📊 Market Efficiency Analysis:")

    # Get final prices for both teams
    sea_final = df[df['team'] == 'Seattle'].iloc[-1] if not df[df['team'] == 'Seattle'].empty else None
    ari_final = df[df['team'] == 'Arizona'].iloc[-1] if not df[df['team'] == 'Arizona'].empty else None

    if sea_final is not None and ari_final is not None:
        total_prob = sea_final['implied_prob'] + ari_final['implied_prob']
        print(f"  Total probability: {total_prob:.1f}%")

        # Check bid-ask for arbitrage
        total_ask = sea_final['yes_ask'] + ari_final['yes_ask']
        if total_ask < 1.0:
            profit = 1.0 - total_ask
            print(f"  💰 ARBITRAGE: Buy both for ${total_ask:.3f}, guaranteed ${profit:.3f} profit!")
        elif total_prob < 98:
            print(f"  📉 Market inefficiency: {100-total_prob:.1f}% gap")
        elif total_prob > 102:
            print(f"  📈 Overpriced: {total_prob-100:.1f}% over 100%")
        else:
            print(f"  ✅ Market is efficiently priced")

    # Save data
    filename = f"seahawks_cardinals_pricing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(filename, index=False)
    print(f"\n💾 Data saved to: {filename}")

    # Show price trajectory
    print("\n📈 Price Trajectory (every 3 seconds):")
    print("-"*60)

    for team in ['Seattle', 'Arizona']:
        team_data = df[df['team'] == team]
        if not team_data.empty:
            print(f"\n{team}:")
            # Show every 3rd data point
            for i in range(0, len(team_data), 3):
                row = team_data.iloc[i]
                print(f"  [{row['timestamp'].strftime('%H:%M:%S')}] "
                      f"${row['yes_mid']:.3f} ({row['implied_prob']:.1f}%)")

else:
    print("❌ No data collected")

print("\n" + "="*60)