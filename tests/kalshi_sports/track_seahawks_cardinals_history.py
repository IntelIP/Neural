#!/usr/bin/env python3
"""
Track pricing history for Seahawks vs Cardinals game over time
"""

import asyncio
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
from neural.data_collection import get_game_markets


async def track_pricing_history(event_ticker="KXNFLGAME-25SEP25SEAARI", duration_minutes=5):
    """Track pricing changes over time"""
    print(f"📊 Tracking pricing history for {event_ticker}")
    print(f"Duration: {duration_minutes} minutes")
    print("=" * 60)

    history = []
    start_time = datetime.now()
    end_time = start_time + timedelta(minutes=duration_minutes)

    print(f"\nStarted at: {start_time.strftime('%H:%M:%S')}")
    print(f"Will end at: {end_time.strftime('%H:%M:%S')}")
    print("\nPress Ctrl+C to stop early\n")

    interval_seconds = 30  # Check every 30 seconds

    try:
        while datetime.now() < end_time:
            # Fetch current market data
            markets = await get_game_markets(event_ticker)

            if not markets.empty:
                # Get Seattle and Arizona markets
                sea_market = markets[markets['ticker'].str.contains('SEA', na=False)].iloc[0] if not markets[markets['ticker'].str.contains('SEA', na=False)].empty else None
                ari_market = markets[markets['ticker'].str.contains('ARI', na=False)].iloc[0] if not markets[markets['ticker'].str.contains('ARI', na=False)].empty else None

                timestamp = datetime.now()

                # Record data point
                data_point = {
                    'timestamp': timestamp,
                    'time_str': timestamp.strftime('%H:%M:%S'),
                    'seattle_yes_bid': sea_market['yes_bid'] / 100 if sea_market is not None else None,
                    'seattle_yes_ask': sea_market['yes_ask'] / 100 if sea_market is not None else None,
                    'seattle_implied_prob': sea_market['yes_ask'] if sea_market is not None else None,
                    'seattle_volume': sea_market['volume'] if sea_market is not None else None,
                    'arizona_yes_bid': ari_market['yes_bid'] / 100 if ari_market is not None else None,
                    'arizona_yes_ask': ari_market['yes_ask'] / 100 if ari_market is not None else None,
                    'arizona_implied_prob': ari_market['yes_ask'] if ari_market is not None else None,
                    'arizona_volume': ari_market['volume'] if ari_market is not None else None,
                }

                history.append(data_point)

                # Display current prices
                print(f"[{data_point['time_str']}] SEA: ${data_point['seattle_yes_ask']:.2f} ({data_point['seattle_implied_prob']:.1f}%) | "
                      f"ARI: ${data_point['arizona_yes_ask']:.2f} ({data_point['arizona_implied_prob']:.1f}%) | "
                      f"Vol: SEA {data_point['seattle_volume']:,} / ARI {data_point['arizona_volume']:,}")

                # Check for significant changes
                if len(history) > 1:
                    prev = history[-2]
                    sea_change = (data_point['seattle_implied_prob'] - prev['seattle_implied_prob'])
                    ari_change = (data_point['arizona_implied_prob'] - prev['arizona_implied_prob'])

                    if abs(sea_change) > 1:  # More than 1% change
                        direction = "📈" if sea_change > 0 else "📉"
                        print(f"    {direction} Seattle moved {sea_change:+.1f}%")

                    if abs(ari_change) > 1:
                        direction = "📈" if ari_change > 0 else "📉"
                        print(f"    {direction} Arizona moved {ari_change:+.1f}%")

            # Wait for next interval
            await asyncio.sleep(interval_seconds)

    except KeyboardInterrupt:
        print("\n\n⏹️ Stopped by user")

    # Save history
    if history:
        df = pd.DataFrame(history)
        filename = f"pricing_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename, index=False)
        print(f"\n💾 History saved to: {filename}")

        # Analyze pricing trends
        analyze_pricing_trends(df)

        # Create visualization
        create_pricing_chart(df)

    return df if history else pd.DataFrame()


def analyze_pricing_trends(df):
    """Analyze pricing trends from history"""
    print("\n📈 Pricing Analysis:")
    print("-" * 40)

    if df.empty:
        print("No data to analyze")
        return

    # Seattle analysis
    print("\n🏈 Seattle Seahawks:")
    print(f"  Starting Price: ${df['seattle_yes_ask'].iloc[0]:.2f} ({df['seattle_implied_prob'].iloc[0]:.1f}%)")
    print(f"  Ending Price:   ${df['seattle_yes_ask'].iloc[-1]:.2f} ({df['seattle_implied_prob'].iloc[-1]:.1f}%)")
    print(f"  Price Range:    ${df['seattle_yes_ask'].min():.2f} - ${df['seattle_yes_ask'].max():.2f}")
    print(f"  Avg Spread:     ${(df['seattle_yes_ask'] - df['seattle_yes_bid']).mean():.3f}")
    print(f"  Total Volume:   {df['seattle_volume'].iloc[-1] - df['seattle_volume'].iloc[0]:,} contracts")

    # Arizona analysis
    print("\n🏈 Arizona Cardinals:")
    print(f"  Starting Price: ${df['arizona_yes_ask'].iloc[0]:.2f} ({df['arizona_implied_prob'].iloc[0]:.1f}%)")
    print(f"  Ending Price:   ${df['arizona_yes_ask'].iloc[-1]:.2f} ({df['arizona_implied_prob'].iloc[-1]:.1f}%)")
    print(f"  Price Range:    ${df['arizona_yes_ask'].min():.2f} - ${df['arizona_yes_ask'].max():.2f}")
    print(f"  Avg Spread:     ${(df['arizona_yes_ask'] - df['arizona_yes_bid']).mean():.3f}")
    print(f"  Total Volume:   {df['arizona_volume'].iloc[-1] - df['arizona_volume'].iloc[0]:,} contracts")

    # Movement analysis
    sea_net_change = df['seattle_implied_prob'].iloc[-1] - df['seattle_implied_prob'].iloc[0]
    ari_net_change = df['arizona_implied_prob'].iloc[-1] - df['arizona_implied_prob'].iloc[0]

    print("\n📊 Net Movement:")
    print(f"  Seattle:  {sea_net_change:+.1f}% {'📈' if sea_net_change > 0 else '📉' if sea_net_change < 0 else '➡️'}")
    print(f"  Arizona:  {ari_net_change:+.1f}% {'📈' if ari_net_change > 0 else '📉' if ari_net_change < 0 else '➡️'}")

    # Volatility
    sea_volatility = df['seattle_implied_prob'].std()
    ari_volatility = df['arizona_implied_prob'].std()
    print(f"\n📊 Volatility (std dev):")
    print(f"  Seattle:  {sea_volatility:.2f}%")
    print(f"  Arizona:  {ari_volatility:.2f}%")


def create_pricing_chart(df):
    """Create visualization of pricing history"""
    if df.empty:
        print("No data to visualize")
        return

    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Convert timestamp to matplotlib format
        times = pd.to_datetime(df['timestamp'])

        # Plot implied probabilities
        ax1.plot(times, df['seattle_implied_prob'], 'b-', label='Seattle', linewidth=2)
        ax1.plot(times, df['arizona_implied_prob'], 'r-', label='Arizona', linewidth=2)
        ax1.fill_between(times, df['seattle_implied_prob'], df['arizona_implied_prob'],
                         where=(df['seattle_implied_prob'] >= df['arizona_implied_prob']),
                         alpha=0.3, color='blue', label='SEA Favored')
        ax1.fill_between(times, df['seattle_implied_prob'], df['arizona_implied_prob'],
                         where=(df['seattle_implied_prob'] < df['arizona_implied_prob']),
                         alpha=0.3, color='red', label='ARI Favored')

        ax1.set_ylabel('Implied Win Probability (%)')
        ax1.set_title('Seahawks vs Cardinals - Win Probability Over Time')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([40, 60])  # Adjust based on data

        # Plot spreads
        sea_spread = df['seattle_yes_ask'] - df['seattle_yes_bid']
        ari_spread = df['arizona_yes_ask'] - df['arizona_yes_bid']

        ax2.plot(times, sea_spread * 100, 'b-', label='Seattle Spread', alpha=0.7)
        ax2.plot(times, ari_spread * 100, 'r-', label='Arizona Spread', alpha=0.7)
        ax2.set_ylabel('Bid-Ask Spread (cents)')
        ax2.set_xlabel('Time')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)

        # Format x-axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.xticks(rotation=45)

        plt.tight_layout()
        filename = f"pricing_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=100)
        print(f"\n📊 Chart saved to: {filename}")
        plt.show()

    except ImportError:
        print("\n⚠️ Matplotlib not installed. Install with: pip install matplotlib")


async def get_snapshot():
    """Get a single snapshot of current prices"""
    event_ticker = "KXNFLGAME-25SEP25SEAARI"

    print("\n📸 Current Market Snapshot")
    print("=" * 60)

    markets = await get_game_markets(event_ticker)

    if markets.empty:
        print("❌ No market data available")
        return

    # Display all markets for this game
    print(f"\n🏈 {event_ticker}")
    print(f"Total Markets: {len(markets)}\n")

    for idx, market in markets.iterrows():
        print(f"Market {idx + 1}: {market['title']}")
        print(f"  Ticker: {market['ticker']}")
        print(f"  Status: {market['status']}")
        print(f"  YES: ${market['yes_ask']/100:.2f} (Bid: ${market['yes_bid']/100:.2f})")
        print(f"  NO:  ${market['no_ask']/100:.2f} (Bid: ${market['no_bid']/100:.2f})")
        print(f"  Implied Prob: {market['yes_ask']:.1f}%")
        print(f"  Volume: {market['volume']:,}")
        print(f"  Open Interest: {market['open_interest']:,}")

        # Check for arbitrage
        total = (market['yes_ask'] + market['no_ask']) / 100
        if total < 1.0:
            profit = 1.0 - total
            print(f"  💰 ARBITRAGE: Total = ${total:.3f}, Profit = ${profit:.3f}")
        print()


async def main():
    """Main function"""
    print("\n🚀 Neural SDK - Seahawks vs Cardinals Pricing History Tracker\n")

    print("Choose an option:")
    print("1. Get current snapshot")
    print("2. Track pricing for 5 minutes")
    print("3. Track pricing for custom duration")

    choice = input("\nEnter choice (1-3): ").strip()

    if choice == "1":
        await get_snapshot()
    elif choice == "2":
        await track_pricing_history(duration_minutes=5)
    elif choice == "3":
        try:
            minutes = int(input("Enter duration in minutes: "))
            await track_pricing_history(duration_minutes=minutes)
        except ValueError:
            print("Invalid duration")
    else:
        print("Invalid choice")


if __name__ == "__main__":
    asyncio.run(main())