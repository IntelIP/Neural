#!/usr/bin/env python3
"""
Test FIX streaming for Cardinals vs Seahawks real-time pricing
"""

import asyncio
from datetime import datetime
from dotenv import load_dotenv
from neural.trading.fix_streaming import FIXStreamingClient, stream_market_data

load_dotenv()


async def test_streaming():
    """Test streaming market data for Cardinals vs Seahawks"""
    print("🏈 Cardinals vs Seahawks - Real-Time Streaming Test")
    print("="*60)

    # The two markets for this game
    symbols = [
        "KXNFLGAME-25SEP25SEAARI-SEA",  # Seattle to win
        "KXNFLGAME-25SEP25SEAARI-ARI"   # Arizona to win
    ]

    print(f"\n📊 Streaming markets:")
    for symbol in symbols:
        print(f"  - {symbol}")

    print("\n🔄 Starting real-time stream (30 seconds)...")
    print("Press Ctrl+C to stop early\n")

    # Stream for 30 seconds
    df = await stream_market_data(
        symbols=symbols,
        duration_seconds=30,
        on_update=None  # Using default print handler
    )

    if not df.empty:
        print(f"\n✅ Collected {len(df)} data points")

        # Analyze the data
        print("\n📈 Streaming Analysis:")
        for symbol in symbols:
            symbol_data = df[df['symbol'] == symbol]
            if not symbol_data.empty:
                team = "Seattle" if "SEA" in symbol else "Arizona"
                print(f"\n{team}:")
                print(f"  Data points: {len(symbol_data)}")
                print(f"  Avg Bid: ${symbol_data['bid'].mean():.3f}")
                print(f"  Avg Ask: ${symbol_data['ask'].mean():.3f}")
                print(f"  Avg Spread: ${symbol_data['spread'].mean():.3f}")
                print(f"  Implied Prob Range: {symbol_data['implied_prob'].min():.1f}% - {symbol_data['implied_prob'].max():.1f}%")

        # Save data
        filename = f"streaming_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename, index=False)
        print(f"\n💾 Data saved to: {filename}")
    else:
        print("\n⚠️ No data collected - market might be closed or connection issues")


async def test_with_callbacks():
    """Test streaming with custom callbacks"""
    print("\n🎯 Advanced Streaming Test with Callbacks")
    print("="*60)

    # Track spreads and changes
    last_prices = {}

    def on_market_update(snapshot):
        """Custom handler for market updates"""
        symbol = snapshot.symbol
        team = "SEA" if "SEA" in symbol else "ARI"

        # Check for price changes
        if symbol in last_prices:
            last_mid = last_prices[symbol]
            change = snapshot.mid_price - last_mid
            if abs(change) > 0.001:  # More than 0.1 cent move
                direction = "📈" if change > 0 else "📉"
                print(f"  {direction} {team} moved ${change:+.3f} to ${snapshot.mid_price:.3f}")

        last_prices[symbol] = snapshot.mid_price

        # Alert on wide spreads
        if snapshot.spread > 0.03:  # More than 3 cents
            print(f"  ⚠️ Wide spread on {team}: ${snapshot.spread:.3f}")

    def on_error(error_msg):
        """Handle errors"""
        print(f"  ❌ Error: {error_msg}")

    # Create client with callbacks
    client = FIXStreamingClient(
        on_market_data=on_market_update,
        on_error=on_error,
        auto_reconnect=True
    )

    try:
        async with client:
            # Subscribe to both teams
            await client.subscribe("KXNFLGAME-25SEP25SEAARI-SEA")
            await client.subscribe("KXNFLGAME-25SEP25SEAARI-ARI")

            print("\n🔄 Monitoring price movements (20 seconds)...")
            await asyncio.sleep(20)

            # Get final snapshots
            snapshots = client.get_all_snapshots()
            if snapshots:
                print("\n📸 Final Market Snapshot:")
                for symbol, snapshot in snapshots.items():
                    team = "Seattle" if "SEA" in symbol else "Arizona"
                    print(f"  {team}: ${snapshot.mid_price:.3f} ({snapshot.implied_probability:.1f}%) Spread: ${snapshot.spread:.3f}")

    except Exception as e:
        print(f"\n❌ Streaming failed: {e}")


async def main():
    """Run all streaming tests"""
    print("\n🚀 Neural SDK - FIX Streaming Infrastructure Test\n")

    # Note: FIX streaming requires market data subscription permissions
    # This test will attempt to connect but may not receive data without proper entitlements

    print("⚠️ Note: FIX market data requires subscription entitlements from Kalshi")
    print("Contact Kalshi to enable FIX market data for your account\n")

    # Test 1: Basic streaming
    await test_streaming()

    # Test 2: Advanced with callbacks
    await test_with_callbacks()

    print("\n✅ Streaming infrastructure test complete!")


if __name__ == "__main__":
    asyncio.run(main())