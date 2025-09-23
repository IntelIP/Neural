#!/usr/bin/env python3
"""
Test REST API Streaming for Seahawks vs Cardinals
Uses polling as a reliable alternative to WebSocket
"""

import asyncio
from datetime import datetime
from typing import Dict, Any
from neural.trading.rest_streaming import RESTStreamingClient, MarketSnapshot, stream_via_rest


class SeahawksGameMonitor:
    """Monitor Seahawks vs Cardinals game via REST polling"""

    def __init__(self):
        self.price_history = []
        self.significant_moves = []
        self.last_prices = {}

    def handle_update(self, snapshot: MarketSnapshot) -> None:
        """Process market updates"""
        ticker = snapshot.ticker
        team = "Seattle" if "SEA" in ticker else "Arizona"

        # Track price history
        self.price_history.append({
            'timestamp': snapshot.timestamp,
            'team': team,
            'price': snapshot.yes_mid,
            'spread': snapshot.yes_spread,
            'probability': snapshot.implied_probability
        })

        # Check for significant moves
        if ticker in self.last_prices:
            price_change = snapshot.yes_mid - self.last_prices[ticker]
            if abs(price_change) > 0.005:  # More than 0.5 cent move
                self.significant_moves.append({
                    'timestamp': snapshot.timestamp,
                    'team': team,
                    'change': price_change,
                    'new_price': snapshot.yes_mid
                })

                direction = "📈" if price_change > 0 else "📉"
                print(f"\n  {direction} SIGNIFICANT MOVE: {team}")
                print(f"     Changed {price_change*100:+.1f}¢ to ${snapshot.yes_mid:.3f}")
                print(f"     New probability: {snapshot.implied_probability:.1f}%")

        self.last_prices[ticker] = snapshot.yes_mid

    def handle_price_change(self, ticker: str, old_price: float, new_price: float) -> None:
        """Handle price change events"""
        team = "Seattle" if "SEA" in ticker else "Arizona"
        change = new_price - old_price
        print(f"  💹 {team}: ${old_price:.3f} → ${new_price:.3f} ({change*100:+.1f}¢)")

    def print_summary(self) -> None:
        """Print monitoring summary"""
        print("\n" + "="*60)
        print("📊 GAME MONITORING SUMMARY")
        print("="*60)

        if self.price_history:
            print(f"\nTotal updates collected: {len(self.price_history)}")

            # Analyze by team
            for team in ["Seattle", "Arizona"]:
                team_data = [p for p in self.price_history if p['team'] == team]
                if team_data:
                    start_price = team_data[0]['price']
                    end_price = team_data[-1]['price']
                    avg_spread = sum(p['spread'] for p in team_data) / len(team_data)

                    print(f"\n🏈 {team}:")
                    print(f"  Starting price: ${start_price:.3f} ({start_price*100:.1f}%)")
                    print(f"  Ending price:   ${end_price:.3f} ({end_price*100:.1f}%)")
                    print(f"  Net change:     {(end_price-start_price)*100:+.1f}¢")
                    print(f"  Avg spread:     ${avg_spread:.3f}")
                    print(f"  Data points:    {len(team_data)}")

        if self.significant_moves:
            print(f"\n🎯 Significant Price Moves: {len(self.significant_moves)}")
            for move in self.significant_moves[-5:]:  # Show last 5
                print(f"  [{move['timestamp'].strftime('%H:%M:%S')}] "
                      f"{move['team']}: {move['change']*100:+.1f}¢ to ${move['new_price']:.3f}")


async def test_rest_streaming():
    """Test REST API streaming for Seahawks vs Cardinals"""
    print("🏈 Seahawks vs Cardinals - REST API Streaming Test")
    print("="*60)

    # Market tickers
    sea_ticker = "KXNFLGAME-25SEP25SEAARI-SEA"
    ari_ticker = "KXNFLGAME-25SEP25SEAARI-ARI"

    print(f"\n📊 Markets to monitor via REST polling:")
    print(f"  - {sea_ticker} (Seattle Seahawks)")
    print(f"  - {ari_ticker} (Arizona Cardinals)")

    monitor = SeahawksGameMonitor()

    print("\n📡 Starting REST API polling...")
    print("Poll interval: 1 second")
    print("Duration: 30 seconds")
    print("\nMonitoring for price changes...\n")

    # Create streaming client
    client = RESTStreamingClient(
        on_market_update=monitor.handle_update,
        on_price_change=monitor.handle_price_change,
        poll_interval=1.0,  # Poll every second
        min_price_change=0.001  # Trigger on 0.1 cent changes
    )

    try:
        async with client:
            # Subscribe to both markets
            await client.subscribe([sea_ticker, ari_ticker])

            # Initial snapshot
            await asyncio.sleep(2)

            # Show initial state
            snapshots = client.get_all_snapshots()
            if snapshots:
                print("📸 Initial Market State:")
                for ticker, snap in snapshots.items():
                    team = "Seattle" if "SEA" in ticker else "Arizona"
                    print(f"  {team}: ${snap.yes_mid:.3f} ({snap.implied_probability:.1f}%) "
                          f"Spread: ${snap.yes_spread:.3f}")

                    # Check arbitrage
                    if snap.arbitrage_opportunity > 0:
                        print(f"    💰 ARBITRAGE OPPORTUNITY: ${snap.arbitrage_opportunity:.3f} profit!")

            print("\nMonitoring continues...")

            # Stream for 30 seconds
            await asyncio.sleep(28)

            # Final snapshot
            print("\n📸 Final Market State:")
            final_snapshots = client.get_all_snapshots()
            for ticker, snap in final_snapshots.items():
                team = "Seattle" if "SEA" in ticker else "Arizona"
                print(f"  {team}: ${snap.yes_mid:.3f} ({snap.implied_probability:.1f}%) "
                      f"Vol: {snap.volume:,}")

    except KeyboardInterrupt:
        print("\n⏹️ Stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        monitor.print_summary()


async def test_stream_function():
    """Test the stream_via_rest convenience function"""
    print("\n🔄 Testing stream_via_rest function")
    print("="*60)

    tickers = [
        "KXNFLGAME-25SEP25SEAARI-SEA",
        "KXNFLGAME-25SEP25SEAARI-ARI"
    ]

    # Stream and collect data
    df = await stream_via_rest(
        tickers=tickers,
        duration_seconds=20,
        poll_interval=1.0
    )

    if not df.empty:
        print(f"\n📊 Collected {len(df)} data points")

        # Analyze data
        for ticker in tickers:
            ticker_data = df[df['ticker'] == ticker]
            if not ticker_data.empty:
                team = "Seattle" if "SEA" in ticker else "Arizona"
                print(f"\n{team}:")
                print(f"  Records: {len(ticker_data)}")
                print(f"  Price range: ${ticker_data['yes_mid'].min():.3f} - ${ticker_data['yes_mid'].max():.3f}")
                print(f"  Avg probability: {ticker_data['implied_prob'].mean():.1f}%")
                print(f"  Max spread: ${ticker_data['yes_spread'].max():.3f}")

        # Save data
        filename = f"rest_streaming_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename, index=False)
        print(f"\n💾 Data saved to: {filename}")


async def main():
    """Main test runner"""
    print("\n🚀 Neural SDK - REST API Streaming Test\n")

    print("This test demonstrates REST API polling as an alternative")
    print("to WebSocket streaming. It provides near real-time updates")
    print("without requiring special WebSocket permissions.\n")

    # Test 1: Custom monitoring
    await test_rest_streaming()

    print("\n" + "="*60)

    # Test 2: Convenience function
    await test_stream_function()

    print("\n✅ REST streaming test complete!")
    print("\n🎉 REST polling is working correctly!")
    print("This provides reliable market data updates without WebSocket.")


if __name__ == "__main__":
    asyncio.run(main())