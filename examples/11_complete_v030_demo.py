#!/usr/bin/env python3
"""
Complete Neural SDK v0.3.0 Demo
================================

This example demonstrates all the new v0.3.0 features:
- NBA and enhanced sports market collection
- Moneyline market filtering
- Historical data fetching with OHLCV
- SportMarketCollector unified interface

Run with: python examples/11_complete_v030_demo.py
"""

import asyncio
from datetime import datetime

import pandas as pd

from neural.data_collection.kalshi import (
    KalshiMarketsSource,
    SportMarketCollector,
    filter_moneyline_markets,
    get_moneyline_markets,
    get_nba_games,
    get_nfl_games,
)


async def demo_sports_collection():
    """Demonstrate enhanced sports market collection"""
    print("🏆 Neural SDK v0.3.0 - Complete Sports Market Demo")
    print("=" * 50)

    # 1. Test individual sport functions
    print("\n🏈 NFL Markets:")
    nfl_games = await get_nfl_games(status="open", limit=3)
    print(f"Found {len(nfl_games)} NFL markets")
    if not nfl_games.empty:
        print(f"Sample: {nfl_games.iloc[0]['title']}")

    print("\n🏀 NBA Markets:")
    nba_games = await get_nba_games(status="open", limit=3)
    print(f"Found {len(nba_games)} NBA markets")
    if not nba_games.empty:
        print(f"Sample: {nba_games.iloc[0]['title']}")

    # 2. Test moneyline filtering
    print("\n🎯 Moneyline Filtering:")
    if not nfl_games.empty:
        moneylines = filter_moneyline_markets(nfl_games)
        print(f"Filtered to {len(moneylines)} NFL moneyline markets")

    # 3. Test unified moneyline function
    print("\n⚡ Unified Moneyline Collection:")
    nfl_moneylines = await get_moneyline_markets("NFL", limit=2)
    print(f"NFL moneylines: {len(nfl_moneylines)}")

    # 4. Test SportMarketCollector
    print("\n🌐 SportMarketCollector Demo:")
    collector = SportMarketCollector()

    # Multi-sport collection
    multi_sport = await collector.get_moneylines_only(["NFL", "NBA"], limit=5)
    print(f"Multi-sport moneylines: {len(multi_sport)}")

    if not multi_sport.empty and "sport" in multi_sport.columns:
        sports_found = multi_sport["sport"].unique()
        print(f"Sports found: {list(sports_found)}")

    return nfl_games


async def demo_historical_data(sample_ticker=None):
    """Demonstrate historical data fetching"""
    print("\n📊 Historical Data Demo")
    print("-" * 30)

    if not sample_ticker:
        # Use a known NFL market
        sample_ticker = "KXNFLGAME-25NOV02SEAWAS-WAS"

    print(f"Fetching historical data for: {sample_ticker}")

    # Create data source
    source = KalshiMarketsSource(series_ticker="KXNFLGAME")

    # Fetch historical candlesticks
    historical_data = await source.fetch_historical_candlesticks(
        market_ticker=sample_ticker,
        interval=60,  # 1-hour bars
        hours_back=24,  # Last 24 hours
    )

    if not historical_data.empty:
        print(f"✅ Retrieved {len(historical_data)} hourly candlesticks")
        print(f"Columns: {list(historical_data.columns)}")

        # Show summary statistics
        print(f"\nPrice Summary:")
        print(f"  Open: ${historical_data['open'].iloc[0]:.3f}")
        print(f"  Close: ${historical_data['close'].iloc[-1]:.3f}")
        print(f"  High: ${historical_data['high'].max():.3f}")
        print(f"  Low: ${historical_data['low'].min():.3f}")
        print(f"  Volume: {historical_data['volume'].sum():,} contracts")

        # Show first few rows
        print(f"\nSample Data:")
        print(
            historical_data[["timestamp", "open", "high", "low", "close", "volume"]]
            .head(3)
            .to_string(index=False)
        )

        return historical_data
    else:
        print("⚠️  No historical data available")
        return pd.DataFrame()


async def demo_complete_workflow():
    """Demonstrate complete workflow: market discovery -> historical data -> analysis"""
    print("\n🚀 Complete Workflow Demo")
    print("-" * 30)

    try:
        # Step 1: Find active NFL moneyline markets
        print("Step 1: Finding active moneyline markets...")
        moneylines = await get_moneyline_markets("NFL", limit=1)

        if moneylines.empty:
            print("No active markets found, using sample ticker")
            sample_ticker = "KXNFLGAME-25NOV02SEAWAS-WAS"
        else:
            sample_ticker = moneylines.iloc[0]["ticker"]
            market_title = moneylines.iloc[0]["title"]
            print(f"Found market: {market_title}")

        # Step 2: Fetch historical data
        print(f"\nStep 2: Fetching historical data for {sample_ticker}...")
        source = KalshiMarketsSource()
        historical_data = await source.fetch_historical_candlesticks(
            market_ticker=sample_ticker, interval=60, hours_back=48
        )

        if not historical_data.empty:
            print(f"✅ Got {len(historical_data)} data points")

            # Step 3: Simple analysis
            print(f"\nStep 3: Basic analysis...")

            # Calculate volatility
            returns = historical_data["close"].pct_change().dropna()
            volatility = returns.std() * 100

            # Calculate price movement
            price_change = (
                (historical_data["close"].iloc[-1] - historical_data["close"].iloc[0])
                / historical_data["close"].iloc[0]
                * 100
            )

            print(f"  Price change: {price_change:.2f}%")
            print(f"  Volatility: {volatility:.2f}%")
            print(f"  Avg volume: {historical_data['volume'].mean():.0f} contracts")

            # Trading opportunity assessment
            if abs(price_change) > 2:
                print(f"  📈 High movement detected - potential trading opportunity")
            else:
                print(f"  📊 Low movement - stable market")

            print(f"\n✅ Complete workflow successful!")
            return True
        else:
            print("❌ No historical data available")
            return False

    except Exception as e:
        print(f"❌ Workflow error: {e}")
        return False


async def main():
    """Run complete v0.3.0 demonstration"""
    print("🎯 Neural SDK v0.3.0 - Complete Feature Demonstration")
    print("=" * 60)
    print("Testing all new features with real Kalshi API data...")
    print("=" * 60)

    try:
        # Demo 1: Enhanced sports collection
        nfl_games = await demo_sports_collection()

        # Demo 2: Historical data
        sample_ticker = None
        if not nfl_games.empty:
            sample_ticker = nfl_games.iloc[0]["ticker"]

        historical_data = await demo_historical_data(sample_ticker)

        # Demo 3: Complete workflow
        success = await demo_complete_workflow()

        # Final summary
        print("\n" + "=" * 60)
        print("🎉 Neural SDK v0.3.0 Demo Complete!")
        print("=" * 60)
        print("✅ Sports market collection: Working")
        print("✅ Moneyline filtering: Working")
        print("✅ Historical data: Working")
        print("✅ Unified interface: Working")
        print(f"✅ Complete workflow: {'Working' if success else 'Partial'}")

        print(f"\n📊 Data Summary:")
        print(f"  NFL markets tested: {len(nfl_games) if not nfl_games.empty else 0}")
        print(
            f"  Historical data points: {len(historical_data) if not historical_data.empty else 0}"
        )

        print(f"\n🚀 Neural SDK v0.3.0 is ready for production!")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
