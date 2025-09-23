#!/usr/bin/env python3
"""
Example: Mean Reversion Trading Strategy

Demonstrates using the analysis stack to trade price divergences.
"""

import asyncio
import pandas as pd
from datetime import datetime, timedelta
from neural.data_collection import KalshiMarketsSource, get_game_markets
from neural.analysis.strategies import MeanReversionStrategy
from neural.analysis.execution import OrderManager
from neural.trading import TradingClient


async def run_mean_reversion_strategy():
    """Run mean reversion strategy on live NFL games"""
    print("🎯 Mean Reversion Strategy Example")
    print("=" * 50)

    # Initialize strategy with conservative parameters
    strategy = MeanReversionStrategy(
        name="NFL Mean Reversion",
        initial_capital=1000,
        divergence_threshold=0.05,  # 5% divergence triggers signal
        reversion_target=0.5,  # Target 50% reversion
        use_sportsbook=True,
        lookback_periods=20,
        max_position_size=0.1,  # 10% max per position
        stop_loss=0.2,  # 20% stop loss
        take_profit=0.5,  # 50% take profit
        use_kelly=True,
        kelly_fraction=0.25  # Conservative Kelly
    )

    # Initialize order manager (dry run for demo)
    order_manager = OrderManager(
        trading_client=None,  # Would pass real client here
        dry_run=True,  # Simulate orders
        require_confirmation=False
    )

    # Get live NFL games
    print("\n📊 Fetching live NFL markets...")
    source = KalshiMarketsSource(
        series_ticker="KXNFLGAME",
        status=None,
        use_authenticated=True
    )

    games_df = await source.fetch()

    if games_df.empty:
        print("No active NFL games found")
        return

    print(f"Found {len(games_df)} NFL markets")

    # Group by event (game)
    events = games_df.groupby('event_ticker').first()
    print(f"\n🏈 Analyzing {len(events)} games for mean reversion...")

    for event_ticker, _ in events.iterrows():
        print(f"\n📈 Game: {event_ticker}")

        # Get detailed market data for this game
        try:
            market_data = await get_game_markets(event_ticker)

            if market_data.empty:
                print(f"  ⚠️ No market data available")
                continue

            # Prepare data for strategy
            market_df = pd.DataFrame({
                'ticker': market_data['ticker'],
                'yes_ask': market_data['yes_ask'] / 100,  # Convert to decimal
                'no_ask': market_data['no_ask'] / 100,
                'volume': market_data['volume']
            })

            # Generate trading signal
            signal = strategy.analyze(market_df)

            # Display signal
            if signal.type.value != "hold":
                print(f"  ✅ Signal: {signal.type.value.upper()}")
                print(f"     Ticker: {signal.ticker}")
                print(f"     Size: {signal.size} contracts")
                print(f"     Confidence: {signal.confidence:.1%}")

                if signal.metadata:
                    print(f"     Fair Value: ${signal.metadata.get('fair_value', 0):.2f}")
                    print(f"     Divergence: {signal.metadata.get('divergence', 0):.3f}")

                # Execute signal (simulated)
                result = await order_manager.execute_signal(signal)
                if result:
                    print(f"     Order: {result.get('status', 'executed')}")
            else:
                print(f"  ⏸️ No signal (holding)")

            # Show current metrics
            print(f"\n  📊 Strategy Metrics:")
            metrics = strategy.get_performance_metrics()
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"     {key}: {value:.2f}")
                else:
                    print(f"     {key}: {value}")

        except Exception as e:
            print(f"  ❌ Error analyzing game: {e}")
            continue

    # Show portfolio summary
    print("\n" + "=" * 50)
    print("📋 Portfolio Summary:")
    portfolio = order_manager.get_portfolio_summary()

    print(f"   Active Positions: {portfolio['positions']}")
    print(f"   Total Value: ${portfolio['total_value']:.2f}")
    print(f"   Total P&L: ${portfolio['total_pnl']:.2f}")
    print(f"   Total Orders: {portfolio['total_orders']}")

    if portfolio['active_positions']:
        print("\n   Position Details:")
        for ticker, pos in portfolio['active_positions'].items():
            print(f"   - {ticker}:")
            print(f"     Side: {pos['side'].upper()}")
            print(f"     Size: {pos['size']} contracts")
            print(f"     Entry: ${pos['entry_price']:.2f}")
            print(f"     Current: ${pos['current_price']:.2f}")
            print(f"     P&L: ${pos['pnl']:.2f} ({pos['pnl_pct']:.1f}%)")


async def backtest_mean_reversion():
    """Backtest mean reversion strategy on historical data"""
    print("\n📊 Backtesting Mean Reversion Strategy")
    print("=" * 50)

    from neural.analysis.backtesting import Backtester

    # Create strategy
    strategy = MeanReversionStrategy(
        divergence_threshold=0.05,
        initial_capital=10000
    )

    # Initialize backtester
    backtester = Backtester(
        initial_capital=10000,
        fee_rate=0.0  # Kalshi fees handled by strategy
    )

    # Run backtest on recent data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    print(f"Period: {start_date.date()} to {end_date.date()}")
    print("Markets: NFL games")

    try:
        result = await backtester.backtest(
            strategy=strategy,
            start_date=start_date,
            end_date=end_date,
            markets=["KXNFLGAME"]  # NFL games only
        )

        # Display results
        print("\n📈 Backtest Results:")
        print(f"   Total Return: {result.total_return:.2f}%")
        print(f"   Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"   Max Drawdown: {result.max_drawdown:.2f}%")
        print(f"   Win Rate: {result.win_rate:.1%}")
        print(f"   Total Trades: {result.total_trades}")
        print(f"   Profit Factor: {result.profit_factor:.2f}")

        # Show trade distribution
        if result.trades:
            trades_df = pd.DataFrame(result.trades)
            print("\n📊 Trade Analysis:")
            print(f"   Average Win: ${trades_df[trades_df['pnl'] > 0]['pnl'].mean():.2f}")
            print(f"   Average Loss: ${trades_df[trades_df['pnl'] < 0]['pnl'].mean():.2f}")
            print(f"   Best Trade: ${trades_df['pnl'].max():.2f}")
            print(f"   Worst Trade: ${trades_df['pnl'].min():.2f}")

    except Exception as e:
        print(f"❌ Backtest failed: {e}")


async def main():
    """Run all mean reversion examples"""
    print("\n🚀 Neural SDK - Mean Reversion Strategy Examples\n")

    # Example 1: Live trading (simulated)
    await run_mean_reversion_strategy()

    # Example 2: Backtesting
    await backtest_mean_reversion()

    print("\n✅ Examples complete!")


if __name__ == "__main__":
    asyncio.run(main())