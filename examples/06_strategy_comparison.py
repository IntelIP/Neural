#!/usr/bin/env python3
"""
Example: Compare Multiple Trading Strategies

Demonstrates backtesting and comparing different strategies.
"""

import asyncio
import pandas as pd
from datetime import datetime, timedelta
from neural.analysis.backtesting import Backtester
from neural.analysis.strategies import (
    MeanReversionStrategy,
    MomentumStrategy,
    ArbitrageStrategy,
    create_strategy
)


async def compare_strategies():
    """Compare multiple trading strategies"""
    print("🔬 Strategy Comparison Example")
    print("=" * 50)

    # Initialize backtester
    backtester = Backtester(initial_capital=10000)

    # Define strategies to test
    strategies = [
        # Conservative mean reversion
        MeanReversionStrategy(
            name="Conservative MR",
            divergence_threshold=0.08,
            max_position_size=0.05,
            stop_loss=0.2
        ),

        # Aggressive mean reversion
        MeanReversionStrategy(
            name="Aggressive MR",
            divergence_threshold=0.03,
            max_position_size=0.15,
            use_kelly=True
        ),

        # Momentum strategy
        MomentumStrategy(
            name="Momentum",
            lookback_periods=10,
            momentum_threshold=0.1,
            use_rsi=True
        ),

        # Arbitrage strategy
        ArbitrageStrategy(
            name="Arbitrage",
            min_arbitrage_profit=0.01,
            speed_priority=True
        ),

        # Using preset
        create_strategy("conservative", name="Preset Conservative"),
        create_strategy("aggressive", name="Preset Aggressive")
    ]

    # Test period
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    print(f"\n📅 Test Period: {start_date.date()} to {end_date.date()}")
    print(f"💰 Initial Capital: $10,000")
    print(f"🎯 Testing {len(strategies)} strategies\n")

    # Run comparison
    try:
        comparison_df = await backtester.compare_strategies(
            strategies=strategies,
            start_date=start_date,
            end_date=end_date,
            markets=["KXNFLGAME", "KXNBA"]  # NFL and NBA
        )

        # Display results table
        print("\n📊 Performance Comparison:")
        print("-" * 80)

        # Format and display results
        display_columns = [
            'total_return',
            'sharpe_ratio',
            'max_drawdown',
            'win_rate',
            'total_trades',
            'profit_factor'
        ]

        for col in display_columns:
            if col in comparison_df.columns:
                print(f"\n{col.replace('_', ' ').title()}:")
                for strategy_name, value in comparison_df[col].items():
                    if isinstance(value, float):
                        if 'rate' in col or 'ratio' in col:
                            print(f"  {strategy_name:20s}: {value:>7.2f}")
                        elif 'return' in col or 'drawdown' in col:
                            print(f"  {strategy_name:20s}: {value:>7.1f}%")
                        else:
                            print(f"  {strategy_name:20s}: {value:>7.2f}")
                    else:
                        print(f"  {strategy_name:20s}: {value:>7}")

        # Find best strategy
        print("\n🏆 Best Performers:")
        print(f"  Highest Return: {comparison_df['total_return'].idxmax()} "
              f"({comparison_df['total_return'].max():.1f}%)")
        print(f"  Best Sharpe: {comparison_df['sharpe_ratio'].idxmax()} "
              f"({comparison_df['sharpe_ratio'].max():.2f})")
        print(f"  Lowest Drawdown: {comparison_df['max_drawdown'].idxmin()} "
              f"({comparison_df['max_drawdown'].min():.1f}%)")

    except Exception as e:
        print(f"❌ Comparison failed: {e}")


async def optimize_strategy_parameters():
    """Optimize strategy parameters"""
    print("\n🔧 Strategy Parameter Optimization")
    print("=" * 50)

    backtester = Backtester(initial_capital=10000)

    # Parameter grid for optimization
    divergence_thresholds = [0.03, 0.05, 0.08, 0.10]
    position_sizes = [0.05, 0.10, 0.15, 0.20]

    print("Testing parameter combinations:")
    print(f"  Divergence Thresholds: {divergence_thresholds}")
    print(f"  Position Sizes: {position_sizes}")

    best_return = -float('inf')
    best_params = {}
    results = []

    end_date = datetime.now()
    start_date = end_date - timedelta(days=14)

    for divergence in divergence_thresholds:
        for position_size in position_sizes:
            # Create strategy with parameters
            strategy = MeanReversionStrategy(
                name=f"MR_{divergence}_{position_size}",
                divergence_threshold=divergence,
                max_position_size=position_size,
                initial_capital=10000
            )

            try:
                # Backtest
                result = await backtester.backtest(
                    strategy=strategy,
                    start_date=start_date,
                    end_date=end_date,
                    markets=["KXNFLGAME"]
                )

                # Store results
                param_result = {
                    'divergence': divergence,
                    'position_size': position_size,
                    'total_return': result.total_return,
                    'sharpe_ratio': result.sharpe_ratio,
                    'max_drawdown': result.max_drawdown,
                    'total_trades': result.total_trades
                }
                results.append(param_result)

                # Track best
                if result.total_return > best_return:
                    best_return = result.total_return
                    best_params = param_result

                print(f"  D={divergence:.2f}, P={position_size:.2f}: "
                      f"Return={result.total_return:.1f}%, "
                      f"Sharpe={result.sharpe_ratio:.2f}")

            except Exception as e:
                print(f"  D={divergence:.2f}, P={position_size:.2f}: Failed - {e}")

    # Display optimization results
    if results:
        results_df = pd.DataFrame(results)

        print("\n📈 Optimization Results:")
        print(f"\n🏆 Best Parameters:")
        print(f"  Divergence Threshold: {best_params['divergence']:.2f}")
        print(f"  Position Size: {best_params['position_size']:.2f}")
        print(f"  Total Return: {best_params['total_return']:.1f}%")
        print(f"  Sharpe Ratio: {best_params['sharpe_ratio']:.2f}")

        # Show heatmap (text version)
        print("\n📊 Return Heatmap:")
        pivot = results_df.pivot(
            index='divergence',
            columns='position_size',
            values='total_return'
        )
        print(pivot.to_string(float_format=lambda x: f'{x:>6.1f}%'))


async def risk_analysis():
    """Analyze risk metrics for strategies"""
    print("\n⚠️ Risk Analysis Example")
    print("=" * 50)

    # Create strategies with different risk profiles
    strategies = {
        "Low Risk": MeanReversionStrategy(
            max_position_size=0.02,
            stop_loss=0.1,
            min_edge=0.05,
            use_kelly=False
        ),
        "Medium Risk": MeanReversionStrategy(
            max_position_size=0.10,
            stop_loss=0.2,
            min_edge=0.03,
            use_kelly=True,
            kelly_fraction=0.25
        ),
        "High Risk": MomentumStrategy(
            max_position_size=0.20,
            stop_loss=0.3,
            min_edge=0.02,
            use_kelly=True,
            kelly_fraction=0.5
        )
    }

    backtester = Backtester(initial_capital=10000)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    print("\n📊 Risk Metrics by Strategy:")
    print("-" * 60)

    for name, strategy in strategies.items():
        try:
            result = await backtester.backtest(
                strategy=strategy,
                start_date=start_date,
                end_date=end_date,
                markets=["KXNFLGAME"]
            )

            print(f"\n{name}:")
            print(f"  Max Drawdown: {result.max_drawdown:.1f}%")
            print(f"  Volatility: {result.metrics.get('volatility', 0):.2f}")
            print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
            print(f"  Sortino Ratio: {result.metrics.get('sortino_ratio', 0):.2f}")
            print(f"  Win Rate: {result.win_rate:.1%}")
            print(f"  Risk/Reward: 1:{result.profit_factor:.1f}")
            print(f"  Avg Loss: ${result.metrics.get('avg_loss', 0):.2f}")
            print(f"  Max Loss: ${result.metrics.get('max_loss', 0):.2f}")

        except Exception as e:
            print(f"\n{name}: Analysis failed - {e}")


async def main():
    """Run all comparison examples"""
    print("\n🚀 Neural SDK - Strategy Analysis Examples\n")

    # Example 1: Compare multiple strategies
    await compare_strategies()

    # Example 2: Optimize parameters
    await optimize_strategy_parameters()

    # Example 3: Risk analysis
    await risk_analysis()

    print("\n✅ All analyses complete!")


if __name__ == "__main__":
    asyncio.run(main())