"""
Comprehensive Backtesting Example

Demonstrates how to use the Neural SDK backtesting module
to test trading strategies with historical data.
"""

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Import the SDK
from neural_sdk.backtesting import BacktestConfig, BacktestEngine, DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def momentum_strategy(market_data):
    """
    Simple momentum trading strategy.

    Logic:
    - Buy when price is trending up and below 0.6
    - Sell when price is trending down and above 0.4
    - Use simple moving average for trend detection

    Args:
        market_data: Dictionary with current market information

    Returns:
        Trading signal dictionary or None
    """
    # Get current data
    current_prices = market_data["prices"]
    portfolio = market_data["portfolio"]

    # Simple strategy: look for NFL winner markets
    for symbol in current_prices:
        if "NFL" in symbol and "WINNER" in symbol:
            price = current_prices[symbol]

            # Get current position
            position_size = portfolio.get_position_size(symbol)

            # Buy signal: price low and not already long
            if price < 0.4 and position_size == 0:
                return {
                    "action": "BUY",
                    "market": symbol,
                    "size": 100,  # Buy 100 contracts
                    "reason": f"Price {price:.2f} below buy threshold",
                }

            # Sell signal: price high and currently long
            elif price > 0.7 and position_size > 0:
                return {
                    "action": "SELL",
                    "market": symbol,
                    "size": position_size,  # Sell all
                    "reason": f"Price {price:.2f} above sell threshold",
                }

    return None


def mean_reversion_strategy(market_data):
    """
    Mean reversion strategy example.

    Assumes prices will revert to fair value (0.5) over time.

    Args:
        market_data: Dictionary with current market information

    Returns:
        Trading signal dictionary or None
    """
    current_prices = market_data["prices"]
    portfolio = market_data["portfolio"]

    for symbol in current_prices:
        price = current_prices[symbol]
        position_size = portfolio.get_position_size(symbol)

        # Calculate distance from fair value (0.5)
        distance_from_fair = abs(price - 0.5)

        # Only trade if price is significantly away from fair value
        if distance_from_fair < 0.15:
            continue

        # Buy undervalued markets
        if price < 0.35 and position_size == 0:
            return {
                "action": "BUY",
                "market": symbol,
                "size": 50,
                "reason": f"Undervalued at {price:.2f}",
            }

        # Sell overvalued markets
        elif price > 0.65 and position_size == 0:
            # Short selling (if supported) or just avoid
            continue

        # Take profits when price moves toward fair value
        elif position_size > 0 and 0.45 <= price <= 0.55:
            return {
                "action": "SELL",
                "market": symbol,
                "size": position_size,
                "reason": f"Taking profit at {price:.2f}",
            }

    return None


def run_basic_backtest():
    """Run a basic backtest with sample data."""

    print("=== Basic Backtest Example ===\n")

    # 1. Configure the backtest
    config = BacktestConfig(
        start_date="2024-01-01",
        end_date="2024-03-31",
        initial_capital=10000.0,
        commission=0.02,  # Kalshi's 2% fee
        slippage=0.01,  # 1% slippage
        position_size_limit=0.10,  # Max 10% per position
        daily_loss_limit=0.15,  # Stop at 15% daily loss
    )

    # 2. Initialize the backtesting engine
    engine = BacktestEngine(config)

    # 3. Add a trading strategy
    engine.add_strategy(momentum_strategy)

    # 4. Create sample data (in real use, load from your data source)
    print("Creating sample data...")
    sample_data = create_sample_data()
    engine.data = sample_data

    # 5. Run the backtest
    print("Running backtest...")
    results = engine.run()

    # 6. Analyze results
    print("\n=== BACKTEST RESULTS ===")
    print(f"Initial Capital: ${config.initial_capital:,.2f}")
    print(f"Final Value: ${results.portfolio_value.iloc[-1]:,.2f}")
    print(f"Total Return: {results.total_return:.2f}%")
    print(f"Number of Trades: {results.num_trades}")

    print(f"\n=== KEY METRICS ===")
    for metric, value in results.metrics.items():
        if isinstance(value, float):
            print(f"{metric.replace('_', ' ').title()}: {value:.4f}")

    # 7. Export detailed report
    results.export_report("backtest_report.html")
    print(f"\nDetailed report saved to: backtest_report.html")

    return results


def run_strategy_comparison():
    """Compare multiple strategies on the same data."""

    print("\n=== Strategy Comparison Example ===\n")

    # Common configuration
    config = BacktestConfig(
        start_date="2024-01-01",
        end_date="2024-06-30",
        initial_capital=10000.0,
        commission=0.02,
        slippage=0.01,
    )

    strategies = {
        "Momentum": momentum_strategy,
        "Mean Reversion": mean_reversion_strategy,
    }

    results = {}
    sample_data = create_sample_data()

    for name, strategy_func in strategies.items():
        print(f"Testing {name} strategy...")

        engine = BacktestEngine(config)
        engine.add_strategy(strategy_func)
        engine.data = sample_data

        result = engine.run()
        results[name] = result

        print(f"  - Total Return: {result.total_return:.2f}%")
        print(f"  - Sharpe Ratio: {result.metrics.get('sharpe_ratio', 0):.3f}")
        print(f"  - Max Drawdown: {result.metrics.get('max_drawdown', 0):.2f}%")
        print(f"  - Win Rate: {result.metrics.get('win_rate', 0):.1f}%")

    # Find best strategy
    best_strategy = max(results.keys(), key=lambda x: results[x].total_return)

    print(f"\nBest performing strategy: {best_strategy}")
    print(f"Return: {results[best_strategy].total_return:.2f}%")

    return results


def run_data_source_example():
    """Example of loading data from different sources."""

    print("\n=== Data Source Example ===\n")

    # Initialize data loader
    loader = DataLoader(cache_dir="~/.kalshi_sdk/cache")

    # Example 1: Load from CSV file
    try:
        print("Loading from CSV...")
        # Create sample CSV first
        create_sample_csv("sample_trades.csv")

        data = loader.load("csv", path="sample_trades.csv")
        print(f"Loaded {len(data)} records from CSV")

    except Exception as e:
        print(f"CSV loading failed: {e}")

    # Example 2: Load from Parquet (more efficient for large datasets)
    try:
        print("Loading from Parquet...")

        # Convert CSV to Parquet for demonstration
        from kalshi_trading_sdk.backtesting.providers import ParquetProvider

        provider = ParquetProvider()
        provider.convert_csv_to_parquet("sample_trades.csv", "sample_trades.parquet")

        data = loader.load("parquet", path="sample_trades.parquet")
        print(f"Loaded {len(data)} records from Parquet")

    except Exception as e:
        print(f"Parquet loading failed: {e}")

    # Example 3: Cache information
    cache_info = loader.get_cache_info()
    print(
        f"\nCache info: {cache_info.get('files', 0)} files, "
        f"{cache_info.get('total_size_mb', 0):.1f} MB"
    )

    return data


def create_sample_data():
    """Create sample market data for backtesting."""

    # Generate timestamps (1 hour intervals for 3 months)
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 3, 31)
    timestamps = pd.date_range(start_date, end_date, freq="1H")

    # Sample NFL markets
    markets = [
        "NFL-KC-BUF-WINNER",
        "NFL-KC-BUF-TOTAL",
        "NFL-SF-DAL-WINNER",
        "NFL-SF-DAL-TOTAL",
    ]

    data_rows = []

    for market in markets:
        # Create realistic price movement (random walk with mean reversion)
        initial_price = np.random.uniform(0.35, 0.65)
        prices = [initial_price]

        for i in range(1, len(timestamps)):
            # Random walk with mean reversion to 0.5
            current_price = prices[-1]
            drift = (0.5 - current_price) * 0.01  # Mean reversion
            noise = np.random.normal(0, 0.005)  # Random noise

            new_price = current_price + drift + noise
            new_price = max(0.01, min(0.99, new_price))  # Keep in bounds
            prices.append(new_price)

        # Create data rows
        for timestamp, price in zip(timestamps, prices):
            data_rows.append(
                {
                    "timestamp": timestamp,
                    "symbol": market,
                    "price": price,
                    "volume": np.random.randint(50, 500),
                }
            )

    return pd.DataFrame(data_rows)


def create_sample_csv(filename):
    """Create a sample CSV file for demonstration."""

    sample_data = create_sample_data()
    sample_data.to_csv(filename, index=False)
    print(f"Created sample CSV: {filename}")


def analyze_strategy_performance(results):
    """Detailed analysis of strategy performance."""

    print("\n=== DETAILED PERFORMANCE ANALYSIS ===\n")

    # Portfolio value over time
    portfolio_values = results.portfolio_value

    # Calculate key statistics
    total_days = len(portfolio_values)
    trading_days = (portfolio_values.index[-1] - portfolio_values.index[0]).days

    print(f"Trading Period: {trading_days} days ({total_days} data points)")
    print(f"Start Value: ${portfolio_values.iloc[0]:,.2f}")
    print(f"End Value: ${portfolio_values.iloc[-1]:,.2f}")
    print(f"Peak Value: ${portfolio_values.max():,.2f}")

    # Daily statistics
    daily_returns = results.daily_returns.dropna()
    if len(daily_returns) > 0:
        print(f"\nDaily Return Statistics:")
        print(f"  Mean: {daily_returns.mean():.3f}%")
        print(f"  Std Dev: {daily_returns.std():.3f}%")
        print(f"  Best Day: +{daily_returns.max():.2f}%")
        print(f"  Worst Day: {daily_returns.min():.2f}%")

        positive_days = (daily_returns > 0).sum()
        print(
            f"  Positive Days: {positive_days}/{len(daily_returns)} ({positive_days/len(daily_returns)*100:.1f}%)"
        )

    # Trade analysis
    if len(results.trades) > 0:
        print(f"\nTrade Analysis:")
        trades_df = pd.DataFrame(
            [
                {
                    "timestamp": t.timestamp,
                    "symbol": t.symbol,
                    "action": t.action,
                    "price": t.price,
                    "pnl": t.pnl,
                }
                for t in results.trades
            ]
        )

        buy_trades = trades_df[trades_df["action"] == "BUY"]
        sell_trades = trades_df[trades_df["action"] == "SELL"]

        print(f"  Total Trades: {len(results.trades)}")
        print(f"  Buy Orders: {len(buy_trades)}")
        print(f"  Sell Orders: {len(sell_trades)}")

        if len(sell_trades) > 0:
            profitable_trades = sell_trades[sell_trades["pnl"] > 0]
            print(f"  Profitable Trades: {len(profitable_trades)}/{len(sell_trades)}")

            if len(profitable_trades) > 0:
                print(f"  Average Win: ${profitable_trades['pnl'].mean():.2f}")
                print(f"  Largest Win: ${profitable_trades['pnl'].max():.2f}")

            losing_trades = sell_trades[sell_trades["pnl"] < 0]
            if len(losing_trades) > 0:
                print(f"  Average Loss: ${losing_trades['pnl'].mean():.2f}")
                print(f"  Largest Loss: ${losing_trades['pnl'].min():.2f}")


if __name__ == "__main__":
    # Run all examples

    print("Kalshi Trading SDK - Backtesting Examples")
    print("=" * 50)

    # 1. Basic backtest
    basic_results = run_basic_backtest()

    # 2. Strategy comparison
    comparison_results = run_strategy_comparison()

    # 3. Data loading examples
    data_example = run_data_source_example()

    # 4. Detailed analysis
    analyze_strategy_performance(basic_results)

    print("\n" + "=" * 50)
    print("Examples completed successfully!")
    print("\nNext steps:")
    print("1. Modify the strategies to implement your own logic")
    print("2. Load your own historical data")
    print("3. Experiment with different configuration parameters")
    print("4. Use the performance metrics to optimize your strategies")
