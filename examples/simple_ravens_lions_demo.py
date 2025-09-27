#!/usr/bin/env python3
"""
Simplified Ravens vs Lions Trading Algorithm Demo

This demonstrates the core concepts of building trading algorithms using the Neural SDK
without the complex dependencies. It shows:

1. Strategy implementation concepts
2. Data collection patterns
3. Performance analysis ideas
4. Visualization approaches

For the full working example, see ravens_lions_algorithm.py
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleRavensStrategy:
    """
    Simplified Ravens win strategy demonstrating core concepts.
    """

    def __init__(self):
        self.name = "RavensWinStrategy"
        self.capital = 10000.0
        self.trades = []

    def analyze_market(self, ravens_price: float, lions_price: float) -> Optional[Dict[str, Any]]:
        """
        Simple analysis: Buy Ravens if price is below 0.45 (undervalued)
        """
        if ravens_price < 0.45:
            confidence = min(0.8, (0.45 - ravens_price) * 2)
            position_size = int((self.capital * 0.1) / ravens_price)  # 10% of capital

            return {
                'action': 'BUY_RAVENS',
                'ticker': 'KXNFLGAME-25SEP22DETBAL-BAL',
                'price': ravens_price,
                'size': position_size,
                'confidence': confidence,
                'reason': f'Ravens undervalued at {ravens_price:.3f}'
            }
        return None

    def record_trade(self, trade: Dict[str, Any]):
        """Record a completed trade"""
        self.trades.append(trade)
        self.capital += trade.get('pnl', 0)


class SimpleLionsStrategy:
    """
    Simplified Lions win strategy demonstrating momentum concepts.
    """

    def __init__(self):
        self.name = "LionsWinStrategy"
        self.capital = 10000.0
        self.price_history = []
        self.trades = []

    def analyze_market(self, ravens_price: float, lions_price: float) -> Optional[Dict[str, Any]]:
        """
        Simple momentum: Buy Lions if price increased in last few observations
        """
        self.price_history.append(lions_price)
        if len(self.price_history) < 3:
            return None

        # Check for upward momentum
        recent_prices = self.price_history[-3:]
        momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]

        if momentum > 0.02 and lions_price < 0.6:  # 2% upward momentum
            confidence = min(0.8, momentum * 10)
            position_size = int((self.capital * 0.1) / lions_price)

            return {
                'action': 'BUY_LIONS',
                'ticker': 'KXNFLGAME-25SEP22DETBAL-DET',
                'price': lions_price,
                'size': position_size,
                'confidence': confidence,
                'reason': f'Lions momentum {momentum:.1%}'
            }
        return None

    def record_trade(self, trade: Dict[str, Any]):
        """Record a completed trade"""
        self.trades.append(trade)
        self.capital += trade.get('pnl', 0)


class RavensLionsDemo:
    """
    Demonstration of Ravens vs Lions trading algorithm concepts.
    """

    def __init__(self):
        self.event_ticker = "KXNFLGAME-25SEP22DETBAL"
        self.strategies = {
            'ravens': SimpleRavensStrategy(),
            'lions': SimpleLionsStrategy()
        }
        self.market_data = []

    async def simulate_market_data(self, hours: int = 2) -> pd.DataFrame:
        """
        Simulate realistic market data for the Ravens vs Lions game.
        In a real implementation, this would fetch from Kalshi API.
        """
        logger.info(f"Simulating {hours} hours of market data...")

        # Start with realistic opening prices
        ravens_price = 0.52  # Ravens favored slightly
        lions_price = 0.48

        data_points = []
        start_time = datetime.now()

        # Simulate price movements over time
        for i in range(hours * 60):  # 1 data point per minute
            timestamp = start_time + timedelta(minutes=i)

            # Add some random walk with mean reversion
            ravens_change = np.random.normal(0, 0.01)  # Small random changes
            lions_change = np.random.normal(0, 0.01)

            # Mean reversion toward fair value (sum = 1.0)
            total = ravens_price + lions_price
            if total > 1.0:
                # Too high, mean revert down
                ravens_price -= 0.001
                lions_price -= 0.001
            elif total < 1.0:
                # Too low, mean revert up
                ravens_price += 0.001
                lions_price += 0.001

            ravens_price = np.clip(ravens_price + ravens_change, 0.01, 0.99)
            lions_price = np.clip(lions_price + lions_change, 0.01, 0.99)

            # Ensure they sum to approximately 1.0 (accounting for fees)
            total = ravens_price + lions_price
            if abs(total - 1.0) > 0.05:  # If too far off
                adjustment = (1.0 - total) / 2
                ravens_price += adjustment
                lions_price += adjustment

            data_points.append({
                'timestamp': timestamp,
                'ravens_price': round(ravens_price, 3),
                'lions_price': round(lions_price, 3),
                'total_probability': round(ravens_price + lions_price, 3),
                'spread_ravens': round(abs(ravens_price - 0.5), 3),
                'spread_lions': round(abs(lions_price - 0.5), 3)
            })

            # Small delay to simulate real-time data
            await asyncio.sleep(0.01)

        df = pd.DataFrame(data_points)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')

        self.market_data = df
        logger.info(f"Generated {len(df)} market data points")
        return df

    async def run_trading_simulation(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run trading simulation using both strategies.
        """
        logger.info("Running trading simulation...")

        results = {}

        for strategy_name, strategy in self.strategies.items():
            logger.info(f"Simulating {strategy_name} strategy...")

            trades = []
            capital_history = [strategy.capital]

            for timestamp, row in market_data.iterrows():
                # Get signals from strategy
                signal = strategy.analyze_market(row['ravens_price'], row['lions_price'])

                if signal:
                    # Simulate trade execution
                    entry_price = signal['price']
                    position_size = signal['size']

                    # Simulate holding for some time (simplified)
                    # In reality, you'd track actual market prices
                    exit_price = entry_price * (1 + np.random.normal(0.02, 0.05))  # Some drift

                    # Calculate P&L (simplified, ignoring fees)
                    if signal['action'] == 'BUY_RAVENS':
                        pnl = position_size * (exit_price - entry_price)
                    else:  # BUY_LIONS
                        pnl = position_size * (entry_price - exit_price)  # Short YES means long NO

                    trade = {
                        'timestamp': timestamp,
                        'action': signal['action'],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'size': position_size,
                        'pnl': pnl,
                        'confidence': signal['confidence']
                    }

                    trades.append(trade)
                    strategy.record_trade(trade)
                    capital_history.append(strategy.capital)

            results[strategy_name] = {
                'final_capital': strategy.capital,
                'total_trades': len(trades),
                'winning_trades': len([t for t in trades if t['pnl'] > 0]),
                'total_pnl': sum(t['pnl'] for t in trades),
                'win_rate': len([t for t in trades if t['pnl'] > 0]) / max(len(trades), 1),
                'avg_trade_pnl': np.mean([t['pnl'] for t in trades]) if trades else 0,
                'capital_history': capital_history,
                'trades': trades
            }

            logger.info(f"✅ {strategy_name}: ${strategy.capital:.2f} final, "
                       f"{len(trades)} trades, "
                       f"{results[strategy_name]['win_rate']:.1%} win rate")

        return results

    def create_performance_comparison(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create performance comparison and analysis.
        """
        logger.info("Creating performance analysis...")

        ravens_result = results.get('ravens', {})
        lions_result = results.get('lions', {})

        analysis = {
            'summary': {
                'ravens_final_capital': ravens_result.get('final_capital', 10000),
                'lions_final_capital': lions_result.get('final_capital', 10000),
                'ravens_total_trades': ravens_result.get('total_trades', 0),
                'lions_total_trades': lions_result.get('total_trades', 0),
                'ravens_win_rate': ravens_result.get('win_rate', 0),
                'lions_win_rate': lions_result.get('win_rate', 0),
                'better_strategy': 'ravens' if ravens_result.get('final_capital', 0) > lions_result.get('final_capital', 0) else 'lions'
            },
            'key_insights': [
                "This demonstrates the core concepts of strategy implementation",
                "Real strategies would use actual market data from Kalshi API",
                "Backtesting would validate performance on historical data",
                "Risk management is crucial for live trading",
                "Visualization helps understand strategy behavior"
            ]
        }

        return analysis

    def print_results(self, market_data: pd.DataFrame, results: Dict[str, Any], analysis: Dict[str, Any]):
        """
        Print comprehensive results summary.
        """
        print("\n" + "="*80)

        print("\n📊 MARKET DATA SUMMARY:")
        print(f"   Total data points: {len(market_data)}")
        print(".3f")
        print(".3f")
        print(".3f")
        print("\n💰 STRATEGY PERFORMANCE:")
        summary = analysis.get('summary', {})
        print(".2f")
        print(".2f")
        print(f"   Ravens Trades: {summary.get('ravens_total_trades', 0)}")
        print(f"   Lions Trades: {summary.get('lions_total_trades', 0)}")
        print(".1%")
        print(".1%")
        print(f"\n🎯 BETTER PERFORMING STRATEGY: {summary.get('better_strategy', 'unknown').upper()}")

        print("\n🔑 KEY CONCEPTS DEMONSTRATED:")
        for insight in analysis.get('key_insights', []):
            print(f"   • {insight}")

        print("\n📈 SAMPLE TRADES:")
        for strategy_name, result in results.items():
            trades = result.get('trades', [])
            if trades:
                print(f"\n   {strategy_name.upper()} Strategy Sample Trades:")
                for i, trade in enumerate(trades[:3]):  # Show first 3 trades
                    print(f"     Trade {i+1}: {trade['action']} @ ${trade['entry_price']:.3f} "
                          f"→ ${trade['exit_price']:.3f} (P&L: ${trade['pnl']:.2f})")

        print("\n" + "="*80)
        print("✅ Demo completed! This shows the core concepts.")
        print("   For full implementation, see ravens_lions_algorithm.py")
        print("="*80)


async def main():
    """Main demonstration function."""
    print("🏈 Starting Ravens vs Lions Trading Algorithm Demo")
    print("This demonstrates core trading concepts without complex dependencies")

    # Initialize demo
    demo = RavensLionsDemo()

    try:
        # Step 1: Simulate market data
        print("\n📊 Step 1: Simulating Market Data...")
        market_data = await demo.simulate_market_data(hours=2)

        # Step 2: Run trading simulation
        print("\n🔬 Step 2: Running Trading Simulation...")
        results = await demo.run_trading_simulation(market_data)

        # Step 3: Create performance analysis
        print("\n📈 Step 3: Analyzing Performance...")
        analysis = demo.create_performance_comparison(results)

        # Step 4: Display results
        demo.print_results(market_data, results, analysis)

    except Exception as e:
        logger.error(f"Error running demo: {e}")
        print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())