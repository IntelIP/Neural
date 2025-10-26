#!/usr/bin/env python3
"""
Example: Live Trading Bot with Analysis Stack

Complete automated trading bot using the Neural SDK analysis stack.
"""

import asyncio
from datetime import datetime

import pandas as pd

from neural.analysis.execution import OrderManager
from neural.analysis.risk import PositionSizer
from neural.analysis.strategies import ArbitrageStrategy, MeanReversionStrategy
from neural.data_collection import KalshiMarketsSource, get_game_markets
from neural.trading import TradingClient


class TradingBot:
    """
    Automated trading bot with multiple strategies.
    """

    def __init__(
        self,
        trading_client: TradingClient,
        initial_capital: float = 1000,
        max_positions: int = 10,
        risk_per_trade: float = 0.02,
        dry_run: bool = True,
    ):
        """
        Initialize trading bot.

        Args:
            trading_client: Neural trading client
            initial_capital: Starting capital
            max_positions: Maximum concurrent positions
            risk_per_trade: Risk per trade (2% default)
            dry_run: Simulate trades if True
        """
        self.trading_client = trading_client
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.risk_per_trade = risk_per_trade
        self.dry_run = dry_run

        # Initialize strategies
        self.strategies = {
            "mean_reversion": MeanReversionStrategy(
                initial_capital=initial_capital,
                max_position_size=risk_per_trade * 2,
                divergence_threshold=0.05,
                stop_loss=0.2,
            ),
            "arbitrage": ArbitrageStrategy(
                initial_capital=initial_capital, min_arbitrage_profit=0.01, speed_priority=True
            ),
        }

        # Initialize order manager
        self.order_manager = OrderManager(
            trading_client=trading_client if not dry_run else None,
            dry_run=dry_run,
            require_confirmation=False,
        )

        # Position sizer
        self.position_sizer = PositionSizer(initial_capital=initial_capital, default_method="kelly")

        # Performance tracking
        self.start_time = datetime.now()
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0

    async def scan_markets(self) -> pd.DataFrame:
        """Scan for tradeable markets"""
        print("🔍 Scanning markets...")

        source = KalshiMarketsSource(series_ticker="KXNFLGAME", status=None, use_authenticated=True)

        markets_df = await source.fetch()

        if not markets_df.empty:
            print(f"  Found {len(markets_df)} NFL markets")
        else:
            print("  No active markets found")

        return markets_df

    async def analyze_market(self, event_ticker: str, strategy_name: str) -> dict | None:
        """
        Analyze a single market with specified strategy.

        Args:
            event_ticker: Event to analyze
            strategy_name: Strategy to use

        Returns:
            Signal and analysis results
        """
        try:
            # Get market data
            market_data = await get_game_markets(event_ticker)

            if market_data.empty:
                return None

            # Prepare data
            market_df = pd.DataFrame(
                {
                    "ticker": market_data["ticker"],
                    "yes_ask": market_data["yes_ask"] / 100,
                    "no_ask": market_data["no_ask"] / 100,
                    "volume": market_data["volume"],
                }
            )

            # Get strategy
            strategy = self.strategies.get(strategy_name)
            if not strategy:
                return None

            # Check for arbitrage first (special case)
            if strategy_name == "arbitrage":
                # Quick YES+NO check
                latest = market_df.iloc[-1]
                total_cost = latest["yes_ask"] + latest["no_ask"]
                if total_cost < 0.99:  # Arbitrage opportunity
                    signal = strategy.analyze(market_df)
                    return {
                        "event": event_ticker,
                        "strategy": strategy_name,
                        "signal": signal,
                        "arbitrage_profit": 1.0 - total_cost,
                    }

            # Regular strategy analysis
            signal = strategy.analyze(market_df)

            if signal.type.value != "hold":
                return {
                    "event": event_ticker,
                    "strategy": strategy_name,
                    "signal": signal,
                    "market_data": market_df.iloc[-1].to_dict(),
                }

        except Exception as e:
            print(f"  ❌ Error analyzing {event_ticker}: {e}")

        return None

    async def execute_trades(self, analysis: dict) -> bool:
        """Execute trade from analysis"""
        signal = analysis["signal"]

        # Risk checks
        if len(self.order_manager.active_positions) >= self.max_positions:
            print(f"  ⚠️ Max positions reached ({self.max_positions})")
            return False

        # Size position based on confidence and risk
        original_size = signal.size
        adjusted_size = self.position_sizer.calculate_size(
            method="kelly",
            edge=signal.metadata.get("edge", 0.03) if signal.metadata else 0.03,
            odds=1.0,
            kelly_fraction=0.25,
        )

        # Apply risk limit
        max_risk = self.position_sizer.current_capital * self.risk_per_trade
        if adjusted_size * signal.entry_price > max_risk:
            adjusted_size = int(max_risk / signal.entry_price)

        signal.size = adjusted_size

        print("\n💰 Executing Trade:")
        print(f"  Strategy: {analysis['strategy']}")
        print(f"  Market: {signal.ticker}")
        print(f"  Action: {signal.type.value}")
        print(f"  Size: {signal.size} contracts (was {original_size})")
        print(f"  Confidence: {signal.confidence:.1%}")

        # Execute order
        result = await self.order_manager.execute_signal(signal)

        if result and result.get("status") != "failed":
            self.total_trades += 1
            print("  ✅ Order executed")
            return True
        else:
            print("  ❌ Order failed")
            return False

    async def monitor_positions(self):
        """Monitor and manage existing positions"""
        if not self.order_manager.active_positions:
            return

        print("\n📊 Position Monitoring:")

        for ticker, position in self.order_manager.active_positions.items():
            # Check stop loss and take profit
            for strategy in self.strategies.values():
                if strategy.should_close_position(position):
                    print(f"  Closing {ticker}: Hit stop/target")
                    from neural.analysis.strategies.base import Signal, SignalType

                    close_signal = Signal(
                        signal_type=SignalType.CLOSE,
                        market_id=ticker,
                        recommended_size=0,
                        confidence=1.0,
                    )
                    await self.order_manager.execute_signal(close_signal)

                    # Update performance
                    if position.pnl > 0:
                        self.winning_trades += 1
                    self.total_pnl += position.pnl
                    self.position_sizer.update_performance(position.pnl)
                    break

    async def run_cycle(self):
        """Run one complete trading cycle"""
        print(f"\n{'=' * 60}")
        print(f"🔄 Trading Cycle - {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'=' * 60}")

        # Scan markets
        markets_df = await self.scan_markets()
        if markets_df.empty:
            return

        # Get unique events
        events = markets_df["event_ticker"].unique()

        # Analyze each event with each strategy
        opportunities = []

        for event in events[:5]:  # Limit to 5 events for demo
            print(f"\n📈 Analyzing {event}...")

            # Try each strategy
            for strategy_name in self.strategies.keys():
                analysis = await self.analyze_market(event, strategy_name)
                if analysis:
                    opportunities.append(analysis)
                    print(f"  ✅ {strategy_name}: Signal found!")

        # Execute best opportunities
        if opportunities:
            print(f"\n🎯 Found {len(opportunities)} opportunities")

            # Sort by confidence or arbitrage profit
            opportunities.sort(
                key=lambda x: x.get("arbitrage_profit", 0) or x["signal"].confidence, reverse=True
            )

            # Execute top opportunities
            for opp in opportunities[:3]:  # Top 3 opportunities
                if len(self.order_manager.active_positions) < self.max_positions:
                    await self.execute_trades(opp)
                    await asyncio.sleep(1)  # Rate limiting

        # Monitor existing positions
        await self.monitor_positions()

        # Display status
        self.display_status()

    def display_status(self):
        """Display bot status"""
        runtime = (datetime.now() - self.start_time).total_seconds() / 60
        win_rate = self.winning_trades / max(self.total_trades, 1)

        portfolio = self.order_manager.get_portfolio_summary()

        print(f"\n{'=' * 60}")
        print("📊 Bot Status:")
        print(f"  Runtime: {runtime:.1f} minutes")
        print(f"  Mode: {'SIMULATION' if self.dry_run else 'LIVE'}")
        print(f"  Capital: ${self.position_sizer.current_capital:.2f}")
        print(f"  Positions: {portfolio['positions']}/{self.max_positions}")
        print(f"  Total Trades: {self.total_trades}")
        print(f"  Win Rate: {win_rate:.1%}")
        print(f"  Total P&L: ${self.total_pnl:.2f}")
        print(f"  Portfolio Value: ${portfolio['total_value']:.2f}")

        if portfolio["active_positions"]:
            print("\n  Active Positions:")
            for ticker, pos in portfolio["active_positions"].items():
                print(
                    f"    {ticker}: {pos['side']} x{pos['size']} "
                    f"@ ${pos['entry_price']:.2f} "
                    f"(P&L: ${pos['pnl']:.2f})"
                )

    async def run(self, cycles: int = None, interval: int = 60):
        """
        Run the trading bot.

        Args:
            cycles: Number of cycles to run (None = infinite)
            interval: Seconds between cycles
        """
        print("\n🤖 Neural Trading Bot Starting...")
        print(f"Capital: ${self.initial_capital}")
        print(f"Risk per trade: {self.risk_per_trade:.1%}")
        print(f"Max positions: {self.max_positions}")
        print(f"Mode: {'SIMULATION' if self.dry_run else '⚠️ LIVE TRADING'}")

        if not self.dry_run:
            confirm = input("\n⚠️ LIVE TRADING MODE - Continue? (y/n): ")
            if confirm.lower() != "y":
                print("Cancelled.")
                return

        cycle_count = 0

        try:
            while cycles is None or cycle_count < cycles:
                await self.run_cycle()
                cycle_count += 1

                if cycles is None or cycle_count < cycles:
                    print(f"\n⏳ Waiting {interval} seconds...")
                    await asyncio.sleep(interval)

        except KeyboardInterrupt:
            print("\n\n🛑 Bot stopped by user")

        finally:
            # Final summary
            print("\n" + "=" * 60)
            print("📊 Final Summary:")
            self.display_status()

            # Close all positions
            if self.order_manager.active_positions:
                print("\n🔒 Closing all positions...")
                await self.order_manager.close_all_positions()

            print("\n✅ Bot shutdown complete")


async def main():
    """Run the trading bot"""
    print("\n🚀 Neural SDK - Automated Trading Bot\n")

    # Initialize trading client (would use real credentials in production)
    # For demo, we'll use None and run in dry_run mode
    trading_client = None  # TradingClient() in production

    # Create bot
    bot = TradingBot(
        trading_client=trading_client,
        initial_capital=1000,
        max_positions=5,
        risk_per_trade=0.02,
        dry_run=True,  # Set to False for live trading
    )

    # Run bot for 3 cycles with 30 second intervals (demo)
    await bot.run(cycles=3, interval=30)


if __name__ == "__main__":
    asyncio.run(main())
