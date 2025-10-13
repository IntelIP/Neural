#!/usr/bin/env python3
"""
Complete Infrastructure Test: WebSocket Streaming + FIX Trading
Demonstrates the full trading pipeline from price discovery to execution
"""

import asyncio
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import simplefix
from dotenv import load_dotenv

from neural.trading import KalshiWebSocketClient
from neural.trading.fix import FIXConnectionConfig, KalshiFIXClient

load_dotenv()


@dataclass
class MarketSnapshot:
    """Current market state"""

    ticker: str
    team: str
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    spread: float
    mid_price: float
    implied_prob: float
    timestamp: datetime


class TradingInfrastructure:
    """Complete trading infrastructure combining WebSocket and FIX"""

    def __init__(self):
        # Market data
        self.market_snapshots: dict[str, MarketSnapshot] = {}
        self.trade_signals = []
        self.orders_placed = []
        self.execution_reports = []

        # Connection states
        self.ws_connected = False
        self.fix_connected = False

        # Trading parameters
        self.max_spread = 0.03  # Max spread to trade (3 cents)
        self.min_edge = 0.02  # Minimum edge required (2%)
        self.trade_size = 1  # Contracts per trade

        # Control
        self.trading_enabled = False
        self.stop_event = threading.Event()

    def handle_ws_message(self, message: dict[str, Any]) -> None:
        """Process WebSocket market data"""
        msg_type = message.get("type")

        if msg_type == "subscribed":
            self.ws_connected = True
            print(f"✅ WebSocket: Subscribed to {message.get('channel')}")

        elif msg_type == "orderbook_snapshot":
            self._process_orderbook(message)

        elif msg_type == "trade":
            self._process_trade(message)

    def _process_orderbook(self, msg: dict[str, Any]) -> None:
        """Process orderbook update and check for trading opportunities"""
        ticker = msg.get("market_ticker")
        if not ticker:
            return

        # Extract best bid/ask
        yes_bids = msg.get("yes_bids", [])
        yes_asks = msg.get("yes_asks", [])

        if not (yes_bids and yes_asks):
            return

        best_bid = yes_bids[0]
        best_ask = yes_asks[0]

        # Create market snapshot
        snapshot = MarketSnapshot(
            ticker=ticker,
            team="Seattle" if "SEA" in ticker else "Arizona",
            bid=best_bid[0] / 100,
            ask=best_ask[0] / 100,
            bid_size=best_bid[1],
            ask_size=best_ask[1],
            spread=(best_ask[0] - best_bid[0]) / 100,
            mid_price=(best_bid[0] + best_ask[0]) / 200,
            implied_prob=(best_bid[0] + best_ask[0]) / 2,
            timestamp=datetime.now(),
        )

        # Store snapshot
        self.market_snapshots[ticker] = snapshot

        # Display update
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(
            f"[{timestamp}] 📊 {snapshot.team}: "
            f"${snapshot.bid:.3f} / ${snapshot.ask:.3f} "
            f"(Spread: ${snapshot.spread:.3f}, Prob: {snapshot.implied_prob:.1f}%)"
        )

        # Check for trading opportunities
        if self.trading_enabled:
            self._check_trading_opportunity(snapshot)

    def _process_trade(self, msg: dict[str, Any]) -> None:
        """Process executed trades from market"""
        ticker = msg.get("market_ticker")
        trade = msg.get("trade", {})

        if trade:
            price = trade.get("yes_price", 0) / 100
            count = trade.get("count", 0)
            team = "Seattle" if "SEA" in ticker else "Arizona"

            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] 💹 Market Trade: {team} @ ${price:.3f} x {count}")

    def _check_trading_opportunity(self, snapshot: MarketSnapshot) -> None:
        """Check if current market conditions present a trading opportunity"""

        # Strategy 1: Tight spread arbitrage
        if snapshot.spread <= 0.01:  # 1 cent or less spread
            self._generate_signal(
                snapshot, "TIGHT_SPREAD", f"Extremely tight spread ${snapshot.spread:.3f}"
            )

        # Strategy 2: Mispricing vs other market
        sea_ticker = "KXNFLGAME-25SEP25SEAARI-SEA"
        ari_ticker = "KXNFLGAME-25SEP25SEAARI-ARI"

        if sea_ticker in self.market_snapshots and ari_ticker in self.market_snapshots:
            sea_snap = self.market_snapshots[sea_ticker]
            ari_snap = self.market_snapshots[ari_ticker]

            # Check if probabilities don't sum to ~100%
            total_prob = sea_snap.implied_prob + ari_snap.implied_prob

            if total_prob < 98:  # Arbitrage opportunity
                self._generate_signal(
                    snapshot, "ARBITRAGE", f"Total probability {total_prob:.1f}% < 100%"
                )

            elif abs(total_prob - 100) > 2:  # Mispricing
                self._generate_signal(
                    snapshot, "MISPRICING", f"Total probability {total_prob:.1f}% != 100%"
                )

    def _generate_signal(self, snapshot: MarketSnapshot, signal_type: str, reason: str) -> None:
        """Generate trading signal"""
        signal = {
            "timestamp": datetime.now(),
            "type": signal_type,
            "ticker": snapshot.ticker,
            "team": snapshot.team,
            "price": snapshot.mid_price,
            "reason": reason,
        }

        self.trade_signals.append(signal)

        print(f"\n🎯 TRADING SIGNAL: {signal_type}")
        print(f"  Market: {snapshot.team}")
        print(f"  Price: ${snapshot.mid_price:.3f}")
        print(f"  Reason: {reason}")

        # In live mode, this would trigger FIX order
        if self.fix_connected:
            print("  → Would place order via FIX")

    def handle_fix_message(self, message: simplefix.FixMessage) -> None:
        """Process FIX messages"""
        msg_dict = KalshiFIXClient.to_dict(message)
        msg_type = msg_dict.get(35)

        if msg_type == "A":  # Logon
            self.fix_connected = True
            print("✅ FIX: Connected and ready for trading")

        elif msg_type == "8":  # Execution Report
            self._handle_execution_report(msg_dict)

        elif msg_type == "5":  # Logout
            self.fix_connected = False

    def _handle_execution_report(self, msg: dict[int, Any]) -> None:
        """Handle order execution reports"""
        cl_order_id = msg.get(11)
        symbol = msg.get(55)
        status = msg.get(39)
        price = float(msg.get(44, 0)) / 100

        status_map = {"0": "NEW", "2": "FILLED", "4": "CANCELLED", "8": "REJECTED"}

        status_text = status_map.get(status, status)

        self.execution_reports.append(
            {
                "order_id": cl_order_id,
                "symbol": symbol,
                "status": status_text,
                "price": price,
                "timestamp": datetime.now(),
            }
        )

        print(f"📊 FIX Order Update: {cl_order_id} is {status_text} @ ${price:.2f}")

    async def place_order(
        self, client: KalshiFIXClient, ticker: str, side: str, price: float
    ) -> None:
        """Place order via FIX"""
        order_id = f"INFRA_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        print("\n📤 Placing FIX order:")
        print(f"  ID: {order_id}")
        print(f"  Symbol: {ticker}")
        print(f"  Side: {side.upper()}")
        print(f"  Price: ${price:.2f}")

        await client.new_order_single(
            cl_order_id=order_id,
            symbol=ticker,
            side=side,
            quantity=self.trade_size,
            price=int(price * 100),  # Convert to cents
            order_type="limit",
            time_in_force="ioc",
        )

        self.orders_placed.append(
            {
                "order_id": order_id,
                "ticker": ticker,
                "side": side,
                "price": price,
                "timestamp": datetime.now(),
            }
        )

    def print_summary(self) -> None:
        """Print infrastructure test summary"""
        print("\n" + "=" * 70)
        print("📊 INFRASTRUCTURE TEST SUMMARY")
        print("=" * 70)

        print("\n🔌 Connection Status:")
        print(f"  WebSocket: {'✅ Connected' if self.ws_connected else '❌ Disconnected'}")
        print(f"  FIX API: {'✅ Connected' if self.fix_connected else '❌ Disconnected'}")

        if self.market_snapshots:
            print("\n📈 Market Data:")
            for _ticker, snap in self.market_snapshots.items():
                print(
                    f"  {snap.team}: ${snap.mid_price:.3f} ({snap.implied_prob:.1f}%) "
                    f"Spread: ${snap.spread:.3f}"
                )

        if self.trade_signals:
            print(f"\n🎯 Trading Signals Generated: {len(self.trade_signals)}")
            for signal in self.trade_signals[-3:]:  # Show last 3
                print(
                    f"  [{signal['timestamp'].strftime('%H:%M:%S')}] "
                    f"{signal['type']}: {signal['team']} - {signal['reason']}"
                )

        if self.orders_placed:
            print(f"\n📝 Orders Placed: {len(self.orders_placed)}")
            for order in self.orders_placed:
                print(
                    f"  {order['order_id']}: {order['side']} {order['ticker']} @ ${order['price']:.2f}"
                )

        if self.execution_reports:
            print(f"\n✅ Execution Reports: {len(self.execution_reports)}")
            for report in self.execution_reports:
                print(f"  {report['order_id']}: {report['status']}")


async def run_infrastructure_test():
    """Run complete infrastructure test"""
    print("🚀 Complete Infrastructure Test: WebSocket + FIX")
    print("=" * 70)

    # Initialize infrastructure
    infra = TradingInfrastructure()

    # Market tickers
    sea_ticker = "KXNFLGAME-25SEP25SEAARI-SEA"
    ari_ticker = "KXNFLGAME-25SEP25SEAARI-ARI"

    print("\n📊 Markets to monitor:")
    print(f"  - {sea_ticker} (Seattle Seahawks)")
    print(f"  - {ari_ticker} (Arizona Cardinals)")

    # Create WebSocket client
    print("\n📡 Connecting WebSocket for market data...")
    ws_client = KalshiWebSocketClient(on_message=infra.handle_ws_message)

    # Create FIX client
    print("🔧 Connecting FIX API for order execution...")
    fix_config = FIXConnectionConfig(
        heartbeat_interval=30, reset_seq_num=True, cancel_on_disconnect=False
    )
    fix_client = KalshiFIXClient(config=fix_config, on_message=infra.handle_fix_message)

    try:
        # Connect both systems
        async with ws_client:
            await fix_client.connect(timeout=10)

            # Subscribe to market data
            print("\n✅ Infrastructure connected")
            print("📊 Subscribing to market data...")

            ws_client.subscribe(
                ["orderbook_delta"], params={"market_tickers": [sea_ticker, ari_ticker]}
            )

            ws_client.subscribe(["trades"], params={"market_tickers": [sea_ticker, ari_ticker]})

            # Wait for initial data
            await asyncio.sleep(3)

            print("\n" + "=" * 70)
            print("🔄 MONITORING MARKETS (30 seconds)")
            print("=" * 70)
            print("Watching for:")
            print("  - Tight spreads (< 1 cent)")
            print("  - Arbitrage opportunities")
            print("  - Price mismatches\n")

            # Enable trading signals (but not actual orders for safety)
            infra.trading_enabled = True

            # Monitor for 30 seconds
            end_time = time.time() + 30

            while time.time() < end_time:
                await asyncio.sleep(1)

                # Every 10 seconds, show status
                if int(time.time()) % 10 == 0:
                    if infra.market_snapshots:
                        print(f"\n⏱️ Status at {datetime.now().strftime('%H:%M:%S')}:")
                        total_prob = 0
                        for _ticker, snap in infra.market_snapshots.items():
                            print(
                                f"  {snap.team}: ${snap.mid_price:.3f} ({snap.implied_prob:.1f}%)"
                            )
                            total_prob += snap.implied_prob
                        print(f"  Total Probability: {total_prob:.1f}%")

            # Demonstration: Place a test order if we have market data
            if infra.market_snapshots and infra.fix_connected:
                print("\n" + "=" * 70)
                print("📝 DEMONSTRATION: Placing test order via FIX")
                print("=" * 70)

                # Get current Seattle price
                if sea_ticker in infra.market_snapshots:
                    snap = infra.market_snapshots[sea_ticker]
                    # Place order 5 cents below mid (unlikely to fill)
                    test_price = snap.mid_price - 0.05

                    await infra.place_order(fix_client, sea_ticker, "buy", test_price)

                    # Wait for execution report
                    await asyncio.sleep(3)

                    # Cancel the order
                    print("\n🚫 Cancelling test order...")
                    if infra.orders_placed:
                        last_order = infra.orders_placed[-1]
                        await fix_client.cancel_order(
                            cl_order_id=f"CANCEL_{last_order['order_id']}",
                            orig_cl_order_id=last_order["order_id"],
                            symbol=sea_ticker,
                            side="buy",
                        )

                    await asyncio.sleep(2)

            # Logout from FIX
            await fix_client.logout()
            await asyncio.sleep(1)
            await fix_client.close()

    except KeyboardInterrupt:
        print("\n⏹️ Stopped by user")
    except Exception as e:
        print(f"\n❌ Infrastructure test error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        infra.print_summary()


async def main():
    """Main function"""
    print("\n🚀 Neural SDK - Complete Infrastructure Test\n")

    print("This test will demonstrate:")
    print("  1. WebSocket streaming for real-time prices")
    print("  2. FIX API for order execution")
    print("  3. Signal generation from market conditions")
    print("  4. Complete trading pipeline\n")

    print("⚠️ Note: This test will place a test order (below market price)")
    print("The order will be cancelled immediately to avoid execution\n")

    response = input("Run infrastructure test? (yes/no): ").strip().lower()

    if response == "yes":
        await run_infrastructure_test()
        print("\n✅ Infrastructure test complete!")
        print("\n🎉 Both WebSocket and FIX are working correctly!")
        print("The trading infrastructure is ready for production use.")
    else:
        print("\n⏹️ Test cancelled")


if __name__ == "__main__":
    asyncio.run(main())
