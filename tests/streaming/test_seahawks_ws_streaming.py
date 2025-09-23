#!/usr/bin/env python3
"""
Test WebSocket streaming for Seahawks vs Cardinals real-time market data
"""

import asyncio
import json
import threading
import time
from datetime import datetime
from typing import Dict, Any
from dotenv import load_dotenv
from neural.trading import KalshiWebSocketClient

load_dotenv()


class MarketDataStreamer:
    """Stream and analyze market data for Seahawks vs Cardinals"""

    def __init__(self):
        self.market_data = {}
        self.trade_history = []
        self.last_update = {}
        self.subscription_ids = []

    def handle_message(self, message: Dict[str, Any]) -> None:
        """Process incoming WebSocket messages"""
        msg_type = message.get("type")
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]

        if msg_type == "subscribed":
            sid = message.get("sid")
            channel = message.get("channel")
            self.subscription_ids.append(sid)
            print(f"[{timestamp}] ✅ Subscribed to {channel} (sid: {sid})")

        elif msg_type == "orderbook_snapshot":
            self._handle_orderbook_snapshot(timestamp, message)

        elif msg_type == "orderbook_delta":
            self._handle_orderbook_delta(timestamp, message)

        elif msg_type == "trade":
            self._handle_trade(timestamp, message)

        elif msg_type == "error":
            print(f"[{timestamp}] ❌ Error: {message.get('msg')}")

    def _handle_orderbook_snapshot(self, timestamp: str, msg: Dict[str, Any]) -> None:
        """Handle orderbook snapshot"""
        market_ticker = msg.get("market_ticker")

        # Extract best bid/ask
        yes_bids = msg.get("yes_bids", [])
        yes_asks = msg.get("yes_asks", [])

        if yes_bids and yes_asks:
            best_bid = yes_bids[0]
            best_ask = yes_asks[0]

            bid_price = best_bid[0] / 100  # Convert cents to dollars
            ask_price = best_ask[0] / 100
            bid_size = best_bid[1]
            ask_size = best_ask[1]

            spread = ask_price - bid_price
            mid_price = (bid_price + ask_price) / 2
            implied_prob = mid_price * 100  # Convert to percentage

            # Determine team from ticker
            team = "Seattle" if "SEA" in market_ticker else "Arizona"

            # Store market data
            self.market_data[market_ticker] = {
                'team': team,
                'bid': bid_price,
                'ask': ask_price,
                'bid_size': bid_size,
                'ask_size': ask_size,
                'spread': spread,
                'mid': mid_price,
                'implied_prob': implied_prob,
                'timestamp': datetime.now()
            }

            # Display update
            print(f"\n[{timestamp}] 📊 {team} Orderbook:")
            print(f"  Bid: ${bid_price:.3f} x {bid_size} | Ask: ${ask_price:.3f} x {ask_size}")
            print(f"  Spread: ${spread:.3f} | Mid: ${mid_price:.3f}")
            print(f"  Implied Win Probability: {implied_prob:.1f}%")

            # Check for significant changes
            if market_ticker in self.last_update:
                last = self.last_update[market_ticker]
                prob_change = implied_prob - last['implied_prob']
                if abs(prob_change) > 0.5:  # More than 0.5% change
                    direction = "📈" if prob_change > 0 else "📉"
                    print(f"  {direction} MOVEMENT: {prob_change:+.2f}% from {last['implied_prob']:.1f}%")

            self.last_update[market_ticker] = self.market_data[market_ticker].copy()

    def _handle_orderbook_delta(self, timestamp: str, msg: Dict[str, Any]) -> None:
        """Handle incremental orderbook updates"""
        market_ticker = msg.get("market_ticker")
        team = "Seattle" if "SEA" in market_ticker else "Arizona"

        # Process bid/ask updates
        yes_bid_deltas = msg.get("yes_bid_deltas", [])
        yes_ask_deltas = msg.get("yes_ask_deltas", [])

        if yes_bid_deltas or yes_ask_deltas:
            print(f"[{timestamp}] 🔄 {team} Update:")

            for delta in yes_bid_deltas:
                price = delta[0] / 100
                size = delta[1]
                action = "ADD" if size > 0 else "REMOVE"
                print(f"    Bid {action}: ${price:.3f} x {abs(size)}")

            for delta in yes_ask_deltas:
                price = delta[0] / 100
                size = delta[1]
                action = "ADD" if size > 0 else "REMOVE"
                print(f"    Ask {action}: ${price:.3f} x {abs(size)}")

    def _handle_trade(self, timestamp: str, msg: Dict[str, Any]) -> None:
        """Handle executed trades"""
        market_ticker = msg.get("market_ticker")
        team = "Seattle" if "SEA" in market_ticker else "Arizona"

        trade = msg.get("trade", {})
        price = trade.get("yes_price", 0) / 100
        count = trade.get("count", 0)
        taker_side = trade.get("taker_side")

        self.trade_history.append({
            'timestamp': datetime.now(),
            'team': team,
            'price': price,
            'count': count,
            'side': taker_side
        })

        side_emoji = "🟢" if taker_side == "yes" else "🔴"
        print(f"[{timestamp}] {side_emoji} TRADE: {team} @ ${price:.3f} x {count} contracts")

        # Alert on large trades
        if count >= 100:
            print(f"  ⚡ LARGE TRADE ALERT: {count} contracts!")

    def print_summary(self) -> None:
        """Print streaming session summary"""
        print("\n" + "="*60)
        print("📊 STREAMING SESSION SUMMARY")
        print("="*60)

        if self.market_data:
            print("\n📈 Final Market State:")
            for ticker, data in self.market_data.items():
                print(f"\n{data['team']}:")
                print(f"  Mid Price: ${data['mid']:.3f}")
                print(f"  Implied Probability: {data['implied_prob']:.1f}%")
                print(f"  Spread: ${data['spread']:.3f}")

        if self.trade_history:
            print(f"\n📊 Trade Statistics:")
            print(f"  Total trades: {len(self.trade_history)}")

            for team in ["Seattle", "Arizona"]:
                team_trades = [t for t in self.trade_history if t['team'] == team]
                if team_trades:
                    total_volume = sum(t['count'] for t in team_trades)
                    avg_price = sum(t['price'] * t['count'] for t in team_trades) / total_volume
                    print(f"\n  {team}:")
                    print(f"    Trades: {len(team_trades)}")
                    print(f"    Volume: {total_volume} contracts")
                    print(f"    Avg Price: ${avg_price:.3f}")


async def stream_seahawks_cardinals():
    """Stream market data for Seahawks vs Cardinals"""
    print("🏈 Seahawks vs Cardinals - WebSocket Streaming Test")
    print("="*60)

    # The market tickers for this game
    sea_ticker = "KXNFLGAME-25SEP25SEAARI-SEA"
    ari_ticker = "KXNFLGAME-25SEP25SEAARI-ARI"

    print(f"\n📡 Connecting to Kalshi WebSocket...")
    print(f"Markets to stream:")
    print(f"  - {sea_ticker} (Seattle to win)")
    print(f"  - {ari_ticker} (Arizona to win)")

    streamer = MarketDataStreamer()
    stop_event = threading.Event()

    def shutdown_handler(signum, frame):
        stop_event.set()
        print("\n\n⏹️ Stopping stream...")

    import signal
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        with KalshiWebSocketClient(on_message=streamer.handle_message) as client:
            print("\n✅ Connected to Kalshi WebSocket")

            # Subscribe to orderbook updates for both markets
            print("\n📊 Subscribing to orderbook data...")
            client.subscribe(
                ["orderbook_delta"],
                params={"market_tickers": [sea_ticker, ari_ticker]}
            )

            # Subscribe to trades
            print("💹 Subscribing to trade data...")
            client.subscribe(
                ["trades"],
                params={"market_tickers": [sea_ticker, ari_ticker]}
            )

            print("\n🔄 Streaming real-time market data...")
            print("Press Ctrl+C to stop\n")

            # Stream for 60 seconds or until interrupted
            duration = 60
            end_time = time.time() + duration

            while time.time() < end_time and not stop_event.is_set():
                await asyncio.sleep(0.1)

                # Periodically show current state
                if int(time.time()) % 10 == 0:
                    if streamer.market_data:
                        print(f"\n⏱️ Current State at {datetime.now().strftime('%H:%M:%S')}:")
                        for ticker, data in streamer.market_data.items():
                            print(f"  {data['team']}: ${data['mid']:.3f} ({data['implied_prob']:.1f}%)")

            # Unsubscribe before closing
            print("\n📤 Unsubscribing from channels...")
            for sid in streamer.subscription_ids:
                if sid:
                    client.unsubscribe([sid])

    except KeyboardInterrupt:
        print("\n⏹️ Stopped by user")
    except Exception as e:
        print(f"\n❌ Streaming error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        streamer.print_summary()


async def main():
    """Main function"""
    print("\n🚀 Neural SDK - WebSocket Infrastructure Test\n")

    await stream_seahawks_cardinals()

    print("\n✅ WebSocket streaming test complete!")


if __name__ == "__main__":
    asyncio.run(main())