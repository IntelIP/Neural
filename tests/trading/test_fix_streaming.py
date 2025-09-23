#!/usr/bin/env python3
"""
Test FIX API Streaming Connection for Kalshi

This script tests the FIX protocol connection to stream real-time market data.
"""

import asyncio
import os
from datetime import datetime
from typing import Dict, Any
import simplefix
from neural.trading.fix import KalshiFIXClient, FIXConnectionConfig


class MarketDataHandler:
    """Handle incoming FIX messages for market data"""

    def __init__(self):
        self.message_count = 0
        self.last_heartbeat = None
        self.market_updates = []

    def on_message(self, message: simplefix.FixMessage) -> None:
        """Process incoming FIX message"""
        self.message_count += 1

        # Convert to dictionary for easier access
        msg_dict = KalshiFIXClient.to_dict(message)
        msg_type = msg_dict.get(35)  # Tag 35 is MsgType

        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]

        # Handle different message types
        if msg_type == 'A':  # Logon
            print(f"[{timestamp}] ✅ LOGON successful")

        elif msg_type == '5':  # Logout
            print(f"[{timestamp}] 👋 LOGOUT received")

        elif msg_type == '0':  # Heartbeat
            self.last_heartbeat = datetime.now()
            print(f"[{timestamp}] 💓 Heartbeat received")

        elif msg_type == '1':  # Test Request
            print(f"[{timestamp}] 🧪 Test request received")

        elif msg_type == '3':  # Reject
            reason = msg_dict.get(58, "Unknown")
            print(f"[{timestamp}] ❌ REJECT: {reason}")

        elif msg_type == '8':  # Execution Report
            self._handle_execution_report(timestamp, msg_dict)

        elif msg_type == 'W':  # Market Data Snapshot/Full Refresh
            self._handle_market_data(timestamp, msg_dict)

        elif msg_type == 'X':  # Market Data Incremental Refresh
            self._handle_market_update(timestamp, msg_dict)

        else:
            print(f"[{timestamp}] 📨 Message type {msg_type}: {msg_dict}")

    def _handle_execution_report(self, timestamp: str, msg: Dict[int, Any]) -> None:
        """Handle execution report (order updates)"""
        order_id = msg.get(11)  # ClOrdID
        status = msg.get(39)  # OrdStatus
        symbol = msg.get(55)  # Symbol

        status_map = {
            '0': 'NEW',
            '1': 'PARTIALLY_FILLED',
            '2': 'FILLED',
            '4': 'CANCELLED',
            '8': 'REJECTED'
        }

        status_text = status_map.get(status, status)
        print(f"[{timestamp}] 📊 ORDER UPDATE: {symbol} - Order {order_id} is {status_text}")

    def _handle_market_data(self, timestamp: str, msg: Dict[int, Any]) -> None:
        """Handle market data snapshot"""
        symbol = msg.get(55)  # Symbol
        bid_price = msg.get(132)  # BidPx
        ask_price = msg.get(133)  # OfferPx
        bid_size = msg.get(134)  # BidSize
        ask_size = msg.get(135)  # OfferSize

        if symbol:
            print(f"[{timestamp}] 💹 MARKET DATA for {symbol}:")
            if bid_price and ask_price:
                print(f"    Bid: ${float(bid_price)/100:.2f} x {bid_size}")
                print(f"    Ask: ${float(ask_price)/100:.2f} x {ask_size}")
                spread = (float(ask_price) - float(bid_price)) / 100
                print(f"    Spread: ${spread:.2f}")

            self.market_updates.append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'bid': bid_price,
                'ask': ask_price,
                'bid_size': bid_size,
                'ask_size': ask_size
            })

    def _handle_market_update(self, timestamp: str, msg: Dict[int, Any]) -> None:
        """Handle incremental market data update"""
        print(f"[{timestamp}] 🔄 MARKET UPDATE: {msg}")

    def print_summary(self) -> None:
        """Print session summary"""
        print("\n" + "="*60)
        print("📊 SESSION SUMMARY")
        print("="*60)
        print(f"Total messages received: {self.message_count}")
        print(f"Market updates received: {len(self.market_updates)}")
        if self.last_heartbeat:
            print(f"Last heartbeat: {self.last_heartbeat.strftime('%H:%M:%S')}")

        if self.market_updates:
            print("\n📈 Latest Market Snapshot:")
            latest = self.market_updates[-1]
            print(f"  Symbol: {latest['symbol']}")
            print(f"  Bid: ${float(latest['bid'])/100:.2f} x {latest['bid_size']}")
            print(f"  Ask: ${float(latest['ask'])/100:.2f} x {latest['ask_size']}")


async def test_fix_connection():
    """Test basic FIX connection and market data subscription"""

    print("🚀 Kalshi FIX API Streaming Test")
    print("="*60)

    # Get credentials from environment
    api_key = os.getenv('KALSHI_API_KEY_ID')

    if not api_key:
        print("❌ KALSHI_API_KEY_ID not set in environment")
        return

    print(f"📔 Using API Key: {api_key}")

    # Create handler for incoming messages
    handler = MarketDataHandler()

    # Configure FIX connection
    config = FIXConnectionConfig(
        host="fix.elections.kalshi.com",
        port=8228,
        target_comp_id="KalshiNR",
        sender_comp_id=api_key,
        heartbeat_interval=30,
        reset_seq_num=True,
        listener_session=False,  # Set to True to receive market data
        use_tls=True
    )

    # Create FIX client
    client = KalshiFIXClient(
        config=config,
        on_message=handler.on_message
    )

    print(f"\n📡 Connecting to {config.host}:{config.port}...")

    try:
        # Connect to FIX gateway
        async with client:
            print("✅ Connected to Kalshi FIX gateway")
            print("🔄 Streaming market data...")
            print("\nPress Ctrl+C to stop\n")

            # Send test request to verify connection
            await client.test_request("TEST123")

            # For market data subscription, we would send a Market Data Request (V) message
            # This would require implementing the subscription message format

            # Keep connection alive
            await asyncio.sleep(60)  # Run for 1 minute

    except KeyboardInterrupt:
        print("\n\n⏹️ Stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        handler.print_summary()


async def test_order_flow():
    """Test order placement via FIX"""

    print("\n🎯 Testing Order Flow")
    print("="*60)

    api_key = os.getenv('KALSHI_API_KEY_ID')
    if not api_key:
        print("❌ KALSHI_API_KEY_ID not set")
        return

    handler = MarketDataHandler()
    config = FIXConnectionConfig(
        sender_comp_id=api_key,
        cancel_on_disconnect=True  # Cancel orders on disconnect
    )

    client = KalshiFIXClient(config=config, on_message=handler.on_message)

    try:
        async with client:
            print("✅ Connected for order testing")

            # Example: Place a limit order for Seahawks
            symbol = "KXNFLGAME-25SEP25SEAARI-SEA"

            print(f"\n📝 Placing test order for {symbol}")

            await client.new_order_single(
                cl_order_id=f"TEST_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                symbol=symbol,
                side="buy",
                quantity=1,
                price=45,  # $0.45 in cents
                order_type="limit",
                time_in_force="ioc"  # Immediate or cancel
            )

            # Wait for execution report
            await asyncio.sleep(5)

    except Exception as e:
        print(f"❌ Order test failed: {e}")


async def main():
    """Main test function"""

    print("\n🔧 Neural SDK - FIX API Infrastructure Test\n")

    # Test 1: Basic connection
    await test_fix_connection()

    # Test 2: Order flow (optional, commented out for safety)
    # Uncomment to test order placement
    # await test_order_flow()

    print("\n✅ FIX infrastructure test complete")


if __name__ == "__main__":
    asyncio.run(main())