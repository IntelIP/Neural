#!/usr/bin/env python3
"""
Test FIX API Order Execution for Seahawks vs Cardinals
"""

import asyncio
from datetime import datetime
from typing import Any

import pytest
import simplefix
from dotenv import load_dotenv

from neural.auth.env import get_api_key_id, get_private_key_material
from neural.trading.fix import FIXConnectionConfig, KalshiFIXClient

load_dotenv()

pytestmark = pytest.mark.skip(reason="Requires Kalshi API credentials")


class OrderExecutionTester:
    """Test FIX order execution capabilities"""

    def __init__(self):
        self.orders = {}
        self.execution_reports = []
        self.connected = False
        self.message_count = 0

    def handle_message(self, message: simplefix.FixMessage) -> None:
        """Process incoming FIX messages"""
        self.message_count += 1

        msg_dict = KalshiFIXClient.to_dict(message)
        msg_type = msg_dict.get(35)
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        if msg_type == "A":  # Logon
            self.connected = True
            print(f"[{timestamp}] ✅ FIX LOGON successful - Ready for trading")

        elif msg_type == "8":  # Execution Report
            self._handle_execution_report(timestamp, msg_dict)

        elif msg_type == "9":  # Order Cancel Reject
            self._handle_cancel_reject(timestamp, msg_dict)

        elif msg_type == "3":  # Reject
            reason = msg_dict.get(58, "Unknown reason")
            print(f"[{timestamp}] ❌ REJECT: {reason}")

        elif msg_type == "5":  # Logout
            self.connected = False
            print(f"[{timestamp}] 👋 Logout acknowledged")

        elif msg_type == "0":  # Heartbeat
            print(f"[{timestamp}] 💓 Heartbeat")

    def _handle_execution_report(self, timestamp: str, msg: dict[int, Any]) -> None:
        """Handle execution reports (order updates)"""
        cl_order_id = msg.get(11)  # ClOrdID
        order_id = msg.get(37)  # OrderID
        symbol = msg.get(55)  # Symbol
        side = msg.get(54)  # Side (1=Buy, 2=Sell)
        status = msg.get(39)  # OrdStatus
        exec_type = msg.get(150)  # ExecType
        price = msg.get(44)  # Price
        qty = msg.get(38)  # OrderQty
        leaves_qty = msg.get(151)  # LeavesQty
        cum_qty = msg.get(14)  # CumQty
        avg_px = msg.get(6)  # AvgPx

        # Map status codes to readable strings
        status_map = {
            "0": "NEW",
            "1": "PARTIALLY_FILLED",
            "2": "FILLED",
            "4": "CANCELLED",
            "6": "PENDING_CANCEL",
            "8": "REJECTED",
            "C": "EXPIRED",
        }

        exec_type_map = {
            "0": "NEW",
            "4": "CANCELLED",
            "8": "REJECTED",
            "C": "EXPIRED",
            "F": "TRADE",
            "I": "ORDER_STATUS",
        }

        status_text = status_map.get(status, status)
        exec_type_text = exec_type_map.get(exec_type, exec_type)
        side_text = "BUY" if side == "1" else "SELL"

        # Convert price from cents to dollars
        price_dollars = float(price) / 100 if price else 0

        print(f"\n[{timestamp}] 📊 EXECUTION REPORT:")
        print(f"  Order ID: {cl_order_id}")
        print(f"  Exchange Order ID: {order_id}")
        print(f"  Symbol: {symbol}")
        print(f"  Side: {side_text}")
        print(f"  Price: ${price_dollars:.2f}")
        print(f"  Quantity: {qty}")
        print(f"  Status: {status_text}")
        print(f"  Exec Type: {exec_type_text}")

        if cum_qty and int(cum_qty) > 0:
            fill_price = float(avg_px) / 100 if avg_px else price_dollars
            print(f"  Filled: {cum_qty} @ ${fill_price:.2f}")

        if leaves_qty:
            print(f"  Remaining: {leaves_qty}")

        # Store execution report
        self.execution_reports.append(
            {
                "timestamp": datetime.now(),
                "cl_order_id": cl_order_id,
                "order_id": order_id,
                "symbol": symbol,
                "side": side_text,
                "price": price_dollars,
                "quantity": qty,
                "status": status_text,
                "exec_type": exec_type_text,
            }
        )

        # Update order tracking
        self.orders[cl_order_id] = {
            "status": status_text,
            "order_id": order_id,
            "filled": cum_qty or 0,
        }

        # Alert on fills
        if exec_type == "F":
            print("  ✅ FILL CONFIRMED!")

        # Alert on rejects
        if status == "8":
            reject_reason = msg.get(103)  # OrdRejReason
            print(f"  ❌ ORDER REJECTED: {reject_reason}")

    def _handle_cancel_reject(self, timestamp: str, msg: dict[int, Any]) -> None:
        """Handle order cancel rejection"""
        cl_order_id = msg.get(11)
        reason = msg.get(102)  # CxlRejReason

        reason_map = {
            "1": "Unknown order",
            "2": "Broker Option",
            "3": "Order already pending cancel",
            "6": "Duplicate ClOrdID",
        }

        reason_text = reason_map.get(reason, reason)

        print(f"\n[{timestamp}] ❌ CANCEL REJECTED:")
        print(f"  Order ID: {cl_order_id}")
        print(f"  Reason: {reason_text}")


async def test_order_placement():
    """Test placing orders via FIX API"""
    print("🎯 FIX Order Execution Test")
    print("=" * 60)

    # Get credentials
    api_key = get_api_key_id()
    get_private_key_material()

    print(f"📔 Using API Key: {api_key[:10]}...")

    # Create execution tester
    tester = OrderExecutionTester()

    # Configure FIX connection
    config = FIXConnectionConfig(
        heartbeat_interval=30,
        reset_seq_num=True,
        cancel_on_disconnect=False,  # Keep orders alive for testing
    )

    # Create FIX client
    client = KalshiFIXClient(config=config, on_message=tester.handle_message)

    try:
        # Connect to FIX gateway
        print("\n📡 Connecting to FIX gateway...")
        await client.connect(timeout=10)

        # Wait for logon confirmation
        await asyncio.sleep(2)

        if not tester.connected:
            print("❌ Failed to establish FIX session")
            return

        print("\n" + "=" * 60)
        print("🔧 TESTING ORDER EXECUTION")
        print("=" * 60)

        # Markets for Seahawks vs Cardinals
        sea_symbol = "KXNFLGAME-25SEP25SEAARI-SEA"
        ari_symbol = "KXNFLGAME-25SEP25SEAARI-ARI"

        # Test 1: Place a small limit order (IOC to avoid leaving orders)
        print("\n📝 TEST 1: Placing limit order for Seattle...")
        print(f"  Symbol: {sea_symbol}")
        print("  Side: BUY")
        print("  Price: $0.45 (45% probability)")
        print("  Quantity: 1 contract")
        print("  Time in Force: IOC (Immediate or Cancel)")

        test_order_id = f"TEST_SEA_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        await client.new_order_single(
            cl_order_id=test_order_id,
            symbol=sea_symbol,
            side="buy",
            quantity=1,
            price=45,  # 45 cents = $0.45
            order_type="limit",
            time_in_force="ioc",  # Immediate or cancel - won't leave order open
        )

        # Wait for execution report
        await asyncio.sleep(3)

        # Test 2: Place a sell order
        print("\n📝 TEST 2: Placing sell order for Arizona...")
        print(f"  Symbol: {ari_symbol}")
        print("  Side: SELL")
        print("  Price: $0.55 (55% probability)")
        print("  Quantity: 1 contract")
        print("  Time in Force: IOC")

        test_order_id_2 = f"TEST_ARI_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        await client.new_order_single(
            cl_order_id=test_order_id_2,
            symbol=ari_symbol,
            side="sell",
            quantity=1,
            price=55,  # 55 cents = $0.55
            order_type="limit",
            time_in_force="ioc",
        )

        # Wait for execution report
        await asyncio.sleep(3)

        # Test 3: Test order cancellation (if we have a GTC order)
        print("\n📝 TEST 3: Testing order cancellation...")
        print("  Placing GTC order to test cancellation...")

        test_order_id_3 = f"TEST_CANCEL_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        await client.new_order_single(
            cl_order_id=test_order_id_3,
            symbol=sea_symbol,
            side="buy",
            quantity=1,
            price=40,  # Low price unlikely to fill
            order_type="limit",
            time_in_force="gtc",  # Good till cancelled
        )

        # Wait for order confirmation
        await asyncio.sleep(2)

        # Cancel the order
        print(f"  Cancelling order {test_order_id_3}...")
        await client.cancel_order(
            cl_order_id=f"CANCEL_{test_order_id_3}",
            orig_cl_order_id=test_order_id_3,
            symbol=sea_symbol,
            side="buy",
        )

        # Wait for cancel confirmation
        await asyncio.sleep(2)

        # Print summary
        print("\n" + "=" * 60)
        print("📊 ORDER EXECUTION TEST SUMMARY")
        print("=" * 60)
        print(f"Total messages received: {tester.message_count}")
        print(f"Orders placed: {len(tester.orders)}")
        print(f"Execution reports: {len(tester.execution_reports)}")

        if tester.orders:
            print("\n📋 Order Status:")
            for order_id, info in tester.orders.items():
                print(f"  {order_id}: {info['status']}")

        # Logout
        print("\n📤 Sending logout...")
        await client.logout()
        await asyncio.sleep(1)

    except asyncio.TimeoutError:
        print("\n⏱️ Connection timeout - check credentials and network")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        await client.close()


async def test_order_status():
    """Test querying order status"""
    print("\n📋 Testing Order Status Query")
    print("=" * 60)

    config = FIXConnectionConfig(reset_seq_num=True)
    tester = OrderExecutionTester()
    client = KalshiFIXClient(config=config, on_message=tester.handle_message)

    try:
        await client.connect(timeout=10)
        await asyncio.sleep(2)

        if tester.connected:
            # Request order status for all orders
            print("📊 Requesting order status...")
            await client.order_status_request(
                cl_order_id=f"STATUS_REQ_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                symbol="*",  # All symbols
                side="buy",
            )

            await asyncio.sleep(3)

            await client.logout()

    except Exception as e:
        print(f"❌ Status query failed: {e}")
    finally:
        await client.close()


async def main():
    """Main test function"""
    print("\n🚀 Neural SDK - FIX Order Execution Test\n")

    print("⚠️ Note: This will place real orders (using IOC to minimize risk)")
    print("Make sure you have:")
    print("  1. Valid API credentials")
    print("  2. FIX API access enabled")
    print("  3. Some balance in your account\n")

    # For automated testing, skip interactive prompt
    # In manual testing, uncomment the following lines:
    # response = input("Continue with order execution test? (yes/no): ").strip().lower()
    # if response == "yes":

    # Test order placement
    await test_order_placement()

    # Test order status
    await test_order_status()

    print("\n✅ FIX order execution test complete!")


if __name__ == "__main__":
    asyncio.run(main())
