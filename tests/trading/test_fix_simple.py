#!/usr/bin/env python3
"""
Simple FIX API test using existing Neural SDK infrastructure
"""

import asyncio
import os
from datetime import datetime
from dotenv import load_dotenv
import simplefix
from neural.trading.fix import KalshiFIXClient, FIXConnectionConfig
from neural.auth.env import get_api_key_id, get_private_key_material

# Load environment variables
load_dotenv()


async def test_fix_basic():
    """Test basic FIX connection with minimal setup"""

    print("🔧 Testing FIX API Connection")
    print("="*60)

    try:
        # Try to get credentials using the SDK's built-in methods
        api_key = get_api_key_id()
        print(f"✅ Found API Key: {api_key[:10]}...")

        # Test that we can load the private key
        private_key_pem = get_private_key_material()
        print(f"✅ Found Private Key: {len(private_key_pem)} bytes")

    except Exception as e:
        print(f"❌ Failed to get credentials: {e}")
        print("\nMake sure you have:")
        print("  1. KALSHI_API_KEY_ID environment variable set")
        print("  2. KALSHI_PRIVATE_KEY_PATH pointing to your .pem file")
        return

    print("\n📡 Attempting FIX connection...")

    # Create a simple message handler
    messages_received = []

    def handle_message(msg: simplefix.FixMessage):
        msg_dict = KalshiFIXClient.to_dict(msg)
        msg_type = msg_dict.get(35)
        timestamp = datetime.now().strftime('%H:%M:%S')

        messages_received.append(msg_type)

        if msg_type == 'A':
            print(f"[{timestamp}] ✅ LOGON SUCCESS - FIX connection established!")
        elif msg_type == '5':
            print(f"[{timestamp}] 👋 Logout acknowledged")
        elif msg_type == '0':
            print(f"[{timestamp}] 💓 Heartbeat")
        elif msg_type == '3':
            print(f"[{timestamp}] ❌ Reject: {msg_dict.get(58)}")
        else:
            print(f"[{timestamp}] 📨 Message type: {msg_type}")

    # Create FIX client with minimal config
    config = FIXConnectionConfig(
        reset_seq_num=True,  # Reset sequence numbers
        heartbeat_interval=30
    )

    client = KalshiFIXClient(
        config=config,
        on_message=handle_message
    )

    try:
        # Connect and stay connected for 5 seconds
        await client.connect(timeout=10)
        print("\n🎉 FIX CONNECTION SUCCESSFUL!")
        print("Waiting 5 seconds to receive heartbeats...")

        await asyncio.sleep(5)

        print("\n📤 Sending logout...")
        await client.logout()
        await client.close()

        print(f"\n📊 Summary:")
        print(f"  Messages received: {len(messages_received)}")
        print(f"  Message types: {set(messages_received)}")

    except asyncio.TimeoutError:
        print("\n⏱️ Connection timeout - check credentials and network")
    except Exception as e:
        print(f"\n❌ Connection failed: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Main test runner"""
    print("\n🚀 Neural SDK - FIX Infrastructure Test\n")
    await test_fix_basic()
    print("\n✅ Test complete")


if __name__ == "__main__":
    asyncio.run(main())