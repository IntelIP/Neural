#!/usr/bin/env python3
"""
Simple WebSocket Test with Correct Endpoint
"""

import asyncio
import logging
from neural.trading import KalshiWebSocketClient

# Enable debug logging to see what's happening
logging.basicConfig(level=logging.DEBUG)

def handle_message(msg):
    """Handle incoming WebSocket messages"""
    print(f"📨 Message: {msg}")

    msg_type = msg.get("type")
    if msg_type == "subscribed":
        print(f"✅ Successfully subscribed! SID: {msg.get('sid')}")
    elif msg_type == "error":
        print(f"❌ Error: {msg.get('msg')}")
    elif msg_type == "orderbook_snapshot":
        ticker = msg.get("market_ticker")
        print(f"📊 Orderbook for {ticker}")

async def test_websocket():
    """Test WebSocket connection with correct endpoint"""
    print("🚀 Testing Kalshi WebSocket Connection")
    print("="*50)

    print("\n📡 Connection Details:")
    print("  Endpoint: wss://api.elections.kalshi.com/trade-api/ws/v2")
    print("  Path: /trade-api/ws/v2")
    print("  Method: GET")

    try:
        # Create WebSocket client
        ws = KalshiWebSocketClient(on_message=handle_message)

        print("\n🔄 Connecting...")
        ws.connect(block=True)

        print("✅ WebSocket connected successfully!")

        # Subscribe to market data
        sea_ticker = "KXNFLGAME-25SEP25SEAARI-SEA"
        print(f"\n📊 Subscribing to: {sea_ticker}")

        req_id = ws.subscribe(
            ["orderbook_delta"],
            params={"market_tickers": [sea_ticker]}
        )
        print(f"  Request ID: {req_id}")

        # Wait for messages
        print("\n⏳ Waiting for messages (10 seconds)...")
        await asyncio.sleep(10)

        # Close connection
        print("\n👋 Closing connection...")
        ws.close()

        print("✅ Test complete - WebSocket works!")

    except Exception as e:
        print(f"\n❌ Connection failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_websocket())