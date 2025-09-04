#!/usr/bin/env python3
"""
Simple WebSocket Demo

The simplest possible example of using Neural SDK WebSocket functionality.
Perfect for getting started with live market streaming.
"""

import asyncio
import logging
from neural_sdk import NeuralSDK

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


async def main():
    """Simple WebSocket streaming demo."""
    
    print("🚀 Neural SDK WebSocket Demo")
    print("=" * 40)
    
    # Initialize SDK
    sdk = NeuralSDK.from_env()
    
    # Create WebSocket
    websocket = sdk.create_websocket()
    
    # Handle market updates
    @websocket.on_market_data
    async def on_price_update(market_data):
        ticker = market_data.get('market_ticker', 'Unknown')
        price = market_data.get('yes_price', 0)
        volume = market_data.get('volume', 0)
        
        print(f"📊 {ticker}: ${price:.4f} (Vol: {volume})")
    
    # Handle connection events
    @websocket.on_connection
    async def on_connection(event):
        status = event.get('status', 'unknown')
        print(f"🔌 Connection: {status}")
    
    try:
        # Connect
        print("Connecting to WebSocket...")
        await websocket.connect()
        
        # Subscribe to NFL markets
        print("Subscribing to NFL markets...")
        await websocket.subscribe_markets(['KXNFLGAME*'])
        
        # Stream for 30 seconds
        print("🎮 Streaming live data for 30 seconds...")
        print("(Press Ctrl+C to stop early)")
        
        await asyncio.sleep(30)
        
        # Show final status
        status = websocket.get_status()
        print(f"\n📈 Final Status: {status}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        print("Disconnecting...")
        await websocket.disconnect()
        print("✅ Demo completed!")


if __name__ == "__main__":
    asyncio.run(main())
