#!/usr/bin/env python3
"""
Eagles Trade Execution Guide

This script shows you exactly how to execute a $5 Eagles win trade
using your verified Neural SDK. YOU control when to execute.
"""

import asyncio
import logging
from neural_sdk import NeuralSDK

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def find_eagles_markets():
    """Find available Eagles markets."""
    logger.info("🔍 Searching for Eagles markets...")
    
    sdk = NeuralSDK.from_env(prefix="KALSHI_")
    
    # Connect to get market data
    websocket = sdk.create_websocket()
    
    try:
        await websocket.connect()
        
        # Subscribe to NFL markets
        await websocket.subscribe_markets(['KXNFLGAME*'])
        logger.info("✅ Connected to NFL markets")
        
        logger.info("📊 Available market patterns to look for:")
        logger.info("  - KXNFLGAME-PHI-WIN (Eagles to win)")
        logger.info("  - KXNFLGAME-*PHI* (Eagles-related markets)")
        
        # Monitor for a few seconds to see available markets
        await asyncio.sleep(3)
        
    finally:
        await websocket.disconnect()


def show_trade_execution_code():
    """Show the exact code to execute Eagles trade."""
    
    print("\n" + "="*60)
    print("🦅 EAGLES TRADE EXECUTION CODE")
    print("="*60)
    print("""
# STEP 1: Initialize your SDK
from neural_sdk import NeuralSDK
sdk = NeuralSDK.from_env(prefix="KALSHI_")

# STEP 2: Create your Eagles trading signal
eagles_signal = sdk.create_signal(
    action='BUY',
    market_ticker='KXNFLGAME-PHI-WIN',  # Replace with actual Eagles market
    side='YES',                         # YES = Eagles win
    quantity=50,                        # ~$5 worth (adjust based on price)
    price_limit=0.55,                   # Maximum price you'll pay
    confidence=0.8,                     # Your confidence level
    reason='Eagles look strong tonight'
)

# STEP 3: Execute the trade (REAL MONEY)
# UNCOMMENT THE LINE BELOW WHEN YOU'RE READY:
# trade_result = await sdk.execute_signal(eagles_signal)

print(f"Signal created: {eagles_signal.action} {eagles_signal.market_ticker}")
print(f"Quantity: {eagles_signal.quantity} contracts")
print(f"Max price: ${eagles_signal.price_limit:.4f}")
""")
    print("="*60)


async def main():
    """Main execution guide."""
    
    print("🦅 EAGLES TRADE EXECUTION GUIDE")
    print("💰 $5 Eagles Win Contracts")
    print("="*60)
    
    # Find available markets
    await find_eagles_markets()
    
    # Show execution code
    show_trade_execution_code()
    
    print("\n📋 TO EXECUTE YOUR EAGLES TRADE:")
    print("1. Copy the code above")
    print("2. Find the actual Eagles market ticker")
    print("3. Adjust quantity based on current price (~$5 worth)")
    print("4. Uncomment the execute_signal line")
    print("5. Run the code when YOU are ready")
    
    print("\n⚠️  IMPORTANT:")
    print("- This will use REAL MONEY from your Kalshi account")
    print("- Double-check the market ticker and price")
    print("- Only execute when YOU decide to trade")
    print("- Monitor your position after execution")
    
    print(f"\n🎯 Your system is ready - the choice is yours!")


if __name__ == "__main__":
    asyncio.run(main())