#!/usr/bin/env python3
"""
Market Investigation Script
Diagnose why the Cowboys vs Eagles trader can't find markets
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from neural_sdk import NeuralSDK
from neural_sdk.data_pipeline.data_sources.kalshi.client import KalshiClient

async def main():
    print("🔍 INVESTIGATING MARKET DATA ISSUES")
    print("=" * 60)
    
    try:
        # Initialize SDK and client
        sdk = NeuralSDK.from_env()
        client = KalshiClient()
        
        print("✅ SDK and client initialized")
        print()
        
        # Step 1: Check current positions (what we know works)
        print("📊 STEP 1: Current Portfolio Positions")
        print("-" * 40)
        try:
            positions = await sdk.get_positions()
            print(f"✅ Found {len(positions)} current positions:")
            
            for pos in positions:
                print(f"  🎯 {pos.ticker}")
                print(f"      Market: {pos.market_name}")
                print(f"      Shares: {pos.position}")
                print(f"      Status: Active (we have positions)")
                print()
        except Exception as e:
            print(f"❌ Error getting positions: {e}")
        
        # Step 2: Query available markets from Kalshi API
        print("🔍 STEP 2: Available Markets from Kalshi API")
        print("-" * 40)
        try:
            # Get all NFL markets
            nfl_markets = client.get_all_markets(series_ticker="KXNFL")
            print(f"✅ Found {len(nfl_markets)} total NFL markets")
            print()
            
            # Filter for Cowboys vs Eagles related
            cowboys_eagles_markets = []
            for market in nfl_markets:
                ticker = market.get('ticker', '')
                title = market.get('title', '')
                status = market.get('status', 'unknown')
                
                # Look for Dallas/Cowboys and Philadelphia/Eagles
                if any(keyword in ticker.upper() or keyword in title.upper() for keyword in 
                       ['DALPHI', 'DAL', 'PHI', 'COWBOYS', 'EAGLES']):
                    cowboys_eagles_markets.append(market)
            
            print(f"🏈 Cowboys vs Eagles Markets: {len(cowboys_eagles_markets)}")
            print("-" * 40)
            
            if cowboys_eagles_markets:
                for market in cowboys_eagles_markets[:10]:  # Show first 10
                    ticker = market.get('ticker', 'Unknown')
                    title = market.get('title', 'No title')
                    status = market.get('status', 'unknown')
                    
                    print(f"📋 {ticker}")
                    print(f"    Title: {title}")
                    print(f"    Status: {status}")
                    print()
            else:
                print("❌ No Cowboys vs Eagles markets found")
                
                # Show recent NFL markets for context
                print("\n🏈 Recent NFL Markets (for reference):")
                print("-" * 40)
                for market in nfl_markets[:5]:
                    print(f"📋 {market.get('ticker', 'Unknown')}")
                    print(f"    Title: {market.get('title', 'No title')}")
                    print(f"    Status: {market.get('status', 'unknown')}")
                    print()
                    
        except Exception as e:
            print(f"❌ Error querying markets: {e}")
            import traceback
            traceback.print_exc()
        
        # Step 3: Check specific problematic ticker
        print("🎯 STEP 3: Check Specific Problematic Market")
        print("-" * 40)
        problematic_ticker = "KXNFLCOMBO-25SEP04DALPHI-DAL-DALCLAMB88-47"
        print(f"Checking: {problematic_ticker}")
        
        try:
            market_info = client.get_market(problematic_ticker)
            print(f"✅ Market found:")
            print(f"    Status: {market_info.get('status', 'unknown')}")
            print(f"    Title: {market_info.get('title', 'No title')}")
            print(f"    Close Date: {market_info.get('close_date', 'Unknown')}")
        except Exception as e:
            print(f"❌ Market not found or error: {e}")
            print("This explains why the trader can't find it!")
        
        print()
        
        # Step 4: Market Status Analysis
        print("📈 STEP 4: Market Status Analysis")
        print("-" * 40)
        
        # Check if game date has passed
        from datetime import datetime
        game_date = "2024-09-04"
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        print(f"Game Date: {game_date}")
        print(f"Current Date: {current_date}")
        
        if current_date > game_date:
            print("⚠️  ISSUE IDENTIFIED: Game date has passed!")
            print("   Markets may be settled/expired")
        else:
            print("✅ Game date is current or future")
        
        print()
        
        # Step 5: Recommendations
        print("💡 RECOMMENDATIONS:")
        print("-" * 40)
        
        if current_date > game_date:
            print("1. 🕒 Markets are likely EXPIRED (game was on Sep 4, 2024)")
            print("2. 🔄 Update trader to look for CURRENT games/markets")
            print("3. 📅 Add market expiration checking before trading")
            print("4. 🎯 Use dynamic market discovery instead of hardcoded tickers")
        
        if positions:
            print("5. 📊 You still have positions in these markets:")
            for pos in positions[:3]:
                print(f"   • {pos.market_name} ({pos.position} shares)")
            print("   Consider closing these positions if markets are settled")
        
        print("\n🔧 NEXT STEPS:")
        print("1. Update trader to use current/future NFL games")
        print("2. Add market status validation before trading")
        print("3. Implement dynamic market discovery")
        print("4. Add better error handling for expired markets")
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        if 'client' in locals():
            client.close()
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)