#!/usr/bin/env python3
"""
Final Infrastructure Test - Verify all components work
"""

print("\n🚀 Neural SDK - Infrastructure Components Test\n")
print("="*70)

# Test 1: REST API Data Collection
print("\n📊 TEST 1: REST API Market Data")
print("-"*40)
try:
    from neural.data_collection import get_game_markets
    import asyncio

    async def test_rest():
        markets = await get_game_markets("KXNFLGAME-25SEP25SEAARI")
        if not markets.empty:
            print("✅ REST API: Working")
            print(f"  Found {len(markets)} markets")
            for _, m in markets.iterrows():
                team = "Seattle" if "SEA" in m['ticker'] else "Arizona"
                print(f"  {team}: ${m['yes_ask']/100:.2f} ({m['yes_ask']:.0f}%)")
            return True
        else:
            print("❌ REST API: No markets found")
            return False

    rest_works = asyncio.run(test_rest())
except Exception as e:
    print(f"❌ REST API: Failed - {e}")
    rest_works = False

# Test 2: FIX Connection
print("\n🔧 TEST 2: FIX API Connection")
print("-"*40)
try:
    from neural.trading.fix import KalshiFIXClient, FIXConnectionConfig

    async def test_fix():
        config = FIXConnectionConfig(
            heartbeat_interval=30,
            reset_seq_num=True
        )

        connected = False
        def handle_msg(msg):
            nonlocal connected
            msg_dict = KalshiFIXClient.to_dict(msg)
            if msg_dict.get(35) == 'A':  # Logon
                connected = True

        client = KalshiFIXClient(config=config, on_message=handle_msg)

        try:
            await client.connect(timeout=5)
            await asyncio.sleep(2)

            if connected:
                print("✅ FIX API: Connected successfully")
                await client.logout()
                await client.close()
                return True
            else:
                print("❌ FIX API: Connected but no logon")
                await client.close()
                return False
        except Exception as e:
            print(f"❌ FIX API: Connection failed - {e}")
            return False

    fix_works = asyncio.run(test_fix())
except Exception as e:
    print(f"❌ FIX API: Failed - {e}")
    fix_works = False

# Test 3: WebSocket (expected to fail without special permissions)
print("\n📡 TEST 3: WebSocket Connection")
print("-"*40)
try:
    from neural.trading import KalshiWebSocketClient

    ws_connected = False

    def handle_ws(msg):
        global ws_connected
        if msg.get("type") == "subscribed":
            ws_connected = True

    try:
        ws = KalshiWebSocketClient(on_message=handle_ws)
        ws.connect(block=True)
        print("⚠️ WebSocket: Connected (unexpected)")
        ws.close()
        ws_works = True
    except Exception as e:
        print(f"⚠️ WebSocket: Not available - {str(e)[:50]}...")
        print("  (This is expected without special permissions)")
        ws_works = False
except Exception as e:
    print(f"⚠️ WebSocket: Module error - {e}")
    ws_works = False

# Summary
print("\n" + "="*70)
print("📊 INFRASTRUCTURE STATUS SUMMARY")
print("="*70)

components = [
    ("REST API (Market Data)", rest_works, "Primary data source"),
    ("FIX API (Order Execution)", fix_works, "Ultra-fast trading"),
    ("WebSocket (Streaming)", ws_works, "Optional - needs permissions")
]

working = 0
for name, status, purpose in components:
    symbol = "✅" if status else "❌"
    working += 1 if status else 0
    print(f"{symbol} {name:25} - {purpose}")

print("\n" + "-"*70)

if rest_works and fix_works:
    print("\n🎉 SUCCESS! Core infrastructure is operational!")
    print("\nYou have everything needed for trading:")
    print("  • REST API provides reliable market data")
    print("  • FIX API enables fast order execution")
    print("  • Complete pipeline: Data → Analysis → Execution")
    print("\n📈 Ready to build trading strategies!")

elif rest_works:
    print("\n⚠️ PARTIAL SUCCESS")
    print("\nREST API is working - you can:")
    print("  • Fetch market data")
    print("  • Analyze prices")
    print("  • Build strategies")
    print("\nFIX API needs credentials for order execution")

else:
    print("\n❌ Infrastructure needs configuration")
    print("\nCheck:")
    print("  • API credentials in .env or secrets/")
    print("  • Network connectivity")
    print("  • Dependencies installed")

print("\n" + "="*70)