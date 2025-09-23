#!/usr/bin/env python3
"""
Debug WebSocket Authentication
"""

from neural.auth.env import get_api_key_id, get_private_key_material
from neural.auth.signers.kalshi import KalshiSigner

# Test authentication headers
print("🔐 Testing WebSocket Authentication")
print("="*50)

# Get credentials
api_key = get_api_key_id()
private_key_pem = get_private_key_material()

print(f"\n📔 API Key: {api_key[:10]}...")
print(f"🔑 Private Key: {len(private_key_pem)} bytes loaded")

# Create signer
signer = KalshiSigner(api_key, private_key_pem)

# Generate headers for WebSocket
path = "/trade-api/ws/v2"
method = "GET"

headers = signer.headers(method, path)

print(f"\n📝 Generated Headers for WebSocket:")
print(f"  Path: {path}")
print(f"  Method: {method}")
print(f"\nHeaders:")
for key, value in headers.items():
    if "SIGNATURE" in key:
        print(f"  {key}: {value[:20]}...")
    else:
        print(f"  {key}: {value}")

# Verify URL construction
from urllib.parse import urlparse, urlunparse
base_url = "https://api.elections.kalshi.com"
parsed = urlparse(base_url)
ws_url = urlunparse(("wss", parsed.netloc, path, "", "", ""))

print(f"\n🌐 WebSocket URL: {ws_url}")

# Try raw websocket-client connection
import websocket
import ssl

print("\n🔄 Testing raw WebSocket connection...")

# Format headers for websocket-client
header_list = [f"{k}: {v}" for k, v in headers.items()]
print(f"\nHeader format for websocket-client:")
for h in header_list:
    if "SIGNATURE" in h:
        print(f"  {h[:40]}...")
    else:
        print(f"  {h}")

try:
    ws = websocket.create_connection(
        ws_url,
        header=header_list,
        sslopt={"cert_reqs": ssl.CERT_NONE}
    )
    print("\n✅ WebSocket connected successfully!")

    # Send a subscribe command
    import json
    subscribe_msg = json.dumps({
        "id": 1,
        "cmd": "subscribe",
        "params": {
            "channels": ["ticker"],
            "market_tickers": ["KXNFLGAME-25SEP25SEAARI-SEA"]
        }
    })

    print(f"\n📤 Sending: {subscribe_msg}")
    ws.send(subscribe_msg)

    # Receive response
    print("\n📥 Waiting for response...")
    response = ws.recv()
    print(f"Response: {response}")

    ws.close()

except Exception as e:
    print(f"\n❌ Connection failed: {e}")
    import traceback
    traceback.print_exc()