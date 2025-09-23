#!/usr/bin/env python3
"""
Verify API credentials work with REST API
"""

import requests
from neural.auth.env import get_api_key_id, get_private_key_material
from neural.auth.signers.kalshi import KalshiSigner

print("🔐 Verifying API Credentials")
print("="*50)

# Get credentials
api_key = get_api_key_id()
private_key_pem = get_private_key_material()

print(f"\n📔 API Key: {api_key[:10]}...")

# Create signer
signer = KalshiSigner(api_key, private_key_pem)

# Test with REST API first
path = "/trade-api/v2/exchange/status"
method = "GET"
url = f"https://api.elections.kalshi.com{path}"

headers = signer.headers(method, path)

print(f"\n📡 Testing REST API:")
print(f"  URL: {url}")
print(f"  Method: {method}")

try:
    response = requests.get(url, headers=headers)
    print(f"\n  Status: {response.status_code}")

    if response.status_code == 200:
        print("  ✅ REST API authentication works!")
        data = response.json()
        print(f"  Exchange Status: {data}")
    else:
        print(f"  ❌ REST API failed: {response.text}")

except Exception as e:
    print(f"  ❌ Request failed: {e}")

# Now test WebSocket authentication
print("\n" + "="*50)
print("📡 WebSocket Authentication Test")
print("="*50)

# According to docs, WebSocket might need different timing or format
import time

# Try with fresh timestamp
timestamp_ms = int(time.time() * 1000)
ws_path = "/trade-api/ws/v2"
ws_method = "GET"

# Manually construct signature to debug
msg = f"{timestamp_ms}{ws_method.upper()}{ws_path}".encode("utf-8")
print(f"\n🔐 Signature components:")
print(f"  Timestamp: {timestamp_ms}")
print(f"  Method: {ws_method.upper()}")
print(f"  Path: {ws_path}")
print(f"  Message to sign: {msg.decode()}")

# Generate signature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
import base64

private_key = serialization.load_pem_private_key(private_key_pem, password=None)
signature = private_key.sign(
    msg,
    padding.PSS(
        mgf=padding.MGF1(hashes.SHA256()),
        salt_length=padding.PSS.DIGEST_LENGTH
    ),
    hashes.SHA256()
)
signature_b64 = base64.b64encode(signature).decode("utf-8")

ws_headers = {
    "KALSHI-ACCESS-KEY": api_key,
    "KALSHI-ACCESS-TIMESTAMP": str(timestamp_ms),
    "KALSHI-ACCESS-SIGNATURE": signature_b64
}

print("\n📝 WebSocket Headers:")
for k, v in ws_headers.items():
    if "SIGNATURE" in k:
        print(f"  {k}: {v[:30]}...")
    else:
        print(f"  {k}: {v}")

# Test WebSocket with these headers
import websocket
import ssl

ws_url = "wss://api.elections.kalshi.com/trade-api/ws/v2"
print(f"\n🌐 WebSocket URL: {ws_url}")

header_list = [f"{k}: {v}" for k, v in ws_headers.items()]

print("\n🔄 Attempting WebSocket connection...")
try:
    ws = websocket.create_connection(
        ws_url,
        header=header_list,
        sslopt={"cert_reqs": ssl.CERT_NONE},
        timeout=10
    )
    print("✅ WebSocket connected successfully!")
    ws.close()

except Exception as e:
    print(f"❌ WebSocket failed: {str(e)[:200]}")

    # Check if it's an auth issue or something else
    if "403" in str(e):
        print("\n⚠️ 403 Forbidden - Authentication rejected")
        print("Possible causes:")
        print("  1. WebSocket requires special permissions")
        print("  2. API key doesn't have WebSocket access")
        print("  3. Different auth method needed for WebSocket")
    elif "404" in str(e):
        print("\n⚠️ 404 Not Found - Wrong endpoint")
    else:
        print("\n⚠️ Unknown error")