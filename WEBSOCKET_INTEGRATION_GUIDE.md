# Kalshi WebSocket Integration Guide

**Last Updated:** October 11, 2025  
**Status:** Production Ready (with workarounds)

---

## Overview

This guide documents the working approach for integrating with Kalshi's WebSocket API for real-time market data, based on successful live testing that achieved **8.53 price updates per second** during the Alabama vs Missouri game.

## Table of Contents

1. [Authentication Setup](#authentication-setup)
2. [SSL/TLS Configuration](#ssltls-configuration)
3. [Connection Establishment](#connection-establishment)
4. [Subscription Patterns](#subscription-patterns)
5. [Message Handling](#message-handling)
6. [Error Handling](#error-handling)
7. [Performance Optimization](#performance-optimization)
8. [Known Issues](#known-issues)

---

## Authentication Setup

### Requirements

1. Kalshi API Key ID
2. Kalshi Private Key (PEM format)
3. Python packages: `websockets`, `cryptography`, `certifi`

### PSS Signature Generation

Kalshi WebSocket authentication requires PSS (Probabilistic Signature Scheme) signatures:

```python
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
import base64
import time

# Load private key
private_key_pem = Path("path/to/private_key.pem").read_bytes()
private_key = serialization.load_pem_private_key(
    private_key_pem,
    password=None
)

# Create signature function
def sign_pss_text(text: str) -> str:
    """Generate PSS signature for authentication."""
    message = text.encode('utf-8')
    signature = private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH
        ),
        hashes.SHA256()
    )
    return base64.b64encode(signature).decode('utf-8')

# Generate authentication headers
timestamp = str(int(time.time() * 1000))  # Milliseconds
msg_string = timestamp + "GET" + "/trade-api/ws/v2"
signature = sign_pss_text(msg_string)

ws_headers = {
    "KALSHI-ACCESS-KEY": api_key_id,
    "KALSHI-ACCESS-SIGNATURE": signature,
    "KALSHI-ACCESS-TIMESTAMP": timestamp,
}
```

### Important Notes

- Timestamp must be in **milliseconds**
- Message string format: `{timestamp}GET/trade-api/ws/v2` (no spaces)
- Signature must use PSS padding (not PKCS1)
- Headers must be included in initial WebSocket handshake

---

## SSL/TLS Configuration

### Using certifi for Certificate Verification

```python
import ssl
import certifi

# Create SSL context with proper certificate bundle
ssl_context = ssl.create_default_context(cafile=certifi.where())

# Use with websockets library
import websockets
async with websockets.connect(
    ws_url,
    additional_headers=ws_headers,
    ssl=ssl_context  # Proper SSL verification
) as websocket:
    # Connected!
```

### Installation

```bash
pip install certifi
```

### Why This is Necessary

- macOS and some systems don't have proper CA certificates by default
- `certifi` provides Mozilla's curated certificate bundle
- Prevents `SSLCertVerificationError` issues
- Required for production use (don't disable SSL verification)

---

## Connection Establishment

### Complete Connection Example

```python
import asyncio
import websockets
import ssl
import certifi
import json

async def connect_kalshi_websocket(api_key_id, private_key_pem):
    """Connect to Kalshi WebSocket with authentication."""
    
    # 1. Generate authentication headers
    timestamp = str(int(time.time() * 1000))
    msg_string = timestamp + "GET" + "/trade-api/ws/v2"
    signature = sign_pss_text(msg_string)
    
    headers = {
        "KALSHI-ACCESS-KEY": api_key_id,
        "KALSHI-ACCESS-SIGNATURE": signature,
        "KALSHI-ACCESS-TIMESTAMP": timestamp,
    }
    
    # 2. Configure SSL
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    
    # 3. Connect
    ws_url = "wss://api.elections.kalshi.com/trade-api/ws/v2"
    
    async with websockets.connect(
        ws_url,
        additional_headers=headers,
        ssl=ssl_context
    ) as websocket:
        print("✅ Connected to Kalshi WebSocket!")
        
        # Connection is ready for subscriptions
        return websocket
```

### Connection URL

- **Production:** `wss://api.elections.kalshi.com/trade-api/ws/v2`
- **Demo:** `wss://demo-api.elections.kalshi.com/trade-api/ws/v2`

---

## Subscription Patterns

### Pattern 1: Subscribe to Specific Markets (RECOMMENDED)

Use this pattern to receive updates only for markets you're interested in:

```python
subscribe_msg = {
    "id": 1,
    "cmd": "subscribe",
    "params": {
        "channels": ["orderbook_delta"],  # Channel type
        "market_tickers": [
            "KXNCAAFGAME-25OCT11ALAMIZZ-ALA",  # Specific market
            "KXNFLGAME-25OCT13-SF-KC"           # Another market
        ]
    }
}
await websocket.send(json.dumps(subscribe_msg))
```

**Available Channels:**
- `orderbook_delta` - Real-time orderbook changes (incremental)
- `orderbook_snapshot` - Full orderbook state
- `trade` - Executed trades
- `fill` - Your order fills (if trading)

**Key Points:**
- ✅ Server-side filtering (efficient)
- ✅ Only receive relevant data
- ✅ Low bandwidth usage
- ⚠️ MUST include both `channels` AND `market_tickers`

### Pattern 2: Subscribe to All Markets

Use this if you need data from many markets:

```python
subscribe_msg = {
    "id": 1,
    "cmd": "subscribe",
    "params": {
        "channels": ["ticker"]  # Special channel for all markets
    }
}
await websocket.send(json.dumps(subscribe_msg))
```

**Key Points:**
- ✅ Receives updates for all active markets
- ❌ High bandwidth (~190KB in 10 seconds in testing)
- ❌ Requires client-side filtering
- ⚠️ Only use if you actually need all markets

### Common Subscription Errors

**Error: "Params required"**
```python
# ❌ WRONG: Missing market_tickers
{"params": {"channels": ["orderbook_delta"]}}

# ✅ CORRECT: Include both
{"params": {"channels": ["orderbook_delta"], "market_tickers": ["TICKER"]}}
```

**Error: "Unknown channel name"**
```python
# ❌ WRONG: Old ticker:MARKET format
{"params": {"channels": ["ticker:KXNCAAFGAME-..."]}}

# ✅ CORRECT: Use separate market_tickers param
{"params": {"channels": ["orderbook_delta"], "market_tickers": ["KXNCAAFGAME-..."]}}
```

---

## Message Handling

### Message Types

#### 1. Subscription Confirmation

```json
{
  "type": "subscribed",
  "channel": "orderbook_delta",
  "id": 1
}
```

#### 2. Orderbook Snapshot

First message after subscription contains full orderbook state:

```json
{
  "type": "orderbook_snapshot",
  "sid": 1,
  "seq": 1,
  "msg": {
    "market_ticker": "KXNCAAFGAME-25OCT11ALAMIZZ-ALA",
    "market_id": "...",
    "yes": [[1, 250082], [2, 3200], ...],  // [price_cents, quantity]
    "no": [[1, 162774], [2, 9000], ...],
    "yes_dollars": [["0.0100", 250082], ...],  // Human-readable
    "no_dollars": [["0.0100", 162774], ...]
  }
}
```

#### 3. Orderbook Delta

Subsequent messages contain only changes:

```json
{
  "type": "orderbook_delta",
  "sid": 1,
  "seq": 2,
  "msg": {
    "market_ticker": "KXNCAAFGAME-25OCT11ALAMIZZ-ALA",
    "market_id": "...",
    "price": 3,              // Price level in cents
    "price_dollars": "0.0300",
    "delta": 1079,           // Change in quantity (+/-)
    "side": "yes",           // "yes" or "no"
    "ts": "2025-10-11T18:26:53.361842Z"
  }
}
```

#### 4. Error Messages

```json
{
  "type": "error",
  "id": 1,
  "msg": {
    "code": 8,
    "msg": "Unknown channel name"
  }
}
```

### Message Processing Example

```python
async def handle_messages(websocket, target_ticker):
    """Process incoming WebSocket messages."""
    
    async for message in websocket:
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            
            if msg_type == "subscribed":
                print(f"✅ Subscribed to {data.get('channel')}")
            
            elif msg_type == "orderbook_snapshot":
                ob_data = data.get("msg", {})
                market = ob_data.get("market_ticker")
                
                if market == target_ticker:
                    # Process full orderbook
                    yes_levels = ob_data.get("yes_dollars", [])
                    no_levels = ob_data.get("no_dollars", [])
                    
                    best_yes = float(yes_levels[-1][0]) if yes_levels else 0
                    best_no = float(no_levels[-1][0]) if no_levels else 0
                    
                    print(f"Orderbook: YES={best_yes:.2f}, NO={best_no:.2f}")
            
            elif msg_type == "orderbook_delta":
                ob_data = data.get("msg", {})
                market = ob_data.get("market_ticker")
                
                if market == target_ticker:
                    # Process orderbook change
                    price = ob_data.get("price_dollars")
                    delta = ob_data.get("delta")
                    side = ob_data.get("side")
                    
                    print(f"Delta: {side.upper()} @ ${price} ({delta:+d})")
            
            elif msg_type == "error":
                print(f"❌ Error: {data.get('msg')}")
        
        except Exception as e:
            print(f"⚠️ Error processing message: {e}")
```

---

## Error Handling

### Connection Errors

```python
import websockets.exceptions

try:
    async with websockets.connect(...) as websocket:
        await handle_messages(websocket)

except websockets.exceptions.InvalidStatusCode as e:
    if e.status_code == 403:
        print("❌ Authentication failed - check API key and signature")
    elif e.status_code == 401:
        print("❌ Unauthorized - invalid credentials")
    else:
        print(f"❌ Connection failed: {e}")

except websockets.exceptions.WebSocketException as e:
    print(f"❌ WebSocket error: {e}")

except Exception as e:
    print(f"❌ Unexpected error: {e}")
```

### Reconnection Logic

```python
async def websocket_with_reconnection(api_key_id, private_key_pem, max_retries=5):
    """WebSocket with automatic reconnection."""
    
    retry_count = 0
    backoff = 1  # seconds
    
    while retry_count < max_retries:
        try:
            async with connect_kalshi_websocket(api_key_id, private_key_pem) as ws:
                # Reset counters on successful connection
                retry_count = 0
                backoff = 1
                
                await handle_messages(ws)
        
        except Exception as e:
            retry_count += 1
            print(f"⚠️ Connection lost ({retry_count}/{max_retries}): {e}")
            
            if retry_count < max_retries:
                print(f"⏳ Reconnecting in {backoff}s...")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)  # Exponential backoff, max 60s
            else:
                print("❌ Max retries reached, giving up")
                raise
```

---

## Performance Optimization

### Achieved Performance

In live testing (Alabama vs Missouri game, October 11, 2025):
- **8.53 updates per second**
- **1,387 updates in 2.7 minutes**
- **Zero dropped messages**
- **Sub-millisecond processing latency**

### Best Practices

1. **Use Specific Market Subscriptions**
   ```python
   # ✅ Good: Only subscribe to markets you need
   {"market_tickers": ["TICKER1", "TICKER2"]}
   
   # ❌ Bad: Subscribe to all then filter
   {"channels": ["ticker"]}  # Wastes bandwidth
   ```

2. **Process Messages Efficiently**
   ```python
   # ✅ Good: Quick filtering
   if msg_type in ["orderbook_delta", "orderbook_snapshot"]:
       market = data["msg"]["market_ticker"]
       if market in target_tickers:
           process_update(data)
   
   # ❌ Bad: Complex processing in message loop
   if msg_type == "orderbook_delta":
       # Don't do heavy computation here!
       analyze_entire_market(data)  # Blocks message loop
   ```

3. **Concurrent Processing**
   ```python
   # ✅ Good: Offload heavy work
   async def handle_message(data):
       if needs_heavy_processing(data):
           asyncio.create_task(process_in_background(data))
   ```

4. **Database Writes**
   ```python
   # ✅ Good: Batch writes or use queue
   write_queue = []
   
   if len(write_queue) >= 10:
       db.bulk_insert(write_queue)
       write_queue.clear()
   ```

---

## Known Issues

### Issue #1: Neural SDK WebSocket Authentication Fails

**Status:** 🔴 BLOCKING SDK USAGE

The Neural SDK's `KalshiWebSocketSupervisor` fails with 403 Forbidden despite correct credentials.

**Workaround:** Use raw `websockets` library as shown in this guide.

**Impact:** Cannot use SDK's built-in reconnection logic and health metrics.

See [BETA_BUGS_TRACKING.md - Bug #11](/BETA_BUGS_TRACKING.md#bug-11-neural-sdk-websocket-authentication-fails-with-kalshiwebsocketsupervisor) for details.

### Issue #2: SDK subscribe() Missing market_tickers Parameter

**Status:** ⚠️ WORKAROUND AVAILABLE

The SDK's `subscribe()` method doesn't accept `market_tickers` parameter.

**Workaround:** Send raw subscription messages as shown above.

**Impact:** Must bypass SDK for subscriptions.

See [BETA_BUGS_TRACKING.md - Bug #14](/BETA_BUGS_TRACKING.md#bug-14-sdk-subscribe-missing-market_tickers-parameter-support) for details.

---

## Complete Working Example

```python
#!/usr/bin/env python3
"""
Complete Kalshi WebSocket Example
Tested and working as of October 11, 2025
"""
import asyncio
import websockets
import ssl
import certifi
import json
import time
from pathlib import Path
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
import base64

# Configuration
API_KEY_ID = "your-api-key-id"
PRIVATE_KEY_PATH = Path("path/to/private_key.pem")
TARGET_TICKER = "KXNCAAFGAME-25OCT11ALAMIZZ-ALA"
WS_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"

def sign_pss_text(private_key, text: str) -> str:
    """Generate PSS signature."""
    message = text.encode('utf-8')
    signature = private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH
        ),
        hashes.SHA256()
    )
    return base64.b64encode(signature).decode('utf-8')

async def main():
    """Main WebSocket client."""
    
    # Load private key
    private_key_pem = PRIVATE_KEY_PATH.read_bytes()
    private_key = serialization.load_pem_private_key(private_key_pem, password=None)
    
    # Generate auth headers
    timestamp = str(int(time.time() * 1000))
    msg_string = timestamp + "GET" + "/trade-api/ws/v2"
    signature = sign_pss_text(private_key, msg_string)
    
    headers = {
        "KALSHI-ACCESS-KEY": API_KEY_ID,
        "KALSHI-ACCESS-SIGNATURE": signature,
        "KALSHI-ACCESS-TIMESTAMP": timestamp,
    }
    
    # Configure SSL
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    
    # Connect
    print("🔌 Connecting to Kalshi WebSocket...")
    async with websockets.connect(WS_URL, additional_headers=headers, ssl=ssl_context) as ws:
        print("✅ Connected!")
        
        # Subscribe
        subscribe_msg = {
            "id": 1,
            "cmd": "subscribe",
            "params": {
                "channels": ["orderbook_delta"],
                "market_tickers": [TARGET_TICKER]
            }
        }
        await ws.send(json.dumps(subscribe_msg))
        print(f"📡 Subscribed to {TARGET_TICKER}")
        
        # Handle messages
        message_count = 0
        async for message in ws:
            data = json.loads(message)
            msg_type = data.get("type")
            
            if msg_type in ["orderbook_delta", "orderbook_snapshot"]:
                message_count += 1
                if message_count % 10 == 0:
                    print(f"📊 Received {message_count} updates")
            
            elif msg_type == "subscribed":
                print(f"✅ Subscription confirmed!")
            
            elif msg_type == "error":
                print(f"❌ Error: {data.get('msg')}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Troubleshooting

### Problem: 403 Forbidden

**Causes:**
1. Invalid API key
2. Incorrect signature generation
3. Wrong timestamp format
4. Missing authentication headers

**Solution:** Verify signature generation matches example above.

### Problem: "Params required"

**Cause:** Missing `market_tickers` parameter.

**Solution:** Include both `channels` AND `market_tickers` in subscription.

### Problem: SSL Certificate Error

**Cause:** Missing or incorrect CA certificates.

**Solution:** Install and use `certifi`:
```bash
pip install certifi
```

### Problem: No Messages Received

**Causes:**
1. Market not active
2. Incorrect ticker
3. Subscription not confirmed

**Solution:** Check subscription confirmation message and verify market is trading.

---

## Additional Resources

- [Kalshi WebSocket API Documentation](https://trading-api.readme.io/reference/marketdatawebsocket)
- [BETA_BUGS_TRACKING.md](/BETA_BUGS_TRACKING.md) - Known SDK issues
- [Live Testing Results](/LIVE_TESTING_FINDINGS.md) - Performance data

---

**Document Version:** 1.0  
**Tested On:** October 11, 2025  
**Next Review:** After SDK beta update

