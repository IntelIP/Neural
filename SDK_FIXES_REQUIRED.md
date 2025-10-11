# Neural SDK - Required Fixes for Beta Update

**SDK Version:** Neural v0.1.0 (Beta)  
**Last Updated:** October 11, 2025  
**Total Issues:** 15 bugs documented

---

## Priority 1: CRITICAL (Blocking Core Functionality)

### 1. Fix KalshiWebSocketClient Authentication

**File:** `neural/trading/websocket.py`  
**Severity:** 🔴 CRITICAL  
**Status:** Blocking SDK WebSocket usage

**Issue:**
`KalshiWebSocketSupervisor` fails with 403 Forbidden despite valid credentials.

**Root Cause:**
Authentication headers not properly set during WebSocket handshake.

**Required Changes:**
1. Ensure PSS signature generation matches Kalshi's requirements
2. Add authentication headers to initial HTTP upgrade request
3. Test with actual Kalshi production credentials
4. Verify SSL/TLS configuration with certifi

**Testing:**
```python
# Should work after fix
supervisor = KalshiWebSocketSupervisor(
    api_key_id="valid-key",
    private_key_pem=private_key_bytes
)
await supervisor.start()  # Should not get 403
```

**Reference:** Bug #11 in BETA_BUGS_TRACKING.md

---

### 2. Fix get_nfl_games() and get_cfb_games() Methods

**File:** `neural/data_collection/kalshi.py`  
**Severity:** 🔴 CRITICAL  
**Status:** Methods completely unusable

**Issue:**
Methods expect `series_ticker` field that doesn't exist in API response.

**Error:**
```python
KeyError: 'series_ticker'
```

**Required Changes:**
1. Remove `series_ticker` parameter usage
2. Use `event_ticker` field (which actually exists)
3. Add proper error handling for missing fields
4. Update method signatures to match actual Kalshi API

**Fix Example:**
```python
# Current (BROKEN):
def get_nfl_games(self):
    return self.get_markets(series_ticker="KXNFLGAME")  # WRONG FIELD

# Fixed:
def get_nfl_games(self):
    markets = self.get_markets_by_sport(sport="football", limit=1000)
    return [m for m in markets.get('markets', []) 
            if 'KXNFLGAME' in m.get('ticker', '')]
```

**Reference:** Bug #12 in BETA_BUGS_TRACKING.md

---

### 3. Add NumPy 2.x Compatibility

**File:** `setup.py` or requirements  
**Severity:** 🔴 CRITICAL  
**Status:** Crashes on import with NumPy 2.x

**Issue:**
SDK compiled against NumPy 1.x, fails with NumPy 2.3.3+

**Required Changes:**
1. Recompile SDK against NumPy 2.0 API
2. Add explicit `numpy<2.0` dependency in setup.py if not recompiling
3. Add version compatibility check on import
4. Document NumPy requirements clearly

**Short-term Fix:**
```python
# In setup.py
install_requires=[
    'numpy>=1.24.0,<2.0',  # Explicit version constraint
    ...
]
```

**Long-term Fix:**
Recompile all C extensions against NumPy 2.0.

**Reference:** Bug #13 in BETA_BUGS_TRACKING.md

---

## Priority 2: IMPORTANT (Reduced Functionality)

### 4. Add market_tickers Parameter to subscribe()

**File:** `neural/trading/websocket.py`  
**Severity:** 🟠 HIGH  
**Status:** Cannot filter subscriptions efficiently

**Issue:**
`subscribe()` method doesn't accept `market_tickers` for filtered subscriptions.

**Required Changes:**
```python
def subscribe(
    self, 
    channels: list[str], 
    market_tickers: Optional[list[str]] = None,
    params: Optional[Dict[str, Any]] = None,
    request_id: Optional[int] = None
) -> int:
    """Subscribe to channels with optional market filtering."""
    req_id = request_id or self._next_id()
    
    subscribe_params = {"channels": channels}
    if market_tickers:
        subscribe_params["market_tickers"] = market_tickers
    if params:
        subscribe_params.update(params)
    
    payload = {
        "id": req_id,
        "cmd": "subscribe",
        "params": subscribe_params
    }
    self.send(payload)
    return req_id
```

**Reference:** Bug #14 in BETA_BUGS_TRACKING.md

---

### 5. Improve WebSocket Error Messages

**File:** `neural/trading/websocket.py`  
**Severity:** 🟠 MEDIUM  
**Status:** Hard to debug issues

**Issue:**
Generic error messages, no context about what failed.

**Required Changes:**
1. Add specific error messages for common failures
2. Include authentication details in debug logs
3. Better exception handling with context
4. Log WebSocket handshake details

**Example:**
```python
try:
    await self.connect()
except websockets.exceptions.InvalidStatusCode as e:
    if e.status_code == 403:
        logger.error(
            "WebSocket authentication failed. "
            "Check API key and private key. "
            f"URL: {self.url}, "
            f"Key ID: {self.api_key_id[:8]}..."
        )
    raise
```

---

### 6. Add Reconnection Logic to Supervisor

**File:** `neural/trading/websocket.py`  
**Severity:** 🟠 MEDIUM  
**Status:** No automatic reconnection

**Issue:**
Supervisor doesn't automatically reconnect on connection loss.

**Required Changes:**
1. Add exponential backoff reconnection
2. Configurable max retries
3. Preserve subscription state across reconnects
4. Health checks and monitoring

**Example:**
```python
class KalshiWebSocketSupervisor:
    async def _reconnect_loop(self):
        retry_count = 0
        backoff = 1.0
        
        while retry_count < self.max_retries:
            try:
                await self.client.connect()
                # Restore subscriptions
                await self._restore_subscriptions()
                retry_count = 0
                backoff = 1.0
            except Exception as e:
                retry_count += 1
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)
```

---

## Priority 3: ENHANCEMENTS (Nice to Have)

### 7. Add Market Filtering Helpers

**File:** `neural/data_collection/kalshi.py`  
**Severity:** 🟢 LOW  
**Status:** Would improve developer experience

**Suggested Addition:**
```python
def get_sports_markets(self, sport: str, event_ticker: str = None, 
                      status: str = None, limit: int = 1000):
    """
    Get markets for a sport with optional filtering.
    
    Args:
        sport: 'football', 'basketball', etc.
        event_ticker: Filter by event (e.g., 'KXNFLGAME')
        status: Filter by status ('open', 'closed', etc.)
        limit: Max markets to return
    
    Returns:
        dict: Markets matching criteria
    """
    markets = self.get_markets_by_sport(sport=sport, limit=limit)
    
    filtered = markets.get('markets', [])
    
    if event_ticker:
        filtered = [m for m in filtered 
                   if event_ticker in m.get('ticker', '')]
    
    if status:
        filtered = [m for m in filtered 
                   if m.get('status') == status]
    
    return {'markets': filtered}
```

---

### 8. Improve SSL/TLS Documentation

**File:** Documentation/examples  
**Severity:** 🟢 LOW  
**Status:** Confusing for users

**Required:**
1. Document need for certifi package
2. Provide SSL configuration examples
3. Explain certificate verification
4. Add troubleshooting guide

**Example Documentation:**
```markdown
## SSL/TLS Setup

Install certifi for proper certificate verification:

pip install certifi

Configure SSL context:

import ssl
import certifi

ssl_context = ssl.create_default_context(cafile=certifi.where())

Use with WebSocket:

await websockets.connect(url, ssl=ssl_context)
```

---

### 9. Add Connection Pooling Examples

**File:** Examples directory  
**Severity:** 🟢 LOW  
**Status:** Would help with performance

**Suggested Example:**
```python
# examples/connection_pooling.py
import asyncio
from neural import TradingClient

class ConnectionPool:
    """Manage multiple WebSocket connections efficiently."""
    
    def __init__(self, api_key_id, private_key, max_connections=5):
        self.api_key_id = api_key_id
        self.private_key = private_key
        self.max_connections = max_connections
        self.connections = []
    
    async def get_connection(self):
        # Implementation
        pass
```

---

## Testing Requirements

For each fix, add:

1. **Unit Tests**
   - Test with valid credentials
   - Test with invalid credentials
   - Test error conditions
   - Test edge cases

2. **Integration Tests**
   - Test against production API
   - Test reconnection logic
   - Test subscription management
   - Test concurrent operations

3. **Performance Tests**
   - Message throughput
   - Memory usage
   - Connection stability
   - Latency measurements

---

## Documentation Updates

### API Reference
- Complete method signatures
- Parameter descriptions
- Return value documentation
- Usage examples
- Error conditions

### Guides
- Getting started tutorial
- WebSocket integration guide
- Market discovery guide
- Error handling guide
- Performance optimization

### Known Issues
- Document current bugs
- Provide workarounds
- Link to issue tracker
- Update with fixes

---

## Release Checklist

Before next beta release:

- [ ] Fix all Priority 1 issues
- [ ] Add tests for critical paths
- [ ] Update documentation
- [ ] Run integration tests against production
- [ ] Verify examples work
- [ ] Update changelog
- [ ] Bump version number
- [ ] Tag release

---

## Contributing Fixes

We're ready to contribute fixes back to Neural SDK:

### Our Working Solutions

1. **WebSocket Authentication** - Working raw websockets implementation
2. **Market Discovery** - Working `get_markets_by_sport()` wrapper
3. **NumPy Compatibility** - Tested version constraints
4. **market_tickers Support** - Working subscription format

### Code Available

All working implementations are in:
- `nfl/run_live_test.py` - Working WebSocket
- `nfl/game_discovery.py` - Working market discovery
- `nfl/test_kalshi_ws_raw.py` - Test scripts

**Ready to contribute back to SDK repository when maintainers are ready.**

---

## Additional Resources

- [BETA_BUGS_TRACKING.md](/BETA_BUGS_TRACKING.md) - Complete bug list
- [WEBSOCKET_INTEGRATION_GUIDE.md](/WEBSOCKET_INTEGRATION_GUIDE.md) - Working patterns
- [LIVE_TESTING_FINDINGS.md](/LIVE_TESTING_FINDINGS.md) - Testing results

---

**Document Version:** 1.0  
**Last Updated:** October 11, 2025  
**Next Review:** With beta update release

