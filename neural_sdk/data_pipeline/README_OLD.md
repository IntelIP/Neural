# Kalshi WebSocket Stream Runner

Standalone service for streaming real-time Kalshi market data via WebSocket.

## Features

- RSA-PSS authentication for secure connection
- Real-time price updates (ticker data)
- Order book depth streaming
- Trade execution monitoring
- Automatic price conversion from centi-cents to dollars
- Support for both demo and production environments
- Interactive market and channel selection

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Authentication

Set the following environment variables:

```bash
# Required for authenticated access
export KALSHI_API_KEY_ID="your-api-key-id"

# Private key - use one of these methods:
# Method 1: Direct key string
export KALSHI_PRIVATE_KEY="-----BEGIN RSA PRIVATE KEY-----
...your private key content...
-----END RSA PRIVATE KEY-----"

# Method 2: Path to key file
export KALSHI_PRIVATE_KEY_FILE="/path/to/your/private_key.pem"

# Environment (demo or prod)
export KALSHI_ENV="demo"  # or "prod" for production
```

### 3. Run the Stream

```bash
python kalshi_stream.py
```

## Usage

1. **Select Markets**: Choose from available Kalshi markets or enter "all"
2. **Select Channels**: Pick data channels to subscribe to:
   - `ticker`: Real-time price updates
   - `orderbook_delta`: Order book changes
   - `trade`: Executed trades
   - `market_lifecycle`: Market status changes

3. **Stream Data**: Watch real-time updates with automatic price conversion

## WebSocket URLs

- **Demo**: `wss://demo-api.kalshi.co/trade-api/ws/v2`
- **Production**: `wss://api.elections.kalshi.com/trade-api/ws/v2`

## Authentication

The script uses RSA-PSS signature authentication as required by Kalshi:
- Signs requests with your private key
- Includes timestamp for replay protection
- Sends signed headers during WebSocket handshake

## Price Conversion

All prices from Kalshi are in centi-cents (1/10000 of a dollar):
- The script automatically converts to dollars for display
- Example: 5000 centi-cents = $0.50

## Notes

- Without authentication, you can only access public market data
- Authentication tokens expire every 30 minutes
- The script includes automatic reconnection logic
- Use Ctrl+C to stop streaming