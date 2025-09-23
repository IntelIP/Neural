# Kalshi Integration Guide

## Overview

The Neural SDK now includes full integration with Kalshi's prediction markets API, enabling you to:
- Fetch real-time market data for CFB and NFL games
- Analyze sentiment and identify trading opportunities
- Execute trades programmatically with proper risk management

## Architecture

```
neural/
├── kalshi/
│   ├── client.py           # High-level Kalshi API client
│   ├── markets.py          # Market data structures
│   └── fees.py             # Fee calculations & Kelly sizing
│
neural_sdk/
├── data_sources/
│   └── kalshi/
│       └── rest_adapter.py # Low-level REST API adapter
│
examples/
└── sports_sentiment_trading.py  # Complete trading example
```

## Setup

### 1. API Credentials

You need two things from Kalshi:
1. **API Key**: Your Kalshi API key
2. **RSA Private Key**: For request signing

Set these as environment variables:

```bash
export KALSHI_API_KEY="your_api_key_here"
export KALSHI_PRIVATE_KEY_PATH="/path/to/your/private_key.pem"
```

### 2. Installation

The Kalshi integration is included in the Neural SDK. No additional packages needed.

## Usage

### Basic Market Fetching

```python
from neural.kalshi import KalshiClient

# Initialize client (uses environment variables)
client = KalshiClient()

# Fetch CFB markets
cfb_markets = await client.get_cfb_markets()

# Fetch NFL markets
nfl_markets = await client.get_nfl_markets()

# Get specific market
market = await client.get_market_by_ticker('NCAAFSPREAD-24DEC14-MICHIGAN-OHIOSTATE-7')
```

### Real Kalshi Ticker Format

Kalshi uses specific ticker formats:

**CFB Markets:**
- Spread: `NCAAFSPREAD-24DEC14-MICHIGAN-OHIOSTATE-7`
- Win: `NCAAFWIN-24DEC14-TEXAS-OKLAHOMA`

**NFL Markets:**
- Win: `NFLWIN-24DEC15-COWBOYS-EAGLES`
- Spread: `NFLSPREAD-24DEC15-CHIEFS-BILLS-3`

### Sports Sentiment Trading Example

Run the complete example:

```bash
# Run for both CFB and NFL
python examples/sports_sentiment_trading.py --sport both

# CFB only
python examples/sports_sentiment_trading.py --sport cfb

# NFL only
python examples/sports_sentiment_trading.py --sport nfl
```

#### Demo Mode

If you don't have API credentials yet, the example runs in demo mode showing:
- Real ticker formats
- Example market data
- Sentiment analysis simulation
- Signal generation with Kelly sizing

#### With Real API

When you have credentials, the system will:
1. Connect to Kalshi API
2. Fetch live market data
3. Analyze Twitter sentiment (if Twitter API configured)
4. Generate trading signals based on edge detection
5. Calculate position sizes using Kelly Criterion

### Strategy Components

#### 1. Market Analysis

```python
from neural.kalshi import KalshiClient, calculate_expected_value

client = KalshiClient()
market = await client.get_market_by_ticker('NCAAFSPREAD-24DEC14-MICHIGAN-OHIOSTATE-7')

yes_price = market['yes_price'] / 100  # Convert to probability
no_price = market['no_price'] / 100

# Calculate expected value
ev = calculate_expected_value(
    win_prob=0.65,  # Your estimated probability
    market_price=yes_price,
    payout=1.0
)
```

#### 2. Position Sizing

```python
from neural.kalshi import calculate_kelly_fraction

# Calculate optimal position size
kelly = calculate_kelly_fraction(
    win_prob=0.65,
    win_amount=1.0,
    loss_amount=1.0
)

# Apply safety factor (Quarter Kelly)
position_size = kelly * 0.25
```

#### 3. Fee Calculations

```python
from neural.kalshi import calculate_kalshi_fee

# Kalshi fee formula: 0.07 × P × (1-P)
fee = calculate_kalshi_fee(price=0.45)  # $0.45 = 45 cents
```

## Architecture Details

### KalshiClient

High-level client providing:
- Async/await support
- Automatic pagination
- Market filtering and search
- Team extraction from tickers
- Error handling

### KalshiRESTAdapter

Low-level REST adapter with:
- RSA-PSS signature authentication
- Request signing
- Rate limiting
- Retry logic
- WebSocket support (for real-time data)

### SportsSentimentStrategy

Complete trading strategy implementing:
- Sentiment analysis (Twitter/mock)
- Edge detection (minimum 5% default)
- Kelly Criterion position sizing
- Risk limits (max 10% per position)
- Signal generation with metadata

## API Methods

### Client Methods

- `get_cfb_markets(week=None)` - Fetch CFB markets
- `get_nfl_markets(week=None)` - Fetch NFL markets
- `get_market_by_ticker(ticker)` - Get specific market
- `get_market_orderbook(ticker)` - Get orderbook depth
- `get_market_history(ticker, limit=100)` - Get trade history
- `search_markets(query, status='open')` - Search markets

### Market Response Format

```python
{
    'ticker': 'NCAAFSPREAD-24DEC14-MICHIGAN-OHIOSTATE-7',
    'title': 'Will Michigan beat Ohio State by more than 7 points?',
    'yes_price': 45,      # Cents
    'no_price': 55,       # Cents
    'volume': 125000,     # Total volume
    'open_interest': 5000,
    'expiration': '2024-12-14T23:59:59Z',
    'status': 'open',
    'spread': 7.0,        # Extracted from ticker
    'teams': {
        'home': 'OHIOSTATE',
        'away': 'MICHIGAN'
    }
}
```

## Best Practices

1. **Risk Management**
   - Use Kelly Criterion with safety factor (1/4 Kelly)
   - Set maximum position limits (10% recommended)
   - Require minimum edge (5% default)

2. **API Usage**
   - Cache frequently accessed markets
   - Use pagination for large result sets
   - Handle rate limits gracefully

3. **Strategy Development**
   - Backtest with historical data
   - Start with paper trading
   - Monitor sentiment confidence levels
   - Track performance metrics

## Troubleshooting

### Common Issues

1. **Import Error: No module named 'neural_sdk'**
   - Ensure you're in the project root
   - Add project to Python path

2. **Authentication Failed**
   - Verify API key is correct
   - Check private key file path and permissions
   - Ensure key is in PEM format

3. **No Markets Found**
   - Check market status (may be closed)
   - Verify date ranges
   - Some markets may not be available yet

## Next Steps

1. Get Kalshi API credentials from https://kalshi.com/api
2. Set up Twitter API for real sentiment analysis
3. Run backtests with historical data
4. Implement additional strategies (momentum, arbitrage, etc.)
5. Add visualization dashboard for monitoring

## Support

For issues or questions:
- Check examples in `/examples/sports_sentiment_trading.py`
- Review test files in `/tests/`
- Consult Kalshi API docs at https://docs.kalshi.com