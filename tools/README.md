# Neural SDK Tools

This directory contains utilities and verification tools for the Neural SDK.

## Verification Tools

### `verification/final_verification.py`
Comprehensive verification script that tests all critical components:
- Authentication with Kalshi API
- WebSocket connectivity
- Trading signal creation
- Risk management configuration
- NFL market subscription

Run before deploying to production:
```bash
python tools/verification/final_verification.py
```

### `verification/production_test.py`
Advanced production testing with real API calls:
- Tests actual trading system components
- Verifies strategy framework
- Confirms portfolio access
- Validates signal execution pathway

Use for thorough system validation:
```bash
python tools/verification/production_test.py
```

## Usage

These tools are designed to verify that your Neural SDK installation is working correctly with real Kalshi API credentials. Run them after:

1. Installing the SDK
2. Configuring your `.env` file
3. Before starting live trading

All tools require your `KALSHI_*` environment variables to be properly set.