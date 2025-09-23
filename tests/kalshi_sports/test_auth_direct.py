#!/usr/bin/env python3
"""Test authenticated API directly with REST calls."""

import os
import requests
from dotenv import load_dotenv
from neural.auth.signers.kalshi import KalshiSigner
from neural.auth.env import get_api_key_id, get_private_key_material, get_base_url

# Load environment variables
load_dotenv()

def test_authenticated_markets():
    """Test markets endpoint with authentication."""

    try:
        # Get credentials
        api_key = get_api_key_id()
        private_key = get_private_key_material()

        print(f"Using API key: {api_key[:10]}...")

        # Create signer
        signer = KalshiSigner(api_key, private_key)

        # Test markets endpoint
        base_url = get_base_url()
        path = "/trade-api/v2/markets"
        url = f"{base_url}{path}"

        # Get auth headers
        headers = signer.headers("GET", path)
        headers["Accept"] = "application/json"

        print(f"\nFetching markets from: {url}")
        print("With authenticated headers...")

        # Make request
        params = {
            "limit": 100,
            "status": "open"
        }

        response = requests.get(url, headers=headers, params=params)
        print(f"Response status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            markets = data.get("markets", [])

            print(f"\nTotal markets returned: {len(markets)}")

            # Show first few titles
            print("\nFirst 10 market titles:")
            for m in markets[:10]:
                print(f"  - {m.get('title', 'N/A')}")

            # Search for football
            football = [m for m in markets if 'football' in m.get('title', '').lower() or 'nfl' in m.get('title', '').lower()]
            print(f"\nFootball markets found: {len(football)}")
            for m in football[:5]:
                print(f"  - {m.get('title', 'N/A')}")
                print(f"    Ticker: {m.get('ticker', 'N/A')}")

        else:
            print(f"Error: {response.text}")

    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have credentials set in environment or secrets folder:")
        print("  - KALSHI_API_KEY_ID")
        print("  - KALSHI_PRIVATE_KEY_PATH or KALSHI_PRIVATE_KEY_BASE64")

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING AUTHENTICATED API")
    print("=" * 60)
    test_authenticated_markets()