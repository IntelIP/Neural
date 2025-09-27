#!/usr/bin/env python3
"""Test if sports markets require authentication."""

import os
import asyncio
import pytest
from dotenv import load_dotenv
import asyncio
import pytest
import pandas as pd
from neural.data_collection.kalshi import get_nfl_games, get_cfb_games
from neural.trading import TradingClient

# Load environment variables
load_dotenv()


def test_authenticated_api():
    """Test the authenticated API directly."""
    print("Testing authenticated API...")

    try:
        # Initialize authenticated client
        client = TradingClient()

        # Try to get markets without filters first
        print("\n1. Getting ALL markets (authenticated)...")
        all_markets = client.markets.get_markets(limit=100)
        print(f"   Total markets returned: {len(all_markets.get('markets', []))}")

        # Show first few market titles
        if all_markets.get("markets"):
            print("\n   First 5 market titles:")
            for m in all_markets["markets"][:5]:
                print(f"     - {m.get('title', 'N/A')}")
                if "event_ticker" in m:
                    print(f"       Event: {m['event_ticker']}")

        # Search for football in titles
        football_markets = [
            m
            for m in all_markets.get("markets", [])
            if any(
                word in m.get("title", "").lower()
                for word in ["football", "nfl", "ravens", "detroit", "baltimore"]
            )
        ]

        print(f"\n2. Football markets found: {len(football_markets)}")
        for m in football_markets[:3]:
            print(f"   - {m.get('title', 'N/A')}")
            print(f"     Ticker: {m.get('ticker', 'N/A')}")
            print(f"     Event: {m.get('event_ticker', 'N/A')}")

# Note: Events and series endpoints not available in current TradingClient; focus on markets
print("\n3. Skipping events/series tests (not implemented in TradingClient)")

        client.close()

    except Exception as e:
        print(f"Authentication error: {e}")
        print("\nMake sure you have valid credentials in:")
        print("  - Environment variables: KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH")
        print("  - Or in secrets/ folder")


@pytest.mark.integration
async def test_nfl_integration():
    """Test NFL games integration with authentication."""
    if not os.getenv("KALSHI_API_KEY_ID"):
        pytest.skip("No Kalshi credentials provided for integration test")

    print("\nTesting NFL games integration...")

    try:
        df = await asyncio.wait_for(get_nfl_games(limit=20), timeout=15.0)

        assert not df.empty, "Expected non-empty NFL DataFrame"
        assert "ticker" in df.columns
        assert "title" in df.columns
        assert "yes_bid" in df.columns
        assert "home_team" in df.columns or "away_team" in df.columns  # At least one team parsed

        # Check for NFL content
        nfl_mask = df["title"].str.contains("NFL", case=False, na=False) | df[
            "series_ticker"
        ].str.contains("KXNFLGAME", na=False)
        nfl_df = df[nfl_mask]
        assert len(nfl_df) > 0, "Expected at least one NFL market"

        print(f"   NFL markets found: {len(nfl_df)}")
        if not nfl_df.empty:
            print("\n   Sample NFL markets:")
            for _, row in nfl_df.head(2).iterrows():
                print(f"     - {row['title']}")
                if pd.notna(row.get("home_team")):
                    print(f"       Teams: {row.get('home_team')} vs {row.get('away_team')}")

    except asyncio.TimeoutError:
        pytest.fail("Test timed out after 15 seconds")
    except Exception as e:
        pytest.fail(f"NFL integration failed: {e}")


@pytest.mark.integration
async def test_cfb_integration():
    """Test CFB games integration with authentication."""
    if not os.getenv("KALSHI_API_KEY_ID"):
        pytest.skip("No Kalshi credentials provided for integration test")

    print("\nTesting CFB games integration...")

    try:
        df = await asyncio.wait_for(get_cfb_games(limit=20), timeout=15.0)

        assert not df.empty, "Expected non-empty CFB DataFrame"
        assert "ticker" in df.columns
        assert "title" in df.columns
        assert "yes_bid" in df.columns
        assert "home_team" in df.columns or "away_team" in df.columns  # At least one team parsed

        # Check for CFB content
        cfb_mask = df["title"].str.contains("NCAA|College Football", case=False, na=False) | df[
            "series_ticker"
        ].str.contains("KXNCAAFGAME", na=False)
        cfb_df = df[cfb_mask]
        assert len(cfb_df) > 0, "Expected at least one CFB market"

        print(f"   CFB markets found: {len(cfb_df)}")
        if not cfb_df.empty:
            print("\n   Sample CFB markets:")
            for _, row in cfb_df.head(2).iterrows():
                print(f"     - {row['title']}")
                if pd.notna(row.get("home_team")):
                    print(f"       Teams: {row.get('home_team')} vs {row.get('away_team')}")

    except asyncio.TimeoutError:
        pytest.fail("Test timed out after 15 seconds")
    except Exception as e:
        pytest.fail(f"CFB integration failed: {e}")


@pytest.mark.parametrize(
    "sport_func, expected_keyword", [(get_nfl_games, "NFL"), (get_cfb_games, "NCAA")]
)
@pytest.mark.integration
async def test_sports_integration(sport_func, expected_keyword):
    """Parametrized test for sports integrations."""
    if not os.getenv("KALSHI_API_KEY_ID"):
        pytest.skip("No Kalshi credentials provided for integration test")

    df = await asyncio.wait_for(sport_func(limit=10), timeout=10.0)

    assert not df.empty
    mask = df["title"].str.contains(expected_keyword, case=False, na=False)
    assert mask.any(), f"Expected '{expected_keyword}' in titles"


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING AUTHENTICATED API FOR SPORTS MARKETS")
    print("=" * 60)

    # Test synchronous authenticated API
    test_authenticated_api()

    # Test async integrations
    asyncio.run(test_nfl_integration())
    asyncio.run(test_cfb_integration())

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
