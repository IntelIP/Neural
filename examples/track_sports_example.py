#!/usr/bin/env python3
"""
Sports Tracking Example
Demonstrates standardized approach to track NFL and CFP markets
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_pipeline.orchestration.unified_stream_manager import StreamManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def track_nfl_example():
    """Example: Track all NFL markets"""
    logger.info("=== NFL Market Tracking Example ===")

    # Initialize stream manager
    stream_manager = StreamManager()

    try:
        # Start the stream manager
        await stream_manager.start()

        # Track all NFL markets (limit to 10 for demo)
        tracked_markets = await stream_manager.track_nfl_markets(limit=10)
        logger.info(f"Tracking {len(tracked_markets)} NFL markets")

        # Run for 30 seconds to see live data
        logger.info("Streaming live data for 30 seconds...")
        await asyncio.sleep(30)

    finally:
        await stream_manager.stop()


async def track_eagles_cowboys_example():
    """Example: Track specific Eagles vs Cowboys game"""
    logger.info("=== Eagles vs Cowboys Game Tracking ===")

    # Initialize stream manager
    stream_manager = StreamManager()

    try:
        # Start the stream manager
        await stream_manager.start()

        # Track Eagles vs Cowboys markets specifically
        tracked_markets = await stream_manager.track_game_markets("nfl", "PHI", "DAL")
        logger.info(f"Tracking {len(tracked_markets)} Eagles vs Cowboys markets")

        for ticker in tracked_markets:
            logger.info(f"  - {ticker}")

        # Run for 30 seconds to see live data
        logger.info("Streaming live Eagles vs Cowboys data for 30 seconds...")
        await asyncio.sleep(30)

    finally:
        await stream_manager.stop()


async def track_team_example():
    """Example: Track all markets for a specific team"""
    logger.info("=== Team Market Tracking (Eagles) ===")

    # Initialize stream manager
    stream_manager = StreamManager()

    try:
        # Start the stream manager
        await stream_manager.start()

        # Track all Eagles markets
        tracked_markets = await stream_manager.track_team_markets("nfl", "PHI")
        logger.info(f"Tracking {len(tracked_markets)} Eagles markets")

        # Run for 30 seconds to see live data
        logger.info("Streaming live Eagles data for 30 seconds...")
        await asyncio.sleep(30)

    finally:
        await stream_manager.stop()


async def discover_markets_example():
    """Example: Just discover markets without tracking"""
    logger.info("=== Market Discovery Example ===")

    from src.data_pipeline.data_sources.kalshi.market_discovery import (
        discover_nfl_markets,
    )

    # Discover NFL markets
    markets = await discover_nfl_markets()
    logger.info(f"Found {len(markets)} NFL markets")

    # Show first 5 markets
    for market in markets[:5]:
        logger.info(f"  {market.display_name} - {market.ticker}")
        if market.yes_bid and market.yes_ask:
            logger.info(f"    Bid: ${market.yes_bid:.3f}, Ask: ${market.yes_ask:.3f}")


async def main():
    """Run examples"""
    logger.info("🏈 Sports Tracking Examples")
    logger.info("=" * 50)

    try:
        # Example 1: Market discovery only
        await discover_markets_example()
        await asyncio.sleep(2)

        # Example 2: Track specific game
        await track_eagles_cowboys_example()
        await asyncio.sleep(2)

        # Example 3: Track team markets
        await track_team_example()
        await asyncio.sleep(2)

        # Example 4: Track all NFL (uncomment to run)
        # await track_nfl_example()

    except KeyboardInterrupt:
        logger.info("Stopped by user")
    except Exception as e:
        logger.error(f"Example failed: {e}")


if __name__ == "__main__":
    print("🏈 NFL Market Tracking Examples")
    print("This demonstrates the standardized sports tracking approach")
    print("Press Ctrl+C to stop\n")

    asyncio.run(main())
