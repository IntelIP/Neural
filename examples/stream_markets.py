"""
Example: Stream real-time market data from Kalshi
"""

import asyncio
import logging

from data_pipeline.streaming import KalshiWebSocket
from data_pipeline.utils import setup_logging


async def main():
    # Setup logging
    setup_logging(level="INFO")
    logger = logging.getLogger(__name__)

    # Create WebSocket client
    ws = KalshiWebSocket()

    try:
        # Connect to WebSocket
        await ws.connect()

        # Subscribe to some test markets
        test_markets = ["KXQUICKSETTLE-23DEC29-T0001", "KXQUICKSETTLE-23DEC29-T0002"]

        await ws.subscribe_markets(test_markets)

        # Run for 60 seconds
        logger.info("Streaming market data for 60 seconds...")
        await asyncio.sleep(60)

    finally:
        # Disconnect
        await ws.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
