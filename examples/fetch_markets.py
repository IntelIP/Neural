"""
Example: Fetch markets using the HTTP client
"""

import logging

from data_pipeline.data_sources.kalshi import KalshiClient
from data_pipeline.utils import setup_logging


def main():
    # Setup logging
    setup_logging(level="INFO")
    logger = logging.getLogger(__name__)

    # Create HTTP client
    client = KalshiClient()

    try:
        # Get first 10 markets
        logger.info("Fetching first 10 markets...")
        response = client.get_markets(limit=10)

        markets = response.get("markets", [])
        logger.info(f"Found {len(markets)} markets")

        for market in markets:
            ticker = market.get("ticker")
            title = market.get("title")
            status = market.get("status")
            logger.info(f"  {ticker}: {title} (Status: {status})")

        # Get all markets with pagination
        logger.info("\nFetching all markets with pagination...")
        all_markets = client.get_all_markets(batch_size=100)
        logger.info(f"Total markets found: {len(all_markets)}")

    finally:
        client.close()


if __name__ == "__main__":
    main()
