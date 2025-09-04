"""
Kalshi WebSocket Infrastructure - Main Entry Point
"""

import asyncio
import argparse
import logging
from typing import List, Optional

from src.data_pipeline.streaming import KalshiWebSocket
from src.data_pipeline.data_sources.kalshi import KalshiClient
from src.data_pipeline.utils import setup_logging


async def stream_markets(markets: List[str], duration: Optional[int] = None):
    """
    Stream real-time market data
    
    Args:
        markets: List of market tickers to stream
        duration: Optional duration in seconds (runs forever if None)
    """
    logger = logging.getLogger(__name__)
    
    # Create WebSocket client
    ws = KalshiWebSocket()
    
    try:
        # Connect and subscribe
        await ws.connect()
        await ws.subscribe_markets(markets)
        
        if duration:
            logger.info(f"Streaming {len(markets)} markets for {duration} seconds...")
            await asyncio.sleep(duration)
        else:
            logger.info(f"Streaming {len(markets)} markets (press Ctrl+C to stop)...")
            await ws.run_forever()
            
    except KeyboardInterrupt:
        logger.info("Streaming stopped by user")
    finally:
        await ws.disconnect()


def list_markets(limit: int = 100):
    """
    List available markets
    
    Args:
        limit: Maximum number of markets to display
    """
    logger = logging.getLogger(__name__)
    
    client = KalshiClient()
    
    try:
        response = client.get_markets(limit=limit)
        markets = response.get('markets', [])
        
        logger.info(f"Found {len(markets)} markets:")
        for market in markets:
            ticker = market.get('ticker')
            title = market.get('title')
            status = market.get('status')
            print(f"  {ticker}: {title} (Status: {status})")
            
    finally:
        client.close()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Kalshi WebSocket Infrastructure")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Stream command
    stream_parser = subparsers.add_parser("stream", help="Stream market data")
    stream_parser.add_argument(
        "markets",
        nargs="+",
        help="Market tickers to stream"
    )
    stream_parser.add_argument(
        "--duration",
        type=int,
        help="Duration in seconds (runs forever if not specified)"
    )
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available markets")
    list_parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of markets to display"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    
    if args.command == "stream":
        asyncio.run(stream_markets(args.markets, args.duration))
    elif args.command == "list":
        list_markets(args.limit)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()