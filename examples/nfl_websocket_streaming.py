#!/usr/bin/env python3
"""
NFL WebSocket Streaming Example

This example demonstrates how to use the Neural SDK's WebSocket functionality
to stream live NFL market data and handle real-time price updates.

Features demonstrated:
- Creating WebSocket connections
- Subscribing to NFL markets
- Handling market data events
- Game-specific streaming
- Team-specific filtering
"""

import asyncio
import logging
from typing import Dict, Any

from neural_sdk import NeuralSDK

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def basic_websocket_example():
    """Basic WebSocket streaming example."""
    
    logger.info("=== Basic WebSocket Example ===")
    
    # Initialize SDK
    sdk = NeuralSDK.from_env()
    
    # Create WebSocket client
    websocket = sdk.create_websocket()
    
    # Set up event handlers
    @websocket.on_market_data
    async def handle_market_updates(market_data: Dict[str, Any]):
        """Handle real-time market data updates."""
        ticker = market_data.get('market_ticker', 'Unknown')
        yes_price = market_data.get('yes_price', 0)
        volume = market_data.get('volume', 0)
        
        logger.info(f"📊 {ticker}: ${yes_price:.4f} (Volume: {volume})")
    
    @websocket.on_trade
    async def handle_trades(trade_data: Dict[str, Any]):
        """Handle trade executions."""
        ticker = trade_data.get('market_ticker', 'Unknown')
        size = trade_data.get('size', 0)
        price = trade_data.get('price', 0)
        
        logger.info(f"💰 Trade: {ticker} - {size} contracts @ ${price:.4f}")
    
    @websocket.on_connection
    async def handle_connection_events(event_data: Dict[str, Any]):
        """Handle connection status changes."""
        status = event_data.get('status', 'unknown')
        logger.info(f"🔌 Connection: {status}")
    
    try:
        # Connect to WebSocket
        await websocket.connect()
        
        # Subscribe to NFL markets (pattern matching)
        await websocket.subscribe_markets(['KXNFLGAME*'])
        
        # Stream for 30 seconds
        logger.info("🎮 Streaming NFL markets for 30 seconds...")
        await asyncio.sleep(30)
        
        # Get status
        status = websocket.get_status()
        logger.info(f"📈 WebSocket Status: {status}")
        
    finally:
        await websocket.disconnect()


async def nfl_game_streaming_example():
    """NFL game-specific streaming example."""
    
    logger.info("=== NFL Game Streaming Example ===")
    
    # Initialize SDK
    sdk = NeuralSDK.from_env()
    
    # Create NFL-specific stream
    nfl_stream = sdk.create_nfl_stream()
    
    try:
        # Connect
        await nfl_stream.connect()
        
        # Subscribe to specific game (you'd get this from current NFL schedule)
        game_id = "25SEP04DALPHI"  # Eagles vs Cowboys example
        await nfl_stream.subscribe_to_game(game_id)
        
        # Stream for 60 seconds
        logger.info(f"🏈 Streaming NFL game {game_id} for 60 seconds...")
        await asyncio.sleep(60)
        
        # Get game summary
        summary = nfl_stream.get_game_summary(game_id)
        if summary:
            logger.info("🎯 Game Summary:")
            logger.info(f"  Teams: {summary['away_team']} @ {summary['home_team']}")
            logger.info(f"  Markets: {summary['markets_count']}")
            logger.info(f"  Win Probability: {summary.get('win_probability', 'N/A')}")
        
        # Get active games
        active_games = nfl_stream.get_active_games()
        logger.info(f"📋 Active Games: {active_games}")
        
    finally:
        await nfl_stream.disconnect()


async def team_specific_streaming_example():
    """Team-specific streaming example."""
    
    logger.info("=== Team Streaming Example ===")
    
    # Initialize SDK
    sdk = NeuralSDK.from_env()
    
    # Create NFL stream
    nfl_stream = sdk.create_nfl_stream()
    
    # Track team-specific data
    eagles_data = {}
    
    @nfl_stream.websocket.on_market_data
    async def track_eagles_markets(market_data: Dict[str, Any]):
        """Track Eagles-specific market data."""
        ticker = market_data.get('market_ticker', '')
        
        if 'PHI' in ticker.upper():
            eagles_data[ticker] = {
                'yes_price': market_data.get('yes_price'),
                'volume': market_data.get('volume', 0),
                'timestamp': market_data.get('timestamp')
            }
            
            logger.info(f"🦅 Eagles Market: {ticker} = ${market_data.get('yes_price', 0):.4f}")
    
    try:
        # Connect
        await nfl_stream.connect()
        
        # Subscribe to Eagles markets
        await nfl_stream.subscribe_to_team("PHI")
        
        # Stream for 45 seconds
        logger.info("🦅 Streaming Eagles markets for 45 seconds...")
        await asyncio.sleep(45)
        
        # Show Eagles market summary
        logger.info(f"📊 Eagles Markets Tracked: {len(eagles_data)}")
        for ticker, data in eagles_data.items():
            logger.info(f"  {ticker}: ${data['yes_price']:.4f} (Vol: {data['volume']})")
        
    finally:
        await nfl_stream.disconnect()


async def sdk_integrated_streaming_example():
    """Example using SDK's integrated streaming methods."""
    
    logger.info("=== SDK Integrated Streaming Example ===")
    
    # Initialize SDK
    sdk = NeuralSDK.from_env()
    
    # Use SDK's built-in streaming event handlers
    @sdk.on_market_data
    async def handle_sdk_market_data(market_data):
        """Handle market data through SDK."""
        ticker = market_data.get('market_ticker', 'Unknown')
        yes_price = market_data.get('yes_price', 0)
        
        # Only log significant price movements
        if yes_price < 0.1 or yes_price > 0.9:
            logger.info(f"🚨 Extreme Price: {ticker} = ${yes_price:.4f}")
    
    @sdk.on_trade
    async def handle_sdk_trades(trade_data):
        """Handle trades through SDK."""
        ticker = trade_data.get('market_ticker', 'Unknown')
        size = trade_data.get('size', 0)
        
        if size > 100:  # Large trades
            logger.info(f"🐋 Large Trade: {ticker} - {size} contracts")
    
    try:
        # Start streaming using SDK's convenience method
        await sdk.start_streaming(['KXNFLGAME*'])
        
        # Stream for 30 seconds
        logger.info("⚡ Using SDK integrated streaming for 30 seconds...")
        await asyncio.sleep(30)
        
    finally:
        await sdk.stop_streaming()


async def price_alert_example():
    """Example with price alerts and filtering."""
    
    logger.info("=== Price Alert Example ===")
    
    # Initialize SDK
    sdk = NeuralSDK.from_env()
    websocket = sdk.create_websocket()
    
    # Track price history for alerts
    price_history = {}
    alert_threshold = 0.05  # 5% price movement
    
    @websocket.on_market_data
    async def price_alert_handler(market_data: Dict[str, Any]):
        """Handle price alerts."""
        ticker = market_data.get('market_ticker', '')
        yes_price = market_data.get('yes_price')
        
        if not yes_price or 'NFL' not in ticker.upper():
            return
        
        # Check for significant price movement
        if ticker in price_history:
            prev_price = price_history[ticker]
            price_change = abs(yes_price - prev_price)
            
            if price_change >= alert_threshold:
                direction = "📈 UP" if yes_price > prev_price else "📉 DOWN"
                logger.warning(
                    f"{direction} ALERT: {ticker} moved {price_change:.3f} "
                    f"({prev_price:.3f} → {yes_price:.3f})"
                )
        
        price_history[ticker] = yes_price
    
    try:
        # Connect and subscribe
        await websocket.connect()
        await websocket.subscribe_markets(['KXNFLGAME*'])
        
        # Stream with alerts for 60 seconds
        logger.info("🚨 Monitoring for price alerts for 60 seconds...")
        await asyncio.sleep(60)
        
        logger.info(f"📊 Tracked {len(price_history)} NFL markets")
        
    finally:
        await websocket.disconnect()


async def main():
    """Run all examples."""
    
    print("=" * 60)
    print("Neural SDK WebSocket Streaming Examples")
    print("=" * 60)
    
    try:
        # Run examples sequentially
        await basic_websocket_example()
        await asyncio.sleep(2)
        
        await nfl_game_streaming_example()
        await asyncio.sleep(2)
        
        await team_specific_streaming_example()
        await asyncio.sleep(2)
        
        await sdk_integrated_streaming_example()
        await asyncio.sleep(2)
        
        await price_alert_example()
        
        print("\n✅ All examples completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        logger.error(f"Example error: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
