#!/usr/bin/env python3
"""
Test Redis Integration - Verify WebSocket to Redis pipeline
Simple test to ensure data flows from WebSockets through Redis to agents
"""

import asyncio
import json
import logging
import redis.asyncio as redis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def redis_subscriber():
    """Subscribe to Redis channels and print messages"""
    redis_url = "redis://localhost:6379"
    client = redis.from_url(redis_url)
    
    # Subscribe to all channels
    channels = [
        "kalshi:markets",
        "kalshi:trades", 
        "kalshi:orderbook",
        "kalshi:signals",
        "espn:games",
        "twitter:sentiment"
    ]
    
    pubsub = client.pubsub()
    await pubsub.subscribe(*channels)
    
    logger.info(f"Subscribed to channels: {channels}")
    logger.info("Listening for messages... (Press Ctrl+C to stop)")
    
    try:
        async for message in pubsub.listen():
            if message['type'] == 'message':
                channel = message['channel'].decode('utf-8')
                data = json.loads(message['data'])
                
                print(f"\n{'='*50}")
                print(f"Channel: {channel}")
                print(f"Timestamp: {data.get('timestamp', 'N/A')}")
                print(f"Type: {data.get('type', 'N/A')}")
                
                # Pretty print the data
                if 'data' in data:
                    print(f"Data: {json.dumps(data['data'], indent=2)[:200]}...")
                
                print(f"{'='*50}")
                
    except KeyboardInterrupt:
        logger.info("Stopping subscriber...")
    finally:
        await pubsub.unsubscribe(*channels)
        await client.close()


async def test_publisher():
    """Publish test messages to verify pipeline"""
    from data_pipeline.orchestration.redis_event_publisher import RedisPublisher
    
    publisher = RedisPublisher()
    await publisher.connect()
    
    # Test market update
    await publisher.publish_market_update({
        "market_ticker": "TEST-MARKET",
        "yes_price": 0.65,
        "no_price": 0.35,
        "volume": 10000,
        "test": True
    })
    logger.info("Published test market update")
    
    # Test trade
    await publisher.publish_trade({
        "market_ticker": "TEST-MARKET",
        "side": "yes",
        "price": 0.65,
        "quantity": 100,
        "test": True
    })
    logger.info("Published test trade")
    
    # Test ESPN update
    await publisher.publish_espn_update({
        "game_id": "TEST-GAME",
        "home_score": 21,
        "away_score": 14,
        "quarter": 2,
        "test": True
    })
    logger.info("Published test ESPN update")
    
    # Test signal
    await publisher.publish_signal({
        "market_ticker": "TEST-MARKET",
        "action": "BUY",
        "confidence": 0.85,
        "reason": "Test signal",
        "test": True
    })
    logger.info("Published test signal")
    
    await asyncio.sleep(1)  # Give subscriber time to receive
    await publisher.disconnect()


async def test_stream_manager():
    """Test StreamManager with Redis integration"""
    from data_pipeline.orchestration.unified_stream_manager import StreamManager
    
    # Create manager
    manager = StreamManager()
    
    # Initialize and start
    await manager.initialize()
    await manager.start()
    
    logger.info("StreamManager started with Redis bridge")
    
    # Track a test market
    await manager.track_market(
        market_ticker="INXD-25JAN02-B4.99",
        game_id="test_game_123",
        sport="football",
        home_team="Team A",
        away_team="Team B"
    )
    
    # Run for 30 seconds
    await asyncio.sleep(30)
    
    # Stop
    await manager.stop()
    logger.info("StreamManager stopped")


async def main():
    """Main test function"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python test_redis_integration.py [subscriber|publisher|stream]")
        print("  subscriber - Listen to Redis channels")
        print("  publisher  - Send test messages")
        print("  stream     - Test full StreamManager integration")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    if mode == "subscriber":
        await redis_subscriber()
    elif mode == "publisher":
        await test_publisher()
    elif mode == "stream":
        await test_stream_manager()
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
