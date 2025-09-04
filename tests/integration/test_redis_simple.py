#!/usr/bin/env python3
"""Simple Redis test - unbuffered output"""

import asyncio
import json
import sys
import redis.asyncio as redis
from neural_sdk.data_pipeline.orchestration.redis_event_publisher import RedisPublisher

# Force unbuffered output
sys.stdout = sys.stderr

async def test_pub_sub():
    """Test publish and subscribe in same process"""
    
    print("Starting Redis pub/sub test...")
    
    # Create publisher
    publisher = RedisPublisher()
    await publisher.connect()
    print("✓ Publisher connected")
    
    # Create subscriber
    redis_client = redis.from_url("redis://localhost:6379")
    pubsub = redis_client.pubsub()
    await pubsub.subscribe("kalshi:markets", "kalshi:signals")
    print("✓ Subscriber connected to channels")
    
    # Publish test messages
    print("\nPublishing test messages...")
    
    await publisher.publish_market_update({
        "market_ticker": "TEST-MARKET-1",
        "yes_price": 0.65,
        "no_price": 0.35,
        "volume": 1000
    })
    print("  → Published market update")
    
    await publisher.publish_signal({
        "action": "BUY",
        "market_ticker": "TEST-MARKET-1",
        "confidence": 0.85
    })
    print("  → Published signal")
    
    # Read messages
    print("\nReading messages...")
    message_count = 0
    
    # Use get_message with timeout instead of listen
    for _ in range(10):  # Try up to 10 times
        message = await pubsub.get_message(timeout=0.1)
        if message and message['type'] == 'message':
            channel = message['channel'].decode('utf-8')
            data = json.loads(message['data'])
            print(f"\n✓ Received on {channel}:")
            print(f"  Type: {data.get('type')}")
            print(f"  Data: {json.dumps(data.get('data', {}), indent=2)}")
            message_count += 1
    
    print(f"\n✓ Total messages received: {message_count}")
    
    # Cleanup
    await pubsub.unsubscribe()
    await redis_client.close()
    await publisher.disconnect()
    
    return message_count > 0

async def test_stream_manager_redis():
    """Test StreamManager Redis integration"""
    from neural_sdk.data_pipeline.orchestration.unified_stream_manager import StreamManager
    
    print("\nTesting StreamManager with Redis...")
    
    # Create subscriber first
    redis_client = redis.from_url("redis://localhost:6379")
    pubsub = redis_client.pubsub()
    await pubsub.subscribe("kalshi:markets", "kalshi:signals", "espn:games")
    print("✓ Subscriber ready")
    
    # Create and start manager
    manager = StreamManager()
    await manager.initialize()
    
    # Start just the Redis bridge (not full stream)
    await manager.redis_bridge.start()
    print("✓ Redis bridge started")
    
    # Manually emit a test event
    from neural_sdk.data_pipeline.orchestration.unified_stream_manager import UnifiedEvent, EventType, DataSource
    from datetime import datetime
    
    test_event = UnifiedEvent(
        type=EventType.PRICE_UPDATE,
        source=DataSource.KALSHI,
        market_ticker="TEST-SM-MARKET",
        data={
            "market_ticker": "TEST-SM-MARKET",
            "yes_price": 0.75,
            "no_price": 0.25
        },
        impact_score=0.8,  # High impact to trigger signal
        timestamp=datetime.utcnow()
    )
    
    await manager._emit_event(test_event)
    print("✓ Emitted test event")
    
    # Read messages
    print("\nChecking for messages...")
    message_count = 0
    
    for _ in range(10):
        message = await pubsub.get_message(timeout=0.1)
        if message and message['type'] == 'message':
            channel = message['channel'].decode('utf-8')
            data = json.loads(message['data'])
            print(f"✓ Received on {channel}: {data.get('type')}")
            message_count += 1
    
    print(f"✓ Messages from StreamManager: {message_count}")
    
    # Cleanup
    await manager.redis_bridge.stop()
    await pubsub.unsubscribe()
    await redis_client.close()
    
    return message_count > 0

async def main():
    """Run all tests"""
    print("=" * 50)
    print("REDIS INTEGRATION TEST")
    print("=" * 50)
    
    # Test 1: Basic pub/sub
    test1_passed = await test_pub_sub()
    
    # Test 2: StreamManager integration
    test2_passed = await test_stream_manager_redis()
    
    # Results
    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
    print(f"Basic Pub/Sub: {'✓ PASSED' if test1_passed else '✗ FAILED'}")
    print(f"StreamManager: {'✓ PASSED' if test2_passed else '✗ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n✅ ALL TESTS PASSED - Redis integration working!")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))