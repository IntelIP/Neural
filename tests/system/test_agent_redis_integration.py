#!/usr/bin/env python3
"""
Test Agent Redis Integration
Comprehensive test of agents consuming from Redis streams
"""

import asyncio
import logging
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_full_pipeline():
    """Test the complete pipeline: StreamManager → Redis → Agents"""
    
    logger.info("=" * 60)
    logger.info("AGENT REDIS INTEGRATION TEST")
    logger.info("=" * 60)
    
    # Import components
    from data_pipeline.orchestration.unified_stream_manager import StreamManager
    from agent_consumers.DataCoordinator.data_coordinator import DataCoordinatorAgent
    from agent_consumers.MarketEngineer.market_engineer import MarketEngineerAgent
    from agent_consumers.base_consumer import AgentRedisOrchestrator
    
    # 1. Start StreamManager with Redis bridge
    logger.info("\n1. Starting StreamManager with Redis bridge...")
    stream_manager = StreamManager()
    await stream_manager.initialize()
    await stream_manager.redis_bridge.start()
    logger.info("✓ StreamManager Redis bridge started")
    
    # 2. Start DataCoordinator with Redis consumer
    logger.info("\n2. Starting DataCoordinator with Redis consumer...")
    data_coordinator = DataCoordinatorAgent()
    await data_coordinator.start()
    logger.info("✓ DataCoordinator started with Redis consumer")
    
    # 3. Start MarketEngineer with Redis consumer
    logger.info("\n3. Starting MarketEngineer with Redis consumer...")
    market_engineer = MarketEngineerAgent()
    await market_engineer.start()
    logger.info("✓ MarketEngineer started with Redis consumer")
    
    # 4. Track a test market
    logger.info("\n4. Tracking test market...")
    await data_coordinator.track_market(
        market_ticker="TEST-MARKET-2025",
        game_id="test_game_123",
        sport="football",
        home_team="Team A",
        away_team="Team B"
    )
    logger.info("✓ Market tracking initiated")
    
    # 5. Simulate market data
    logger.info("\n5. Simulating market data...")
    from data_pipeline.orchestration.unified_stream_manager import UnifiedEvent, EventType, DataSource
    
    # Simulate price update
    test_event = UnifiedEvent(
        type=EventType.PRICE_UPDATE,
        source=DataSource.KALSHI,
        market_ticker="TEST-MARKET-2025",
        data={
            "market_ticker": "TEST-MARKET-2025",
            "yes_price": 0.45,
            "no_price": 0.52,  # Arbitrage opportunity!
            "volume": 10000
        },
        impact_score=0.8,
        timestamp=datetime.utcnow()
    )
    
    await stream_manager._emit_event(test_event)
    logger.info("✓ Price update emitted (arbitrage opportunity)")
    
    # Wait for processing
    await asyncio.sleep(2)
    
    # 6. Check agent statistics
    logger.info("\n6. Agent Statistics:")
    logger.info("-" * 40)
    
    # DataCoordinator stats
    dc_status = await data_coordinator.get_status()
    logger.info(f"DataCoordinator:")
    logger.info(f"  Events received: {dc_status['events_received']}")
    logger.info(f"  Events routed: {dc_status['events_routed']}")
    logger.info(f"  Tracked markets: {len(dc_status['tracked_markets'])}")
    
    if hasattr(data_coordinator, 'redis_consumer'):
        redis_stats = data_coordinator.redis_consumer.get_stats()
        logger.info(f"  Redis messages: {redis_stats['messages_received']}")
    
    # MarketEngineer stats
    if hasattr(market_engineer, 'redis_consumer'):
        me_stats = market_engineer.redis_consumer.get_stats()
        logger.info(f"\nMarketEngineer:")
        logger.info(f"  Messages received: {me_stats['messages_received']}")
        logger.info(f"  Messages processed: {me_stats['messages_processed']}")
        logger.info(f"  Opportunities found: {market_engineer.redis_consumer.opportunities_found}")
    
    # 7. Simulate ESPN game event
    logger.info("\n7. Simulating ESPN game event...")
    game_event = UnifiedEvent(
        type=EventType.BIG_PLAY,
        source=DataSource.ESPN,
        market_ticker="TEST-MARKET-2025",
        game_id="test_game_123",
        data={
            "game_id": "test_game_123",
            "event": "TOUCHDOWN",
            "home_score": 21,
            "away_score": 14,
            "description": "Team A scores touchdown!"
        },
        impact_score=0.9,
        timestamp=datetime.utcnow()
    )
    
    await stream_manager._emit_event(game_event)
    logger.info("✓ ESPN touchdown event emitted")
    
    # Wait for processing
    await asyncio.sleep(2)
    
    # 8. Final statistics
    logger.info("\n8. Final Statistics:")
    logger.info("-" * 40)
    
    stream_stats = stream_manager.get_statistics()
    logger.info(f"StreamManager:")
    logger.info(f"  Events processed: {stream_stats['events_processed']}")
    logger.info(f"  Markets tracked: {len(stream_stats['markets'])}")
    
    if stream_manager.redis_publisher:
        redis_pub_stats = stream_manager.redis_publisher.get_stats()
        logger.info(f"  Redis messages published: {redis_pub_stats['messages_published']}")
    
    # 9. Cleanup
    logger.info("\n9. Cleaning up...")
    await data_coordinator.stop()
    await market_engineer.stop()
    await stream_manager.redis_bridge.stop()
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ INTEGRATION TEST COMPLETE")
    logger.info("=" * 60)
    
    return True


async def test_redis_orchestrator():
    """Test the AgentRedisOrchestrator for running multiple agents"""
    
    logger.info("\n" + "=" * 60)
    logger.info("TESTING AGENT ORCHESTRATOR")
    logger.info("=" * 60)
    
    from agent_consumers.base_consumer import AgentRedisOrchestrator, BaseAgentRedisConsumer
    from data_pipeline.orchestration.redis_event_publisher import RedisPublisher
    
    # Create test consumers
    class TestConsumer1(BaseAgentRedisConsumer):
        async def process_message(self, channel: str, data: dict):
            logger.info(f"Consumer1 processed: {channel}")
    
    class TestConsumer2(BaseAgentRedisConsumer):
        async def process_message(self, channel: str, data: dict):
            logger.info(f"Consumer2 processed: {channel}")
    
    # Create orchestrator
    orchestrator = AgentRedisOrchestrator()
    
    # Register consumers
    consumer1 = TestConsumer1("TestAgent1")
    consumer2 = TestConsumer2("TestAgent2")
    
    # Connect first
    await consumer1.connect()
    await consumer2.connect()
    
    # Subscribe to channels
    await consumer1.subscribe(["kalshi:markets"])
    await consumer2.subscribe(["kalshi:signals"])
    
    orchestrator.register_consumer(consumer1)
    orchestrator.register_consumer(consumer2)
    
    # Start consuming
    import asyncio
    task1 = asyncio.create_task(consumer1.start_consuming())
    task2 = asyncio.create_task(consumer2.start_consuming())
    logger.info("✓ Orchestrator started all consumers")
    
    # Publish test messages
    publisher = RedisPublisher()
    await publisher.connect()
    
    await publisher.publish_market_update({"market_ticker": "TEST", "yes_price": 0.5})
    await publisher.publish_signal({"action": "BUY", "confidence": 0.9})
    
    # Wait for processing
    await asyncio.sleep(1)
    
    # Check health
    health = await orchestrator.health_check_all()
    logger.info(f"Health check: {health}")
    
    # Get stats
    stats = orchestrator.get_all_stats()
    for agent_name, agent_stats in stats.items():
        logger.info(f"{agent_name}: {agent_stats['messages_received']} messages")
    
    # Stop all
    task1.cancel()
    task2.cancel()
    await consumer1.disconnect()
    await consumer2.disconnect()
    await publisher.disconnect()
    
    logger.info("✓ Orchestrator test complete")


async def main():
    """Main test runner"""
    
    if len(sys.argv) > 1 and sys.argv[1] == "orchestrator":
        await test_redis_orchestrator()
    else:
        await test_full_pipeline()


if __name__ == "__main__":
    asyncio.run(main())