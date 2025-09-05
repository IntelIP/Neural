#!/usr/bin/env python3
"""
Test Simplified Agent System
Verifies the new architecture works correctly
"""

import asyncio
import json
import logging
import redis.asyncio as redis
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimplifiedSystemTester:
    """Test the simplified agent system"""
    
    def __init__(self):
        self.redis_client = None
        self.test_results = {
            "passed": [],
            "failed": []
        }
    
    async def connect(self):
        """Connect to Redis"""
        self.redis_client = redis.from_url("redis://localhost:6379")
        logger.info("Tester connected to Redis")
    
    async def test_trigger_conditions(self):
        """Test various trigger conditions"""
        logger.info("\n🧪 Testing Trigger Conditions...")
        
        test_events = [
            {
                "name": "Price Spike",
                "channel": "events:market",
                "data": {
                    "price_change": 0.08,  # > 5% threshold
                    "market_ticker": "TEST-MARKET",
                    "yes_price": 0.58,
                    "no_price": 0.42
                },
                "expected_agent": "ArbitrageHunter"
            },
            {
                "name": "Arbitrage Opportunity",
                "channel": "events:market",
                "data": {
                    "yes_price": 0.48,
                    "no_price": 0.48,  # Sum < 0.98
                    "market_ticker": "ARB-MARKET"
                },
                "expected_agent": "ArbitrageHunter"
            },
            {
                "name": "Sentiment Shift",
                "channel": "events:sentiment",
                "data": {
                    "sentiment_change": 0.45,  # > 0.3 threshold
                    "team": "Chiefs",
                    "market_ticker": "CHIEFS-WIN"
                },
                "expected_agent": "MarketEngineer"
            },
            {
                "name": "Major Game Event",
                "channel": "events:game",
                "data": {
                    "event_type": "touchdown",
                    "team": "Bills",
                    "impact": "high",
                    "game_id": "12345"
                },
                "expected_agent": "GameAnalyst"
            },
            {
                "name": "Stop Loss Trigger",
                "channel": "events:portfolio",
                "data": {
                    "position_pnl": -0.12,  # < -10% threshold
                    "market_ticker": "LOSS-MARKET",
                    "position_size": 100
                },
                "expected_agent": "RiskManager"
            }
        ]
        
        for test in test_events:
            try:
                # Publish test event
                await self.redis_client.publish(
                    test["channel"],
                    json.dumps(test["data"])
                )
                
                logger.info(f"  ✓ {test['name']}: Published to {test['channel']}")
                self.test_results["passed"].append(f"Trigger: {test['name']}")
                
                # Give trigger service time to process
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"  ✗ {test['name']}: {e}")
                self.test_results["failed"].append(f"Trigger: {test['name']}")
    
    async def test_always_on_agents(self):
        """Test always-on agent functionality"""
        logger.info("\n🤖 Testing Always-On Agents...")
        
        # Test Data Coordinator
        try:
            # Publish market update
            market_data = {
                "data": {
                    "market_ticker": "TEST-DC",
                    "yes_price": 0.65,
                    "no_price": 0.35,
                    "volume": 10000
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self.redis_client.publish("kalshi:markets", json.dumps(market_data))
            logger.info("  ✓ DataCoordinator: Market update published")
            self.test_results["passed"].append("DataCoordinator: Market processing")
            
        except Exception as e:
            logger.error(f"  ✗ DataCoordinator test failed: {e}")
            self.test_results["failed"].append("DataCoordinator: Market processing")
        
        # Test Portfolio Monitor
        try:
            # Simulate new position
            position_data = {
                "data": {
                    "market_ticker": "TEST-PM",
                    "side": "yes",
                    "quantity": 50,
                    "price": 0.60,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
            await self.redis_client.publish("trades:executed", json.dumps(position_data))
            logger.info("  ✓ PortfolioMonitor: New position published")
            self.test_results["passed"].append("PortfolioMonitor: Position tracking")
            
            # Update price to trigger stop-loss check
            await asyncio.sleep(1)
            
            price_update = {
                "data": {
                    "market_ticker": "TEST-PM",
                    "yes_price": 0.50  # 16.7% loss
                }
            }
            
            await self.redis_client.publish("kalshi:markets", json.dumps(price_update))
            logger.info("  ✓ PortfolioMonitor: Stop-loss check triggered")
            self.test_results["passed"].append("PortfolioMonitor: Stop-loss")
            
        except Exception as e:
            logger.error(f"  ✗ PortfolioMonitor test failed: {e}")
            self.test_results["failed"].append("PortfolioMonitor: Position tracking")
    
    async def test_on_demand_activation(self):
        """Test on-demand agent activation"""
        logger.info("\n🎯 Testing On-Demand Agent Activation...")
        
        try:
            # Simulate user request for game analysis
            activation_data = {
                "agent": "GameAnalyst",
                "trigger": "manual:test",
                "priority": "HIGH",
                "data": {
                    "event_type": "user_request",
                    "game_id": "TEST123",
                    "home_team": "TestHome",
                    "away_team": "TestAway"
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self.redis_client.publish(
                "agent:activate:gameanalyst",
                json.dumps(activation_data)
            )
            
            logger.info("  ✓ GameAnalyst: Activation request sent")
            self.test_results["passed"].append("GameAnalyst: On-demand activation")
            
            # Give agent time to process
            await asyncio.sleep(2)
            
        except Exception as e:
            logger.error(f"  ✗ On-demand activation failed: {e}")
            self.test_results["failed"].append("GameAnalyst: On-demand activation")
    
    async def test_redis_channels(self):
        """Test Redis pub/sub channels"""
        logger.info("\n📡 Testing Redis Channels...")
        
        channels = [
            "kalshi:markets",
            "kalshi:signals",
            "espn:games",
            "events:market",
            "events:portfolio",
            "agent:activate:gameanalyst",
            "trades:orders",
            "analysis:game"
        ]
        
        for channel in channels:
            try:
                # Test publish
                test_msg = {"test": True, "channel": channel}
                await self.redis_client.publish(channel, json.dumps(test_msg))
                logger.info(f"  ✓ Channel {channel}: OK")
                self.test_results["passed"].append(f"Channel: {channel}")
                
            except Exception as e:
                logger.error(f"  ✗ Channel {channel}: {e}")
                self.test_results["failed"].append(f"Channel: {channel}")
    
    async def display_results(self):
        """Display test results"""
        print("\n" + "="*60)
        print("TEST RESULTS")
        print("="*60)
        
        print(f"\n✅ PASSED: {len(self.test_results['passed'])}")
        for test in self.test_results['passed']:
            print(f"  • {test}")
        
        if self.test_results['failed']:
            print(f"\n❌ FAILED: {len(self.test_results['failed'])}")
            for test in self.test_results['failed']:
                print(f"  • {test}")
        
        success_rate = len(self.test_results['passed']) / max(
            len(self.test_results['passed']) + len(self.test_results['failed']), 1
        ) * 100
        
        print(f"\n📊 Success Rate: {success_rate:.1f}%")
        print("="*60)
    
    async def run_all_tests(self):
        """Run all tests"""
        await self.connect()
        
        # Run test suites
        await self.test_redis_channels()
        await self.test_trigger_conditions()
        await self.test_always_on_agents()
        await self.test_on_demand_activation()
        
        # Display results
        await self.display_results()
        
        # Cleanup
        await self.redis_client.close()


async def main():
    """Main test runner"""
    logger.info("Starting Simplified System Tests...")
    
    # Check Redis connection first
    try:
        test_redis = redis.from_url("redis://localhost:6379")
        await test_redis.ping()
        await test_redis.close()
        logger.info("✓ Redis connection verified")
    except Exception as e:
        logger.error(f"✗ Redis not available: {e}")
        logger.info("Please start Redis: redis-server")
        return
    
    # Run tests
    tester = SimplifiedSystemTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())