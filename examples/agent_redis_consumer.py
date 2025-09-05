#!/usr/bin/env python3
"""
Agent Redis Consumer - Example of how agents consume WebSocket data via Redis
This shows the pattern agents will use to receive real-time market data
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict

import redis.asyncio as redis

logger = logging.getLogger(__name__)


class AgentRedisConsumer:
    """Base class for agents to consume Redis streams"""

    def __init__(self, agent_name: str, redis_url: str = "redis://localhost:6379"):
        self.agent_name = agent_name
        self.redis_url = redis_url
        self.redis_client = None
        self.pubsub = None
        self.is_running = False

    async def connect(self):
        """Connect to Redis"""
        self.redis_client = redis.from_url(self.redis_url)
        self.pubsub = self.redis_client.pubsub()
        logger.info(f"{self.agent_name} connected to Redis")

    async def subscribe(self, channels: list):
        """Subscribe to Redis channels"""
        await self.pubsub.subscribe(*channels)
        logger.info(f"{self.agent_name} subscribed to: {channels}")

    async def start_consuming(self):
        """Start consuming messages"""
        self.is_running = True

        async for message in self.pubsub.listen():
            if not self.is_running:
                break

            if message["type"] == "message":
                channel = message["channel"].decode("utf-8")
                data = json.loads(message["data"])

                # Route to appropriate handler
                await self.handle_message(channel, data)

    async def handle_message(self, channel: str, data: Dict[str, Any]):
        """Override in subclass to handle messages"""
        pass

    async def stop(self):
        """Stop consuming and disconnect"""
        self.is_running = False
        if self.pubsub:
            await self.pubsub.unsubscribe()
        if self.redis_client:
            await self.redis_client.close()


class DataEngineerConsumer(AgentRedisConsumer):
    """Example: DataEngineer agent consuming market data"""

    def __init__(self):
        super().__init__("DataEngineer")
        self.market_data = {}

    async def handle_message(self, channel: str, data: Dict[str, Any]):
        """Process incoming market data"""

        if channel == "kalshi:markets":
            # Store latest market data
            market_ticker = data["data"].get("market_ticker")
            if market_ticker:
                self.market_data[market_ticker] = {
                    "yes_price": data["data"].get("yes_price"),
                    "no_price": data["data"].get("no_price"),
                    "volume": data["data"].get("volume"),
                    "timestamp": data["timestamp"],
                }
                logger.info(f"Updated market data for {market_ticker}")

        elif channel == "kalshi:signals":
            # High-impact event requiring analysis
            await self.analyze_signal(data["data"])

    async def analyze_signal(self, signal: Dict[str, Any]):
        """Analyze high-impact signals"""
        logger.info(f"Analyzing signal: {signal}")
        # Add analysis logic here


class MarketEngineerConsumer(AgentRedisConsumer):
    """Example: MarketEngineer agent identifying opportunities"""

    def __init__(self):
        super().__init__("MarketEngineer")
        self.opportunities = []

    async def handle_message(self, channel: str, data: Dict[str, Any]):
        """Look for trading opportunities"""

        if channel == "kalshi:markets":
            market_data = data["data"]

            # Simple opportunity detection
            yes_price = market_data.get("yes_price", 0)
            no_price = market_data.get("no_price", 0)

            # Check for arbitrage opportunity
            if yes_price > 0 and no_price > 0:
                total = yes_price + no_price
                if total < 0.98:  # Arbitrage opportunity
                    opportunity = {
                        "type": "arbitrage",
                        "market": market_data.get("market_ticker"),
                        "yes_price": yes_price,
                        "no_price": no_price,
                        "profit_potential": 1.0 - total,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                    self.opportunities.append(opportunity)
                    logger.info(f"Found arbitrage opportunity: {opportunity}")

                    # Publish opportunity for TradeExecutor
                    await self.publish_opportunity(opportunity)

        elif channel == "espn:games":
            # React to game events
            game_data = data["data"]
            logger.info(f"Game update: {game_data}")

    async def publish_opportunity(self, opportunity: Dict[str, Any]):
        """Publish trading opportunity"""
        # In real implementation, publish to Redis or call TradeExecutor
        logger.info(f"Publishing opportunity: {opportunity}")


class TradeExecutorConsumer(AgentRedisConsumer):
    """Example: TradeExecutor agent placing trades"""

    def __init__(self):
        super().__init__("TradeExecutor")
        self.pending_trades = []

    async def handle_message(self, channel: str, data: Dict[str, Any]):
        """Execute trades based on signals"""

        if channel == "kalshi:signals":
            signal = data["data"]

            if signal.get("action") == "BUY":
                await self.place_order(
                    market=signal.get("market_ticker"),
                    side="yes" if signal.get("confidence", 0) > 0.5 else "no",
                    quantity=self.calculate_position_size(signal),
                )

    def calculate_position_size(self, signal: Dict[str, Any]) -> int:
        """Calculate position size using Kelly Criterion"""
        # Simplified Kelly calculation
        confidence = signal.get("confidence", 0.5)
        kelly_fraction = confidence - (1 - confidence)

        # Apply Kelly with safety factor
        position_size = int(kelly_fraction * 100 * 0.25)  # 25% of Kelly
        return max(1, min(position_size, 100))  # Between 1 and 100

    async def place_order(self, market: str, side: str, quantity: int):
        """Place order on Kalshi"""
        logger.info(f"Placing order: {market} {side} x{quantity}")
        # In real implementation, call Kalshi API


async def run_agent(agent_class, channels):
    """Run an agent consumer"""
    agent = agent_class()
    await agent.connect()
    await agent.subscribe(channels)

    try:
        await agent.start_consuming()
    except KeyboardInterrupt:
        logger.info(f"Stopping {agent.agent_name}...")
    finally:
        await agent.stop()


async def main():
    """Example of running multiple agents"""
    import sys

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if len(sys.argv) < 2:
        print("Usage: python agent_redis_consumer.py [data|market|trade|all]")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "data":
        # Run DataEngineer
        await run_agent(DataEngineerConsumer, ["kalshi:markets", "kalshi:signals"])

    elif mode == "market":
        # Run MarketEngineer
        await run_agent(MarketEngineerConsumer, ["kalshi:markets", "espn:games"])

    elif mode == "trade":
        # Run TradeExecutor
        await run_agent(TradeExecutorConsumer, ["kalshi:signals"])

    elif mode == "all":
        # Run all agents concurrently
        tasks = [
            asyncio.create_task(
                run_agent(DataEngineerConsumer, ["kalshi:markets", "kalshi:signals"])
            ),
            asyncio.create_task(
                run_agent(MarketEngineerConsumer, ["kalshi:markets", "espn:games"])
            ),
            asyncio.create_task(run_agent(TradeExecutorConsumer, ["kalshi:signals"])),
        ]

        await asyncio.gather(*tasks)

    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
