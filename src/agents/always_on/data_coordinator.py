"""
Data Coordinator Agent - Simplified without Agno
Orchestrates data flow from StreamManager without any database or Agno dependencies
Now with Redis consumer for real-time data processing
"""

import os
import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.data_pipeline.orchestration.unified_stream_manager import StreamManager, EventType, UnifiedEvent
from src.trading.llm_client import get_llm_client
from src.agents.base_consumer import BaseAgentRedisConsumer

logger = logging.getLogger(__name__)


class DataCoordinatorAgent:
    """
    Data Coordinator Agent - Orchestrates data flow from StreamManager
    
    Responsibilities:
    - Consumes unified events from StreamManager
    - Routes high-priority events to appropriate agents
    - Maintains data quality and consistency
    - Handles market subscriptions
    
    NO Agno dependencies - pure Python implementation
    """
    
    def __init__(self):
        """Initialize Data Coordinator Agent"""
        # Initialize LLM client
        self.llm_client = get_llm_client()
        
        # Initialize StreamManager
        self.stream_manager = StreamManager()
        
        # Tracked markets
        self.tracked_markets: Dict[str, Dict] = {}
        
        # Message handler (set by Agentuity)
        self.message_handler = None
        
        # State
        self.is_running = False
        
        # Statistics
        self.events_received = 0
        self.events_routed = 0
    
    async def start(self, agent_context=None):
        """
        Start the Data Coordinator
        
        Args:
            agent_context: Agentuity context for agent communication
        """
        if self.is_running:
            logger.warning("Data Coordinator already running")
            return
        
        self.is_running = True
        logger.info("Starting Data Coordinator Agent...")
        
        # Set agent context for StreamManager if provided
        if agent_context:
            self.stream_manager.agent_context = agent_context
        
        # Initialize and start StreamManager
        await self.stream_manager.initialize()
        await self.stream_manager.start()
        
        # Register event handlers
        self._register_event_handlers()
        
        # Start Redis consumer
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis_consumer = DataCoordinatorRedisConsumer(self, agent_context)
        self.redis_consumer.redis_url = redis_url
        
        await self.redis_consumer.connect()
        await self.redis_consumer.subscribe([
            "kalshi:markets",
            "kalshi:trades", 
            "kalshi:signals",
            "espn:games"
        ])
        
        # Start consuming in background
        asyncio.create_task(self.redis_consumer.start_consuming())
        
        logger.info("Data Coordinator Agent started successfully with Redis consumer")
    
    def _register_event_handlers(self):
        """Register handlers for different event types"""
        # High-priority events
        self.stream_manager.register_handler(
            EventType.DIVERGENCE_DETECTED,
            self._handle_divergence
        )
        
        self.stream_manager.register_handler(
            EventType.INJURY_ALERT,
            self._handle_injury
        )
        
        self.stream_manager.register_handler(
            EventType.SENTIMENT_SHIFT,
            self._handle_sentiment_shift
        )
        
        # Market updates
        self.stream_manager.register_handler(
            EventType.PRICE_UPDATE,
            self._handle_price_update
        )
        
        # Risk events
        self.stream_manager.register_handler(
            EventType.RISK_ALERT,
            self._handle_risk_alert
        )
    
    async def track_market(
        self,
        market_ticker: str,
        game_id: Optional[str] = None,
        home_team: Optional[str] = None,
        away_team: Optional[str] = None,
        sport: str = "nfl"
    ):
        """
        Track a market via StreamManager
        
        Args:
            market_ticker: Kalshi market ticker
            game_id: ESPN game ID
            home_team: Home team name
            away_team: Away team name
            sport: Sport type
        """
        logger.info(f"Tracking market: {market_ticker}")
        
        # Store configuration
        self.tracked_markets[market_ticker] = {
            "game_id": game_id,
            "home_team": home_team,
            "away_team": away_team,
            "sport": sport,
            "started_at": datetime.now()
        }
        
        # Delegate to StreamManager
        await self.stream_manager.track_market(
            market_ticker=market_ticker,
            game_id=game_id,
            home_team=home_team,
            away_team=away_team,
            sport=sport
        )
        
        # Use LLM to analyze the setup
        analysis = await self.analyze_market_setup(
            market_ticker, home_team, away_team, sport
        )
        logger.info(f"Market analysis: {analysis}")
    
    async def analyze_market_setup(
        self,
        market_ticker: str,
        home_team: str,
        away_team: str,
        sport: str
    ) -> str:
        """Analyze market setup using LLM"""
        prompt = f"""
        Analyze this market tracking configuration:
        - Market: {market_ticker}
        - Game: {home_team} vs {away_team}
        - Sport: {sport}
        
        Provide brief assessment of:
        1. Key events to watch for
        2. Potential volatility factors
        3. Data quality requirements
        
        Keep response under 100 words.
        """
        
        try:
            return await self.llm_client.complete(prompt, temperature=0.1)
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return "Market tracking initiated"
    
    async def _handle_divergence(self, event: UnifiedEvent):
        """Handle market divergence events"""
        self.events_received += 1
        
        if event.impact_score > 0.7:
            # High-impact divergence - route to Strategy Analyst
            logger.info(f"High-impact divergence detected: {event.market_ticker}")
            
            if self.message_handler:
                await self.message_handler("divergence_opportunity", {
                    "market_ticker": event.market_ticker,
                    "divergence_data": event.data,
                    "impact_score": event.impact_score,
                    "timestamp": event.timestamp.isoformat()
                })
                self.events_routed += 1
    
    async def _handle_injury(self, event: UnifiedEvent):
        """Handle injury alerts"""
        self.events_received += 1
        
        # Injuries always high priority
        logger.warning(f"INJURY ALERT: {event.data.get('description')}")
        
        if self.message_handler:
            await self.message_handler("injury_alert", {
                "game_id": event.game_id,
                "injury_data": event.data,
                "impact_score": 0.9,  # Injuries are high impact
                "timestamp": event.timestamp.isoformat()
            })
            self.events_routed += 1
    
    async def _handle_sentiment_shift(self, event: UnifiedEvent):
        """Handle sentiment shift events"""
        self.events_received += 1
        
        logger.info(f"Sentiment shift detected: {event.data}")
        
        if self.message_handler:
            await self.message_handler("sentiment_shift", {
                "shift_data": event.data,
                "impact_score": event.impact_score,
                "timestamp": event.timestamp.isoformat()
            })
            self.events_routed += 1
    
    async def _handle_price_update(self, event: UnifiedEvent):
        """Handle price updates"""
        self.events_received += 1
        
        # Only route significant price moves
        if event.market_ticker in self.tracked_markets:
            context = self.stream_manager.get_market_context(event.market_ticker)
            
            if context and context.opportunity_score > 0.5:
                if self.message_handler:
                    await self.message_handler("price_update", {
                        "market_ticker": event.market_ticker,
                        "price_data": event.data,
                        "opportunity_score": context.opportunity_score,
                        "timestamp": event.timestamp.isoformat()
                    })
                    self.events_routed += 1
    
    async def _handle_risk_alert(self, event: UnifiedEvent):
        """Handle risk alerts"""
        self.events_received += 1
        
        logger.error(f"RISK ALERT: {event.data}")
        
        if self.message_handler:
            await self.message_handler("risk_alert", {
                "alert_data": event.data,
                "impact_score": event.impact_score,
                "timestamp": event.timestamp.isoformat()
            })
            self.events_routed += 1
    
    async def get_market_summary(self, market_ticker: str) -> Dict[str, Any]:
        """
        Get comprehensive market summary
        
        Args:
            market_ticker: Market to summarize
            
        Returns:
            Market summary with all data sources
        """
        context = self.stream_manager.get_market_context(market_ticker)
        
        if not context:
            return {"error": "Market not found"}
        
        return {
            "market_ticker": market_ticker,
            "price": {
                "yes": context.yes_price,
                "no": context.no_price,
                "bid": context.yes_bid,
                "ask": context.yes_ask,
                "volume_24h": context.volume_24h
            },
            "game": {
                "home_team": context.home_team,
                "away_team": context.away_team,
                "home_score": context.home_score,
                "away_score": context.away_score,
                "quarter": context.quarter,
                "time": context.time_remaining,
                "win_probability": context.win_probability
            },
            "sentiment": {
                "score": context.sentiment_score,
                "velocity": context.sentiment_velocity,
                "volume": context.tweet_volume,
                "high_impact_tweets": len(context.high_impact_tweets)
            },
            "analysis": {
                "divergence": context.price_sentiment_divergence,
                "opportunity_score": context.opportunity_score,
                "risk_level": context.risk_level
            },
            "last_update": context.last_update.isoformat()
        }
    
    async def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        stream_stats = self.stream_manager.get_statistics()
        
        return {
            "is_running": self.is_running,
            "tracked_markets": list(self.tracked_markets.keys()),
            "events_received": self.events_received,
            "events_routed": self.events_routed,
            "routing_rate": f"{(self.events_routed / max(self.events_received, 1)) * 100:.1f}%",
            "stream_manager": stream_stats
        }
    
    async def stop(self):
        """Stop the Data Coordinator"""
        self.is_running = False
        
        # Stop StreamManager
        await self.stream_manager.stop()
        
        # Stop Redis consumer if running
        if hasattr(self, 'redis_consumer') and self.redis_consumer:
            await self.redis_consumer.disconnect()
        
        logger.info(f"Data Coordinator stopped. Events: {self.events_received} received, {self.events_routed} routed")


class DataCoordinatorRedisConsumer(BaseAgentRedisConsumer):
    """
    Redis consumer for DataCoordinator
    Processes real-time market data and routes to appropriate agents
    """
    
    def __init__(self, coordinator: DataCoordinatorAgent, agent_context=None):
        super().__init__("DataCoordinator", agent_context=agent_context)
        self.coordinator = coordinator
        
    async def process_message(self, channel: str, data: Dict[str, Any]):
        """Process incoming Redis messages"""
        
        if channel == "kalshi:markets":
            await self._handle_market_update(data)
        elif channel == "kalshi:trades":
            await self._handle_trade(data)
        elif channel == "kalshi:signals":
            await self._handle_signal(data)
        elif channel == "espn:games":
            await self._handle_espn_update(data)
    
    async def _handle_market_update(self, data: Dict[str, Any]):
        """Handle market price updates"""
        market_ticker = data.get('data', {}).get('market_ticker')
        
        if market_ticker in self.coordinator.tracked_markets:
            # Update tracked market data
            self.coordinator.tracked_markets[market_ticker]['last_update'] = datetime.now()
            self.coordinator.tracked_markets[market_ticker]['latest_data'] = data.get('data', {})
            
            # Check for significant price movements
            yes_price = data.get('data', {}).get('yes_price', 0)
            if yes_price > 0.8 or yes_price < 0.2:
                # Alert for extreme prices
                await self.publish_signal({
                    "action": "PRICE_EXTREME",
                    "market_ticker": market_ticker,
                    "yes_price": yes_price,
                    "timestamp": datetime.utcnow().isoformat()
                })
    
    async def _handle_trade(self, data: Dict[str, Any]):
        """Handle trade executions"""
        self.coordinator.events_received += 1
        logger.info(f"Trade executed: {data}")
        
        # Could route to RiskManager for position tracking
        
    async def _handle_signal(self, data: Dict[str, Any]):
        """Handle high-impact signals"""
        self.coordinator.events_received += 1
        
        # Route high-confidence signals
        confidence = data.get('data', {}).get('confidence', 0)
        if confidence > 0.8:
            logger.info(f"High confidence signal: {data}")
            self.coordinator.events_routed += 1
            
    async def _handle_espn_update(self, data: Dict[str, Any]):
        """Handle ESPN game updates"""
        game_data = data.get('data', {})
        
        # Find associated market
        for market_ticker, market_info in self.coordinator.tracked_markets.items():
            if market_info.get('game_id') == game_data.get('game_id'):
                # Correlate game data with market
                logger.info(f"Game update for {market_ticker}: {game_data}")
                break


# Create singleton instance only when needed
data_coordinator_agent = None

def get_data_coordinator_agent():
    """Get or create the data coordinator agent instance."""
    global data_coordinator_agent
    if data_coordinator_agent is None:
        data_coordinator_agent = DataCoordinatorAgent()
    return data_coordinator_agent


# Example usage
async def main():
    """Example of running the Data Coordinator"""
    
    # Initialize agent
    agent = DataCoordinatorAgent()
    
    # Start agent
    await agent.start()
    
    # Track a market
    await agent.track_market(
        market_ticker="SUPERBOWL-2025",
        game_id="401547435",
        home_team="Chiefs",
        away_team="Bills",
        sport="nfl"
    )
    
    # Run for a while
    await asyncio.sleep(300)  # 5 minutes
    
    # Get status
    status = await agent.get_status()
    print(f"Agent status: {status}")
    
    # Get market summary
    summary = await agent.get_market_summary("SUPERBOWL-2025")
    print(f"Market summary: {summary}")
    
    # Stop agent
    await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())