#!/usr/bin/env python3
"""
Run Simplified Agent System
Orchestrates always-on and on-demand agents without Agentuity overhead
"""

import asyncio
import logging
import signal
import sys
from typing import List, Dict, Any
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.always_on.data_coordinator import DataCoordinatorAgent
from src.agents.always_on.portfolio_monitor import PortfolioMonitorAgent
from src.agents.on_demand.game_analyst import GameAnalystAgent
from src.agents.trigger_service import TriggerService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimplifiedAgentOrchestrator:
    """
    Orchestrates the simplified agent system
    
    Always-On Agents:
    - DataCoordinator: Collects and routes data
    - PortfolioMonitor: Monitors positions and risk
    
    On-Demand Agents:
    - GameAnalyst: Analyzes specific games
    - ArbitrageHunter: Finds arbitrage opportunities
    - StrategyOptimizer: Optimizes trading strategies
    
    Coordination:
    - TriggerService: Evaluates conditions and activates on-demand agents
    """
    
    def __init__(self):
        self.is_running = False
        
        # Always-on agents
        self.data_coordinator = DataCoordinatorAgent()
        self.portfolio_monitor = PortfolioMonitorAgent()
        
        # On-demand agents
        self.game_analyst = GameAnalystAgent()
        
        # Trigger service
        self.trigger_service = TriggerService()
        
        # Track active tasks
        self.tasks: List[asyncio.Task] = []
    
    async def start(self):
        """Start the agent system"""
        if self.is_running:
            logger.warning("Orchestrator already running")
            return
        
        self.is_running = True
        logger.info("Starting Simplified Agent System...")
        
        try:
            # Connect all components
            await self._connect_all()
            
            # Start always-on agents
            await self._start_always_on_agents()
            
            # Start on-demand agents (listening mode)
            await self._start_on_demand_agents()
            
            # Start trigger service
            await self._start_trigger_service()
            
            logger.info("✅ All agents started successfully")
            
            # Display status
            await self._display_status()
            
        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            await self.stop()
            raise
    
    async def _connect_all(self):
        """Connect all agents to Redis"""
        logger.info("Connecting agents to Redis...")
        
        # Always-on agents don't use Redis directly in simplified version
        # They use StreamManager which handles Redis internally
        
        # Connect on-demand agents
        await self.game_analyst.connect()
        
        # Connect trigger service
        await self.trigger_service.connect()
        
        logger.info("All agents connected")
    
    async def _start_always_on_agents(self):
        """Start always-on agents"""
        logger.info("Starting always-on agents...")
        
        # Start Data Coordinator
        data_task = asyncio.create_task(self.data_coordinator.start())
        self.tasks.append(data_task)
        logger.info("  ✓ DataCoordinator started")
        
        # Start Portfolio Monitor
        await self.portfolio_monitor.connect()
        await self.portfolio_monitor.start()
        logger.info("  ✓ PortfolioMonitor started")
        
        # Give them time to initialize
        await asyncio.sleep(2)
    
    async def _start_on_demand_agents(self):
        """Start on-demand agents in listening mode"""
        logger.info("Starting on-demand agents (listening mode)...")
        
        # Start Game Analyst
        await self.game_analyst.start()
        logger.info("  ✓ GameAnalyst ready")
        
        # Additional on-demand agents would be started here
        # await self.arbitrage_hunter.start()
        # await self.strategy_optimizer.start()
    
    async def _start_trigger_service(self):
        """Start the trigger service"""
        logger.info("Starting trigger service...")
        
        await self.trigger_service.start()
        logger.info("  ✓ TriggerService started")
    
    async def _display_status(self):
        """Display system status"""
        print("\n" + "="*60)
        print("SIMPLIFIED AGENT SYSTEM STATUS")
        print("="*60)
        
        # Data Coordinator status
        dc_status = await self.data_coordinator.get_status()
        print("\n📊 Data Coordinator:")
        print(f"  • Running: {dc_status['is_running']}")
        print(f"  • Markets tracked: {len(dc_status['tracked_markets'])}")
        print(f"  • Events received: {dc_status['events_received']}")
        print(f"  • Events routed: {dc_status['events_routed']}")
        
        # Portfolio Monitor status
        pm_status = self.portfolio_monitor.get_statistics()
        print("\n💼 Portfolio Monitor:")
        print(f"  • Running: {pm_status['is_running']}")
        print(f"  • Trading halted: {pm_status['trading_halted']}")
        print(f"  • Positions: {pm_status['portfolio']['positions']}")
        print(f"  • Daily P&L: ${pm_status['portfolio']['daily_pnl']:.2f}")
        
        # Game Analyst status
        ga_status = self.game_analyst.get_statistics()
        print("\n🎮 Game Analyst:")
        print(f"  • Active: {ga_status['is_active']}")
        print(f"  • Analyses performed: {ga_status['analyses_performed']}")
        
        # Trigger Service status
        ts_status = self.trigger_service.get_statistics()
        print("\n⚡ Trigger Service:")
        print(f"  • Running: {ts_status['is_running']}")
        print(f"  • Triggers configured: {ts_status['triggers_configured']}")
        print(f"  • Activations: {ts_status['activation_stats']}")
        
        print("\n" + "="*60)
    
    async def track_market(
        self,
        market_ticker: str,
        game_id: str = None,
        home_team: str = None,
        away_team: str = None,
        sport: str = "nfl"
    ):
        """
        Track a market across all systems
        
        Args:
            market_ticker: Kalshi market ticker
            game_id: ESPN game ID
            home_team: Home team name
            away_team: Away team name
            sport: Sport type
        """
        logger.info(f"Tracking market: {market_ticker}")
        
        # Track via Data Coordinator
        await self.data_coordinator.track_market(
            market_ticker=market_ticker,
            game_id=game_id,
            home_team=home_team,
            away_team=away_team,
            sport=sport
        )
        
        # Trigger pre-game analysis if game is soon
        await self.trigger_service.manual_trigger(
            "GameAnalyst",
            {
                "game_id": game_id,
                "home_team": home_team,
                "away_team": away_team,
                "market_ticker": market_ticker
            },
            reason="market_tracking_started"
        )
    
    async def analyze_game(self, game_id: str, home_team: str, away_team: str):
        """
        Manually trigger game analysis
        
        Args:
            game_id: Game identifier
            home_team: Home team name
            away_team: Away team name
        """
        logger.info(f"Requesting analysis for {home_team} vs {away_team}")
        
        await self.trigger_service.manual_trigger(
            "GameAnalyst",
            {
                "event_type": "user_request",
                "game_id": game_id,
                "home_team": home_team,
                "away_team": away_team
            },
            reason="user_request"
        )
    
    async def get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio status"""
        return self.portfolio_monitor.get_statistics()
    
    async def stop(self):
        """Stop the agent system"""
        logger.info("Stopping Simplified Agent System...")
        
        self.is_running = False
        
        # Stop all agents
        await self.data_coordinator.stop()
        await self.portfolio_monitor.stop()
        await self.game_analyst.stop()
        await self.trigger_service.stop()
        
        # Cancel all tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        logger.info("✅ All agents stopped")


async def main():
    """Main entry point"""
    orchestrator = SimplifiedAgentOrchestrator()
    
    # Handle shutdown signals
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, shutting down...")
        asyncio.create_task(orchestrator.stop())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start the system
        await orchestrator.start()
        
        # Example: Track a market
        await orchestrator.track_market(
            market_ticker="NFL-CHIEFS-BILLS-CONF",
            game_id="401547435",
            home_team="Chiefs",
            away_team="Bills",
            sport="nfl"
        )
        
        # Example: Request analysis
        await asyncio.sleep(5)
        await orchestrator.analyze_game(
            game_id="401547435",
            home_team="Chiefs",
            away_team="Bills"
        )
        
        # Keep running
        logger.info("System running. Press Ctrl+C to stop.")
        
        while orchestrator.is_running:
            # Periodic status display
            await asyncio.sleep(60)
            await orchestrator._display_status()
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"System error: {e}")
    finally:
        await orchestrator.stop()


if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check for required environment variables
    required_vars = [
        "KALSHI_API_KEY_ID",
        "KALSHI_PRIVATE_KEY",
        "TWITTERAPI_KEY",
        "OPENAI_API_KEY"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.info("Please set them in your .env file")
        sys.exit(1)
    
    # Run the system
    asyncio.run(main())