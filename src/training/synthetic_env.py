"""
Synthetic Training Environment

Provides isolated training environment for agents using synthetic data
with accelerated time replay and performance tracking.
"""

import logging
import asyncio
import random
from typing import List, Dict, Any, Optional, Tuple, Iterator, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import time

from ..synthetic_data.generators import (
    TradingScenario, MarketEvent, SyntheticGame, 
    ScenarioBuilder, TrainingScenarioSet
)
from src.sdk.core.base_adapter import StandardizedEvent

logger = logging.getLogger(__name__)


class EnvironmentState(Enum):
    """Training environment states"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


class InformationLevel(Enum):
    """Levels of information available to agents"""
    FULL = "full"              # All information immediately
    PUBLIC_ONLY = "public"     # Only public information
    DELAYED = "delayed"        # Information with delays
    PRIVATE = "private"        # Access to private information
    ASYMMETRIC = "asymmetric"  # Different agents get different info


@dataclass
class AgentPerformanceMetrics:
    """Performance metrics for agent in training"""
    agent_id: str
    scenario_id: str
    
    # Trading metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    
    # Decision metrics
    decision_latency_ms: List[float] = field(default_factory=list)
    kelly_adherence: float = 0.0
    position_sizing_accuracy: float = 0.0
    
    # Learning metrics
    pattern_recognition_score: float = 0.0
    risk_management_score: float = 0.0
    information_utilization: float = 0.0
    
    # Behavioral metrics
    overconfidence_incidents: int = 0
    panic_trades: int = 0
    fomo_trades: int = 0
    
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


@dataclass
class AgentAction:
    """Action taken by agent during training"""
    agent_id: str
    timestamp: datetime
    action_type: str  # 'trade', 'update_position', 'cancel', 'analyze'
    
    # Trade details
    market_ticker: Optional[str] = None
    side: Optional[str] = None  # 'yes', 'no'
    size: Optional[float] = None
    price: Optional[float] = None
    
    # Decision context
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    available_information: List[str] = field(default_factory=list)
    
    # Execution details
    execution_time_ms: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class EnvironmentConfig:
    """Configuration for training environment"""
    
    # Time simulation
    time_acceleration: float = 10.0  # 10x real-time
    enable_pause: bool = True
    
    # Information flow
    information_level: InformationLevel = InformationLevel.PUBLIC_ONLY
    information_delay_ms: float = 1000  # 1 second delay
    private_information_probability: float = 0.1
    
    # Market simulation
    enable_slippage: bool = True
    slippage_factor: float = 0.02
    min_trade_size: float = 10.0
    max_trade_size: float = 1000.0
    
    # Logging and monitoring
    log_all_actions: bool = True
    track_performance: bool = True
    enable_real_time_analytics: bool = True


class SyntheticTrainingEnvironment:
    """
    Isolated training environment for agent development using synthetic data
    """
    
    def __init__(self, config: EnvironmentConfig = None):
        """
        Initialize training environment
        
        Args:
            config: Environment configuration
        """
        self.config = config or EnvironmentConfig()
        
        # Environment state
        self.state = EnvironmentState.IDLE
        self.current_scenario: Optional[TradingScenario] = None
        self.scenario_start_time: Optional[datetime] = None
        self.current_time: Optional[datetime] = None
        
        # Agent management
        self.registered_agents: Dict[str, Dict[str, Any]] = {}
        self.agent_callbacks: Dict[str, Callable] = {}
        self.agent_metrics: Dict[str, AgentPerformanceMetrics] = {}
        
        # Event management
        self.pending_events: List[Tuple[datetime, MarketEvent]] = []
        self.processed_events: List[Tuple[datetime, MarketEvent]] = []
        self.information_buffer: Dict[str, List[MarketEvent]] = {}  # Agent-specific buffers
        
        # Performance tracking
        self.environment_metrics = {
            "scenarios_completed": 0,
            "total_training_time": 0.0,
            "events_processed": 0,
            "agents_trained": set()
        }
        
        logger.info("Initialized SyntheticTrainingEnvironment")
    
    def register_agent(self, 
                      agent_id: str,
                      agent_callback: Callable,
                      information_level: InformationLevel = None,
                      agent_config: Dict[str, Any] = None) -> bool:
        """
        Register agent for training
        
        Args:
            agent_id: Unique agent identifier
            agent_callback: Function to call with market events
            information_level: Level of information access for this agent
            agent_config: Additional agent configuration
            
        Returns:
            True if registration successful
        """
        if agent_id in self.registered_agents:
            logger.warning(f"Agent {agent_id} already registered")
            return False
        
        self.registered_agents[agent_id] = {
            "callback": agent_callback,
            "information_level": information_level or self.config.information_level,
            "config": agent_config or {},
            "registered_at": datetime.now()
        }
        
        self.agent_callbacks[agent_id] = agent_callback
        self.information_buffer[agent_id] = []
        
        logger.info(f"Registered agent: {agent_id}")
        return True
    
    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister agent from training"""
        if agent_id not in self.registered_agents:
            logger.warning(f"Agent {agent_id} not registered")
            return False
        
        del self.registered_agents[agent_id]
        del self.agent_callbacks[agent_id]
        if agent_id in self.information_buffer:
            del self.information_buffer[agent_id]
        
        logger.info(f"Unregistered agent: {agent_id}")
        return True
    
    async def run_scenario(self, scenario: TradingScenario) -> Dict[str, AgentPerformanceMetrics]:
        """
        Run single training scenario
        
        Args:
            scenario: Trading scenario to execute
            
        Returns:
            Performance metrics for all agents
        """
        if self.state != EnvironmentState.IDLE:
            raise RuntimeError(f"Environment not idle, current state: {self.state}")
        
        if not self.registered_agents:
            raise RuntimeError("No agents registered for training")
        
        logger.info(f"Starting scenario: {scenario.scenario_id}")
        
        # Initialize scenario
        self.state = EnvironmentState.RUNNING
        self.current_scenario = scenario
        self.scenario_start_time = datetime.now()
        self.current_time = scenario.events[0].timestamp if scenario.events else datetime.now()
        
        # Initialize agent metrics
        for agent_id in self.registered_agents:
            self.agent_metrics[agent_id] = AgentPerformanceMetrics(
                agent_id=agent_id,
                scenario_id=scenario.scenario_id
            )
        
        # Prepare event timeline
        await self._prepare_event_timeline(scenario)
        
        try:
            # Run scenario simulation
            await self._execute_scenario()
            
            # Finalize metrics
            self._finalize_agent_metrics()
            
            self.state = EnvironmentState.COMPLETED
            
        except Exception as e:
            logger.error(f"Error running scenario: {e}")
            self.state = EnvironmentState.ERROR
            raise
        
        finally:
            # Clean up
            self._cleanup_scenario()
        
        logger.info(f"Completed scenario: {scenario.scenario_id}")
        return dict(self.agent_metrics)
    
    async def run_training_set(self, 
                             training_set: TrainingScenarioSet,
                             progress_callback: Optional[Callable] = None) -> Dict[str, List[AgentPerformanceMetrics]]:
        """
        Run complete training set
        
        Args:
            training_set: Set of training scenarios
            progress_callback: Optional progress callback function
            
        Returns:
            Performance metrics for all scenarios and agents
        """
        logger.info(f"Starting training set: {training_set.name} ({len(training_set.scenarios)} scenarios)")
        
        all_metrics = {agent_id: [] for agent_id in self.registered_agents}
        completed_scenarios = 0
        
        for scenario in training_set.scenarios:
            try:
                # Run scenario
                scenario_metrics = await self.run_scenario(scenario)
                
                # Collect metrics
                for agent_id, metrics in scenario_metrics.items():
                    all_metrics[agent_id].append(metrics)
                
                completed_scenarios += 1
                
                # Progress callback
                if progress_callback:
                    progress_callback(completed_scenarios, len(training_set.scenarios), scenario.scenario_id)
                
                # Brief pause between scenarios
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Failed to run scenario {scenario.scenario_id}: {e}")
                continue
        
        # Update environment metrics
        self.environment_metrics["scenarios_completed"] += completed_scenarios
        self.environment_metrics["agents_trained"].update(self.registered_agents.keys())
        
        logger.info(f"Completed training set: {completed_scenarios}/{len(training_set.scenarios)} scenarios")
        return all_metrics
    
    async def _prepare_event_timeline(self, scenario: TradingScenario):
        """Prepare chronological event timeline for scenario"""
        
        # Clear previous events
        self.pending_events.clear()
        self.processed_events.clear()
        
        # Classify events by information access
        classified_events = self._classify_events_by_information_access(scenario)
        
        # Build timeline
        for event in scenario.events:
            self.pending_events.append((event.timestamp, event))
        
        # Sort by timestamp
        self.pending_events.sort(key=lambda x: x[0])
        
        logger.debug(f"Prepared timeline with {len(self.pending_events)} events")
    
    def _classify_events_by_information_access(self, scenario: TradingScenario) -> Dict[str, List[MarketEvent]]:
        """Classify events by information access level"""
        
        classified = {
            "public": scenario.public_events,
            "private": scenario.private_events, 
            "delayed": scenario.delayed_events
        }
        
        # If no explicit classification, classify all as public
        if not classified["public"] and not classified["private"] and not classified["delayed"]:
            classified["public"] = scenario.events
        
        return classified
    
    async def _execute_scenario(self):
        """Execute scenario with time simulation"""
        
        start_time = time.time()
        
        while self.pending_events and self.state == EnvironmentState.RUNNING:
            # Get next event
            event_time, event = self.pending_events.pop(0)
            
            # Simulate time advancement
            await self._advance_time_to(event_time)
            
            # Process event
            await self._process_event(event)
            
            # Check for pause
            while self.state == EnvironmentState.PAUSED:
                await asyncio.sleep(0.1)
            
            # Brief processing delay
            await asyncio.sleep(0.01)
        
        # Update environment metrics
        self.environment_metrics["total_training_time"] += time.time() - start_time
        self.environment_metrics["events_processed"] += len(self.processed_events)
    
    async def _advance_time_to(self, target_time: datetime):
        """Advance simulation time to target with acceleration"""
        
        if not self.current_time:
            self.current_time = target_time
            return
        
        time_diff = (target_time - self.current_time).total_seconds()
        if time_diff <= 0:
            return
        
        # Apply time acceleration
        sleep_time = time_diff / self.config.time_acceleration
        await asyncio.sleep(sleep_time)
        
        self.current_time = target_time
    
    async def _process_event(self, event: MarketEvent):
        """Process market event and distribute to agents"""
        
        # Determine which agents should receive this event
        receiving_agents = self._determine_event_recipients(event)
        
        # Distribute to agents
        tasks = []
        for agent_id in receiving_agents:
            task = self._send_event_to_agent(agent_id, event)
            tasks.append(task)
        
        # Execute all agent notifications concurrently
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Record processed event
        self.processed_events.append((datetime.now(), event))
    
    def _determine_event_recipients(self, event: MarketEvent) -> List[str]:
        """Determine which agents should receive event based on information levels"""
        
        recipients = []
        
        for agent_id, agent_info in self.registered_agents.items():
            info_level = agent_info["information_level"]
            
            should_receive = False
            
            if info_level == InformationLevel.FULL:
                should_receive = True
            elif info_level == InformationLevel.PUBLIC_ONLY:
                should_receive = event in self.current_scenario.public_events
            elif info_level == InformationLevel.PRIVATE:
                should_receive = (event in self.current_scenario.public_events or 
                                event in self.current_scenario.private_events)
            elif info_level == InformationLevel.DELAYED:
                # Add delay for delayed events
                if event in self.current_scenario.delayed_events:
                    # Buffer the event for later delivery
                    asyncio.create_task(self._deliver_delayed_event(agent_id, event))
                    continue
                else:
                    should_receive = True
            elif info_level == InformationLevel.ASYMMETRIC:
                # Random information access
                should_receive = random.random() > 0.3
            
            if should_receive:
                recipients.append(agent_id)
        
        return recipients
    
    async def _deliver_delayed_event(self, agent_id: str, event: MarketEvent):
        """Deliver event with delay"""
        delay_seconds = self.config.information_delay_ms / 1000.0
        await asyncio.sleep(delay_seconds)
        
        if self.state == EnvironmentState.RUNNING:
            await self._send_event_to_agent(agent_id, event)
    
    async def _send_event_to_agent(self, agent_id: str, event: MarketEvent):
        """Send event to specific agent"""
        
        try:
            callback = self.agent_callbacks[agent_id]
            
            # Convert to StandardizedEvent format
            standardized_event = self._convert_to_standardized_event(event)
            
            # Record start time for latency measurement
            start_time = time.time()
            
            # Call agent
            await callback(standardized_event)
            
            # Record execution time
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Update agent metrics
            if agent_id in self.agent_metrics:
                self.agent_metrics[agent_id].decision_latency_ms.append(execution_time_ms)
            
        except Exception as e:
            logger.error(f"Error sending event to agent {agent_id}: {e}")
    
    def _convert_to_standardized_event(self, event: MarketEvent) -> StandardizedEvent:
        """Convert MarketEvent to StandardizedEvent"""
        from src.sdk.core.base_adapter import EventType
        
        return StandardizedEvent(
            source="synthetic_training_env",
            event_type=EventType.MARKET_EVENT,
            timestamp=event.timestamp,
            game_id=self.current_scenario.game.game_id if self.current_scenario else "",
            data={
                "market_ticker": event.market_ticker,
                "event_type": event.event_type.value,
                "description": event.description,
                "price_impact": event.price_impact,
                "volume_impact": event.volume_impact,
                "confidence_impact": event.confidence_impact,
                "synthetic": True
            },
            confidence=0.99,
            impact="high" if abs(event.price_impact) > 0.1 else "medium" if abs(event.price_impact) > 0.05 else "low",
            metadata={
                "training_environment": True,
                "scenario_id": self.current_scenario.scenario_id if self.current_scenario else "",
                "current_time": self.current_time.isoformat() if self.current_time else ""
            },
            raw_data=event
        )
    
    async def record_agent_action(self, agent_action: AgentAction):
        """Record action taken by agent during training"""
        
        # Update performance metrics
        if agent_action.agent_id in self.agent_metrics:
            metrics = self.agent_metrics[agent_action.agent_id]
            
            if agent_action.action_type == "trade":
                metrics.total_trades += 1
                
                # Record execution latency
                if agent_action.execution_time_ms:
                    metrics.decision_latency_ms.append(agent_action.execution_time_ms)
            
            # Log action if configured
            if self.config.log_all_actions:
                logger.debug(f"Agent {agent_action.agent_id} action: {agent_action.action_type}")
    
    def pause_environment(self):
        """Pause training environment"""
        if self.state == EnvironmentState.RUNNING:
            self.state = EnvironmentState.PAUSED
            logger.info("Training environment paused")
    
    def resume_environment(self):
        """Resume training environment"""
        if self.state == EnvironmentState.PAUSED:
            self.state = EnvironmentState.RUNNING
            logger.info("Training environment resumed")
    
    def stop_environment(self):
        """Stop training environment"""
        self.state = EnvironmentState.IDLE
        logger.info("Training environment stopped")
    
    def _finalize_agent_metrics(self):
        """Calculate final performance metrics for all agents"""
        
        for agent_id, metrics in self.agent_metrics.items():
            metrics.completed_at = datetime.now()
            
            # Calculate derived metrics
            if metrics.total_trades > 0:
                metrics.win_rate = metrics.winning_trades / metrics.total_trades
                
                if metrics.losing_trades > 0:
                    avg_win = metrics.total_pnl / max(1, metrics.winning_trades)
                    avg_loss = abs(metrics.total_pnl) / metrics.losing_trades
                    metrics.profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
            
            # Calculate average decision latency
            if metrics.decision_latency_ms:
                avg_latency = sum(metrics.decision_latency_ms) / len(metrics.decision_latency_ms)
                logger.debug(f"Agent {agent_id} avg latency: {avg_latency:.1f}ms")
    
    def _cleanup_scenario(self):
        """Clean up after scenario completion"""
        self.current_scenario = None
        self.scenario_start_time = None
        self.current_time = None
        self.pending_events.clear()
        
        # Clear information buffers
        for buffer in self.information_buffer.values():
            buffer.clear()
    
    def get_environment_status(self) -> Dict[str, Any]:
        """Get current environment status"""
        return {
            "state": self.state.value,
            "registered_agents": len(self.registered_agents),
            "current_scenario": self.current_scenario.scenario_id if self.current_scenario else None,
            "pending_events": len(self.pending_events),
            "processed_events": len(self.processed_events),
            "metrics": self.environment_metrics
        }
    
    def get_agent_performance_summary(self, agent_id: str = None) -> Dict[str, Any]:
        """Get performance summary for agent(s)"""
        
        if agent_id:
            if agent_id not in self.agent_metrics:
                return {}
            
            metrics = self.agent_metrics[agent_id]
            return {
                "agent_id": agent_id,
                "total_trades": metrics.total_trades,
                "win_rate": metrics.win_rate,
                "total_pnl": metrics.total_pnl,
                "max_drawdown": metrics.max_drawdown,
                "avg_latency_ms": sum(metrics.decision_latency_ms) / len(metrics.decision_latency_ms) if metrics.decision_latency_ms else 0
            }
        else:
            # Return summary for all agents
            summaries = {}
            for aid, metrics in self.agent_metrics.items():
                summaries[aid] = {
                    "total_trades": metrics.total_trades,
                    "win_rate": metrics.win_rate,
                    "total_pnl": metrics.total_pnl,
                    "avg_latency_ms": sum(metrics.decision_latency_ms) / len(metrics.decision_latency_ms) if metrics.decision_latency_ms else 0
                }
            return summaries


# Example usage and testing
if __name__ == "__main__":
    import sys
    import asyncio
    sys.path.append('/Users/hudson/Documents/GitHub/IntelIP/PROJECTS/Neural/Kalshi_Agentic_Agent')
    
    from src.synthetic_data.generators import SyntheticGameEngine, MarketSimulator, ScenarioBuilder
    from src.synthetic_data.storage.chromadb_manager import ChromaDBManager
    
    async def mock_agent_callback(event: StandardizedEvent):
        """Mock agent callback for testing"""
        logger.info(f"Agent received event: {event.data.get('description', 'No description')}")
        await asyncio.sleep(0.1)  # Simulate processing time
    
    async def test_training_environment():
        # Initialize components
        chromadb = ChromaDBManager()
        game_engine = SyntheticGameEngine(chromadb)
        market_simulator = MarketSimulator()
        scenario_builder = ScenarioBuilder(game_engine, market_simulator, chromadb)
        
        # Create training environment
        config = EnvironmentConfig(
            time_acceleration=100.0,  # Very fast for testing
            information_level=InformationLevel.PUBLIC_ONLY
        )
        env = SyntheticTrainingEnvironment(config)
        
        # Register test agents
        env.register_agent("test_agent_1", mock_agent_callback, InformationLevel.FULL)
        env.register_agent("test_agent_2", mock_agent_callback, InformationLevel.DELAYED)
        
        # Generate test scenario
        game = await game_engine.generate_single_game(
            home_team="KC", 
            away_team="BUF"
        )
        scenario = market_simulator.create_trading_scenario(game)
        
        print(f"Testing with scenario: {scenario.scenario_id}")
        print(f"Events: {len(scenario.events)}")
        
        # Run scenario
        metrics = await env.run_scenario(scenario)
        
        print(f"\nPerformance Results:")
        for agent_id, agent_metrics in metrics.items():
            print(f"  {agent_id}:")
            print(f"    Decision latency: {sum(agent_metrics.decision_latency_ms)/len(agent_metrics.decision_latency_ms) if agent_metrics.decision_latency_ms else 0:.1f}ms")
            print(f"    Events received: {len(agent_metrics.decision_latency_ms)}")
        
        # Environment status
        status = env.get_environment_status()
        print(f"\nEnvironment Status: {status}")
    
    # Run test
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_training_environment())