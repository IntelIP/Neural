"""
Training Bridge Module

Bridges synthetic data pipeline with Redis-based agent infrastructure,
enabling seamless training with generated scenarios.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import logging
import json
import redis.asyncio as redis
from collections import defaultdict

from ..sdk.core.base_adapter import StandardizedEvent
from ..synthetic_data.generators.scenario_builder import TrainingScenarioSet
from ..training.agent_analytics import AgentAnalytics, DecisionMetrics
from ..training.memory_system import AgentMemorySystem
from ..confidence_calibration.calibrator import ConfidenceCalibrator
from ..hybrid_pipeline.data_orchestrator import HybridDataOrchestrator


class TrainingMode(Enum):
    """Training modes for agents"""
    EXPLORATION = "exploration"  # High randomness, learning new patterns
    EXPLOITATION = "exploitation"  # Low randomness, refining strategies
    VALIDATION = "validation"  # No randomness, testing performance
    PRODUCTION_PREP = "production_prep"  # Simulate production conditions


@dataclass
class TrainingConfig:
    """Configuration for training sessions"""
    mode: TrainingMode = TrainingMode.EXPLORATION
    time_acceleration: float = 10.0  # 10x speed
    inject_noise: bool = True  # Add realistic noise to data
    noise_level: float = 0.05  # 5% noise
    track_decisions: bool = True
    update_memory: bool = True
    calibrate_confidence: bool = True
    
    # Redis configuration
    redis_url: str = "redis://localhost:6379"
    training_namespace: str = "training"  # Prefix for training channels
    
    # Performance thresholds
    min_confidence_threshold: float = 0.3
    max_position_size: float = 0.1  # 10% of capital
    stop_loss_threshold: float = 0.2  # 20% drawdown stops training
    
    # Scenario preferences
    edge_case_probability: float = 0.2
    market_volatility_multiplier: float = 1.0


@dataclass
class TrainingSession:
    """Active training session for an agent"""
    session_id: str
    agent_id: str
    start_time: datetime
    config: TrainingConfig
    scenarios_completed: int = 0
    decisions_made: int = 0
    total_pnl: float = 0.0
    errors_encountered: int = 0
    end_time: Optional[datetime] = None
    
    @property
    def duration(self) -> timedelta:
        end = self.end_time or datetime.now()
        return end - self.start_time
    
    @property
    def is_active(self) -> bool:
        return self.end_time is None


class TrainingBridge:
    """
    Central bridge between synthetic data generation and agent training.
    
    Coordinates data flow from synthetic generators through Redis to agents,
    while tracking performance and updating learning systems.
    """
    
    def __init__(
        self,
        orchestrator: HybridDataOrchestrator,
        analytics: AgentAnalytics,
        memory_system: AgentMemorySystem,
        calibrator: ConfidenceCalibrator,
        config: TrainingConfig = None
    ):
        self.orchestrator = orchestrator
        self.analytics = analytics
        self.memory_system = memory_system
        self.calibrator = calibrator
        self.config = config or TrainingConfig()
        self.logger = logging.getLogger(__name__)
        
        # Redis connections
        self.redis_client: Optional[redis.Redis] = None
        self.publisher: Optional[redis.Redis] = None
        
        # Active training sessions
        self.active_sessions: Dict[str, TrainingSession] = {}
        
        # Performance tracking
        self.agent_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Event routing
        self.channel_mappings = {
            "game_event": "espn:games",
            "market_update": "kalshi:markets",
            "trade_executed": "kalshi:trades",
            "signal_generated": "kalshi:signals",
            "sentiment_update": "twitter:sentiment"
        }
        
        # Callbacks for agent responses
        self.response_handlers: Dict[str, Callable] = {}
        
    async def initialize(self) -> None:
        """Initialize the training bridge"""
        try:
            self.logger.info("Initializing Training Bridge")
            
            # Connect to Redis
            await self._connect_redis()
            
            # Initialize orchestrator
            await self.orchestrator.initialize()
            
            # Set up response listeners
            await self._setup_response_listeners()
            
            self.logger.info("Training Bridge initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize training bridge: {e}")
            raise
    
    async def _connect_redis(self) -> None:
        """Connect to Redis for pub/sub operations"""
        try:
            self.redis_client = redis.from_url(self.config.redis_url)
            self.publisher = redis.from_url(self.config.redis_url)
            
            # Test connection
            await self.redis_client.ping()
            
            self.logger.info("Connected to Redis for training bridge")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def _setup_response_listeners(self) -> None:
        """Set up listeners for agent responses during training"""
        try:
            # Subscribe to agent response channels
            pubsub = self.redis_client.pubsub()
            
            response_channels = [
                f"{self.config.training_namespace}:agent_decisions",
                f"{self.config.training_namespace}:agent_signals",
                f"{self.config.training_namespace}:agent_trades"
            ]
            
            await pubsub.subscribe(*response_channels)
            
            # Start listening in background
            asyncio.create_task(self._listen_for_responses(pubsub))
            
            self.logger.info(f"Listening for agent responses on {response_channels}")
            
        except Exception as e:
            self.logger.error(f"Failed to setup response listeners: {e}")
    
    async def _listen_for_responses(self, pubsub) -> None:
        """Listen for agent responses and process them"""
        try:
            async for message in pubsub.listen():
                if message['type'] not in ('message', 'pmessage'):
                    continue
                
                try:
                    channel = message['channel'].decode('utf-8')
                    data = json.loads(message['data'])
                    
                    # Extract agent ID and process response
                    agent_id = data.get('agent_id')
                    if agent_id and agent_id in self.active_sessions:
                        await self._process_agent_response(agent_id, channel, data)
                        
                except Exception as e:
                    self.logger.error(f"Error processing response: {e}")
                    
        except Exception as e:
            self.logger.error(f"Response listener error: {e}")
    
    async def _process_agent_response(self, agent_id: str, channel: str, data: Dict[str, Any]) -> None:
        """Process an agent's response during training"""
        try:
            session = self.active_sessions.get(agent_id)
            if not session:
                return
            
            # Update session statistics
            session.decisions_made += 1
            
            # Extract decision details
            if 'agent_decisions' in channel:
                await self._track_decision(agent_id, data)
            elif 'agent_trades' in channel:
                await self._track_trade(agent_id, data)
            elif 'agent_signals' in channel:
                await self._track_signal(agent_id, data)
            
            # Update metrics
            if 'pnl' in data:
                session.total_pnl += data['pnl']
            
            # Check for training termination conditions
            if await self._should_stop_training(session):
                await self.stop_training_session(agent_id)
                
        except Exception as e:
            self.logger.error(f"Failed to process agent response: {e}")
            session.errors_encountered += 1
    
    async def _track_decision(self, agent_id: str, decision_data: Dict[str, Any]) -> None:
        """Track a decision made by an agent during training"""
        try:
            # Create decision metrics
            decision = DecisionMetrics(
                decision_id=decision_data.get('decision_id', f"{agent_id}_{datetime.now().timestamp()}"),
                agent_id=agent_id,
                scenario_id=decision_data.get('scenario_id', 'unknown'),
                timestamp=datetime.now(),
                market_ticker=decision_data.get('market_ticker', ''),
                decision_type=decision_data.get('decision_type', 'hold'),
                confidence=decision_data.get('confidence', 0.5),
                expected_value=decision_data.get('expected_value', 0.0),
                kelly_fraction=decision_data.get('kelly_fraction', 0.0),
                actual_kelly_used=decision_data.get('actual_kelly_used', 0.0),
                position_size=decision_data.get('position_size', 0.0),
                market_efficiency=decision_data.get('market_efficiency', 0.8),
                information_advantage=decision_data.get('information_advantage', 0.0),
                execution_latency=decision_data.get('latency', 0.0)
            )
            
            # Record in analytics
            await self.analytics.record_decision(decision)
            
            # Update memory if configured
            if self.config.update_memory:
                await self.memory_system.store_agent_experience(
                    agent_id=agent_id,
                    scenario_id=decision.scenario_id,
                    action={
                        'type': decision.decision_type,
                        'confidence': decision.confidence,
                        'position_size': decision.position_size
                    },
                    outcome={'pending': True},  # Will be updated later
                    context=decision_data.get('context', {})
                )
            
            # Calibrate confidence if configured
            if self.config.calibrate_confidence:
                calibrated_score = await self.calibrator.calibrate_confidence(
                    agent_id=agent_id,
                    raw_confidence=decision.confidence,
                    context=decision_data.get('context', {})
                )
                
                # Store calibration result
                self.agent_metrics[agent_id]['calibrated_confidence'] = calibrated_score.calibrated_confidence
                self.agent_metrics[agent_id]['confidence_uncertainty'] = calibrated_score.uncertainty
                
        except Exception as e:
            self.logger.error(f"Failed to track decision: {e}")
    
    async def _track_trade(self, agent_id: str, trade_data: Dict[str, Any]) -> None:
        """Track a trade execution during training"""
        try:
            # Update agent metrics
            self.agent_metrics[agent_id]['trades_executed'] = \
                self.agent_metrics[agent_id].get('trades_executed', 0) + 1
            
            # Track trade outcome
            if trade_data.get('status') == 'FILLED':
                self.agent_metrics[agent_id]['successful_trades'] = \
                    self.agent_metrics[agent_id].get('successful_trades', 0) + 1
                    
        except Exception as e:
            self.logger.error(f"Failed to track trade: {e}")
    
    async def _track_signal(self, agent_id: str, signal_data: Dict[str, Any]) -> None:
        """Track a signal generated during training"""
        try:
            # Update agent metrics
            self.agent_metrics[agent_id]['signals_generated'] = \
                self.agent_metrics[agent_id].get('signals_generated', 0) + 1
            
            # Track signal quality
            confidence = signal_data.get('confidence', 0.5)
            if confidence > 0.7:
                self.agent_metrics[agent_id]['high_confidence_signals'] = \
                    self.agent_metrics[agent_id].get('high_confidence_signals', 0) + 1
                    
        except Exception as e:
            self.logger.error(f"Failed to track signal: {e}")
    
    async def _should_stop_training(self, session: TrainingSession) -> bool:
        """Check if training should be stopped based on performance"""
        try:
            # Check drawdown threshold
            if session.total_pnl < -self.config.stop_loss_threshold:
                self.logger.warning(f"Stopping training for {session.agent_id}: Drawdown exceeded")
                return True
            
            # Check error rate
            if session.errors_encountered > 100:
                self.logger.warning(f"Stopping training for {session.agent_id}: Too many errors")
                return True
            
            # Check minimum decisions
            if session.decisions_made > 10000:
                self.logger.info(f"Stopping training for {session.agent_id}: Maximum decisions reached")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking stop conditions: {e}")
            return False
    
    async def start_training_session(
        self,
        agent_id: str,
        scenarios: TrainingScenarioSet,
        config: Optional[TrainingConfig] = None
    ) -> str:
        """
        Start a new training session for an agent.
        
        Args:
            agent_id: Unique identifier for the agent
            scenarios: Set of training scenarios to use
            config: Optional training configuration override
            
        Returns:
            Session ID for the training session
        """
        try:
            # Use provided config or default
            session_config = config or self.config
            
            # Create session
            session_id = f"{agent_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            session = TrainingSession(
                session_id=session_id,
                agent_id=agent_id,
                start_time=datetime.now(),
                config=session_config
            )
            
            self.active_sessions[agent_id] = session
            
            # Start scenario injection
            asyncio.create_task(self._run_training_scenarios(session, scenarios))
            
            self.logger.info(f"Started training session {session_id} for agent {agent_id}")
            return session_id
            
        except Exception as e:
            self.logger.error(f"Failed to start training session: {e}")
            raise
    
    async def _run_training_scenarios(self, session: TrainingSession, scenarios: TrainingScenarioSet) -> None:
        """Run training scenarios for a session"""
        try:
            for scenario in scenarios.scenarios:
                if not session.is_active:
                    break
                
                # Convert scenario to events
                events = await self._scenario_to_events(scenario)
                
                # Inject events into Redis with timing
                for event in events:
                    if not session.is_active:
                        break
                    
                    # Apply time acceleration
                    delay = event.metadata.get('delay', 0.1) / session.config.time_acceleration
                    await asyncio.sleep(delay)
                    
                    # Add noise if configured
                    if session.config.inject_noise:
                        event = self._add_noise_to_event(event, session.config.noise_level)
                    
                    # Publish to appropriate channel
                    await self._publish_training_event(event, session)
                
                session.scenarios_completed += 1
                
                # Update progress
                await self._update_training_progress(session)
            
            # Session complete
            if session.is_active:
                await self.stop_training_session(session.agent_id)
                
        except Exception as e:
            self.logger.error(f"Error running training scenarios: {e}")
            session.errors_encountered += 1
    
    async def _scenario_to_events(self, scenario: Any) -> List[StandardizedEvent]:
        """Convert a training scenario to standardized events"""
        try:
            events = []
            
            # Handle different scenario types
            if hasattr(scenario, 'game') and scenario.game:
                # Game scenario
                for play in scenario.game.plays:
                    event = StandardizedEvent(
                        event_id=f"training_{play.play_id}",
                        game_id=scenario.game.game_id,
                        timestamp=play.timestamp,
                        event_type=play.play_type,
                        description=play.description,
                        team_possession=play.team_possession,
                        score_home=play.score_home,
                        score_away=play.score_away,
                        quarter=play.quarter,
                        time_remaining=play.time_remaining,
                        metadata={
                            'training': True,
                            'scenario_id': scenario.scenario_id,
                            'delay': 0.1  # Default delay between events
                        }
                    )
                    events.append(event)
            
            elif hasattr(scenario, 'market_events'):
                # Market scenario
                for market_event in scenario.market_events:
                    event = StandardizedEvent(
                        event_id=f"training_market_{market_event.event_id}",
                        game_id=scenario.market_ticker,
                        timestamp=market_event.timestamp,
                        event_type="market_update",
                        description=f"Price: {market_event.price}",
                        metadata={
                            'training': True,
                            'scenario_id': scenario.scenario_id,
                            'price': market_event.price,
                            'volume': market_event.volume,
                            'delay': 0.05
                        }
                    )
                    events.append(event)
            
            return events
            
        except Exception as e:
            self.logger.error(f"Failed to convert scenario to events: {e}")
            return []
    
    def _add_noise_to_event(self, event: StandardizedEvent, noise_level: float) -> StandardizedEvent:
        """Add realistic noise to training events"""
        try:
            import random
            
            # Add noise to numeric fields
            if hasattr(event, 'score_home') and event.score_home is not None:
                # Don't add noise to scores (they're discrete)
                pass
            
            # Add noise to metadata prices
            if 'price' in event.metadata:
                original_price = event.metadata['price']
                noise = random.gauss(0, noise_level * original_price)
                event.metadata['price'] = max(0.01, min(0.99, original_price + noise))
            
            # Add timing jitter
            if 'delay' in event.metadata:
                original_delay = event.metadata['delay']
                jitter = random.gauss(0, noise_level * original_delay)
                event.metadata['delay'] = max(0.01, original_delay + jitter)
            
            return event
            
        except Exception as e:
            self.logger.error(f"Failed to add noise to event: {e}")
            return event
    
    async def _publish_training_event(self, event: StandardizedEvent, session: TrainingSession) -> None:
        """Publish training event to appropriate Redis channel"""
        try:
            # Determine channel based on event type
            channel = self._get_channel_for_event(event)
            
            # Add training namespace if configured
            if session.config.training_namespace:
                channel = f"{session.config.training_namespace}:{channel}"
            
            # Prepare message
            message = {
                "timestamp": event.timestamp.isoformat() if hasattr(event.timestamp, 'isoformat') else str(event.timestamp),
                "source": "training_bridge",
                "training_session": session.session_id,
                "data": {
                    "event_id": event.event_id,
                    "game_id": event.game_id,
                    "event_type": event.event_type,
                    "description": event.description,
                    "metadata": event.metadata
                }
            }
            
            # Add event-specific data
            if event.event_type in ["score_update", "big_play"]:
                message["data"]["score_home"] = event.score_home
                message["data"]["score_away"] = event.score_away
                message["data"]["quarter"] = event.quarter
                message["data"]["time_remaining"] = event.time_remaining
            
            # Publish to Redis
            await self.publisher.publish(channel, json.dumps(message))
            
        except Exception as e:
            self.logger.error(f"Failed to publish training event: {e}")
    
    def _get_channel_for_event(self, event: StandardizedEvent) -> str:
        """Determine the appropriate Redis channel for an event"""
        event_type = event.event_type.lower() if event.event_type else "unknown"
        
        # Map event types to channels
        if "market" in event_type or "price" in event_type:
            return "kalshi:markets"
        elif "trade" in event_type:
            return "kalshi:trades"
        elif "signal" in event_type:
            return "kalshi:signals"
        elif "sentiment" in event_type:
            return "twitter:sentiment"
        else:
            return "espn:games"  # Default to game events
    
    async def _update_training_progress(self, session: TrainingSession) -> None:
        """Update training progress metrics"""
        try:
            # Calculate progress metrics
            elapsed_time = (datetime.now() - session.start_time).total_seconds()
            scenarios_per_minute = (session.scenarios_completed / elapsed_time) * 60 if elapsed_time > 0 else 0
            
            # Log progress
            self.logger.info(
                f"Training progress for {session.agent_id}: "
                f"{session.scenarios_completed} scenarios, "
                f"{session.decisions_made} decisions, "
                f"P&L: {session.total_pnl:.2f}, "
                f"Rate: {scenarios_per_minute:.1f} scenarios/min"
            )
            
            # Update agent metrics
            self.agent_metrics[session.agent_id]['scenarios_completed'] = session.scenarios_completed
            self.agent_metrics[session.agent_id]['training_pnl'] = session.total_pnl
            
        except Exception as e:
            self.logger.error(f"Failed to update training progress: {e}")
    
    async def stop_training_session(self, agent_id: str) -> Dict[str, Any]:
        """
        Stop a training session and return final metrics.
        
        Args:
            agent_id: Agent whose training session to stop
            
        Returns:
            Final training metrics and summary
        """
        try:
            session = self.active_sessions.get(agent_id)
            if not session:
                return {"error": f"No active session for agent {agent_id}"}
            
            # Mark session as ended
            session.end_time = datetime.now()
            
            # Generate final analytics
            final_metrics = {
                "session_id": session.session_id,
                "agent_id": agent_id,
                "duration": str(session.duration),
                "scenarios_completed": session.scenarios_completed,
                "decisions_made": session.decisions_made,
                "total_pnl": session.total_pnl,
                "errors_encountered": session.errors_encountered,
                "agent_metrics": dict(self.agent_metrics.get(agent_id, {}))
            }
            
            # Trigger final calibration update if needed
            if session.config.calibrate_confidence and session.decisions_made > 50:
                await self._trigger_calibration_update(agent_id)
            
            # Clean up session
            del self.active_sessions[agent_id]
            
            self.logger.info(f"Stopped training session for agent {agent_id}")
            return final_metrics
            
        except Exception as e:
            self.logger.error(f"Failed to stop training session: {e}")
            return {"error": str(e)}
    
    async def _trigger_calibration_update(self, agent_id: str) -> None:
        """Trigger confidence calibration update for an agent"""
        try:
            # Get recent decisions from analytics
            recent_analytics = await self.analytics.get_agent_analytics(
                agent_id, timedelta(hours=1)
            )
            
            # Check if calibration update is needed
            if await self.calibrator.should_update_calibration(agent_id):
                self.logger.info(f"Triggering calibration update for agent {agent_id}")
                # Calibration will be updated in next decision cycle
                
        except Exception as e:
            self.logger.error(f"Failed to trigger calibration update: {e}")
    
    def register_response_handler(self, event_type: str, handler: Callable) -> None:
        """Register a handler for specific agent response types"""
        self.response_handlers[event_type] = handler
        self.logger.info(f"Registered response handler for {event_type}")
    
    async def get_training_status(self) -> Dict[str, Any]:
        """Get status of all active training sessions"""
        try:
            status = {
                "active_sessions": len(self.active_sessions),
                "sessions": {}
            }
            
            for agent_id, session in self.active_sessions.items():
                status["sessions"][agent_id] = {
                    "session_id": session.session_id,
                    "started": session.start_time.isoformat(),
                    "duration": str(session.duration),
                    "scenarios_completed": session.scenarios_completed,
                    "decisions_made": session.decisions_made,
                    "current_pnl": session.total_pnl,
                    "mode": session.config.mode.value
                }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Failed to get training status: {e}")
            return {"error": str(e)}
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the training bridge"""
        try:
            self.logger.info("Shutting down Training Bridge")
            
            # Stop all active sessions
            for agent_id in list(self.active_sessions.keys()):
                await self.stop_training_session(agent_id)
            
            # Close Redis connections
            if self.redis_client:
                await self.redis_client.close()
            if self.publisher:
                await self.publisher.close()
            
            self.logger.info("Training Bridge shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")