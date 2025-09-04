"""
Agent Training Harness

Orchestrates complete training sessions for agents, coordinating
between synthetic data generation, agent execution, and performance monitoring.
"""

import asyncio
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import redis.asyncio as redis

from ..synthetic_data.generators.game_engine import SyntheticGameEngine
from ..synthetic_data.generators.market_simulator import MarketSimulator
# Training module imports commented out - not yet available
from ..confidence_calibration.calibrator import ConfidenceCalibrator
# Training module imports commented out - not yet available
from .synthetic_injector import SyntheticDataInjector, InjectionConfig, EventTiming
from .training_bridge import TrainingBridge, TrainingMode

logger = logging.getLogger(__name__)


class TrainingScenario(Enum):
    """Pre-defined training scenarios"""
    BASIC_GAME = "basic_game"
    CLOSE_GAME = "close_game"
    BLOWOUT = "blowout"
    COMEBACK = "comeback"
    HIGH_VOLATILITY = "high_volatility"
    LOW_LIQUIDITY = "low_liquidity"
    NEWS_DRIVEN = "news_driven"
    INJURY_SCENARIO = "injury_scenario"
    WEATHER_IMPACT = "weather_impact"
    MOMENTUM_SHIFT = "momentum_shift"


@dataclass
class HarnessConfig:
    """Configuration for training harness"""
    redis_url: str = "redis://localhost:6379"
    
    # Training parameters
    scenarios_per_session: int = 10
    warmup_scenarios: int = 2
    evaluation_scenarios: int = 3
    
    # Timing configuration
    scenario_spacing_seconds: float = 5.0
    decision_timeout_seconds: float = 30.0
    between_session_delay: float = 60.0
    
    # Performance thresholds
    min_accuracy_threshold: float = 0.6
    min_profit_threshold: float = -0.1
    max_drawdown_threshold: float = 0.2
    
    # Adaptive training
    enable_adaptive_difficulty: bool = True
    difficulty_adjustment_rate: float = 0.1
    performance_window_size: int = 5
    
    # Monitoring
    enable_real_time_monitoring: bool = True
    checkpoint_frequency: int = 5
    performance_report_frequency: int = 10


@dataclass
class TrainingMetrics:
    """Metrics tracked during training"""
    scenario_count: int = 0
    decision_count: int = 0
    successful_trades: int = 0
    failed_trades: int = 0
    
    total_profit: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    
    avg_decision_time: float = 0.0
    avg_confidence: float = 0.0
    confidence_calibration_error: float = 0.0
    
    exploration_rate: float = 0.0
    learning_rate: float = 0.0
    
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ScenarioResult:
    """Result from running a single scenario"""
    scenario_id: str
    scenario_type: TrainingScenario
    start_time: datetime
    end_time: datetime
    
    decisions_made: int
    profit_loss: float
    accuracy: float
    avg_confidence: float
    
    events_processed: int
    errors: List[str]
    
    agent_state: Dict[str, Any]
    market_conditions: Dict[str, Any]


class AgentTrainingHarness:
    """
    Orchestrates complete training sessions for agents.
    
    Coordinates between synthetic data generation, agent execution,
    and performance monitoring to provide comprehensive training.
    """
    
    def __init__(self, config: HarnessConfig):
        self.config = config
        self.redis_client: Optional[redis.Redis] = None
        
        # Core components
        self.training_bridge = TrainingBridge()
        self.data_injector = SyntheticDataInjector(InjectionConfig())
        self.memory_system = AgentMemorySystem()
        self.calibrator = ConfidenceCalibrator()
        self.analytics = AgentAnalytics()
        
        # Generators
        self.game_engine = SyntheticGameEngine()
        self.market_sim = MarketSimulator()
        
        # State tracking
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.scenario_queue: List[TrainingScenario] = []
        self.metrics: Dict[str, TrainingMetrics] = {}
        
        # Performance tracking
        self.performance_history: List[ScenarioResult] = []
        self.current_difficulty: float = 0.5
        
    async def initialize(self):
        """Initialize harness components"""
        self.redis_client = redis.from_url(self.config.redis_url)
        
        # Initialize components
        await self.training_bridge.initialize()
        await self.data_injector.initialize()
        
        # Set up monitoring
        if self.config.enable_real_time_monitoring:
            asyncio.create_task(self._monitor_performance())
            
        logger.info("Training harness initialized")
        
    async def run_training_session(
        self,
        agent_id: str,
        scenarios: List[TrainingScenario],
        training_mode: TrainingMode = TrainingMode.EXPLORATION
    ) -> Dict[str, Any]:
        """
        Run a complete training session for an agent.
        
        Args:
            agent_id: ID of agent to train
            scenarios: List of scenarios to run
            training_mode: Training mode to use
            
        Returns:
            Session results and metrics
        """
        session_id = f"session_{agent_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize session
        self.active_sessions[session_id] = {
            'agent_id': agent_id,
            'start_time': datetime.now(),
            'scenarios': scenarios,
            'mode': training_mode,
            'results': []
        }
        
        self.metrics[session_id] = TrainingMetrics()
        
        try:
            # Run warmup scenarios
            if self.config.warmup_scenarios > 0:
                warmup_scenarios = scenarios[:self.config.warmup_scenarios]
                await self._run_warmup(session_id, agent_id, warmup_scenarios)
            
            # Run training scenarios
            training_scenarios = scenarios[self.config.warmup_scenarios:]
            results = []
            
            for i, scenario in enumerate(training_scenarios):
                # Adjust difficulty if adaptive training enabled
                if self.config.enable_adaptive_difficulty:
                    await self._adjust_difficulty(session_id)
                
                # Run scenario
                result = await self._run_scenario(
                    session_id,
                    agent_id,
                    scenario,
                    training_mode
                )
                
                results.append(result)
                self.performance_history.append(result)
                
                # Update metrics
                await self._update_metrics(session_id, result)
                
                # Checkpoint if needed
                if (i + 1) % self.config.checkpoint_frequency == 0:
                    await self._create_checkpoint(session_id, agent_id)
                
                # Performance report
                if (i + 1) % self.config.performance_report_frequency == 0:
                    await self._generate_performance_report(session_id)
                
                # Delay between scenarios
                await asyncio.sleep(self.config.scenario_spacing_seconds)
            
            # Run evaluation scenarios
            if self.config.evaluation_scenarios > 0:
                eval_results = await self._run_evaluation(
                    session_id,
                    agent_id,
                    scenarios[-self.config.evaluation_scenarios:]
                )
                results.extend(eval_results)
            
            # Generate final report
            report = await self._generate_final_report(session_id, results)
            
            # Clean up session
            self.active_sessions[session_id]['end_time'] = datetime.now()
            self.active_sessions[session_id]['results'] = results
            
            return report
            
        except Exception as e:
            logger.error(f"Training session failed: {e}")
            self.metrics[session_id].errors.append(str(e))
            raise
            
    async def _run_scenario(
        self,
        session_id: str,
        agent_id: str,
        scenario: TrainingScenario,
        mode: TrainingMode
    ) -> ScenarioResult:
        """Run a single training scenario"""
        scenario_id = f"{session_id}_{scenario.value}_{datetime.now().timestamp()}"
        start_time = datetime.now()
        
        try:
            # Generate scenario data
            if scenario in [TrainingScenario.BASIC_GAME, TrainingScenario.CLOSE_GAME,
                           TrainingScenario.BLOWOUT, TrainingScenario.COMEBACK]:
                # Game scenario
                game_data = await self._generate_game_scenario(scenario)
                await self.data_injector.inject_game_scenario(
                    game_data['game'],
                    game_data['plays'],
                    timing=EventTiming.ACCELERATED
                )
                
            else:
                # Trading scenario
                trading_data = await self._generate_trading_scenario(scenario)
                await self.data_injector.inject_trading_scenario(
                    trading_data['events'],
                    timing=EventTiming.ACCELERATED
                )
            
            # Wait for agent decisions
            decisions = await self._collect_agent_decisions(
                agent_id,
                scenario_id,
                timeout=self.config.decision_timeout_seconds
            )
            
            # Calculate performance metrics
            metrics = await self._calculate_scenario_metrics(
                decisions,
                scenario
            )
            
            end_time = datetime.now()
            
            return ScenarioResult(
                scenario_id=scenario_id,
                scenario_type=scenario,
                start_time=start_time,
                end_time=end_time,
                decisions_made=len(decisions),
                profit_loss=metrics['profit_loss'],
                accuracy=metrics['accuracy'],
                avg_confidence=metrics['avg_confidence'],
                events_processed=metrics['events_processed'],
                errors=[],
                agent_state=await self._get_agent_state(agent_id),
                market_conditions=metrics.get('market_conditions', {})
            )
            
        except Exception as e:
            logger.error(f"Scenario {scenario_id} failed: {e}")
            return ScenarioResult(
                scenario_id=scenario_id,
                scenario_type=scenario,
                start_time=start_time,
                end_time=datetime.now(),
                decisions_made=0,
                profit_loss=0.0,
                accuracy=0.0,
                avg_confidence=0.0,
                events_processed=0,
                errors=[str(e)],
                agent_state={},
                market_conditions={}
            )
            
    async def _generate_game_scenario(
        self,
        scenario: TrainingScenario
    ) -> Dict[str, Any]:
        """Generate game scenario data"""
        if scenario == TrainingScenario.BASIC_GAME:
            game = await self.game_engine.generate_game(
                home_team="Team A",
                away_team="Team B",
                home_strength=0.5,
                away_strength=0.5
            )
            plays = await self.game_engine.generate_plays(game, num_plays=50)
            
        elif scenario == TrainingScenario.CLOSE_GAME:
            game = await self.game_engine.generate_game(
                home_team="Team A",
                away_team="Team B",
                home_strength=0.52,
                away_strength=0.48
            )
            plays = await self.game_engine.generate_plays(game, num_plays=60)
            
        elif scenario == TrainingScenario.BLOWOUT:
            game = await self.game_engine.generate_game(
                home_team="Team A",
                away_team="Team B",
                home_strength=0.7,
                away_strength=0.3
            )
            plays = await self.game_engine.generate_plays(game, num_plays=45)
            
        elif scenario == TrainingScenario.COMEBACK:
            game = await self.game_engine.generate_game(
                home_team="Team A",
                away_team="Team B",
                home_strength=0.4,
                away_strength=0.6
            )
            plays = await self.game_engine.generate_plays(game, num_plays=70)
            # Simulate comeback in second half
            for play in plays[35:]:
                if hasattr(play, 'scoring_play'):
                    play.scoring_play = play.scoring_play and play.quarter >= 3
                    
        else:
            raise ValueError(f"Unknown game scenario: {scenario}")
            
        return {'game': game, 'plays': plays}
        
    async def _generate_trading_scenario(
        self,
        scenario: TrainingScenario
    ) -> Dict[str, Any]:
        """Generate trading scenario data"""
        # First generate a game for the trading scenario
        if scenario == TrainingScenario.HIGH_VOLATILITY:
            # Close game with high volatility
            game = await self.game_engine.generate_game(
                home_team="Team A",
                away_team="Team B",
                home_strength=0.51,
                away_strength=0.49
            )
            trading_scenario = self.market_sim.create_trading_scenario(
                game=game,
                scenario_type="high_volatility",
                market_efficiency=0.6
            )
            
        elif scenario == TrainingScenario.LOW_LIQUIDITY:
            # Low-profile game
            game = await self.game_engine.generate_game(
                home_team="Team C",
                away_team="Team D",
                home_strength=0.45,
                away_strength=0.55
            )
            trading_scenario = self.market_sim.create_trading_scenario(
                game=game,
                scenario_type="low_liquidity",
                market_efficiency=0.4
            )
            
        elif scenario == TrainingScenario.NEWS_DRIVEN:
            # Game with injury/news events
            game = await self.game_engine.generate_game(
                home_team="Team E",
                away_team="Team F",
                home_strength=0.6,
                away_strength=0.4
            )
            trading_scenario = self.market_sim.create_trading_scenario(
                game=game,
                scenario_type="news_driven",
                information_delay=0.3,
                market_efficiency=0.7
            )
            
        elif scenario == TrainingScenario.MOMENTUM_SHIFT:
            # Game with momentum shifts
            game = await self.game_engine.generate_game(
                home_team="Team G",
                away_team="Team H",
                home_strength=0.48,
                away_strength=0.52
            )
            trading_scenario = self.market_sim.create_trading_scenario(
                game=game,
                scenario_type="momentum_shift",
                market_efficiency=0.75
            )
            
        else:
            # Default trading scenario
            game = await self.game_engine.generate_game(
                home_team="Team X",
                away_team="Team Y",
                home_strength=0.5,
                away_strength=0.5
            )
            trading_scenario = self.market_sim.create_trading_scenario(
                game=game,
                scenario_type="regular"
            )
            
        # Convert to standardized events
        events = self.market_sim.convert_to_standardized_events(trading_scenario)
        
        return {'events': events}
        
    async def _collect_agent_decisions(
        self,
        agent_id: str,
        scenario_id: str,
        timeout: float
    ) -> List[Dict[str, Any]]:
        """Collect decisions made by agent during scenario"""
        decisions = []
        start_time = datetime.now()
        
        # Subscribe to agent decision channel
        pubsub = self.redis_client.pubsub()
        await pubsub.subscribe(f"agent:{agent_id}:decisions")
        
        try:
            while (datetime.now() - start_time).total_seconds() < timeout:
                message = await pubsub.get_message(timeout=1.0)
                
                if message and message['type'] == 'message':
                    try:
                        decision = json.loads(message['data'])
                        decision['scenario_id'] = scenario_id
                        decision['timestamp'] = datetime.now().isoformat()
                        decisions.append(decision)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid decision data: {message['data']}")
                        
        finally:
            await pubsub.unsubscribe(f"agent:{agent_id}:decisions")
            await pubsub.close()
            
        return decisions
        
    async def _calculate_scenario_metrics(
        self,
        decisions: List[Dict[str, Any]],
        scenario: TrainingScenario
    ) -> Dict[str, Any]:
        """Calculate performance metrics for scenario"""
        if not decisions:
            return {
                'profit_loss': 0.0,
                'accuracy': 0.0,
                'avg_confidence': 0.0,
                'events_processed': 0,
                'market_conditions': {}
            }
            
        # Calculate P&L
        profit_loss = sum(d.get('profit_loss', 0.0) for d in decisions)
        
        # Calculate accuracy (correct predictions)
        correct = sum(1 for d in decisions if d.get('correct', False))
        accuracy = correct / len(decisions) if decisions else 0.0
        
        # Calculate average confidence
        confidences = [d.get('confidence', 0.5) for d in decisions]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        
        # Count events
        events_processed = len(set(d.get('event_id') for d in decisions if d.get('event_id')))
        
        # Extract market conditions
        market_conditions = {
            'volatility': self._estimate_volatility(decisions),
            'trend': self._estimate_trend(decisions),
            'liquidity': self._estimate_liquidity(decisions)
        }
        
        return {
            'profit_loss': profit_loss,
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'events_processed': events_processed,
            'market_conditions': market_conditions
        }
        
    async def _update_metrics(self, session_id: str, result: ScenarioResult):
        """Update session metrics with scenario result"""
        metrics = self.metrics[session_id]
        
        metrics.scenario_count += 1
        metrics.decision_count += result.decisions_made
        
        if result.profit_loss > 0:
            metrics.successful_trades += 1
        else:
            metrics.failed_trades += 1
            
        metrics.total_profit += result.profit_loss
        
        # Update running averages
        alpha = 0.1  # Exponential moving average factor
        metrics.avg_confidence = (1 - alpha) * metrics.avg_confidence + alpha * result.avg_confidence
        
        # Calculate win rate
        total_trades = metrics.successful_trades + metrics.failed_trades
        metrics.win_rate = metrics.successful_trades / total_trades if total_trades > 0 else 0.0
        
        # Track errors
        metrics.errors.extend(result.errors)
        
    async def _adjust_difficulty(self, session_id: str):
        """Adjust scenario difficulty based on performance"""
        metrics = self.metrics[session_id]
        
        # Get recent performance
        recent_results = self.performance_history[-self.config.performance_window_size:]
        if not recent_results:
            return
            
        # Calculate performance score
        avg_accuracy = sum(r.accuracy for r in recent_results) / len(recent_results)
        avg_profit = sum(r.profit_loss for r in recent_results) / len(recent_results)
        
        performance_score = 0.6 * avg_accuracy + 0.4 * (1.0 if avg_profit > 0 else 0.0)
        
        # Adjust difficulty
        if performance_score > 0.7:
            # Increase difficulty
            self.current_difficulty = min(1.0, self.current_difficulty + self.config.difficulty_adjustment_rate)
        elif performance_score < 0.4:
            # Decrease difficulty
            self.current_difficulty = max(0.0, self.current_difficulty - self.config.difficulty_adjustment_rate)
            
        logger.info(f"Adjusted difficulty to {self.current_difficulty:.2f} (performance: {performance_score:.2f})")
        
    async def _create_checkpoint(self, session_id: str, agent_id: str):
        """Create training checkpoint"""
        checkpoint = {
            'session_id': session_id,
            'agent_id': agent_id,
            'timestamp': datetime.now().isoformat(),
            'metrics': self.metrics[session_id].__dict__,
            'difficulty': self.current_difficulty,
            'scenario_count': len(self.performance_history)
        }
        
        # Save to Redis
        await self.redis_client.hset(
            f"training:checkpoints:{session_id}",
            datetime.now().isoformat(),
            json.dumps(checkpoint)
        )
        
        logger.info(f"Created checkpoint for session {session_id}")
        
    async def _generate_performance_report(self, session_id: str):
        """Generate intermediate performance report"""
        metrics = self.metrics[session_id]
        
        report = {
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'scenarios_completed': metrics.scenario_count,
            'total_decisions': metrics.decision_count,
            'win_rate': metrics.win_rate,
            'total_profit': metrics.total_profit,
            'avg_confidence': metrics.avg_confidence,
            'current_difficulty': self.current_difficulty
        }
        
        # Publish report
        await self.redis_client.publish(
            f"training:reports:{session_id}",
            json.dumps(report)
        )
        
        logger.info(f"Performance report: Win rate={metrics.win_rate:.2%}, Profit={metrics.total_profit:.2f}")
        
    async def _generate_final_report(
        self,
        session_id: str,
        results: List[ScenarioResult]
    ) -> Dict[str, Any]:
        """Generate final training report"""
        metrics = self.metrics[session_id]
        session = self.active_sessions[session_id]
        
        # Calculate final statistics
        total_profit = sum(r.profit_loss for r in results)
        avg_accuracy = sum(r.accuracy for r in results) / len(results) if results else 0.0
        
        # Calculate Sharpe ratio
        if len(results) > 1:
            returns = [r.profit_loss for r in results]
            avg_return = sum(returns) / len(returns)
            std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
            sharpe_ratio = avg_return / std_return if std_return > 0 else 0.0
        else:
            sharpe_ratio = 0.0
            
        report = {
            'session_id': session_id,
            'agent_id': session['agent_id'],
            'start_time': session['start_time'].isoformat(),
            'end_time': datetime.now().isoformat(),
            'duration_minutes': (datetime.now() - session['start_time']).total_seconds() / 60,
            
            'scenarios_run': len(results),
            'total_decisions': sum(r.decisions_made for r in results),
            
            'performance': {
                'total_profit': total_profit,
                'avg_accuracy': avg_accuracy,
                'win_rate': metrics.win_rate,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': metrics.max_drawdown,
                'avg_confidence': metrics.avg_confidence
            },
            
            'training_progress': {
                'starting_difficulty': 0.5,
                'ending_difficulty': self.current_difficulty,
                'exploration_rate': metrics.exploration_rate,
                'learning_rate': metrics.learning_rate
            },
            
            'errors': metrics.errors,
            'warnings': metrics.warnings,
            
            'recommendations': self._generate_recommendations(metrics, results)
        }
        
        # Save final report
        await self.redis_client.set(
            f"training:final_report:{session_id}",
            json.dumps(report),
            ex=86400  # Expire after 24 hours
        )
        
        return report
        
    def _generate_recommendations(
        self,
        metrics: TrainingMetrics,
        results: List[ScenarioResult]
    ) -> List[str]:
        """Generate training recommendations based on performance"""
        recommendations = []
        
        if metrics.win_rate < 0.5:
            recommendations.append("Consider additional training on market prediction")
            
        if metrics.avg_confidence > 0.8 and metrics.win_rate < 0.6:
            recommendations.append("Agent may be overconfident - adjust calibration")
            
        if metrics.max_drawdown > self.config.max_drawdown_threshold:
            recommendations.append("Implement stricter risk management")
            
        if metrics.exploration_rate < 0.1:
            recommendations.append("Increase exploration to discover new strategies")
            
        # Scenario-specific recommendations
        scenario_performance = {}
        for result in results:
            if result.scenario_type not in scenario_performance:
                scenario_performance[result.scenario_type] = []
            scenario_performance[result.scenario_type].append(result.accuracy)
            
        for scenario, accuracies in scenario_performance.items():
            avg_accuracy = sum(accuracies) / len(accuracies)
            if avg_accuracy < 0.5:
                recommendations.append(f"Focus training on {scenario.value} scenarios")
                
        return recommendations
        
    async def _run_warmup(
        self,
        session_id: str,
        agent_id: str,
        scenarios: List[TrainingScenario]
    ):
        """Run warmup scenarios"""
        logger.info(f"Running {len(scenarios)} warmup scenarios")
        
        for scenario in scenarios:
            result = await self._run_scenario(
                session_id,
                agent_id,
                scenario,
                TrainingMode.EXPLORATION
            )
            # Don't count warmup in metrics
            logger.debug(f"Warmup scenario {scenario.value}: accuracy={result.accuracy:.2%}")
            
    async def _run_evaluation(
        self,
        session_id: str,
        agent_id: str,
        scenarios: List[TrainingScenario]
    ) -> List[ScenarioResult]:
        """Run evaluation scenarios"""
        logger.info(f"Running {len(scenarios)} evaluation scenarios")
        
        results = []
        for scenario in scenarios:
            result = await self._run_scenario(
                session_id,
                agent_id,
                scenario,
                TrainingMode.VALIDATION
            )
            results.append(result)
            
        return results
        
    async def _get_agent_state(self, agent_id: str) -> Dict[str, Any]:
        """Get current agent state"""
        # Retrieve agent state from Redis
        state_key = f"agent:{agent_id}:state"
        state_data = await self.redis_client.get(state_key)
        
        if state_data:
            return json.loads(state_data)
        return {}
        
    def _estimate_volatility(self, decisions: List[Dict[str, Any]]) -> float:
        """Estimate market volatility from decisions"""
        if len(decisions) < 2:
            return 0.02
            
        prices = [d.get('price', 0.5) for d in decisions if 'price' in d]
        if len(prices) < 2:
            return 0.02
            
        # Calculate standard deviation of price changes
        changes = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
        return sum(changes) / len(changes) if changes else 0.02
        
    def _estimate_trend(self, decisions: List[Dict[str, Any]]) -> float:
        """Estimate market trend from decisions"""
        if len(decisions) < 2:
            return 0.0
            
        prices = [d.get('price', 0.5) for d in decisions if 'price' in d]
        if len(prices) < 2:
            return 0.0
            
        # Simple linear trend
        return (prices[-1] - prices[0]) / len(prices)
        
    def _estimate_liquidity(self, decisions: List[Dict[str, Any]]) -> float:
        """Estimate market liquidity from decisions"""
        volumes = [d.get('volume', 0) for d in decisions if 'volume' in d]
        if not volumes:
            return 0.5
            
        avg_volume = sum(volumes) / len(volumes)
        # Normalize to 0-1 scale (assuming max volume of 10000)
        return min(1.0, avg_volume / 10000)
        
    async def _monitor_performance(self):
        """Real-time performance monitoring"""
        while True:
            try:
                for session_id, session in self.active_sessions.items():
                    if 'end_time' not in session:
                        # Session still active
                        metrics = self.metrics.get(session_id)
                        if metrics:
                            logger.info(
                                f"Session {session_id}: "
                                f"Scenarios={metrics.scenario_count}, "
                                f"Win rate={metrics.win_rate:.2%}, "
                                f"Profit={metrics.total_profit:.2f}"
                            )
                            
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(30)
                
    async def cleanup(self):
        """Clean up resources"""
        if self.redis_client:
            await self.redis_client.close()
            
        logger.info("Training harness cleaned up")