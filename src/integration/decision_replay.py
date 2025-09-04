"""
Decision Replay System

Enables replay of historical trading decisions for analysis,
debugging, and strategy improvement.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import redis.asyncio as redis

from .decision_tracker import DecisionRecord, DecisionOutcome, DecisionType, TrackingConfig
from ..synthetic_data.generators.market_simulator import MarketSimulator, MarketState
from ..training.agent_analytics import AgentAnalytics

logger = logging.getLogger(__name__)


class ReplayMode(Enum):
    """Replay execution modes"""
    HISTORICAL = "historical"  # Replay with historical data
    SIMULATED = "simulated"    # Replay with simulated conditions
    COMPARATIVE = "comparative"  # Compare multiple strategies
    WHATIF = "whatif"          # What-if scenarios


class ReplaySpeed(Enum):
    """Replay speed settings"""
    REALTIME = "realtime"      # 1x speed
    FAST = "fast"              # 10x speed
    MAX = "max"                # As fast as possible
    STEP = "step"              # Step by step


@dataclass
class ReplayConfig:
    """Configuration for decision replay"""
    mode: ReplayMode = ReplayMode.HISTORICAL
    speed: ReplaySpeed = ReplaySpeed.FAST
    
    # Time settings
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Filters
    agent_filter: Optional[str] = None
    market_filter: Optional[str] = None
    outcome_filter: Optional[DecisionOutcome] = None
    
    # Replay options
    skip_failed: bool = False
    apply_slippage: bool = True
    include_market_impact: bool = True
    
    # Analysis options
    calculate_alternatives: bool = True
    track_metrics: bool = True
    generate_report: bool = True
    
    # Comparison settings (for comparative mode)
    comparison_strategies: List[str] = None
    
    # What-if settings
    whatif_scenarios: Dict[str, Any] = None


@dataclass
class ReplayResult:
    """Result from a replay session"""
    session_id: str
    mode: ReplayMode
    decisions_replayed: int
    
    # Performance comparison
    original_pnl: float
    replayed_pnl: float
    improvement: float
    
    # Metrics
    original_sharpe: float
    replayed_sharpe: float
    max_drawdown: float
    
    # Analysis
    better_decisions: List[str]
    worse_decisions: List[str]
    missed_opportunities: List[Dict[str, Any]]
    
    # Detailed results
    decision_results: List[Dict[str, Any]]
    market_conditions: Dict[str, Any]
    
    # Report
    summary: str
    recommendations: List[str]


class DecisionReplayEngine:
    """
    Engine for replaying historical trading decisions.
    
    Enables analysis of past decisions, comparison of strategies,
    and what-if scenario testing.
    """
    
    def __init__(self, tracking_config: TrackingConfig):
        self.tracking_config = tracking_config
        self.redis_client: Optional[redis.Redis] = None
        
        # Components
        self.market_simulator = MarketSimulator()
        self.analytics = None  # Will be injected
        
        # Replay state
        self.current_replay: Optional[Dict[str, Any]] = None
        self.replay_buffer: List[DecisionRecord] = []
        self.market_states: Dict[str, MarketState] = {}
        
        # Callbacks
        self.decision_callbacks: List[Callable] = []
        self.completion_callbacks: List[Callable] = []
        
    async def initialize(self):
        """Initialize replay engine"""
        if self.tracking_config.storage_backend in ["redis", "hybrid"]:
            self.redis_client = redis.from_url(self.tracking_config.redis_url)
            
        logger.info("Decision replay engine initialized")
        
    async def replay_decisions(
        self,
        decisions: List[DecisionRecord],
        config: ReplayConfig
    ) -> ReplayResult:
        """
        Replay a set of historical decisions.
        
        Args:
            decisions: Decisions to replay
            config: Replay configuration
            
        Returns:
            Replay results with analysis
        """
        session_id = f"replay_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize replay session
        self.current_replay = {
            'session_id': session_id,
            'config': config,
            'start_time': datetime.now(),
            'decisions': decisions,
            'results': []
        }
        
        # Sort decisions by timestamp
        decisions = sorted(decisions, key=lambda d: d.timestamp)
        
        # Initialize market states
        await self._initialize_market_states(decisions)
        
        # Replay based on mode
        if config.mode == ReplayMode.HISTORICAL:
            results = await self._replay_historical(decisions, config)
        elif config.mode == ReplayMode.SIMULATED:
            results = await self._replay_simulated(decisions, config)
        elif config.mode == ReplayMode.COMPARATIVE:
            results = await self._replay_comparative(decisions, config)
        elif config.mode == ReplayMode.WHATIF:
            results = await self._replay_whatif(decisions, config)
        else:
            raise ValueError(f"Unknown replay mode: {config.mode}")
            
        # Calculate performance metrics
        replay_result = await self._calculate_replay_results(
            session_id,
            config,
            decisions,
            results
        )
        
        # Generate report if requested
        if config.generate_report:
            replay_result.summary = await self._generate_report(replay_result)
            replay_result.recommendations = await self._generate_recommendations(replay_result)
            
        # Trigger completion callbacks
        for callback in self.completion_callbacks:
            asyncio.create_task(callback(replay_result))
            
        # Clean up
        self.current_replay = None
        
        return replay_result
        
    async def replay_session(
        self,
        session_id: str,
        config: Optional[ReplayConfig] = None
    ) -> ReplayResult:
        """
        Replay all decisions from a training session.
        
        Args:
            session_id: Training session ID
            config: Optional replay configuration
            
        Returns:
            Replay results
        """
        # Load decisions from session
        decisions = await self._load_session_decisions(session_id)
        
        if not decisions:
            raise ValueError(f"No decisions found for session {session_id}")
            
        # Use default config if not provided
        if not config:
            config = ReplayConfig()
            
        return await self.replay_decisions(decisions, config)
        
    async def compare_strategies(
        self,
        decisions: List[DecisionRecord],
        strategies: List[str]
    ) -> Dict[str, ReplayResult]:
        """
        Compare multiple strategies on the same decisions.
        
        Args:
            decisions: Base decisions to replay
            strategies: List of strategy names to compare
            
        Returns:
            Results for each strategy
        """
        config = ReplayConfig(
            mode=ReplayMode.COMPARATIVE,
            comparison_strategies=strategies
        )
        
        results = {}
        for strategy in strategies:
            # Modify decisions based on strategy
            modified_decisions = await self._apply_strategy(decisions, strategy)
            
            # Replay with modified decisions
            result = await self.replay_decisions(modified_decisions, config)
            results[strategy] = result
            
        return results
        
    async def test_whatif_scenario(
        self,
        decisions: List[DecisionRecord],
        scenario: Dict[str, Any]
    ) -> ReplayResult:
        """
        Test what-if scenarios on historical decisions.
        
        Args:
            decisions: Historical decisions
            scenario: What-if scenario parameters
            
        Returns:
            Replay results for scenario
        """
        config = ReplayConfig(
            mode=ReplayMode.WHATIF,
            whatif_scenarios=scenario
        )
        
        return await self.replay_decisions(decisions, config)
        
    async def _replay_historical(
        self,
        decisions: List[DecisionRecord],
        config: ReplayConfig
    ) -> List[Dict[str, Any]]:
        """Replay with historical market data"""
        results = []
        
        for decision in decisions:
            # Skip if configured
            if config.skip_failed and decision.outcome == DecisionOutcome.LOSS:
                continue
                
            # Get historical market state
            market_state = await self._get_historical_market_state(
                decision.market_ticker,
                decision.timestamp
            )
            
            # Replay decision
            result = await self._replay_single_decision(
                decision,
                market_state,
                config
            )
            
            results.append(result)
            
            # Control replay speed
            await self._control_replay_speed(config.speed)
            
            # Trigger callbacks
            for callback in self.decision_callbacks:
                asyncio.create_task(callback(decision, result))
                
        return results
        
    async def _replay_simulated(
        self,
        decisions: List[DecisionRecord],
        config: ReplayConfig
    ) -> List[Dict[str, Any]]:
        """Replay with simulated market conditions"""
        results = []
        
        for decision in decisions:
            # Generate simulated market state
            market_state = await self._generate_simulated_market_state(
                decision.market_ticker,
                decision.timestamp,
                decision.market_context
            )
            
            # Replay decision
            result = await self._replay_single_decision(
                decision,
                market_state,
                config
            )
            
            results.append(result)
            
            # Update simulated market based on decision
            await self._update_simulated_market(market_state, decision, result)
            
        return results
        
    async def _replay_comparative(
        self,
        decisions: List[DecisionRecord],
        config: ReplayConfig
    ) -> List[Dict[str, Any]]:
        """Replay comparing multiple strategies"""
        results = []
        
        for strategy in config.comparison_strategies:
            strategy_results = []
            
            for decision in decisions:
                # Modify decision based on strategy
                modified_decision = await self._modify_decision_for_strategy(
                    decision,
                    strategy
                )
                
                # Get market state
                market_state = await self._get_historical_market_state(
                    decision.market_ticker,
                    decision.timestamp
                )
                
                # Replay modified decision
                result = await self._replay_single_decision(
                    modified_decision,
                    market_state,
                    config
                )
                
                strategy_results.append(result)
                
            results.append({
                'strategy': strategy,
                'results': strategy_results
            })
            
        return results
        
    async def _replay_whatif(
        self,
        decisions: List[DecisionRecord],
        config: ReplayConfig
    ) -> List[Dict[str, Any]]:
        """Replay with what-if scenarios"""
        results = []
        scenario = config.whatif_scenarios
        
        for decision in decisions:
            # Apply what-if modifications
            modified_decision = await self._apply_whatif_scenario(
                decision,
                scenario
            )
            
            # Modify market conditions based on scenario
            market_state = await self._modify_market_for_scenario(
                decision.market_ticker,
                decision.timestamp,
                scenario
            )
            
            # Replay with modifications
            result = await self._replay_single_decision(
                modified_decision,
                market_state,
                config
            )
            
            results.append(result)
            
        return results
        
    async def _replay_single_decision(
        self,
        decision: DecisionRecord,
        market_state: MarketState,
        config: ReplayConfig
    ) -> Dict[str, Any]:
        """Replay a single decision"""
        # Calculate entry with slippage
        if config.apply_slippage:
            entry_price = self._calculate_slippage(
                decision.entry_price,
                decision.decision_type,
                market_state
            )
        else:
            entry_price = decision.entry_price
            
        # Calculate market impact
        if config.include_market_impact:
            market_impact = self._calculate_market_impact(
                decision.position_size,
                market_state
            )
        else:
            market_impact = 0
            
        # Simulate execution
        execution_price = entry_price + market_impact
        
        # Calculate outcome
        if decision.exit_price:
            exit_price = decision.exit_price
        else:
            # Estimate exit based on market conditions
            exit_price = await self._estimate_exit_price(
                decision,
                market_state
            )
            
        # Calculate P&L
        if decision.decision_type == DecisionType.BUY:
            pnl = (exit_price - execution_price) * decision.position_size
        elif decision.decision_type == DecisionType.SELL:
            pnl = (execution_price - exit_price) * decision.position_size
        else:
            pnl = 0
            
        # Calculate alternatives if requested
        alternatives = []
        if config.calculate_alternatives:
            alternatives = await self._calculate_alternative_decisions(
                decision,
                market_state
            )
            
        return {
            'decision_id': decision.decision_id,
            'original_pnl': decision.profit_loss,
            'replayed_pnl': pnl,
            'execution_price': execution_price,
            'exit_price': exit_price,
            'slippage': execution_price - decision.entry_price,
            'market_impact': market_impact,
            'alternatives': alternatives,
            'market_state': market_state.__dict__ if market_state else None
        }
        
    async def _calculate_alternative_decisions(
        self,
        decision: DecisionRecord,
        market_state: MarketState
    ) -> List[Dict[str, Any]]:
        """Calculate alternative decision outcomes"""
        alternatives = []
        
        # Alternative: Different position size
        for size_multiplier in [0.5, 2.0]:
            alt_size = decision.position_size * size_multiplier
            alt_pnl = await self._calculate_pnl_for_size(
                decision,
                alt_size,
                market_state
            )
            alternatives.append({
                'type': 'position_size',
                'multiplier': size_multiplier,
                'size': alt_size,
                'pnl': alt_pnl
            })
            
        # Alternative: Different timing
        for time_offset in [-60, 60]:  # +/- 1 minute
            alt_market = await self._get_offset_market_state(
                decision.market_ticker,
                decision.timestamp + timedelta(seconds=time_offset)
            )
            if alt_market:
                alt_pnl = await self._calculate_pnl_with_market(
                    decision,
                    alt_market
                )
                alternatives.append({
                    'type': 'timing',
                    'offset_seconds': time_offset,
                    'pnl': alt_pnl
                })
                
        # Alternative: Opposite decision
        if decision.decision_type == DecisionType.BUY:
            opposite_type = DecisionType.SELL
        elif decision.decision_type == DecisionType.SELL:
            opposite_type = DecisionType.BUY
        else:
            opposite_type = DecisionType.HOLD
            
        opposite_pnl = await self._calculate_opposite_decision_pnl(
            decision,
            opposite_type,
            market_state
        )
        alternatives.append({
            'type': 'opposite',
            'decision': opposite_type.value,
            'pnl': opposite_pnl
        })
        
        return alternatives
        
    async def _calculate_replay_results(
        self,
        session_id: str,
        config: ReplayConfig,
        decisions: List[DecisionRecord],
        results: List[Dict[str, Any]]
    ) -> ReplayResult:
        """Calculate comprehensive replay results"""
        # Calculate P&L
        original_pnl = sum(d.profit_loss or 0 for d in decisions)
        replayed_pnl = sum(r['replayed_pnl'] for r in results)
        
        # Calculate Sharpe ratio
        original_returns = [d.profit_loss or 0 for d in decisions if d.profit_loss is not None]
        replayed_returns = [r['replayed_pnl'] for r in results]
        
        original_sharpe = self._calculate_sharpe_ratio(original_returns)
        replayed_sharpe = self._calculate_sharpe_ratio(replayed_returns)
        
        # Find better/worse decisions
        better_decisions = []
        worse_decisions = []
        
        for i, (decision, result) in enumerate(zip(decisions, results)):
            if result['replayed_pnl'] > (decision.profit_loss or 0):
                better_decisions.append(decision.decision_id)
            elif result['replayed_pnl'] < (decision.profit_loss or 0):
                worse_decisions.append(decision.decision_id)
                
        # Find missed opportunities
        missed_opportunities = await self._find_missed_opportunities(
            decisions,
            results
        )
        
        # Calculate max drawdown
        max_drawdown = self._calculate_max_drawdown(replayed_returns)
        
        return ReplayResult(
            session_id=session_id,
            mode=config.mode,
            decisions_replayed=len(decisions),
            original_pnl=original_pnl,
            replayed_pnl=replayed_pnl,
            improvement=replayed_pnl - original_pnl,
            original_sharpe=original_sharpe,
            replayed_sharpe=replayed_sharpe,
            max_drawdown=max_drawdown,
            better_decisions=better_decisions,
            worse_decisions=worse_decisions,
            missed_opportunities=missed_opportunities,
            decision_results=results,
            market_conditions={},
            summary="",
            recommendations=[]
        )
        
    async def _generate_report(self, result: ReplayResult) -> str:
        """Generate replay report"""
        report = f"""
        Decision Replay Report
        ======================
        Session: {result.session_id}
        Mode: {result.mode.value}
        Decisions Replayed: {result.decisions_replayed}
        
        Performance Comparison
        ----------------------
        Original P&L: ${result.original_pnl:.2f}
        Replayed P&L: ${result.replayed_pnl:.2f}
        Improvement: ${result.improvement:.2f} ({result.improvement/abs(result.original_pnl)*100:.1f}%)
        
        Risk Metrics
        ------------
        Original Sharpe: {result.original_sharpe:.2f}
        Replayed Sharpe: {result.replayed_sharpe:.2f}
        Max Drawdown: {result.max_drawdown:.2%}
        
        Decision Analysis
        -----------------
        Better Decisions: {len(result.better_decisions)}
        Worse Decisions: {len(result.worse_decisions)}
        Missed Opportunities: {len(result.missed_opportunities)}
        """
        
        return report.strip()
        
    async def _generate_recommendations(self, result: ReplayResult) -> List[str]:
        """Generate recommendations based on replay"""
        recommendations = []
        
        # Performance recommendations
        if result.improvement > 0:
            recommendations.append(
                f"Replay shows potential for ${result.improvement:.2f} improvement"
            )
            
        # Sharpe ratio recommendations
        if result.replayed_sharpe > result.original_sharpe:
            recommendations.append(
                "Risk-adjusted returns improved in replay"
            )
            
        # Decision quality recommendations
        if len(result.worse_decisions) > len(result.better_decisions):
            recommendations.append(
                "Consider reviewing decision criteria - more decisions performed worse in replay"
            )
            
        # Missed opportunities
        if result.missed_opportunities:
            recommendations.append(
                f"Identified {len(result.missed_opportunities)} missed trading opportunities"
            )
            
        # Drawdown recommendations
        if result.max_drawdown > 0.2:
            recommendations.append(
                "High drawdown detected - consider stricter risk management"
            )
            
        return recommendations
        
    async def _initialize_market_states(self, decisions: List[DecisionRecord]):
        """Initialize market states for replay"""
        for decision in decisions:
            if decision.market_ticker not in self.market_states:
                self.market_states[decision.market_ticker] = MarketState(
                    market_ticker=decision.market_ticker,
                    yes_price=decision.entry_price,
                    no_price=1 - decision.entry_price,
                    volume=decision.market_context.volume,
                    volatility=decision.market_context.volatility
                )
                
    async def _get_historical_market_state(
        self,
        market_ticker: str,
        timestamp: datetime
    ) -> Optional[MarketState]:
        """Get historical market state"""
        # Would query historical data source
        # For now, return cached state
        return self.market_states.get(market_ticker)
        
    async def _generate_simulated_market_state(
        self,
        market_ticker: str,
        timestamp: datetime,
        context: Any
    ) -> MarketState:
        """Generate simulated market state"""
        # Use market simulator
        return MarketState(
            market_ticker=market_ticker,
            yes_price=context.price,
            no_price=1 - context.price,
            volume=context.volume,
            volatility=context.volatility,
            momentum=context.momentum,
            liquidity=context.liquidity
        )
        
    def _calculate_slippage(
        self,
        price: float,
        decision_type: DecisionType,
        market_state: MarketState
    ) -> float:
        """Calculate execution slippage"""
        spread = market_state.bid_ask_spread if hasattr(market_state, 'bid_ask_spread') else 0.01
        
        if decision_type == DecisionType.BUY:
            return price + spread / 2
        elif decision_type == DecisionType.SELL:
            return price - spread / 2
        else:
            return price
            
    def _calculate_market_impact(
        self,
        size: float,
        market_state: MarketState
    ) -> float:
        """Calculate market impact of trade"""
        if not market_state:
            return 0
            
        liquidity = market_state.liquidity if hasattr(market_state, 'liquidity') else 0.5
        impact = size * 0.001 / liquidity  # Simple linear impact model
        
        return min(impact, 0.05)  # Cap at 5% impact
        
    async def _estimate_exit_price(
        self,
        decision: DecisionRecord,
        market_state: MarketState
    ) -> float:
        """Estimate exit price for decision"""
        # Simple model: price moves based on decision outcome
        if decision.outcome == DecisionOutcome.PROFITABLE:
            if decision.decision_type == DecisionType.BUY:
                return decision.entry_price * 1.02  # 2% profit
            else:
                return decision.entry_price * 0.98
        elif decision.outcome == DecisionOutcome.LOSS:
            if decision.decision_type == DecisionType.BUY:
                return decision.entry_price * 0.98  # 2% loss
            else:
                return decision.entry_price * 1.02
        else:
            return decision.entry_price
            
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio"""
        if not returns or len(returns) < 2:
            return 0
            
        import numpy as np
        returns_array = np.array(returns)
        
        if returns_array.std() == 0:
            return 0
            
        return returns_array.mean() / returns_array.std() * np.sqrt(252)  # Annualized
        
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not returns:
            return 0
            
        cumulative = []
        cum_sum = 0
        
        for r in returns:
            cum_sum += r
            cumulative.append(cum_sum)
            
        peak = cumulative[0]
        max_dd = 0
        
        for value in cumulative:
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak != 0 else 0
            max_dd = max(max_dd, dd)
            
        return max_dd
        
    async def _control_replay_speed(self, speed: ReplaySpeed):
        """Control replay execution speed"""
        if speed == ReplaySpeed.REALTIME:
            await asyncio.sleep(1.0)
        elif speed == ReplaySpeed.FAST:
            await asyncio.sleep(0.1)
        elif speed == ReplaySpeed.STEP:
            # Would wait for user input
            await asyncio.sleep(0.5)
        # MAX speed - no delay
        
    async def _load_session_decisions(self, session_id: str) -> List[DecisionRecord]:
        """Load all decisions from a session"""
        if not self.redis_client:
            return []
            
        # Get all decision IDs for session
        pattern = f"session:{session_id}:decisions:*"
        keys = []
        
        async for key in self.redis_client.scan_iter(match=pattern):
            keys.append(key)
            
        # Load decisions
        decisions = []
        for key in keys:
            data = await self.redis_client.get(key)
            if data:
                decision = DecisionRecord.from_dict(json.loads(data))
                decisions.append(decision)
                
        return decisions
        
    async def _apply_strategy(
        self,
        decisions: List[DecisionRecord],
        strategy: str
    ) -> List[DecisionRecord]:
        """Apply strategy modifications to decisions"""
        # Strategy-specific modifications
        # This would be customized based on available strategies
        return decisions
        
    async def _modify_decision_for_strategy(
        self,
        decision: DecisionRecord,
        strategy: str
    ) -> DecisionRecord:
        """Modify decision based on strategy"""
        # Create copy and modify
        import copy
        modified = copy.deepcopy(decision)
        
        # Apply strategy-specific changes
        if strategy == "conservative":
            modified.position_size *= 0.5
            modified.confidence *= 0.8
        elif strategy == "aggressive":
            modified.position_size *= 1.5
            modified.confidence *= 1.2
            
        return modified
        
    async def _apply_whatif_scenario(
        self,
        decision: DecisionRecord,
        scenario: Dict[str, Any]
    ) -> DecisionRecord:
        """Apply what-if scenario to decision"""
        import copy
        modified = copy.deepcopy(decision)
        
        # Apply scenario modifications
        if "position_multiplier" in scenario:
            modified.position_size *= scenario["position_multiplier"]
        if "confidence_adjustment" in scenario:
            modified.confidence += scenario["confidence_adjustment"]
            
        return modified
        
    async def _modify_market_for_scenario(
        self,
        market_ticker: str,
        timestamp: datetime,
        scenario: Dict[str, Any]
    ) -> MarketState:
        """Modify market state for scenario"""
        base_state = self.market_states.get(market_ticker)
        if not base_state:
            return None
            
        import copy
        modified = copy.deepcopy(base_state)
        
        # Apply market modifications
        if "volatility_multiplier" in scenario:
            modified.volatility *= scenario["volatility_multiplier"]
        if "liquidity_adjustment" in scenario:
            modified.liquidity += scenario["liquidity_adjustment"]
            
        return modified
        
    async def _update_simulated_market(
        self,
        market_state: MarketState,
        decision: DecisionRecord,
        result: Dict[str, Any]
    ):
        """Update simulated market after decision"""
        # Update market state based on trade
        if decision.decision_type == DecisionType.BUY:
            market_state.yes_price += 0.001 * decision.position_size
        elif decision.decision_type == DecisionType.SELL:
            market_state.yes_price -= 0.001 * decision.position_size
            
        market_state.volume += int(decision.position_size * 1000)
        
    async def _calculate_pnl_for_size(
        self,
        decision: DecisionRecord,
        size: float,
        market_state: MarketState
    ) -> float:
        """Calculate P&L for different position size"""
        # Simple calculation
        if decision.profit_loss:
            return decision.profit_loss * (size / decision.position_size)
        return 0
        
    async def _calculate_pnl_with_market(
        self,
        decision: DecisionRecord,
        market_state: MarketState
    ) -> float:
        """Calculate P&L with different market state"""
        # Would use market state to calculate
        return decision.profit_loss or 0
        
    async def _get_offset_market_state(
        self,
        market_ticker: str,
        timestamp: datetime
    ) -> Optional[MarketState]:
        """Get market state at offset time"""
        # Would query historical data
        return self.market_states.get(market_ticker)
        
    async def _calculate_opposite_decision_pnl(
        self,
        decision: DecisionRecord,
        opposite_type: DecisionType,
        market_state: MarketState
    ) -> float:
        """Calculate P&L for opposite decision"""
        # Invert the P&L
        if decision.profit_loss:
            return -decision.profit_loss
        return 0
        
    async def _find_missed_opportunities(
        self,
        decisions: List[DecisionRecord],
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find missed trading opportunities"""
        missed = []
        
        for i, result in enumerate(results):
            # Check alternatives
            if 'alternatives' in result:
                for alt in result['alternatives']:
                    if alt['pnl'] > result['replayed_pnl'] * 1.5:  # 50% better
                        missed.append({
                            'decision_id': result['decision_id'],
                            'alternative': alt['type'],
                            'potential_pnl': alt['pnl'],
                            'actual_pnl': result['replayed_pnl']
                        })
                        
        return missed
        
    async def cleanup(self):
        """Clean up resources"""
        if self.redis_client:
            await self.redis_client.close()
            
        logger.info("Decision replay engine cleaned up")