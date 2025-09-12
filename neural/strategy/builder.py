"""
Strategy Composition Framework

This module provides tools for building sophisticated multi-strategy systems
by combining, weighting, and orchestrating multiple individual strategies.

Key Features:
- Strategy combination and weighting
- Signal aggregation methods
- Risk-aware portfolio allocation
- Dynamic strategy selection
- Performance-based rebalancing

This framework enables building complex systems while maintaining modularity
and allowing for sophisticated risk management across multiple strategies.
"""

from typing import Dict, List, Optional, Any, Callable
import logging
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
import asyncio
import numpy as np

from neural.analysis.base import AnalysisResult, SignalStrength
from neural.strategy.base import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class AggregationMethod(Enum):
    """Methods for aggregating signals from multiple strategies."""
    WEIGHTED_AVERAGE = "weighted_average"
    MAJORITY_VOTE = "majority_vote"
    HIGHEST_CONFIDENCE = "highest_confidence"
    CONSENSUS_THRESHOLD = "consensus_threshold"
    DYNAMIC_WEIGHTING = "dynamic_weighting"


class AllocationMethod(Enum):
    """Methods for allocating capital across strategies."""
    EQUAL_WEIGHT = "equal_weight"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    PERFORMANCE_WEIGHTED = "performance_weighted"
    RISK_PARITY = "risk_parity"
    KELLY_WEIGHTED = "kelly_weighted"


@dataclass
class StrategyConfig:
    """Configuration for a strategy within the composition."""
    strategy: BaseStrategy
    weight: float = 1.0
    max_allocation: float = 0.25  # Max 25% of total capital
    min_confidence: float = 0.5
    enabled: bool = True
    performance_lookback: int = 30  # Days for performance calculation
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class CompositeSignal:
    """Aggregated signal from multiple strategies."""
    market_id: str
    action: str
    confidence: float
    signal_strength: SignalStrength
    position_size: float
    contributing_strategies: List[str]
    individual_signals: List[Signal]
    aggregation_method: AggregationMethod
    timestamp: datetime
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class StrategyComposer:
    """
    Compose and orchestrate multiple strategies into a unified system.
    
    This class provides sophisticated methods for combining multiple strategies
    while managing risk, allocation, and performance across the entire system.
    """
    
    def __init__(
        self,
        aggregation_method: AggregationMethod = AggregationMethod.WEIGHTED_AVERAGE,
        allocation_method: AllocationMethod = AllocationMethod.CONFIDENCE_WEIGHTED,
        max_total_allocation: float = 0.30,  # Max 30% of total capital deployed
        consensus_threshold: float = 0.6,   # For consensus methods
        rebalance_frequency: int = 7        # Days between rebalancing
    ):
        """
        Initialize strategy composer.
        
        Args:
            aggregation_method: How to combine signals from multiple strategies
            allocation_method: How to allocate capital across strategies
            max_total_allocation: Maximum total capital to deploy across all strategies
            consensus_threshold: Minimum agreement required for consensus methods
            rebalance_frequency: Days between strategy weight rebalancing
        """
        self.strategies: Dict[str, StrategyConfig] = {}
        self.aggregation_method = aggregation_method
        self.allocation_method = allocation_method
        self.max_total_allocation = max_total_allocation
        self.consensus_threshold = consensus_threshold
        self.rebalance_frequency = rebalance_frequency
        
        # Performance tracking
        self.performance_history: Dict[str, List[float]] = {}
        self.last_rebalance = datetime.now()
        
        logger.info(f"Initialized StrategyComposer with {aggregation_method.value} aggregation")
    
    def add_strategy(self, strategy_config: StrategyConfig) -> None:
        """
        Add a strategy to the composition.
        
        Args:
            strategy_config: Configuration for the strategy to add
        """
        strategy_id = strategy_config.strategy.name
        self.strategies[strategy_id] = strategy_config
        self.performance_history[strategy_id] = []
        
        logger.info(f"Added strategy: {strategy_id} with weight {strategy_config.weight}")
    
    def remove_strategy(self, strategy_id: str) -> bool:
        """
        Remove a strategy from the composition.
        
        Args:
            strategy_id: ID of strategy to remove
            
        Returns:
            True if strategy was removed, False if not found
        """
        if strategy_id in self.strategies:
            del self.strategies[strategy_id]
            del self.performance_history[strategy_id]
            logger.info(f"Removed strategy: {strategy_id}")
            return True
        return False
    
    def update_strategy_weight(self, strategy_id: str, weight: float) -> bool:
        """
        Update the weight of a strategy.
        
        Args:
            strategy_id: ID of strategy to update
            weight: New weight for the strategy
            
        Returns:
            True if updated successfully, False if strategy not found
        """
        if strategy_id in self.strategies:
            self.strategies[strategy_id].weight = weight
            logger.info(f"Updated {strategy_id} weight to {weight}")
            return True
        return False
    
    async def analyze(self, market_id: str, market_data: Dict[str, Any]) -> Optional[CompositeSignal]:
        """
        Generate composite signal from all enabled strategies.
        
        Args:
            market_id: Market identifier
            market_data: Dictionary containing market information
            
        Returns:
            CompositeSignal if signals generated, None otherwise
        """
        try:
            # Get signals from all enabled strategies
            individual_signals = await self._collect_individual_signals(market_id, market_data)
            
            if not individual_signals:
                return None
            
            # Aggregate signals using specified method
            composite_signal = self._aggregate_signals(
                market_id, individual_signals, market_data
            )
            
            return composite_signal
            
        except Exception as e:
            logger.error(f"Error generating composite signal for {market_id}: {e}")
            return None
    
    async def _collect_individual_signals(
        self, 
        market_id: str, 
        market_data: Dict[str, Any]
    ) -> List[Signal]:
        """
        Collect signals from all enabled strategies concurrently.
        """
        tasks = []
        
        for strategy_id, config in self.strategies.items():
            if config.enabled:
                task = self._get_strategy_signal(config, market_id, market_data)
                tasks.append(task)
        
        # Execute all strategy analyses concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter valid signals
        valid_signals = []
        for result in results:
            if isinstance(result, Signal):
                valid_signals.append(result)
            elif isinstance(result, Exception):
                logger.warning(f"Strategy analysis error: {result}")
        
        return valid_signals
    
    async def _get_strategy_signal(
        self, 
        config: StrategyConfig, 
        market_id: str, 
        market_data: Dict[str, Any]
    ) -> Optional[Signal]:
        """
        Get signal from individual strategy with confidence filtering.
        """
        try:
            signal = await config.strategy.analyze(market_id, market_data)
            
            if signal and signal.confidence >= config.min_confidence:
                return signal
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error getting signal from {config.strategy.name}: {e}")
            return None
    
    def _aggregate_signals(
        self, 
        market_id: str, 
        signals: List[Signal], 
        market_data: Dict[str, Any]
    ) -> Optional[CompositeSignal]:
        """
        Aggregate individual signals into a composite signal.
        """
        if not signals:
            return None
        
        if self.aggregation_method == AggregationMethod.WEIGHTED_AVERAGE:
            return self._weighted_average_aggregation(market_id, signals, market_data)
        elif self.aggregation_method == AggregationMethod.MAJORITY_VOTE:
            return self._majority_vote_aggregation(market_id, signals, market_data)
        elif self.aggregation_method == AggregationMethod.HIGHEST_CONFIDENCE:
            return self._highest_confidence_aggregation(market_id, signals, market_data)
        elif self.aggregation_method == AggregationMethod.CONSENSUS_THRESHOLD:
            return self._consensus_threshold_aggregation(market_id, signals, market_data)
        elif self.aggregation_method == AggregationMethod.DYNAMIC_WEIGHTING:
            return self._dynamic_weighting_aggregation(market_id, signals, market_data)
        else:
            logger.error(f"Unknown aggregation method: {self.aggregation_method}")
            return None
    
    def _weighted_average_aggregation(
        self, 
        market_id: str, 
        signals: List[Signal], 
        market_data: Dict[str, Any]
    ) -> Optional[CompositeSignal]:
        """
        Aggregate signals using weighted average of confidence and position sizes.
        """
        # Separate signals by action
        yes_signals = [s for s in signals if s.action == 'BUY_YES']
        no_signals = [s for s in signals if s.action == 'BUY_NO']
        
        # Calculate weighted scores for each action
        yes_score = self._calculate_weighted_score(yes_signals)
        no_score = self._calculate_weighted_score(no_signals)
        
        # Determine final action
        if yes_score > no_score and yes_score > 0:
            action = 'BUY_YES'
            final_confidence = yes_score
            contributing_signals = yes_signals
        elif no_score > 0:
            action = 'BUY_NO'
            final_confidence = no_score
            contributing_signals = no_signals
        else:
            return None  # No clear signal
        
        # Calculate composite position size
        position_size = self._calculate_composite_position_size(contributing_signals)
        
        # Determine signal strength
        signal_strength = self._determine_composite_strength(contributing_signals)
        
        return CompositeSignal(
            market_id=market_id,
            action=action,
            confidence=final_confidence,
            signal_strength=signal_strength,
            position_size=position_size,
            contributing_strategies=[s.strategy_id for s in contributing_signals],
            individual_signals=signals,
            aggregation_method=self.aggregation_method,
            timestamp=datetime.now(),
            reasoning=f"Weighted average: {len(contributing_signals)} strategies",
            metadata={
                'yes_score': yes_score,
                'no_score': no_score,
                'total_strategies': len(signals)
            }
        )
    
    def _majority_vote_aggregation(
        self, 
        market_id: str, 
        signals: List[Signal], 
        market_data: Dict[str, Any]
    ) -> Optional[CompositeSignal]:
        """
        Aggregate signals using majority vote with confidence weighting.
        """
        yes_votes = len([s for s in signals if s.action == 'BUY_YES'])
        no_votes = len([s for s in signals if s.action == 'BUY_NO'])
        
        if yes_votes > no_votes:
            action = 'BUY_YES'
            contributing_signals = [s for s in signals if s.action == 'BUY_YES']
        elif no_votes > yes_votes:
            action = 'BUY_NO'
            contributing_signals = [s for s in signals if s.action == 'BUY_NO']
        else:
            return None  # Tie - no clear majority
        
        # Average confidence of majority
        avg_confidence = np.mean([s.confidence for s in contributing_signals])
        
        # Conservative position sizing for majority vote
        position_size = min(
            np.mean([s.position_size for s in contributing_signals]) * 0.8,
            self.max_total_allocation / 2
        )
        
        signal_strength = self._determine_composite_strength(contributing_signals)
        
        return CompositeSignal(
            market_id=market_id,
            action=action,
            confidence=avg_confidence,
            signal_strength=signal_strength,
            position_size=position_size,
            contributing_strategies=[s.strategy_id for s in contributing_signals],
            individual_signals=signals,
            aggregation_method=self.aggregation_method,
            timestamp=datetime.now(),
            reasoning=f"Majority vote: {len(contributing_signals)}/{len(signals)} strategies",
            metadata={
                'yes_votes': yes_votes,
                'no_votes': no_votes
            }
        )
    
    def _highest_confidence_aggregation(
        self, 
        market_id: str, 
        signals: List[Signal], 
        market_data: Dict[str, Any]
    ) -> Optional[CompositeSignal]:
        """
        Use the signal with highest confidence.
        """
        best_signal = max(signals, key=lambda s: s.confidence)
        
        # Scale down position size for single-strategy decision
        position_size = best_signal.position_size * 0.7
        
        return CompositeSignal(
            market_id=market_id,
            action=best_signal.action,
            confidence=best_signal.confidence,
            signal_strength=best_signal.signal_strength,
            position_size=position_size,
            contributing_strategies=[best_signal.strategy_id],
            individual_signals=signals,
            aggregation_method=self.aggregation_method,
            timestamp=datetime.now(),
            reasoning=f"Highest confidence: {best_signal.strategy_id}",
            metadata={
                'best_strategy': best_signal.strategy_id,
                'best_confidence': best_signal.confidence
            }
        )
    
    def _consensus_threshold_aggregation(
        self, 
        market_id: str, 
        signals: List[Signal], 
        market_data: Dict[str, Any]
    ) -> Optional[CompositeSignal]:
        """
        Require consensus above threshold to generate signal.
        """
        yes_signals = [s for s in signals if s.action == 'BUY_YES']
        no_signals = [s for s in signals if s.action == 'BUY_NO']
        
        yes_ratio = len(yes_signals) / len(signals)
        no_ratio = len(no_signals) / len(signals)
        
        if yes_ratio >= self.consensus_threshold:
            action = 'BUY_YES'
            contributing_signals = yes_signals
        elif no_ratio >= self.consensus_threshold:
            action = 'BUY_NO'
            contributing_signals = no_signals
        else:
            return None  # No consensus
        
        # High confidence for consensus
        avg_confidence = np.mean([s.confidence for s in contributing_signals])
        consensus_boost = (len(contributing_signals) / len(signals)) * 0.1
        final_confidence = min(avg_confidence + consensus_boost, 0.95)
        
        position_size = self._calculate_composite_position_size(contributing_signals)
        signal_strength = self._determine_composite_strength(contributing_signals)
        
        return CompositeSignal(
            market_id=market_id,
            action=action,
            confidence=final_confidence,
            signal_strength=signal_strength,
            position_size=position_size,
            contributing_strategies=[s.strategy_id for s in contributing_signals],
            individual_signals=signals,
            aggregation_method=self.aggregation_method,
            timestamp=datetime.now(),
            reasoning=f"Consensus: {len(contributing_signals)}/{len(signals)} strategies",
            metadata={
                'consensus_ratio': len(contributing_signals) / len(signals),
                'threshold': self.consensus_threshold
            }
        )
    
    def _dynamic_weighting_aggregation(
        self, 
        market_id: str, 
        signals: List[Signal], 
        market_data: Dict[str, Any]
    ) -> Optional[CompositeSignal]:
        """
        Use dynamic weighting based on recent strategy performance.
        """
        # Calculate performance-based weights
        performance_weights = self._calculate_performance_weights(signals)
        
        # Apply weights to signals
        weighted_signals = []
        for signal in signals:
            weight = performance_weights.get(signal.strategy_id, 1.0)
            weighted_signal = Signal(
                strategy_id=signal.strategy_id,
                market_id=signal.market_id,
                action=signal.action,
                confidence=signal.confidence * weight,
                signal_strength=signal.signal_strength,
                position_size=signal.position_size * weight,
                timestamp=signal.timestamp,
                reasoning=f"Performance weighted: {weight:.2f}",
                metadata=signal.metadata
            )
            weighted_signals.append(weighted_signal)
        
        # Use weighted average on the adjusted signals
        return self._weighted_average_aggregation(market_id, weighted_signals, market_data)
    
    def _calculate_weighted_score(self, signals: List[Signal]) -> float:
        """
        Calculate weighted score for a set of signals.
        """
        if not signals:
            return 0.0
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        for signal in signals:
            strategy_config = self.strategies.get(signal.strategy_id)
            if strategy_config:
                weight = strategy_config.weight
                total_weight += weight
                weighted_sum += signal.confidence * weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _calculate_composite_position_size(self, signals: List[Signal]) -> float:
        """
        Calculate composite position size with risk management.
        """
        if not signals:
            return 0.0
        
        # Start with average position size
        avg_position = np.mean([s.position_size for s in signals])
        
        # Apply diversification bonus for multiple strategies
        diversification_factor = min(1.0 + (len(signals) - 1) * 0.1, 1.5)
        
        # Apply total allocation limit
        composite_size = min(
            avg_position * diversification_factor,
            self.max_total_allocation
        )
        
        return composite_size
    
    def _determine_composite_strength(self, signals: List[Signal]) -> SignalStrength:
        """
        Determine composite signal strength from individual signals.
        """
        if not signals:
            return SignalStrength.WEAK
        
        # Count strength levels
        strong_count = sum(1 for s in signals if s.signal_strength == SignalStrength.STRONG)
        moderate_count = sum(1 for s in signals if s.signal_strength == SignalStrength.MODERATE)
        
        # Determine based on majority and total count
        total = len(signals)
        strong_ratio = strong_count / total
        moderate_ratio = moderate_count / total
        
        if strong_ratio >= 0.6 or (strong_count >= 2 and total >= 3):
            return SignalStrength.STRONG
        elif strong_ratio + moderate_ratio >= 0.7:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK
    
    def _calculate_performance_weights(self, signals: List[Signal]) -> Dict[str, float]:
        """
        Calculate performance-based weights for strategies.
        
        This is a placeholder for more sophisticated performance tracking.
        """
        weights = {}
        for signal in signals:
            # Default weight of 1.0 - in practice, this would use historical performance
            weights[signal.strategy_id] = 1.0
        return weights
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """
        Get summary of all strategies in the composition.
        """
        summary = {
            'total_strategies': len(self.strategies),
            'enabled_strategies': sum(1 for config in self.strategies.values() if config.enabled),
            'aggregation_method': self.aggregation_method.value,
            'allocation_method': self.allocation_method.value,
            'max_total_allocation': self.max_total_allocation,
            'strategies': {}
        }
        
        for strategy_id, config in self.strategies.items():
            summary['strategies'][strategy_id] = {
                'weight': config.weight,
                'max_allocation': config.max_allocation,
                'enabled': config.enabled,
                'min_confidence': config.min_confidence
            }
        
        return summary
