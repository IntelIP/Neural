"""
Advanced Signal Processing

This module provides sophisticated signal processing capabilities for
trading signal management, including filtering, timing, decay, and aggregation.

Key Features:
- Signal filtering and validation
- Time-based signal decay
- Signal aggregation across timeframes
- Confidence thresholding
- Signal persistence and memory
- Performance-based signal weighting

These tools enable building robust trading systems that can handle
complex signal flows and make intelligent decisions about signal timing
and reliability.
"""

from typing import Dict, List, Optional, Any, Callable, Tuple
import logging
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import numpy as np
from collections import deque

from neural.analysis.base import SignalStrength
from neural.strategy.base import Signal

logger = logging.getLogger(__name__)


class SignalFilter(Enum):
    """Types of signal filters available."""
    CONFIDENCE_THRESHOLD = "confidence_threshold"
    STRENGTH_MINIMUM = "strength_minimum"
    TEMPORAL_CONSISTENCY = "temporal_consistency"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    CORRELATION_FILTER = "correlation_filter"


class DecayFunction(Enum):
    """Signal decay function types."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    STEP = "step"
    SIGMOID = "sigmoid"


@dataclass
class SignalHistory:
    """Historical record of signals for a market."""
    market_id: str
    signals: deque = field(default_factory=lambda: deque(maxlen=100))
    last_signal_time: Optional[datetime] = None
    signal_count: int = 0
    performance_score: float = 0.5  # Track how well signals performed


@dataclass
class ProcessedSignal:
    """Signal with processing metadata."""
    original_signal: Signal
    processed_confidence: float
    decay_factor: float
    filter_status: Dict[str, bool]
    processing_timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class SignalProcessor:
    """
    Advanced signal processing engine.
    
    This class provides sophisticated signal processing capabilities
    including filtering, timing, decay, and aggregation for building
    robust trading systems.
    """
    
    def __init__(
        self,
        decay_function: DecayFunction = DecayFunction.EXPONENTIAL,
        decay_half_life: int = 30,  # minutes
        min_confidence: float = 0.5,
        max_signal_age: int = 120,  # minutes
        enable_temporal_filtering: bool = True
    ):
        """
        Initialize signal processor.
        
        Args:
            decay_function: Function to use for signal decay
            decay_half_life: Half-life for signal decay (minutes)
            min_confidence: Minimum confidence to process signals
            max_signal_age: Maximum age for signals (minutes)
            enable_temporal_filtering: Enable temporal consistency filtering
        """
        self.decay_function = decay_function
        self.decay_half_life = decay_half_life
        self.min_confidence = min_confidence
        self.max_signal_age = max_signal_age
        self.enable_temporal_filtering = enable_temporal_filtering
        
        # Signal history tracking
        self.signal_histories: Dict[str, SignalHistory] = {}
        
        # Filter configurations
        self.filters: Dict[SignalFilter, Dict[str, Any]] = {}
        
        # Performance tracking
        self.strategy_performance: Dict[str, float] = {}
        
        logger.info("Initialized SignalProcessor")
    
    def add_filter(self, filter_type: SignalFilter, **kwargs) -> None:
        """
        Add a signal filter with configuration.
        
        Args:
            filter_type: Type of filter to add
            **kwargs: Filter-specific configuration parameters
        """
        self.filters[filter_type] = kwargs
        logger.info(f"Added {filter_type.value} filter")
    
    def remove_filter(self, filter_type: SignalFilter) -> None:
        """Remove a signal filter."""
        if filter_type in self.filters:
            del self.filters[filter_type]
            logger.info(f"Removed {filter_type.value} filter")
    
    def process_signal(self, signal: Signal) -> Optional[ProcessedSignal]:
        """
        Process a signal through all configured filters and transformations.
        
        Args:
            signal: Raw signal to process
            
        Returns:
            ProcessedSignal if signal passes all filters, None otherwise
        """
        try:
            # Store signal in history
            self._update_signal_history(signal)
            
            # Apply age filter first
            if not self._check_signal_age(signal):
                return None
            
            # Apply decay function
            decay_factor = self._calculate_decay_factor(signal)
            
            # Calculate processed confidence
            processed_confidence = signal.confidence * decay_factor
            
            # Apply all configured filters
            filter_status = {}
            for filter_type in self.filters:
                passed = self._apply_filter(filter_type, signal, processed_confidence)
                filter_status[filter_type.value] = passed
                
                if not passed:
                    logger.debug(f"Signal failed {filter_type.value} filter")
                    return None
            
            # Apply performance-based weighting
            performance_weight = self._get_strategy_performance_weight(signal.strategy_id)
            final_confidence = processed_confidence * performance_weight
            
            # Check final confidence threshold
            if final_confidence < self.min_confidence:
                logger.debug(f"Final confidence {final_confidence:.2f} below threshold")
                return None
            
            return ProcessedSignal(
                original_signal=signal,
                processed_confidence=final_confidence,
                decay_factor=decay_factor,
                filter_status=filter_status,
                processing_timestamp=datetime.now(),
                metadata={
                    'performance_weight': performance_weight,
                    'filters_applied': list(filter_status.keys())
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
            return None
    
    def aggregate_signals(
        self, 
        signals: List[Signal], 
        aggregation_window: int = 15  # minutes
    ) -> Optional[Signal]:
        """
        Aggregate multiple signals from the same market over a time window.
        
        Args:
            signals: List of signals to aggregate
            aggregation_window: Time window for aggregation (minutes)
            
        Returns:
            Aggregated signal or None
        """
        if not signals:
            return None
        
        # Group signals by market
        market_signals = {}
        for signal in signals:
            if signal.market_id not in market_signals:
                market_signals[signal.market_id] = []
            market_signals[signal.market_id].append(signal)
        
        # Process each market separately
        aggregated_signals = []
        for market_id, market_signal_list in market_signals.items():
            aggregated = self._aggregate_market_signals(market_signal_list, aggregation_window)
            if aggregated:
                aggregated_signals.append(aggregated)
        
        # Return the most confident aggregated signal
        if aggregated_signals:
            return max(aggregated_signals, key=lambda s: s.confidence)
        
        return None
    
    def get_signal_consensus(
        self, 
        market_id: str, 
        lookback_minutes: int = 60
    ) -> Dict[str, Any]:
        """
        Calculate consensus metrics for recent signals on a market.
        
        Args:
            market_id: Market to analyze
            lookback_minutes: How far back to look for signals
            
        Returns:
            Dictionary with consensus metrics
        """
        if market_id not in self.signal_histories:
            return {'consensus_score': 0.0, 'signal_count': 0}
        
        history = self.signal_histories[market_id]
        cutoff_time = datetime.now() - timedelta(minutes=lookback_minutes)
        
        # Filter recent signals
        recent_signals = [
            s for s in history.signals 
            if s.timestamp >= cutoff_time
        ]
        
        if not recent_signals:
            return {'consensus_score': 0.0, 'signal_count': 0}
        
        # Calculate consensus metrics
        yes_signals = [s for s in recent_signals if s.action == 'BUY_YES']
        no_signals = [s for s in recent_signals if s.action == 'BUY_NO']
        
        total_signals = len(recent_signals)
        yes_ratio = len(yes_signals) / total_signals
        no_ratio = len(no_signals) / total_signals
        
        # Consensus score (higher when signals agree)
        consensus_score = max(yes_ratio, no_ratio)
        
        # Average confidence
        avg_confidence = np.mean([s.confidence for s in recent_signals])
        
        # Strength distribution
        strength_counts = {
            'STRONG': len([s for s in recent_signals if s.signal_strength == SignalStrength.STRONG]),
            'MODERATE': len([s for s in recent_signals if s.signal_strength == SignalStrength.MODERATE]),
            'WEAK': len([s for s in recent_signals if s.signal_strength == SignalStrength.WEAK])
        }
        
        return {
            'consensus_score': consensus_score,
            'signal_count': total_signals,
            'yes_ratio': yes_ratio,
            'no_ratio': no_ratio,
            'avg_confidence': avg_confidence,
            'strength_distribution': strength_counts,
            'dominant_action': 'BUY_YES' if yes_ratio > no_ratio else 'BUY_NO'
        }
    
    def _update_signal_history(self, signal: Signal) -> None:
        """Update signal history for a market."""
        if signal.market_id not in self.signal_histories:
            self.signal_histories[signal.market_id] = SignalHistory(
                market_id=signal.market_id
            )
        
        history = self.signal_histories[signal.market_id]
        history.signals.append(signal)
        history.last_signal_time = signal.timestamp
        history.signal_count += 1
    
    def _check_signal_age(self, signal: Signal) -> bool:
        """Check if signal is within acceptable age limit."""
        age_minutes = (datetime.now() - signal.timestamp).total_seconds() / 60
        return age_minutes <= self.max_signal_age
    
    def _calculate_decay_factor(self, signal: Signal) -> float:
        """
        Calculate decay factor based on signal age and configured function.
        """
        age_minutes = (datetime.now() - signal.timestamp).total_seconds() / 60
        
        if age_minutes <= 0:
            return 1.0
        
        if self.decay_function == DecayFunction.LINEAR:
            return max(0, 1 - (age_minutes / (self.decay_half_life * 2)))
        
        elif self.decay_function == DecayFunction.EXPONENTIAL:
            return 0.5 ** (age_minutes / self.decay_half_life)
        
        elif self.decay_function == DecayFunction.STEP:
            if age_minutes < self.decay_half_life:
                return 1.0
            elif age_minutes < self.decay_half_life * 2:
                return 0.5
            else:
                return 0.1
        
        elif self.decay_function == DecayFunction.SIGMOID:
            # Sigmoid decay centered at half_life
            x = (age_minutes - self.decay_half_life) / (self.decay_half_life / 4)
            return 1 / (1 + np.exp(x))
        
        else:
            return 1.0  # No decay
    
    def _apply_filter(
        self, 
        filter_type: SignalFilter, 
        signal: Signal, 
        processed_confidence: float
    ) -> bool:
        """
        Apply a specific filter to a signal.
        
        Args:
            filter_type: Type of filter to apply
            signal: Signal to filter
            processed_confidence: Confidence after decay
            
        Returns:
            True if signal passes filter, False otherwise
        """
        filter_config = self.filters.get(filter_type, {})
        
        if filter_type == SignalFilter.CONFIDENCE_THRESHOLD:
            threshold = filter_config.get('threshold', 0.6)
            return processed_confidence >= threshold
        
        elif filter_type == SignalFilter.STRENGTH_MINIMUM:
            min_strength = filter_config.get('min_strength', SignalStrength.WEAK)
            strength_order = {SignalStrength.WEAK: 0, SignalStrength.MODERATE: 1, SignalStrength.STRONG: 2}
            return strength_order.get(signal.signal_strength, 0) >= strength_order.get(min_strength, 0)
        
        elif filter_type == SignalFilter.TEMPORAL_CONSISTENCY:
            return self._check_temporal_consistency(signal, filter_config)
        
        elif filter_type == SignalFilter.VOLATILITY_ADJUSTED:
            return self._check_volatility_filter(signal, filter_config)
        
        elif filter_type == SignalFilter.CORRELATION_FILTER:
            return self._check_correlation_filter(signal, filter_config)
        
        else:
            logger.warning(f"Unknown filter type: {filter_type}")
            return True  # Pass unknown filters
    
    def _check_temporal_consistency(self, signal: Signal, config: Dict[str, Any]) -> bool:
        """
        Check if signal is consistent with recent signal history.
        """
        if signal.market_id not in self.signal_histories:
            return True  # No history to check against
        
        history = self.signal_histories[signal.market_id]
        lookback_minutes = config.get('lookback_minutes', 30)
        min_consistency = config.get('min_consistency', 0.6)
        
        # Get recent signals
        cutoff_time = datetime.now() - timedelta(minutes=lookback_minutes)
        recent_signals = [
            s for s in history.signals 
            if s.timestamp >= cutoff_time
        ]
        
        if len(recent_signals) < 2:
            return True  # Not enough history
        
        # Check consistency with recent signals
        same_action_count = sum(1 for s in recent_signals if s.action == signal.action)
        consistency_ratio = same_action_count / len(recent_signals)
        
        return consistency_ratio >= min_consistency
    
    def _check_volatility_filter(self, signal: Signal, config: Dict[str, Any]) -> bool:
        """
        Adjust signal based on market volatility.
        """
        # Placeholder for volatility-based filtering
        # In practice, this would use market volatility data
        return True
    
    def _check_correlation_filter(self, signal: Signal, config: Dict[str, Any]) -> bool:
        """
        Filter based on correlation with other market signals.
        """
        # Placeholder for correlation filtering
        # In practice, this would analyze correlations across markets
        return True
    
    def _get_strategy_performance_weight(self, strategy_id: str) -> float:
        """
        Get performance-based weight for a strategy.
        """
        # Default performance weight
        return self.strategy_performance.get(strategy_id, 1.0)
    
    def _aggregate_market_signals(
        self, 
        signals: List[Signal], 
        window_minutes: int
    ) -> Optional[Signal]:
        """
        Aggregate signals for a single market within a time window.
        """
        if not signals:
            return None
        
        # Sort by timestamp
        signals.sort(key=lambda s: s.timestamp)
        
        # Group signals within time windows
        current_window = []
        aggregated_signals = []
        window_start = signals[0].timestamp
        
        for signal in signals:
            if (signal.timestamp - window_start).total_seconds() / 60 <= window_minutes:
                current_window.append(signal)
            else:
                # Process current window
                if current_window:
                    agg_signal = self._create_aggregated_signal(current_window)
                    if agg_signal:
                        aggregated_signals.append(agg_signal)
                
                # Start new window
                current_window = [signal]
                window_start = signal.timestamp
        
        # Process final window
        if current_window:
            agg_signal = self._create_aggregated_signal(current_window)
            if agg_signal:
                aggregated_signals.append(agg_signal)
        
        # Return most recent aggregated signal
        return aggregated_signals[-1] if aggregated_signals else None
    
    def _create_aggregated_signal(self, signals: List[Signal]) -> Optional[Signal]:
        """
        Create single aggregated signal from a group of signals.
        """
        if not signals:
            return None
        
        # Separate by action
        yes_signals = [s for s in signals if s.action == 'BUY_YES']
        no_signals = [s for s in signals if s.action == 'BUY_NO']
        
        # Determine dominant action
        if len(yes_signals) > len(no_signals):
            dominant_signals = yes_signals
            action = 'BUY_YES'
        elif len(no_signals) > len(yes_signals):
            dominant_signals = no_signals
            action = 'BUY_NO'
        else:
            # Tie - use highest confidence
            all_signals = yes_signals + no_signals
            best_signal = max(all_signals, key=lambda s: s.confidence)
            dominant_signals = [best_signal]
            action = best_signal.action
        
        # Aggregate characteristics
        avg_confidence = np.mean([s.confidence for s in dominant_signals])
        avg_position_size = np.mean([s.position_size for s in dominant_signals])
        
        # Determine aggregated strength
        strong_count = sum(1 for s in dominant_signals if s.signal_strength == SignalStrength.STRONG)
        if strong_count >= len(dominant_signals) / 2:
            signal_strength = SignalStrength.STRONG
        else:
            moderate_count = sum(1 for s in dominant_signals if s.signal_strength == SignalStrength.MODERATE)
            if moderate_count >= len(dominant_signals) / 2:
                signal_strength = SignalStrength.MODERATE
            else:
                signal_strength = SignalStrength.WEAK
        
        # Create aggregated signal
        return Signal(
            strategy_id="AGGREGATED",
            market_id=signals[0].market_id,
            action=action,
            confidence=avg_confidence,
            signal_strength=signal_strength,
            position_size=avg_position_size,
            timestamp=datetime.now(),
            reasoning=f"Aggregated from {len(dominant_signals)} signals",
            metadata={
                'aggregated_count': len(dominant_signals),
                'total_signals': len(signals),
                'contributing_strategies': [s.strategy_id for s in dominant_signals]
            }
        )
    
    def update_strategy_performance(self, strategy_id: str, performance_score: float) -> None:
        """
        Update performance score for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            performance_score: Performance score (0.0 to 2.0, 1.0 = neutral)
        """
        self.strategy_performance[strategy_id] = performance_score
        logger.info(f"Updated {strategy_id} performance score to {performance_score:.2f}")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get statistics about signal processing.
        """
        total_markets = len(self.signal_histories)
        total_signals = sum(h.signal_count for h in self.signal_histories.values())
        
        return {
            'total_markets_tracked': total_markets,
            'total_signals_processed': total_signals,
            'active_filters': list(self.filters.keys()),
            'decay_function': self.decay_function.value,
            'decay_half_life': self.decay_half_life,
            'min_confidence': self.min_confidence,
            'max_signal_age': self.max_signal_age
        }
