"""
Line Movement Strategy - Educational Example

This strategy demonstrates basic trend following by analyzing significant
price movements (line movements) that might indicate smart money or
information flow.

Key Features:
- Price movement detection
- Basic momentum analysis
- Trend following concepts
- Educational signal generation

Note: This is a simplified educational example. Advanced momentum strategies
should incorporate sophisticated trend analysis, volume confirmation,
and adaptive parameters based on market conditions.
"""

from typing import Dict, Optional, Any, List
import logging
import numpy as np
from datetime import datetime, timedelta

from neural.analysis.base import AnalysisResult, AnalysisType, SignalStrength
from neural.strategy.base import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class LineMovementStrategy(BaseStrategy):
    """
    Educational line movement strategy.
    
    This strategy follows significant price movements that might indicate:
    - Smart money positioning
    - Information flow
    - Market sentiment shifts
    - Momentum building
    
    Key Concepts:
    - Trend identification
    - Momentum measurement
    - Volume confirmation
    - Timing analysis
    
    This is an educational implementation. Advanced strategies should:
    - Use sophisticated trend detection
    - Incorporate multiple timeframes
    - Apply dynamic thresholds
    - Consider market microstructure
    """
    
    def __init__(
        self, 
        movement_threshold: float = 0.04,  # 4% price movement
        time_window: int = 6,  # 6 hours for movement
        volume_confirmation: bool = True,
        min_confidence: float = 0.6
    ):
        """
        Initialize line movement strategy.
        
        Args:
            movement_threshold: Minimum price movement to consider (default: 4%)
            time_window: Time window to measure movement (hours)
            volume_confirmation: Require volume confirmation (default: True)
            min_confidence: Minimum confidence for signals (default: 60%)
        """
        super().__init__("LineMovement")
        self.movement_threshold = movement_threshold
        self.time_window = time_window
        self.volume_confirmation = volume_confirmation
        self.min_confidence = min_confidence
        
        logger.info(f"Initialized {self.name} strategy")
    
    async def analyze(self, market_id: str, market_data: Dict[str, Any]) -> Optional[Signal]:
        """
        Generate signals based on significant line movements.
        
        Args:
            market_id: Market identifier
            market_data: Dictionary containing market information
            
        Returns:
            Signal object or None if no signal generated
        """
        try:
            # Extract price history
            current_price = market_data.get('current_price')
            price_history = market_data.get('price_history', [])
            
            if current_price is None or len(price_history) < 3:
                logger.debug(f"Insufficient price data for line movement: {market_id}")
                return None
            
            # Analyze price movement
            movement_analysis = self._analyze_price_movement(current_price, price_history)
            
            if not movement_analysis['significant_movement']:
                return None
                
            # Confirm with volume if required
            if self.volume_confirmation:
                if not self._confirm_with_volume(market_data, movement_analysis):
                    return None
            
            # Generate signal
            signal_direction = movement_analysis['direction']
            action = 'BUY_YES' if signal_direction == 'up' else 'BUY_NO'
            
            # Calculate signal strength and confidence
            signal_strength = self._calculate_signal_strength(movement_analysis, market_data)
            confidence = self._calculate_confidence(movement_analysis, market_data)
            
            if confidence < self.min_confidence:
                logger.debug(f"Confidence {confidence:.2f} below threshold for {market_id}")
                return None
            
            # Calculate position size
            position_size = self._calculate_position_size(movement_analysis, confidence)
            
            return Signal(
                strategy_id=self.name,
                market_id=market_id,
                action=action,
                confidence=confidence,
                signal_strength=signal_strength,
                position_size=position_size,
                timestamp=datetime.now(),
                reasoning=f"{movement_analysis['direction'].upper()} movement: {movement_analysis['magnitude']:.1%}",
                metadata={
                    'movement_direction': movement_analysis['direction'],
                    'movement_magnitude': movement_analysis['magnitude'],
                    'movement_speed': movement_analysis['speed'],
                    'volume_confirmed': movement_analysis.get('volume_confirmed', False),
                    'current_price': current_price
                }
            )
            
        except Exception as e:
            logger.error(f"Error analyzing line movement for {market_id}: {e}")
            return None
    
    def _analyze_price_movement(self, current_price: float, price_history: List[Dict]) -> Dict[str, Any]:
        """
        Analyze price movement characteristics.
        
        Basic implementation for educational purposes.
        Advanced strategies should use sophisticated trend analysis.
        """
        if len(price_history) < 2:
            return {'significant_movement': False}
        
        # Sort by timestamp to ensure chronological order
        sorted_history = sorted(price_history, key=lambda x: x.get('timestamp', 0))
        
        # Calculate movement from time window start to current
        cutoff_time = datetime.now() - timedelta(hours=self.time_window)
        cutoff_timestamp = cutoff_time.timestamp()
        
        # Find starting price within time window
        start_price = None
        for price_point in sorted_history:
            if price_point.get('timestamp', 0) >= cutoff_timestamp:
                start_price = price_point.get('price', price_point.get('last'))
                break
        
        if start_price is None or start_price == current_price:
            return {'significant_movement': False}
        
        # Calculate movement magnitude
        magnitude = abs(current_price - start_price) / start_price
        direction = 'up' if current_price > start_price else 'down'
        
        # Calculate movement speed (magnitude per hour)
        time_span = min(self.time_window, 
                       (datetime.now().timestamp() - sorted_history[0].get('timestamp', 0)) / 3600)
        speed = magnitude / max(time_span, 0.1)  # Avoid division by zero
        
        # Check if movement is significant
        significant = magnitude >= self.movement_threshold
        
        return {
            'significant_movement': significant,
            'magnitude': magnitude,
            'direction': direction,
            'speed': speed,
            'start_price': start_price,
            'current_price': current_price
        }
    
    def _confirm_with_volume(self, market_data: Dict[str, Any], movement_analysis: Dict[str, Any]) -> bool:
        """
        Confirm price movement with volume analysis.
        
        Basic volume confirmation for educational purposes.
        """
        if not self.volume_confirmation:
            return True
        
        current_volume = market_data.get('current_volume', 0)
        avg_volume = market_data.get('average_volume', 0)
        
        if avg_volume == 0:
            return True  # Can't confirm, but don't reject
        
        # Volume should be elevated for significant movements
        volume_ratio = current_volume / avg_volume
        movement_magnitude = movement_analysis['magnitude']
        
        # Require higher volume for larger movements
        if movement_magnitude > 0.08:  # 8%+ movement
            volume_threshold = 1.5  # 50% above average
        elif movement_magnitude > 0.05:  # 5%+ movement  
            volume_threshold = 1.3  # 30% above average
        else:
            volume_threshold = 1.1  # 10% above average
        
        volume_confirmed = volume_ratio >= volume_threshold
        movement_analysis['volume_confirmed'] = volume_confirmed
        
        return volume_confirmed
    
    def _calculate_signal_strength(
        self, 
        movement_analysis: Dict[str, Any], 
        market_data: Dict[str, Any]
    ) -> SignalStrength:
        """
        Calculate signal strength based on movement characteristics.
        
        Educational implementation for demonstration purposes.
        """
        magnitude = movement_analysis['magnitude']
        speed = movement_analysis['speed']
        
        # Base strength on magnitude
        if magnitude >= 0.10:  # 10%+ movement
            base_strength = SignalStrength.STRONG
        elif magnitude >= 0.06:  # 6%+ movement
            base_strength = SignalStrength.MODERATE
        else:
            base_strength = SignalStrength.WEAK
        
        # Adjust for speed (faster movements might be stronger signals)
        if speed > 0.02:  # 2% per hour
            # Fast movement - potentially stronger
            if base_strength == SignalStrength.MODERATE:
                return SignalStrength.STRONG
        elif speed < 0.005:  # Very slow movement
            # Slow movement - potentially weaker
            if base_strength == SignalStrength.STRONG:
                return SignalStrength.MODERATE
        
        return base_strength
    
    def _calculate_confidence(
        self, 
        movement_analysis: Dict[str, Any], 
        market_data: Dict[str, Any]
    ) -> float:
        """
        Calculate confidence in the line movement signal.
        
        Basic implementation for educational purposes.
        """
        base_confidence = 0.5
        
        # Higher magnitude = higher confidence
        magnitude_boost = min(movement_analysis['magnitude'] * 5, 0.3)
        
        # Volume confirmation boosts confidence
        volume_boost = 0.15 if movement_analysis.get('volume_confirmed', False) else 0
        
        # Speed factor (very fast or very slow movements are less reliable)
        speed = movement_analysis['speed']
        if 0.005 < speed < 0.03:  # Goldilocks zone
            speed_boost = 0.1
        else:
            speed_boost = -0.05
        
        # Market quality factors
        spread = market_data.get('spread', 0.05)
        spread_penalty = min(spread * 2, 0.15)
        
        # Time to event (closer events might be more reliable)
        hours_to_close = market_data.get('hours_to_close', 0)
        if 0 < hours_to_close <= 48:  # Within 48 hours
            time_boost = 0.1
        else:
            time_boost = 0
        
        confidence = (base_confidence + magnitude_boost + volume_boost + 
                     speed_boost - spread_penalty + time_boost)
        
        return max(0.2, min(confidence, 0.9))
    
    def _calculate_position_size(self, movement_analysis: Dict[str, Any], confidence: float) -> float:
        """
        Calculate position size for line movement signal.
        
        Educational implementation - production should use proper risk management.
        """
        base_size = 0.025  # 2.5% base position
        
        # Scale with movement magnitude
        magnitude_multiplier = min(movement_analysis['magnitude'] / 0.04, 2.0)
        
        # Scale with confidence
        confidence_multiplier = confidence / 0.7
        
        # Scale with movement speed (moderate speed preferred)
        speed = movement_analysis['speed']
        if 0.01 < speed < 0.025:  # Optimal speed range
            speed_multiplier = 1.2
        else:
            speed_multiplier = 0.9
        
        position_size = (base_size * magnitude_multiplier * 
                        confidence_multiplier * speed_multiplier)
        
        # Cap position size
        return min(position_size, 0.12)  # Max 12% for momentum trades
    
    def get_required_data_sources(self) -> list:
        """Return list of required data sources for this strategy."""
        return [
            'current_price',
            'price_history',  # List of price points with timestamps
            'current_volume',
            'average_volume',
            'spread',
            'hours_to_close'
        ]
    
    def get_strategy_description(self) -> str:
        """Return human-readable strategy description."""
        return (
            f"Line movement strategy that follows price movements "
            f"≥{self.movement_threshold:.1%} over {self.time_window}h windows. "
            f"Educational example of trend following and momentum analysis."
        )
