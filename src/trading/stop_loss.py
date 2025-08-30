"""
Dynamic stop-loss calculator for sentiment-aware risk management.
Adjusts stop-loss levels based on market conditions and sentiment momentum.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import math
import logging

logger = logging.getLogger(__name__)


class DynamicStopLoss:
    """
    Calculate and manage dynamic stop-losses based on market conditions.
    
    Adjusts stop-loss levels based on:
    - Sentiment momentum (tighten if turning negative)
    - Market volatility (widen in volatile markets)
    - Time decay (tighten as event approaches)
    - Profit level (trailing stops for winners)
    """
    
    @staticmethod
    def calculate_stop_loss(
        entry_price: float,
        current_price: float,
        sentiment_momentum: float,
        volatility: float,
        time_to_event_hours: Optional[float] = None,
        base_stop_pct: float = 0.10,
        position_type: str = "YES"
    ) -> Dict[str, float]:
        """
        Calculate dynamic stop-loss based on market conditions.
        
        Args:
            entry_price: Entry price of position (0-1 for Kalshi)
            current_price: Current market price (0-1)
            sentiment_momentum: Rate of sentiment change (-1 to 1)
            volatility: Market volatility (0 to 1, where 0.5 is normal)
            time_to_event_hours: Hours until event resolution
            base_stop_pct: Base stop-loss percentage (0.10 = 10%)
            position_type: "YES" or "NO" position
        
        Returns:
            Dictionary with stop price and related metrics
        """
        # Validate inputs
        if not 0 < entry_price < 1:
            raise ValueError(f"Entry price must be between 0 and 1, got {entry_price}")
        
        if not 0 < current_price < 1:
            raise ValueError(f"Current price must be between 0 and 1, got {current_price}")
        
        # Calculate base stop distance
        if position_type == "YES":
            # For YES positions, stop is below entry
            base_stop_distance = entry_price * base_stop_pct
        else:
            # For NO positions, stop is above entry (remember: profit when price goes DOWN)
            base_stop_distance = (1 - entry_price) * base_stop_pct
        
        # Adjust for sentiment momentum
        sentiment_multiplier = DynamicStopLoss._calculate_sentiment_multiplier(
            sentiment_momentum, position_type
        )
        
        # Adjust for volatility
        volatility_multiplier = DynamicStopLoss._calculate_volatility_multiplier(volatility)
        
        # Adjust for time decay (tighten stops as event approaches)
        time_multiplier = DynamicStopLoss._calculate_time_multiplier(time_to_event_hours)
        
        # Calculate adjusted stop distance
        adjusted_stop_distance = (
            base_stop_distance * 
            sentiment_multiplier * 
            volatility_multiplier * 
            time_multiplier
        )
        
        # Apply bounds (5% minimum, 20% maximum)
        min_stop = entry_price * 0.05 if position_type == "YES" else (1 - entry_price) * 0.05
        max_stop = entry_price * 0.20 if position_type == "YES" else (1 - entry_price) * 0.20
        adjusted_stop_distance = max(min_stop, min(max_stop, adjusted_stop_distance))
        
        # Calculate actual stop price
        if position_type == "YES":
            stop_price = entry_price - adjusted_stop_distance
            stop_price = max(0.01, stop_price)  # Can't go below $0.01
        else:
            stop_price = entry_price + adjusted_stop_distance
            stop_price = min(0.99, stop_price)  # Can't go above $0.99
        
        # Calculate stop percentage
        stop_pct = adjusted_stop_distance / entry_price if position_type == "YES" else adjusted_stop_distance / (1 - entry_price)
        
        # Determine confidence in stop placement
        confidence = DynamicStopLoss._calculate_stop_confidence(
            sentiment_momentum, volatility, time_to_event_hours
        )
        
        return {
            "stop_price": round(stop_price, 3),
            "stop_distance": round(adjusted_stop_distance, 3),
            "stop_pct": round(stop_pct, 3),
            "entry_price": entry_price,
            "current_price": current_price,
            "position_type": position_type,
            "confidence": round(confidence, 3),
            "adjustments": {
                "sentiment_multiplier": round(sentiment_multiplier, 2),
                "volatility_multiplier": round(volatility_multiplier, 2),
                "time_multiplier": round(time_multiplier, 2)
            },
            "reasoning": DynamicStopLoss._generate_reasoning(
                sentiment_momentum, volatility, time_to_event_hours
            )
        }
    
    @staticmethod
    def calculate_trailing_stop(
        entry_price: float,
        current_price: float,
        highest_price: float,
        position_type: str = "YES",
        trail_pct: float = 0.05,
        breakeven_threshold: float = 0.02
    ) -> Dict[str, float]:
        """
        Calculate trailing stop-loss for profitable positions.
        
        Args:
            entry_price: Original entry price
            current_price: Current market price
            highest_price: Highest price since entry (for YES) or lowest (for NO)
            position_type: "YES" or "NO" position
            trail_pct: Trailing percentage from peak
            breakeven_threshold: Min profit before activating trailing stop
        
        Returns:
            Trailing stop configuration
        """
        if position_type == "YES":
            # Calculate profit
            unrealized_profit = current_price - entry_price
            peak_profit = highest_price - entry_price
            
            # Only trail if we're above breakeven threshold
            if peak_profit > breakeven_threshold:
                # Trail from the highest price
                trailing_stop = highest_price * (1 - trail_pct)
                
                # Ensure stop is at least at breakeven
                trailing_stop = max(trailing_stop, entry_price)
                
                # Can't be above current price
                trailing_stop = min(trailing_stop, current_price - 0.01)
                
                is_active = True
            else:
                # Not enough profit to trail yet
                trailing_stop = entry_price * (1 - 0.10)  # Default stop
                is_active = False
        
        else:  # NO position
            # For NO positions, profit when price goes DOWN
            unrealized_profit = entry_price - current_price
            peak_profit = entry_price - highest_price  # highest_price is actually lowest for NO
            
            if peak_profit > breakeven_threshold:
                # Trail from the lowest price (stored as highest_price)
                trailing_stop = highest_price * (1 + trail_pct)
                
                # Ensure stop is at least at breakeven
                trailing_stop = min(trailing_stop, entry_price)
                
                # Can't be below current price
                trailing_stop = max(trailing_stop, current_price + 0.01)
                
                is_active = True
            else:
                trailing_stop = entry_price * (1 + 0.10)  # Default stop
                is_active = False
        
        return {
            "trailing_stop": round(trailing_stop, 3),
            "is_active": is_active,
            "unrealized_profit": round(unrealized_profit, 3),
            "peak_profit": round(peak_profit, 3),
            "trail_distance": round(abs(current_price - trailing_stop), 3),
            "position_type": position_type
        }
    
    @staticmethod
    def calculate_time_based_stop(
        entry_price: float,
        position_type: str,
        time_to_event_hours: float,
        urgency_factor: float = 1.0
    ) -> Dict[str, float]:
        """
        Calculate time-based stop that tightens as event approaches.
        
        Args:
            entry_price: Entry price of position
            position_type: "YES" or "NO"
            time_to_event_hours: Hours until event resolution
            urgency_factor: Multiplier for urgency (1.0 = normal, 2.0 = urgent)
        
        Returns:
            Time-based stop configuration
        """
        if time_to_event_hours <= 0:
            # Event has passed, exit immediately
            return {
                "stop_price": entry_price,
                "urgency": "IMMEDIATE",
                "reason": "Event has concluded"
            }
        
        # Calculate time-based stop percentage
        # Tighter stops as we approach the event
        if time_to_event_hours > 48:
            # More than 2 days: normal stop
            stop_pct = 0.15
            urgency = "LOW"
        elif time_to_event_hours > 24:
            # 1-2 days: tighter stop
            stop_pct = 0.10
            urgency = "MEDIUM"
        elif time_to_event_hours > 6:
            # 6-24 hours: tight stop
            stop_pct = 0.07
            urgency = "HIGH"
        elif time_to_event_hours > 1:
            # 1-6 hours: very tight stop
            stop_pct = 0.05
            urgency = "VERY_HIGH"
        else:
            # Less than 1 hour: minimal stop
            stop_pct = 0.03
            urgency = "CRITICAL"
        
        # Apply urgency factor
        stop_pct *= (1 / urgency_factor)
        
        # Calculate stop price
        if position_type == "YES":
            stop_price = entry_price * (1 - stop_pct)
            stop_price = max(0.01, stop_price)
        else:
            stop_price = entry_price * (1 + stop_pct)
            stop_price = min(0.99, stop_price)
        
        return {
            "stop_price": round(stop_price, 3),
            "stop_pct": round(stop_pct, 3),
            "time_to_event_hours": time_to_event_hours,
            "urgency": urgency,
            "urgency_factor": urgency_factor
        }
    
    @staticmethod
    def calculate_correlated_stops(
        positions: List[Dict[str, float]],
        correlation_matrix: Dict[Tuple[str, str], float],
        base_stop_pct: float = 0.10
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate stop-losses for correlated positions.
        Tighter stops for highly correlated positions to limit cascade losses.
        
        Args:
            positions: List of position dictionaries
            correlation_matrix: Correlation between markets
            base_stop_pct: Base stop percentage
        
        Returns:
            Stop configuration for each position
        """
        stops = {}
        
        for i, pos in enumerate(positions):
            market = pos["market_ticker"]
            
            # Calculate correlation penalty
            max_correlation = 0
            for j, other_pos in enumerate(positions):
                if i != j:
                    other_market = other_pos["market_ticker"]
                    corr_key = tuple(sorted([market, other_market]))
                    correlation = abs(correlation_matrix.get(corr_key, 0))
                    max_correlation = max(max_correlation, correlation)
            
            # Tighten stop for correlated positions
            # High correlation = tighter stop to prevent cascade
            correlation_multiplier = 1 - (max_correlation * 0.3)  # Up to 30% tighter
            
            adjusted_stop_pct = base_stop_pct * correlation_multiplier
            
            # Calculate stop price
            entry_price = pos["entry_price"]
            position_type = pos["position_type"]
            
            if position_type == "YES":
                stop_price = entry_price * (1 - adjusted_stop_pct)
            else:
                stop_price = entry_price * (1 + adjusted_stop_pct)
            
            stops[market] = {
                "stop_price": round(stop_price, 3),
                "stop_pct": round(adjusted_stop_pct, 3),
                "correlation_penalty": round(1 - correlation_multiplier, 3),
                "max_correlation": round(max_correlation, 3)
            }
        
        return stops
    
    @staticmethod
    def _calculate_sentiment_multiplier(sentiment_momentum: float, position_type: str) -> float:
        """Calculate stop adjustment based on sentiment momentum."""
        if position_type == "YES":
            # For YES positions, negative momentum = tighter stop
            if sentiment_momentum < -0.3:
                return 0.5  # 50% tighter stop
            elif sentiment_momentum < -0.1:
                return 0.75  # 25% tighter
            elif sentiment_momentum > 0.3:
                return 1.5  # 50% wider stop
            elif sentiment_momentum > 0.1:
                return 1.25  # 25% wider
        else:  # NO position
            # For NO positions, positive momentum = tighter stop
            if sentiment_momentum > 0.3:
                return 0.5  # 50% tighter stop
            elif sentiment_momentum > 0.1:
                return 0.75  # 25% tighter
            elif sentiment_momentum < -0.3:
                return 1.5  # 50% wider stop
            elif sentiment_momentum < -0.1:
                return 1.25  # 25% wider
        
        return 1.0  # No adjustment
    
    @staticmethod
    def _calculate_volatility_multiplier(volatility: float) -> float:
        """Calculate stop adjustment based on volatility."""
        # Higher volatility = wider stops to avoid whipsaws
        # volatility: 0 = low, 0.5 = normal, 1.0 = high
        
        if volatility < 0.3:
            return 0.8  # Low volatility: tighter stop
        elif volatility < 0.5:
            return 1.0  # Normal volatility: no adjustment
        elif volatility < 0.7:
            return 1.3  # High volatility: wider stop
        else:
            return 1.5  # Very high volatility: much wider stop
    
    @staticmethod
    def _calculate_time_multiplier(time_to_event_hours: Optional[float]) -> float:
        """Calculate stop adjustment based on time to event."""
        if time_to_event_hours is None:
            return 1.0
        
        if time_to_event_hours < 6:
            return 0.5  # Very close to event: tight stop
        elif time_to_event_hours < 24:
            return 0.75  # Within a day: tighter stop
        elif time_to_event_hours < 72:
            return 1.0  # 1-3 days: normal
        else:
            return 1.25  # Far from event: wider stop
    
    @staticmethod
    def _calculate_stop_confidence(
        sentiment_momentum: float,
        volatility: float,
        time_to_event_hours: Optional[float]
    ) -> float:
        """Calculate confidence in stop placement."""
        confidence = 1.0
        
        # Lower confidence in high volatility
        confidence -= volatility * 0.3
        
        # Lower confidence with strong momentum (might break stop)
        confidence -= abs(sentiment_momentum) * 0.2
        
        # Higher confidence closer to event (clearer outcome)
        if time_to_event_hours is not None and time_to_event_hours < 24:
            confidence += 0.1
        
        return max(0.3, min(1.0, confidence))
    
    @staticmethod
    def _generate_reasoning(
        sentiment_momentum: float,
        volatility: float,
        time_to_event_hours: Optional[float]
    ) -> str:
        """Generate human-readable reasoning for stop placement."""
        reasons = []
        
        # Sentiment reasoning
        if abs(sentiment_momentum) > 0.3:
            direction = "negative" if sentiment_momentum < 0 else "positive"
            reasons.append(f"Strong {direction} sentiment momentum")
        elif abs(sentiment_momentum) > 0.1:
            direction = "negative" if sentiment_momentum < 0 else "positive"
            reasons.append(f"Moderate {direction} sentiment")
        
        # Volatility reasoning
        if volatility > 0.7:
            reasons.append("High volatility requires wider stop")
        elif volatility < 0.3:
            reasons.append("Low volatility allows tighter stop")
        
        # Time reasoning
        if time_to_event_hours is not None:
            if time_to_event_hours < 6:
                reasons.append("Very close to event - tight risk control")
            elif time_to_event_hours < 24:
                reasons.append("Approaching event - moderate stop")
        
        return "; ".join(reasons) if reasons else "Standard stop placement"


class StopLossManager:
    """
    Manage stop-losses across multiple positions.
    """
    
    def __init__(self):
        """Initialize stop-loss manager."""
        self.active_stops: Dict[str, Dict[str, float]] = {}
        self.stop_history: List[Dict[str, any]] = []
    
    def update_stop(
        self,
        market_ticker: str,
        stop_price: float,
        stop_type: str = "FIXED",
        metadata: Optional[Dict] = None
    ):
        """Update stop-loss for a position."""
        self.active_stops[market_ticker] = {
            "stop_price": stop_price,
            "stop_type": stop_type,
            "updated_at": datetime.now(),
            "metadata": metadata or {}
        }
        
        # Log to history
        self.stop_history.append({
            "market_ticker": market_ticker,
            "stop_price": stop_price,
            "stop_type": stop_type,
            "timestamp": datetime.now(),
            "metadata": metadata
        })
    
    def check_stops(
        self,
        current_prices: Dict[str, float]
    ) -> List[Dict[str, any]]:
        """
        Check if any stops have been triggered.
        
        Args:
            current_prices: Current market prices
        
        Returns:
            List of triggered stops
        """
        triggered = []
        
        for market_ticker, stop_config in self.active_stops.items():
            if market_ticker not in current_prices:
                continue
            
            current_price = current_prices[market_ticker]
            stop_price = stop_config["stop_price"]
            position_type = stop_config.get("metadata", {}).get("position_type", "YES")
            
            # Check if stop is triggered
            if position_type == "YES" and current_price <= stop_price:
                triggered.append({
                    "market_ticker": market_ticker,
                    "stop_price": stop_price,
                    "current_price": current_price,
                    "position_type": position_type,
                    "reason": "Stop-loss triggered (YES position)"
                })
            elif position_type == "NO" and current_price >= stop_price:
                triggered.append({
                    "market_ticker": market_ticker,
                    "stop_price": stop_price,
                    "current_price": current_price,
                    "position_type": position_type,
                    "reason": "Stop-loss triggered (NO position)"
                })
        
        return triggered
    
    def get_stop_summary(self) -> Dict[str, any]:
        """Get summary of all active stops."""
        return {
            "active_stops": len(self.active_stops),
            "stops": self.active_stops,
            "last_update": max(
                (s["updated_at"] for s in self.active_stops.values()),
                default=None
            )
        }


# Example usage
if __name__ == "__main__":
    # Example: Calculate dynamic stop for a position
    stop_calc = DynamicStopLoss()
    
    # YES position with negative sentiment momentum
    stop_config = stop_calc.calculate_stop_loss(
        entry_price=0.45,
        current_price=0.48,
        sentiment_momentum=-0.2,  # Sentiment turning negative
        volatility=0.6,  # Moderate-high volatility
        time_to_event_hours=12,  # 12 hours to event
        base_stop_pct=0.10,
        position_type="YES"
    )
    
    print("Dynamic Stop Configuration:")
    print(f"Entry: ${stop_config['entry_price']:.2f}")
    print(f"Stop: ${stop_config['stop_price']:.2f}")
    print(f"Stop Distance: {stop_config['stop_pct']:.1%}")
    print(f"Confidence: {stop_config['confidence']:.1%}")
    print(f"Reasoning: {stop_config['reasoning']}")
    
    # Example: Trailing stop for profitable position
    trailing = stop_calc.calculate_trailing_stop(
        entry_price=0.40,
        current_price=0.55,
        highest_price=0.60,
        position_type="YES"
    )
    
    print("\nTrailing Stop:")
    print(f"Trailing Stop: ${trailing['trailing_stop']:.2f}")
    print(f"Active: {trailing['is_active']}")
    print(f"Unrealized Profit: ${trailing['unrealized_profit']:.2f}")