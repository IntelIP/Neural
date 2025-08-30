"""
Convert multi-source sentiment into trading probabilities.
Combines sentiment scores, ESPN win probabilities, and orderbook pressure.
"""

from typing import Dict, List, Optional, Tuple
import math
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SentimentProbabilityConverter:
    """
    Convert sentiment signals into actionable trading probabilities.
    
    Combines multiple data sources:
    - Twitter sentiment analysis
    - ESPN win probabilities
    - Kalshi orderbook pressure
    - Historical correlations
    """
    
    @staticmethod
    def sentiment_to_probability(
        sentiment_score: float,  # -1 to 1
        confidence: float = 1.0,  # 0 to 1
        base_prob: float = 0.5,   # Prior probability
        strength: float = 2.0      # Transformation strength
    ) -> float:
        """
        Convert sentiment score to win probability using sigmoid transformation.
        
        Args:
            sentiment_score: Sentiment from -1 (negative) to 1 (positive)
            confidence: Confidence in the sentiment (0-1)
            base_prob: Prior probability before sentiment adjustment
            strength: Steepness of sigmoid transformation
        
        Returns:
            Probability between 0.01 and 0.99
            
        Example:
            sentiment=0.5, confidence=0.8 -> ~70% probability
            sentiment=-0.3, confidence=0.6 -> ~38% probability
        """
        # Validate inputs
        if not -1 <= sentiment_score <= 1:
            logger.warning(f"Sentiment score {sentiment_score} out of range, clamping")
            sentiment_score = max(-1, min(1, sentiment_score))
        
        if not 0 <= confidence <= 1:
            logger.warning(f"Confidence {confidence} out of range, clamping")
            confidence = max(0, min(1, confidence))
        
        # Sigmoid transformation
        # Maps sentiment to (0, 1) range with steepness controlled by strength
        try:
            sigmoid = 1 / (1 + math.exp(-sentiment_score * strength))
        except OverflowError:
            sigmoid = 1.0 if sentiment_score > 0 else 0.0
        
        # Adjust from base probability with confidence weighting
        # High confidence = more deviation from base
        # Low confidence = stay closer to base
        adjustment = (sigmoid - 0.5) * confidence
        probability = base_prob + adjustment
        
        # Bound between 0.01 and 0.99 (never fully certain)
        probability = max(0.01, min(0.99, probability))
        
        logger.debug(
            f"Sentiment->Prob: score={sentiment_score:.2f}, conf={confidence:.2f}, "
            f"base={base_prob:.2f} -> prob={probability:.3f}"
        )
        
        return probability
    
    @staticmethod
    def combine_probabilities(
        sources: Dict[str, Dict[str, float]],
        correlation_matrix: Optional[Dict[Tuple[str, str], float]] = None
    ) -> Dict[str, float]:
        """
        Combine multiple probability sources with confidence weighting.
        
        Args:
            sources: Dictionary of source data:
                {
                    "sentiment": {"prob": 0.65, "weight": 0.4, "confidence": 0.8},
                    "espn": {"prob": 0.60, "weight": 0.3, "confidence": 0.9},
                    "orderbook": {"prob": 0.58, "weight": 0.3, "confidence": 0.7}
                }
            correlation_matrix: Correlation between sources (for adjustment)
        
        Returns:
            Combined probability with aggregate confidence
        """
        if not sources:
            return {"probability": 0.5, "confidence": 0.0, "sources": {}}
        
        # Normalize weights based on confidence
        weighted_sum = 0
        weight_total = 0
        confidence_values = []
        
        for source_name, data in sources.items():
            prob = data.get("prob", 0.5)
            base_weight = data.get("weight", 1.0 / len(sources))
            confidence = data.get("confidence", 1.0)
            
            # Adjust weight by confidence
            effective_weight = base_weight * confidence
            
            # Apply correlation penalty if provided
            if correlation_matrix:
                correlation_penalty = 0
                for other_source in sources:
                    if other_source != source_name:
                        corr_key = tuple(sorted([source_name, other_source]))
                        correlation = correlation_matrix.get(corr_key, 0)
                        correlation_penalty += abs(correlation) * 0.1
                
                effective_weight *= (1 - correlation_penalty)
            
            weighted_sum += prob * effective_weight
            weight_total += effective_weight
            confidence_values.append(confidence)
        
        # Calculate combined probability
        combined_prob = weighted_sum / weight_total if weight_total > 0 else 0.5
        
        # Calculate aggregate confidence
        # Use harmonic mean for conservative confidence estimate
        if confidence_values:
            harmonic_mean = len(confidence_values) / sum(1/c if c > 0 else float('inf') for c in confidence_values)
            avg_confidence = harmonic_mean
        else:
            avg_confidence = 0
        
        return {
            "probability": round(combined_prob, 3),
            "confidence": round(avg_confidence, 3),
            "weight_total": round(weight_total, 3),
            "sources": sources
        }
    
    @staticmethod
    def calculate_sentiment_momentum(
        sentiment_history: List[Dict[str, float]],
        window_minutes: int = 30
    ) -> Dict[str, float]:
        """
        Calculate sentiment momentum and velocity from historical data.
        
        Args:
            sentiment_history: List of {"timestamp": datetime, "sentiment": float}
            window_minutes: Time window for momentum calculation
        
        Returns:
            Momentum metrics for dynamic probability adjustment
        """
        if not sentiment_history or len(sentiment_history) < 2:
            return {
                "momentum": 0.0,
                "velocity": 0.0,
                "acceleration": 0.0,
                "trend_strength": 0.0
            }
        
        # Sort by timestamp
        sorted_history = sorted(sentiment_history, key=lambda x: x["timestamp"])
        
        # Get recent window
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        recent = [h for h in sorted_history if h["timestamp"] > cutoff_time]
        
        if len(recent) < 2:
            recent = sorted_history[-min(10, len(sorted_history)):]
        
        # Calculate momentum (current vs average)
        sentiments = [h["sentiment"] for h in recent]
        current = sentiments[-1]
        average = sum(sentiments) / len(sentiments)
        momentum = current - average
        
        # Calculate velocity (rate of change)
        if len(recent) >= 2:
            time_diff = (recent[-1]["timestamp"] - recent[0]["timestamp"]).total_seconds() / 60
            if time_diff > 0:
                velocity = (sentiments[-1] - sentiments[0]) / time_diff
            else:
                velocity = 0
        else:
            velocity = 0
        
        # Calculate acceleration (change in velocity)
        if len(recent) >= 3:
            mid_point = len(recent) // 2
            first_half_velocity = (sentiments[mid_point] - sentiments[0]) / max(1, mid_point)
            second_half_velocity = (sentiments[-1] - sentiments[mid_point]) / max(1, len(recent) - mid_point)
            acceleration = second_half_velocity - first_half_velocity
        else:
            acceleration = 0
        
        # Calculate trend strength (consistency of direction)
        if len(recent) >= 3:
            increases = sum(1 for i in range(1, len(sentiments)) if sentiments[i] > sentiments[i-1])
            trend_strength = abs(increases / (len(sentiments) - 1) - 0.5) * 2  # 0 = random, 1 = consistent
        else:
            trend_strength = 0
        
        return {
            "momentum": round(momentum, 3),
            "velocity": round(velocity, 4),
            "acceleration": round(acceleration, 4),
            "trend_strength": round(trend_strength, 3)
        }
    
    @staticmethod
    def adjust_probability_for_momentum(
        base_probability: float,
        momentum_metrics: Dict[str, float],
        max_adjustment: float = 0.1
    ) -> float:
        """
        Adjust probability based on sentiment momentum.
        
        Args:
            base_probability: Initial probability estimate
            momentum_metrics: Output from calculate_sentiment_momentum
            max_adjustment: Maximum probability adjustment
        
        Returns:
            Momentum-adjusted probability
        """
        momentum = momentum_metrics.get("momentum", 0)
        velocity = momentum_metrics.get("velocity", 0)
        trend_strength = momentum_metrics.get("trend_strength", 0)
        
        # Calculate adjustment based on momentum and trend
        # Strong, consistent momentum gets more weight
        adjustment = momentum * trend_strength * max_adjustment
        
        # Boost adjustment if velocity confirms momentum direction
        if (momentum > 0 and velocity > 0) or (momentum < 0 and velocity < 0):
            adjustment *= 1.5
        
        # Apply adjustment
        adjusted_prob = base_probability + adjustment
        
        # Bound result
        adjusted_prob = max(0.01, min(0.99, adjusted_prob))
        
        logger.debug(
            f"Momentum adjustment: base={base_probability:.3f}, "
            f"momentum={momentum:.3f}, adjustment={adjustment:.3f}, "
            f"final={adjusted_prob:.3f}"
        )
        
        return adjusted_prob
    
    @staticmethod
    def calculate_edge_confidence(
        probability: float,
        market_price: float,
        confidence: float,
        min_edge: float = 0.05
    ) -> Dict[str, float]:
        """
        Calculate trading edge and adjusted confidence.
        
        Args:
            probability: Our estimated probability
            market_price: Current market price
            confidence: Confidence in our estimate
            min_edge: Minimum required edge
        
        Returns:
            Edge metrics and trading viability
        """
        # Calculate raw edge
        edge = probability - market_price
        
        # Adjust edge by confidence
        confidence_adjusted_edge = edge * confidence
        
        # Calculate edge quality (how much over minimum)
        if abs(edge) > min_edge:
            edge_quality = (abs(edge) - min_edge) / min_edge
        else:
            edge_quality = 0
        
        # Trading confidence (combines edge size and estimate confidence)
        trading_confidence = confidence * min(1.0, edge_quality)
        
        # Determine if tradeable
        is_tradeable = abs(confidence_adjusted_edge) > min_edge
        
        return {
            "raw_edge": round(edge, 3),
            "confidence_adjusted_edge": round(confidence_adjusted_edge, 3),
            "edge_quality": round(edge_quality, 3),
            "trading_confidence": round(trading_confidence, 3),
            "is_tradeable": is_tradeable,
            "direction": "BUY_YES" if edge > 0 else "BUY_NO" if edge < 0 else "HOLD"
        }
    
    @staticmethod
    def sports_specific_adjustments(
        base_probability: float,
        sport: str,
        context: Dict[str, any]
    ) -> float:
        """
        Apply sport-specific probability adjustments.
        
        Args:
            base_probability: Initial probability
            sport: Sport type (NFL, NBA, etc.)
            context: Game context (home/away, injuries, weather, etc.)
        
        Returns:
            Adjusted probability with sport-specific factors
        """
        adjusted_prob = base_probability
        
        if sport.upper() == "NFL":
            # Home field advantage in NFL (~2.5 points = ~7% probability)
            if context.get("is_home"):
                adjusted_prob += 0.035
            
            # Weather impact (affects passing games)
            weather = context.get("weather", {})
            if weather.get("wind_mph", 0) > 20:
                # Favor running teams in high wind
                if context.get("team_style") == "running":
                    adjusted_prob += 0.05
                elif context.get("team_style") == "passing":
                    adjusted_prob -= 0.05
            
            # Injury adjustments
            if context.get("key_injuries", {}).get("quarterback"):
                adjusted_prob -= 0.15  # QB injury is huge
            if context.get("key_injuries", {}).get("star_player"):
                adjusted_prob -= 0.05  # Other key player
        
        elif sport.upper() == "NBA":
            # Home court advantage in NBA (~3 points = ~10% probability)
            if context.get("is_home"):
                adjusted_prob += 0.05
            
            # Back-to-back games
            if context.get("back_to_back"):
                adjusted_prob -= 0.08  # Fatigue factor
            
            # Rest advantage
            rest_diff = context.get("rest_days_difference", 0)
            adjusted_prob += rest_diff * 0.03  # 3% per day of rest advantage
        
        # Bound probability
        adjusted_prob = max(0.01, min(0.99, adjusted_prob))
        
        return adjusted_prob


class MarketMaker:
    """
    Determine fair market prices based on probability estimates.
    """
    
    @staticmethod
    def probability_to_market_spread(
        probability: float,
        confidence: float,
        base_spread: float = 0.02,
        min_spread: float = 0.01,
        max_spread: float = 0.10
    ) -> Tuple[float, float]:
        """
        Convert probability to bid/ask spread for market making.
        
        Args:
            probability: Fair probability estimate
            confidence: Confidence in estimate (affects spread width)
            base_spread: Base spread width
            min_spread: Minimum spread
            max_spread: Maximum spread
        
        Returns:
            Tuple of (bid_price, ask_price)
        """
        # Wider spread for lower confidence
        spread_multiplier = 2 - confidence  # 1x at conf=1, 2x at conf=0
        spread = base_spread * spread_multiplier
        
        # Adjust spread for extreme probabilities (less liquid)
        extremeness = abs(probability - 0.5) * 2  # 0 at 50%, 1 at 0% or 100%
        spread += extremeness * 0.02
        
        # Apply bounds
        spread = max(min_spread, min(max_spread, spread))
        
        # Calculate bid/ask
        bid = probability - spread / 2
        ask = probability + spread / 2
        
        # Ensure valid prices
        bid = max(0.01, bid)
        ask = min(0.99, ask)
        
        # Ensure spread is maintained
        if ask - bid < min_spread:
            mid = (ask + bid) / 2
            bid = mid - min_spread / 2
            ask = mid + min_spread / 2
        
        return (round(bid, 3), round(ask, 3))


# Example usage
if __name__ == "__main__":
    converter = SentimentProbabilityConverter()
    
    # Example 1: Convert sentiment to probability
    sentiment_prob = converter.sentiment_to_probability(
        sentiment_score=0.3,  # Mildly positive
        confidence=0.8,
        base_prob=0.5
    )
    print(f"Sentiment probability: {sentiment_prob:.3f}")
    
    # Example 2: Combine multiple sources
    sources = {
        "sentiment": {"prob": 0.65, "weight": 0.4, "confidence": 0.8},
        "espn": {"prob": 0.60, "weight": 0.3, "confidence": 0.9},
        "orderbook": {"prob": 0.58, "weight": 0.3, "confidence": 0.7}
    }
    
    combined = converter.combine_probabilities(sources)
    print(f"Combined probability: {combined['probability']:.3f}")
    print(f"Aggregate confidence: {combined['confidence']:.3f}")
    
    # Example 3: Calculate edge
    edge_metrics = converter.calculate_edge_confidence(
        probability=combined['probability'],
        market_price=0.55,
        confidence=combined['confidence'],
        min_edge=0.05
    )
    print(f"Trading edge: {edge_metrics['raw_edge']:.3f}")
    print(f"Is tradeable: {edge_metrics['is_tradeable']}")
    print(f"Direction: {edge_metrics['direction']}")