"""
Sentiment Momentum Template for sports prediction markets.

Combines real-time sentiment analysis with price momentum to generate
trading signals for sports prediction contracts.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from .base import SportsPredictionTemplate

logger = logging.getLogger(__name__)

class SentimentMomentumTemplate(SportsPredictionTemplate):
    """
    Combines price momentum with social sentiment analysis.

    Strategy:
    - Monitor Twitter/X sentiment for teams
    - Buy when sentiment > threshold and price momentum positive
    - Incorporate game time decay for higher conviction near game time
    """

    def __init__(self,
                 sentiment_threshold: float = 0.7,
                 momentum_period: int = 10,
                 sentiment_weight: float = 0.6,
                 game_time_decay: bool = True,
                 **kwargs):

        super().__init__(**kwargs)
        self.sentiment_threshold = sentiment_threshold
        self.momentum_period = momentum_period
        self.sentiment_weight = sentiment_weight
        self.game_time_decay = game_time_decay

        # Initialize sentiment engine (placeholder for now)
        # from ..sentiment.twitter_engine import TwitterSentimentEngine
        # self.sentiment_engine = TwitterSentimentEngine()

    def _get_description(self) -> str:
        return "Combines price momentum with real-time sentiment analysis"

    def _get_parameters(self) -> Dict:
        return {
            'sentiment_threshold': {
                'type': 'float',
                'default': 0.7,
                'min': 0.0,
                'max': 1.0,
                'description': 'Minimum sentiment score to trigger signal'
            },
            'momentum_period': {
                'type': 'int',
                'default': 10,
                'min': 5,
                'max': 50,
                'description': 'Period for momentum calculation'
            },
            'sentiment_weight': {
                'type': 'float',
                'default': 0.6,
                'min': 0.0,
                'max': 1.0,
                'description': 'Weight given to sentiment vs momentum'
            }
        }

    def validate_parameters(self, params: Dict) -> bool:
        """Validate template parameters."""
        if not 0 <= params.get('sentiment_threshold', 0.7) <= 1.0:
            return False
        if not 5 <= params.get('momentum_period', 10) <= 50:
            return False
        if not 0 <= params.get('sentiment_weight', 0.6) <= 1.0:
            return False
        return True

    def generate_signals(self, market_data: Dict, **kwargs) -> List[Dict]:
        """
        Generate trading signals based on sentiment and momentum.

        Args:
            market_data: Dictionary containing market data
            **kwargs: Additional data (sentiment_data, game_info, etc.)
        """
        signals = []
        sentiment_data = kwargs.get('sentiment_data', {})

        for market in market_data.get('markets', []):
            symbol = market.get('ticker', '')
            price = market.get('last_price', 0)

            if not symbol or not price:
                continue

            # Get sentiment for teams in this market
            team_sentiment = self._get_team_sentiment(symbol, sentiment_data)

            # Calculate momentum
            momentum = self._calculate_momentum(symbol, market_data, self.momentum_period)

            # Game time factor
            time_factor = self._calculate_time_decay(market) if self.game_time_decay else 1.0

            # Combined signal
            sentiment_score = team_sentiment * self.sentiment_weight
            momentum_score = momentum * (1 - self.sentiment_weight)
            combined_score = (sentiment_score + momentum_score) * time_factor

            if combined_score > self.sentiment_threshold:
                signals.append({
                    'action': 'BUY',
                    'symbol': symbol,
                    'size': self.calculate_position_size(combined_score, market.get('volatility', 0.1)),
                    'confidence': combined_score,
                    'reason': f'Sentiment: {team_sentiment:.2f}, Momentum: {momentum:.2f}'
                })

        # Apply risk management
        return self.apply_risk_management(signals)

    def _get_team_sentiment(self, symbol: str, sentiment_data: Dict) -> float:
        """Extract team sentiment from market symbol."""
        # Parse team names from symbol (e.g., "KXNFLCHIEFS" -> "chiefs")
        # This would need sport-specific parsing logic

        # For now, return mock sentiment based on symbol
        if 'CHIEFS' in symbol.upper() or 'KANSAS' in symbol.upper():
            return 0.75  # Positive sentiment for Chiefs
        elif 'CHARGERS' in symbol.upper() or 'LAC' in symbol.upper():
            return 0.65  # Slightly positive for Chargers
        else:
            return 0.5  # Neutral sentiment

    def _calculate_momentum(self, symbol: str, market_data: Dict, period: int) -> float:
        """Calculate price momentum."""
        # Get price history for symbol
        prices = market_data.get('price_history', {}).get(symbol, [])

        if len(prices) < period + 1:
            return 0.0

        # Calculate momentum as percentage change
        current_price = prices[-1]
        past_price = prices[-period-1]

        if past_price == 0:
            return 0.0

        momentum = (current_price - past_price) / past_price

        # Normalize to 0-1 scale
        return max(0, min(1, (momentum + 0.1) / 0.2))  # Assuming -10% to +10% range

    def _calculate_time_decay(self, market: Dict) -> float:
        """Calculate time decay factor based on time to game."""
        # Get game time from market data
        game_time_str = market.get('game_time')

        if not game_time_str:
            return 1.0

        try:
            # Parse game time
            if isinstance(game_time_str, str):
                game_time = datetime.fromisoformat(game_time_str.replace('Z', '+00:00'))
            else:
                game_time = game_time_str

            # Calculate hours until game
            now = datetime.now(game_time.tzinfo) if hasattr(game_time, 'tzinfo') else datetime.now()
            hours_until_game = (game_time - now).total_seconds() / 3600

            # Apply decay: closer to game = higher factor
            if hours_until_game <= 1:
                return 1.5  # Last hour
            elif hours_until_game <= 6:
                return 1.2  # Last 6 hours
            elif hours_until_game <= 24:
                return 1.1  # Last 24 hours
            else:
                return 0.8  # More than 24 hours
        except Exception as e:
            logger.warning(f"Error calculating time decay: {e}")
            return 1.0