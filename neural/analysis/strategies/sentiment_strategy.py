"""
Sentiment-Based Trading Strategy

This strategy uses real-time sentiment analysis from Twitter and ESPN data
to identify trading opportunities on Kalshi prediction markets.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from .base import BaseStrategy, Signal, SignalType, StrategyConfig
from ..sentiment import SentimentScore, SentimentStrength
from ...data_collection.aggregator import AggregatedData


class SentimentSignalType(Enum):
    """Types of sentiment-based signals."""
    SENTIMENT_DIVERGENCE = "sentiment_divergence"
    MOMENTUM_SHIFT = "momentum_shift"
    VIRAL_MOMENT = "viral_moment"
    SUSTAINED_TREND = "sustained_trend"
    CONTRARIAN_OPPORTUNITY = "contrarian_opportunity"


@dataclass
class SentimentTradingConfig(StrategyConfig):
    """Configuration for sentiment trading strategy."""
    # Sentiment thresholds
    min_sentiment_strength: float = 0.3  # Minimum sentiment magnitude for trade
    sentiment_divergence_threshold: float = 0.2  # Sentiment vs price divergence
    trend_confirmation_minutes: int = 3  # Minutes to confirm trend
    volatility_threshold: float = 0.1  # Max volatility for confident trades

    # Position sizing
    base_position_size: float = 0.05  # Base position as fraction of capital
    sentiment_multiplier: float = 2.0  # Multiply position by sentiment strength
    max_sentiment_position: float = 0.15  # Max position size for sentiment trades

    # Risk management
    sentiment_stop_loss: float = 0.3  # Stop loss for sentiment trades
    take_profit_multiplier: float = 2.5  # Take profit as multiple of risk
    max_holding_minutes: int = 60  # Max time to hold sentiment-based position

    # Signal filters
    min_twitter_engagement: int = 100  # Minimum total engagement for Twitter signal
    min_espn_plays: int = 3  # Minimum ESPN plays for momentum signal
    min_confidence_threshold: float = 0.6  # Minimum confidence for trades


class SentimentTradingStrategy(BaseStrategy):
    """
    Advanced sentiment-based trading strategy.

    Uses Twitter sentiment, ESPN game momentum, and market price divergence
    to identify high-probability trading opportunities.
    """

    def __init__(
        self,
        name: str = "SentimentTrading",
        config: Optional[SentimentTradingConfig] = None,
        teams: Optional[List[str]] = None,
        market_tickers: Optional[Dict[str, str]] = None
    ):
        if config is None:
            config = SentimentTradingConfig()

        super().__init__(name=name, config=config)
        self.sentiment_config = config
        self.teams = teams or []
        self.market_tickers = market_tickers or {}

        # State tracking
        self.sentiment_history: List[Dict[str, Any]] = []
        self.signal_history: List[Dict[str, Any]] = []
        self.last_trade_time: Optional[datetime] = None

        # Sentiment analysis
        self.sentiment_windows = {
            '1min': [],
            '5min': [],
            '15min': []
        }

    async def analyze(
        self,
        market_data: pd.DataFrame,
        aggregated_data: Optional[AggregatedData] = None,
        **kwargs
    ) -> Optional[Signal]:
        """
        Analyze aggregated sentiment data and generate trading signals.

        Args:
            market_data: Current market prices and volumes
            aggregated_data: Combined Twitter, ESPN, and market data
            **kwargs: Additional parameters

        Returns:
            Trading signal or None
        """
        if not aggregated_data or not aggregated_data.sentiment_metrics:
            return self.hold()

        # Update sentiment history
        self._update_sentiment_history(aggregated_data)

        # Analyze different signal types
        signals = []

        # 1. Sentiment-Price Divergence
        divergence_signal = await self._analyze_sentiment_divergence(
            market_data, aggregated_data
        )
        if divergence_signal:
            signals.append(divergence_signal)

        # 2. Momentum Shift Detection
        momentum_signal = await self._analyze_momentum_shift(
            market_data, aggregated_data
        )
        if momentum_signal:
            signals.append(momentum_signal)

        # 3. Viral Moment Detection
        viral_signal = await self._analyze_viral_moment(
            market_data, aggregated_data
        )
        if viral_signal:
            signals.append(viral_signal)

        # 4. Sustained Trend Trading
        trend_signal = await self._analyze_sustained_trend(
            market_data, aggregated_data
        )
        if trend_signal:
            signals.append(trend_signal)

        # 5. Contrarian Opportunities
        contrarian_signal = await self._analyze_contrarian_opportunity(
            market_data, aggregated_data
        )
        if contrarian_signal:
            signals.append(contrarian_signal)

        # Select best signal
        if signals:
            best_signal = max(signals, key=lambda s: s.confidence * s.recommended_size)
            return best_signal

        return self.hold()

    async def _analyze_sentiment_divergence(
        self,
        market_data: pd.DataFrame,
        aggregated_data: AggregatedData
    ) -> Optional[Signal]:
        """Detect divergence between sentiment and market prices."""
        sentiment_metrics = aggregated_data.sentiment_metrics
        if not sentiment_metrics:
            return None

        combined_sentiment = sentiment_metrics.get('combined_sentiment', 0.0)
        sentiment_strength = abs(combined_sentiment)

        # Get current market prices (mock implementation)
        current_price = self._get_current_market_price(market_data, aggregated_data.teams[0])
        if current_price is None:
            return None

        # Calculate expected price based on sentiment
        # Positive sentiment should correlate with higher win probability
        expected_price = 0.5 + (combined_sentiment * 0.3)  # Scale sentiment to price range
        price_divergence = abs(expected_price - current_price)

        # Check if divergence is significant
        if (price_divergence > self.sentiment_config.sentiment_divergence_threshold and
            sentiment_strength > self.sentiment_config.min_sentiment_strength):

            # Determine trade direction
            if combined_sentiment > 0 and current_price < expected_price:
                # Positive sentiment, underpriced market -> Buy YES
                signal_type = SignalType.BUY_YES
                edge = expected_price - current_price
            elif combined_sentiment < 0 and current_price > expected_price:
                # Negative sentiment, overpriced market -> Buy NO
                signal_type = SignalType.BUY_NO
                edge = current_price - expected_price
            else:
                return None

            # Calculate confidence
            confidence = min(0.9, (
                price_divergence * 2 +
                sentiment_strength +
                aggregated_data.metadata.get('signal_strength', 0.0)
            ) / 3)

            if confidence < self.sentiment_config.min_confidence_threshold:
                return None

            # Calculate position size
            position_size = min(
                self.sentiment_config.max_sentiment_position,
                self.sentiment_config.base_position_size * (1 + sentiment_strength * self.sentiment_config.sentiment_multiplier)
            )

            ticker = self._get_market_ticker(aggregated_data.teams[0])
            if not ticker:
                return None

            return Signal(
                signal_type=signal_type,
                market_id=ticker,
                recommended_size=position_size,
                confidence=confidence,
                edge=edge,
                expected_value=edge * position_size,
                stop_loss_price=current_price * (1 - self.sentiment_config.sentiment_stop_loss) if signal_type == SignalType.BUY_YES else current_price * (1 + self.sentiment_config.sentiment_stop_loss),
                take_profit_price=min(0.95, current_price + edge * self.sentiment_config.take_profit_multiplier) if signal_type == SignalType.BUY_YES else max(0.05, current_price - edge * self.sentiment_config.take_profit_multiplier),
                metadata={
                    'strategy_type': SentimentSignalType.SENTIMENT_DIVERGENCE.value,
                    'sentiment_score': combined_sentiment,
                    'price_divergence': price_divergence,
                    'expected_price': expected_price,
                    'current_price': current_price,
                    'twitter_engagement': aggregated_data.twitter_data.get('total_engagement', 0) if aggregated_data.twitter_data else 0,
                    'espn_momentum': sentiment_metrics.get('espn_momentum', 0)
                }
            )

        return None

    async def _analyze_momentum_shift(
        self,
        market_data: pd.DataFrame,
        aggregated_data: AggregatedData
    ) -> Optional[Signal]:
        """Detect sudden momentum shifts in game or sentiment."""
        sentiment_metrics = aggregated_data.sentiment_metrics
        if not sentiment_metrics or len(self.sentiment_history) < 3:
            return None

        # Calculate momentum change
        current_trend = sentiment_metrics.get('combined_trend', 0.0)
        trend_strength = abs(current_trend)

        # Check for significant momentum shift
        if (trend_strength > 0.1 and  # Significant trend
            aggregated_data.espn_data and
            aggregated_data.espn_data.get('new_plays', [])):

            recent_plays = aggregated_data.espn_data.get('new_plays', [])
            if len(recent_plays) < self.sentiment_config.min_espn_plays:
                return None

            # Analyze play momentum
            play_momentum = np.mean([
                play.get('momentum_score', 0) for play in recent_plays
            ])

            momentum_strength = abs(play_momentum)
            if momentum_strength < 0.3:  # Not strong enough
                return None

            # Get market data
            current_price = self._get_current_market_price(market_data, aggregated_data.teams[0])
            if current_price is None:
                return None

            # Determine trade direction based on momentum
            if play_momentum > 0 and current_trend > 0:
                # Positive momentum building
                signal_type = SignalType.BUY_YES
                confidence = min(0.85, momentum_strength + trend_strength)
            elif play_momentum < 0 and current_trend < 0:
                # Negative momentum building
                signal_type = SignalType.BUY_NO
                confidence = min(0.85, momentum_strength + abs(current_trend))
            else:
                return None

            if confidence < self.sentiment_config.min_confidence_threshold:
                return None

            position_size = min(
                self.sentiment_config.max_sentiment_position,
                self.sentiment_config.base_position_size * (1 + momentum_strength * 1.5)
            )

            ticker = self._get_market_ticker(aggregated_data.teams[0])
            if not ticker:
                return None

            return Signal(
                signal_type=signal_type,
                market_id=ticker,
                recommended_size=position_size,
                confidence=confidence,
                edge=momentum_strength * 0.1,  # Estimated edge from momentum
                metadata={
                    'strategy_type': SentimentSignalType.MOMENTUM_SHIFT.value,
                    'play_momentum': play_momentum,
                    'trend_strength': trend_strength,
                    'recent_plays': len(recent_plays),
                    'momentum_plays': [play.get('description', '')[:50] for play in recent_plays[:3]]
                }
            )

        return None

    async def _analyze_viral_moment(
        self,
        market_data: pd.DataFrame,
        aggregated_data: AggregatedData
    ) -> Optional[Signal]:
        """Detect viral moments with high social media engagement."""
        if not aggregated_data.twitter_data:
            return None

        twitter_data = aggregated_data.twitter_data
        total_engagement = twitter_data.get('total_engagement', 0)
        tweet_count = twitter_data.get('tweet_count', 0)

        # Check for viral threshold
        if (total_engagement < self.sentiment_config.min_twitter_engagement or
            tweet_count < 10):
            return None

        # Calculate engagement velocity
        if len(self.sentiment_history) >= 2:
            prev_engagement = self.sentiment_history[-2].get('twitter_engagement', 0)
            engagement_growth = (total_engagement - prev_engagement) / max(prev_engagement, 1)

            # Viral moment: high engagement growth + strong sentiment
            if engagement_growth > 2.0:  # 200% growth
                sentiment_metrics = aggregated_data.sentiment_metrics
                combined_sentiment = sentiment_metrics.get('combined_sentiment', 0.0)
                sentiment_strength = abs(combined_sentiment)

                if sentiment_strength > 0.4:  # Strong sentiment
                    current_price = self._get_current_market_price(market_data, aggregated_data.teams[0])
                    if current_price is None:
                        return None

                    # Quick momentum trade
                    signal_type = SignalType.BUY_YES if combined_sentiment > 0 else SignalType.BUY_NO
                    confidence = min(0.8, sentiment_strength + min(engagement_growth / 5, 0.3))

                    if confidence < self.sentiment_config.min_confidence_threshold:
                        return None

                    position_size = min(
                        self.sentiment_config.max_sentiment_position * 0.8,  # Smaller position for viral trades
                        self.sentiment_config.base_position_size * (1 + sentiment_strength)
                    )

                    ticker = self._get_market_ticker(aggregated_data.teams[0])
                    if not ticker:
                        return None

                    return Signal(
                        signal_type=signal_type,
                        market_id=ticker,
                        recommended_size=position_size,
                        confidence=confidence,
                        edge=sentiment_strength * 0.15,
                        metadata={
                            'strategy_type': SentimentSignalType.VIRAL_MOMENT.value,
                            'engagement_growth': engagement_growth,
                            'total_engagement': total_engagement,
                            'viral_tweets': twitter_data.get('sample_tweets', [])[:2]
                        }
                    )

        return None

    async def _analyze_sustained_trend(
        self,
        market_data: pd.DataFrame,
        aggregated_data: AggregatedData
    ) -> Optional[Signal]:
        """Trade on sustained sentiment trends."""
        if len(self.sentiment_history) < 5:
            return None

        sentiment_metrics = aggregated_data.sentiment_metrics
        current_sentiment = sentiment_metrics.get('combined_sentiment', 0.0)
        current_trend = sentiment_metrics.get('combined_trend', 0.0)

        # Check for sustained trend over time
        recent_sentiments = [
            item.get('combined_sentiment', 0) for item in self.sentiment_history[-5:]
        ]

        # All recent sentiments should be in same direction
        if current_sentiment > 0.2:
            # Positive trend
            sustained_positive = all(s > 0.1 for s in recent_sentiments)
            if sustained_positive and current_trend > 0.05:
                signal_type = SignalType.BUY_YES
                trend_strength = min(recent_sentiments)  # Weakest positive sentiment
            else:
                return None
        elif current_sentiment < -0.2:
            # Negative trend
            sustained_negative = all(s < -0.1 for s in recent_sentiments)
            if sustained_negative and current_trend < -0.05:
                signal_type = SignalType.BUY_NO
                trend_strength = abs(max(recent_sentiments))  # Weakest negative sentiment
            else:
                return None
        else:
            return None

        current_price = self._get_current_market_price(market_data, aggregated_data.teams[0])
        if current_price is None:
            return None

        confidence = min(0.75, trend_strength + abs(current_trend) * 2)
        if confidence < self.sentiment_config.min_confidence_threshold:
            return None

        position_size = min(
            self.sentiment_config.max_sentiment_position,
            self.sentiment_config.base_position_size * (1 + trend_strength)
        )

        ticker = self._get_market_ticker(aggregated_data.teams[0])
        if not ticker:
            return None

        return Signal(
            signal_type=signal_type,
            market_id=ticker,
            recommended_size=position_size,
            confidence=confidence,
            edge=trend_strength * 0.12,
            metadata={
                'strategy_type': SentimentSignalType.SUSTAINED_TREND.value,
                'trend_duration': len(recent_sentiments),
                'trend_strength': trend_strength,
                'sentiment_consistency': np.std(recent_sentiments)
            }
        )

    async def _analyze_contrarian_opportunity(
        self,
        market_data: pd.DataFrame,
        aggregated_data: AggregatedData
    ) -> Optional[Signal]:
        """Identify contrarian opportunities when sentiment is extreme."""
        sentiment_metrics = aggregated_data.sentiment_metrics
        if not sentiment_metrics:
            return None

        combined_sentiment = sentiment_metrics.get('combined_sentiment', 0.0)
        sentiment_volatility = sentiment_metrics.get('twitter_volatility', 0.0)

        # Look for extreme sentiment with high volatility (potential overreaction)
        if (abs(combined_sentiment) > 0.7 and  # Very extreme sentiment
            sentiment_volatility > 0.3):  # High volatility suggests uncertainty

            current_price = self._get_current_market_price(market_data, aggregated_data.teams[0])
            if current_price is None:
                return None

            # Contrarian logic: bet against extreme sentiment
            if combined_sentiment > 0.7 and current_price > 0.8:
                # Very positive sentiment, very high price -> potential overreaction
                signal_type = SignalType.BUY_NO
                contrarian_edge = (current_price - 0.6) * 0.5  # Estimate mean reversion
            elif combined_sentiment < -0.7 and current_price < 0.2:
                # Very negative sentiment, very low price -> potential overreaction
                signal_type = SignalType.BUY_YES
                contrarian_edge = (0.4 - current_price) * 0.5  # Estimate mean reversion
            else:
                return None

            # Lower confidence for contrarian trades
            confidence = min(0.65, sentiment_volatility + abs(combined_sentiment - 0.7) * 0.5)

            if confidence < 0.5:  # Lower threshold for contrarian
                return None

            position_size = min(
                self.sentiment_config.max_sentiment_position * 0.6,  # Smaller contrarian positions
                self.sentiment_config.base_position_size
            )

            ticker = self._get_market_ticker(aggregated_data.teams[0])
            if not ticker:
                return None

            return Signal(
                signal_type=signal_type,
                market_id=ticker,
                recommended_size=position_size,
                confidence=confidence,
                edge=contrarian_edge,
                metadata={
                    'strategy_type': SentimentSignalType.CONTRARIAN_OPPORTUNITY.value,
                    'extreme_sentiment': combined_sentiment,
                    'sentiment_volatility': sentiment_volatility,
                    'contrarian_rationale': 'Mean reversion from extreme sentiment'
                }
            )

        return None

    def _update_sentiment_history(self, aggregated_data: AggregatedData) -> None:
        """Update sentiment history for trend analysis."""
        sentiment_metrics = aggregated_data.sentiment_metrics
        if not sentiment_metrics:
            return

        history_item = {
            'timestamp': datetime.now(),
            'combined_sentiment': sentiment_metrics.get('combined_sentiment', 0.0),
            'twitter_sentiment': sentiment_metrics.get('twitter_sentiment', 0.0),
            'espn_momentum': sentiment_metrics.get('espn_momentum', 0.0),
            'signal_strength': aggregated_data.metadata.get('signal_strength', 0.0),
            'twitter_engagement': aggregated_data.twitter_data.get('total_engagement', 0) if aggregated_data.twitter_data else 0
        }

        self.sentiment_history.append(history_item)

        # Keep only recent history (last 60 minutes)
        cutoff_time = datetime.now() - timedelta(minutes=60)
        self.sentiment_history = [
            item for item in self.sentiment_history
            if item['timestamp'] >= cutoff_time
        ]

    def _get_current_market_price(self, market_data: pd.DataFrame, team: str) -> Optional[float]:
        """Get current market price for a team."""
        # This is a mock implementation - in practice, extract from market_data
        # Based on the team name and available markets
        if not market_data.empty:
            # Assume market_data has team-specific pricing
            price_column = f"{team.lower().replace(' ', '_')}_price"
            if price_column in market_data.columns:
                return float(market_data[price_column].iloc[-1])

        # Fallback - return mock price for demonstration
        return 0.5  # 50% probability as placeholder

    def _get_market_ticker(self, team: str) -> Optional[str]:
        """Get Kalshi market ticker for a team."""
        return self.market_tickers.get(team)

    def should_exit_position(
        self,
        position: Any,
        current_data: AggregatedData
    ) -> bool:
        """Determine if we should exit a sentiment-based position."""
        if not hasattr(position, 'entry_time') or not hasattr(position, 'metadata'):
            return super().should_close_position(position)

        # Time-based exit for sentiment trades
        hold_duration = datetime.now() - position.entry_time
        if hold_duration.total_seconds() / 60 > self.sentiment_config.max_holding_minutes:
            return True

        # Sentiment reversal exit
        if position.metadata and 'strategy_type' in position.metadata:
            strategy_type = position.metadata['strategy_type']
            if strategy_type in [SentimentSignalType.VIRAL_MOMENT.value, SentimentSignalType.MOMENTUM_SHIFT.value]:
                # Quick exit for momentum-based trades if sentiment reverses
                current_sentiment = current_data.sentiment_metrics.get('combined_sentiment', 0.0)
                entry_sentiment = position.metadata.get('sentiment_score', 0.0)

                if (entry_sentiment > 0 and current_sentiment < -0.2) or (entry_sentiment < 0 and current_sentiment > 0.2):
                    return True

        return super().should_close_position(position)

    def get_strategy_metrics(self) -> Dict[str, Any]:
        """Get sentiment strategy specific metrics."""
        base_metrics = self.get_performance_metrics()

        # Add sentiment-specific metrics
        if self.signal_history:
            signal_types = [sig.get('strategy_type') for sig in self.signal_history]
            signal_type_counts = {
                signal_type.value: signal_types.count(signal_type.value)
                for signal_type in SentimentSignalType
            }
        else:
            signal_type_counts = {}

        sentiment_metrics = {
            'sentiment_signals_generated': len(self.signal_history),
            'signal_type_breakdown': signal_type_counts,
            'avg_sentiment_confidence': np.mean([
                sig.get('confidence', 0) for sig in self.signal_history
            ]) if self.signal_history else 0,
            'sentiment_history_length': len(self.sentiment_history)
        }

        return {**base_metrics, **sentiment_metrics}


# Factory function
def create_sentiment_strategy(
    teams: List[str],
    market_tickers: Dict[str, str],
    **config_kwargs
) -> SentimentTradingStrategy:
    """
    Create a sentiment trading strategy.

    Args:
        teams: List of team names to trade on
        market_tickers: Mapping of team names to Kalshi tickers
        **config_kwargs: Configuration overrides

    Returns:
        Configured sentiment trading strategy
    """
    config = SentimentTradingConfig(**config_kwargs)
    return SentimentTradingStrategy(
        config=config,
        teams=teams,
        market_tickers=market_tickers
    )


# Example usage
if __name__ == "__main__":
    # Create strategy for Ravens vs Lions
    strategy = create_sentiment_strategy(
        teams=["Baltimore Ravens", "Detroit Lions"],
        market_tickers={
            "Baltimore Ravens": "RAVENS_WIN_TICKET",
            "Detroit Lions": "LIONS_WIN_TICKET"
        },
        min_sentiment_strength=0.25,
        sentiment_divergence_threshold=0.15,
        max_sentiment_position=0.12
    )

    print(f"Created strategy: {strategy.name}")
    print(f"Teams: {strategy.teams}")
    print(f"Config: min_sentiment={strategy.sentiment_config.min_sentiment_strength}")