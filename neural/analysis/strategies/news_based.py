"""
News-Based Trading Strategy

Trades on news events, sentiment analysis, and social media momentum.
Particularly effective for injury news, lineup changes, and breaking developments.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
from .base import Strategy, Signal, SignalType


class NewsBasedStrategy(Strategy):
    """
    News and sentiment-based trading strategy.

    Reacts to:
    - Injury reports
    - Lineup changes
    - Weather updates
    - Social media sentiment
    - Breaking news
    """

    def __init__(
        self,
        sentiment_threshold: float = 0.6,  # 60% positive/negative
        news_decay_minutes: int = 30,  # News impact decay
        min_social_volume: int = 100,  # Minimum tweets/posts
        injury_impact_map: Optional[Dict[str, float]] = None,
        weather_impacts: Optional[Dict[str, float]] = None,
        use_sentiment_api: bool = True,
        **kwargs
    ):
        """
        Initialize news-based strategy.

        Args:
            sentiment_threshold: Sentiment threshold to trade
            news_decay_minutes: How fast news impact decays
            min_social_volume: Minimum social media volume
            injury_impact_map: Impact weights for player injuries
            weather_impacts: Weather condition impacts
            use_sentiment_api: Use external sentiment API
            **kwargs: Base strategy parameters
        """
        super().__init__(**kwargs)
        self.sentiment_threshold = sentiment_threshold
        self.news_decay_minutes = news_decay_minutes
        self.min_social_volume = min_social_volume
        self.injury_impact_map = injury_impact_map or {
            'quarterback': 0.15,
            'star_player': 0.10,
            'key_player': 0.07,
            'role_player': 0.03,
            'bench': 0.01
        }
        self.weather_impacts = weather_impacts or {
            'heavy_rain': -0.05,
            'snow': -0.08,
            'high_wind': -0.06,
            'extreme_cold': -0.04,
            'dome': 0.0
        }
        self.use_sentiment_api = use_sentiment_api
        self.recent_news: List[Dict] = []

    def analyze(
        self,
        market_data: pd.DataFrame,
        espn_data: Optional[Dict] = None,
        news_data: Optional[Dict] = None,
        social_data: Optional[Dict] = None,
        **kwargs
    ) -> Signal:
        """
        Analyze news and sentiment for trading signals.

        Args:
            market_data: Market prices
            espn_data: ESPN play-by-play
            news_data: News feed data
            social_data: Social media sentiment
            **kwargs: Additional parameters

        Returns:
            Trading signal
        """
        if market_data.empty:
            return self.hold()

        latest = market_data.iloc[-1]
        ticker = latest['ticker']

        # Check for injury news
        if news_data:
            injury_signal = self._check_injury_news(
                ticker, news_data, market_data
            )
            if injury_signal.type != SignalType.HOLD:
                return injury_signal

        # Check social sentiment
        if social_data:
            sentiment_signal = self._check_social_sentiment(
                ticker, social_data, market_data
            )
            if sentiment_signal.type != SignalType.HOLD:
                return sentiment_signal

        # Check weather updates
        weather_signal = self._check_weather_impact(
            ticker, news_data, market_data
        )
        if weather_signal.type != SignalType.HOLD:
            return weather_signal

        return self.hold(ticker)

    def _check_injury_news(
        self,
        ticker: str,
        news_data: Dict,
        market_data: pd.DataFrame
    ) -> Signal:
        """Check for injury-related news"""
        injuries = news_data.get('injuries', [])

        for injury in injuries:
            # Check if news is fresh
            if not self._is_news_fresh(injury.get('timestamp')):
                continue

            player = injury.get('player', '')
            team = injury.get('team', '')
            severity = injury.get('severity', 'questionable')
            position = injury.get('position', 'role_player')

            # Check if relevant to this game
            if not self._is_relevant_to_ticker(team, ticker):
                continue

            # Calculate impact
            impact = self._calculate_injury_impact(
                position, severity, player
            )

            if abs(impact) < 0.03:  # Minimum 3% impact
                continue

            # Generate signal based on impact
            latest = market_data.iloc[-1]

            if impact < 0:  # Negative for team
                # Buy NO if injury hurts team's chances
                size = self.calculate_position_size(
                    abs(impact), 1.0, 0.8
                )
                if size > 0:
                    return self.buy_no(
                        ticker=ticker,
                        size=size,
                        confidence=0.8,
                        entry_price=latest['no_ask'],
                        news_type='injury',
                        player=player,
                        impact=impact
                    )
            else:  # Positive (opponent injury)
                size = self.calculate_position_size(
                    impact, 1.0, 0.8
                )
                if size > 0:
                    return self.buy_yes(
                        ticker=ticker,
                        size=size,
                        confidence=0.8,
                        entry_price=latest['yes_ask'],
                        news_type='opponent_injury',
                        impact=impact
                    )

        return self.hold(ticker)

    def _check_social_sentiment(
        self,
        ticker: str,
        social_data: Dict,
        market_data: pd.DataFrame
    ) -> Signal:
        """Check social media sentiment"""
        if not social_data:
            return self.hold(ticker)

        # Extract sentiment metrics
        volume = social_data.get('volume', 0)
        sentiment = social_data.get('sentiment', 0.5)  # 0-1 scale
        momentum = social_data.get('momentum', 0)  # Rate of change

        if volume < self.min_social_volume:
            return self.hold(ticker)

        # Check for strong sentiment
        if sentiment > self.sentiment_threshold:
            # Positive sentiment
            edge = (sentiment - 0.5) * 0.2  # Convert to edge
            confidence = self._calculate_sentiment_confidence(
                sentiment, volume, momentum
            )

            latest = market_data.iloc[-1]
            size = self.calculate_position_size(edge, 1.0, confidence)

            if size > 0:
                return self.buy_yes(
                    ticker=ticker,
                    size=size,
                    confidence=confidence,
                    entry_price=latest['yes_ask'],
                    sentiment=sentiment,
                    social_volume=volume,
                    momentum=momentum
                )

        elif sentiment < (1 - self.sentiment_threshold):
            # Negative sentiment
            edge = (0.5 - sentiment) * 0.2
            confidence = self._calculate_sentiment_confidence(
                sentiment, volume, momentum
            )

            latest = market_data.iloc[-1]
            size = self.calculate_position_size(edge, 1.0, confidence)

            if size > 0:
                return self.buy_no(
                    ticker=ticker,
                    size=size,
                    confidence=confidence,
                    entry_price=latest['no_ask'],
                    sentiment=sentiment,
                    social_volume=volume,
                    momentum=momentum
                )

        return self.hold(ticker)

    def _check_weather_impact(
        self,
        ticker: str,
        news_data: Optional[Dict],
        market_data: pd.DataFrame
    ) -> Signal:
        """Check weather-related impacts"""
        if not news_data:
            return self.hold(ticker)

        weather = news_data.get('weather', {})
        if not weather:
            return self.hold(ticker)

        conditions = weather.get('conditions', 'clear')
        wind_speed = weather.get('wind_speed', 0)
        temperature = weather.get('temperature', 70)
        precipitation = weather.get('precipitation', 0)

        # Calculate total weather impact
        impact = 0

        if 'rain' in conditions.lower() and precipitation > 0.5:
            impact += self.weather_impacts.get('heavy_rain', -0.05)

        if 'snow' in conditions.lower():
            impact += self.weather_impacts.get('snow', -0.08)

        if wind_speed > 20:
            impact += self.weather_impacts.get('high_wind', -0.06)

        if temperature < 32:
            impact += self.weather_impacts.get('extreme_cold', -0.04)

        if abs(impact) < 0.03:
            return self.hold(ticker)

        # Weather typically affects scoring (under markets)
        latest = market_data.iloc[-1]

        # For weather, we typically fade the favorite in bad conditions
        if impact < 0:  # Bad weather
            if latest['yes_ask'] > 0.6:  # Favorite
                size = self.calculate_position_size(
                    abs(impact), 1.0, 0.7
                )
                if size > 0:
                    return self.buy_no(
                        ticker=ticker,
                        size=size,
                        confidence=0.7,
                        entry_price=latest['no_ask'],
                        weather_impact=impact,
                        conditions=conditions
                    )

        return self.hold(ticker)

    def _is_news_fresh(self, timestamp: Optional[str]) -> bool:
        """Check if news is recent enough to trade on"""
        if not timestamp:
            return False

        try:
            news_time = datetime.fromisoformat(timestamp)
            age = datetime.now() - news_time
            return age.total_seconds() < self.news_decay_minutes * 60
        except:
            return False

    def _is_relevant_to_ticker(self, team: str, ticker: str) -> bool:
        """Check if team is relevant to ticker"""
        if not team or not ticker:
            return False

        # Extract teams from ticker
        if '-' in ticker:
            parts = ticker.split('-')
            if len(parts) > 1:
                teams = parts[-1]  # e.g., "DETBAL"
                return team[:3].upper() in teams.upper()

        return False

    def _calculate_injury_impact(
        self,
        position: str,
        severity: str,
        player: str
    ) -> float:
        """Calculate market impact of injury"""
        base_impact = self.injury_impact_map.get(
            position.lower(), 0.03
        )

        # Adjust for severity
        severity_multipliers = {
            'out': 1.0,
            'doubtful': 0.8,
            'questionable': 0.4,
            'probable': 0.2
        }

        multiplier = severity_multipliers.get(severity.lower(), 0.5)
        return -base_impact * multiplier  # Negative for team

    def _calculate_sentiment_confidence(
        self,
        sentiment: float,
        volume: int,
        momentum: float
    ) -> float:
        """Calculate confidence from sentiment metrics"""
        confidence = 0.5

        # Sentiment strength
        sentiment_strength = abs(sentiment - 0.5) * 2
        confidence *= (1 + sentiment_strength)

        # Volume factor
        volume_factor = min(volume / 1000, 1.0)  # Cap at 1000
        confidence *= (0.5 + 0.5 * volume_factor)

        # Momentum factor
        if momentum > 0:
            confidence *= 1.2
        elif momentum < 0:
            confidence *= 0.8

        return min(confidence, 0.9)  # Cap at 90%


class BreakingNewsStrategy(NewsBasedStrategy):
    """
    Specialized strategy for immediate reaction to breaking news.

    Trades within seconds of major news breaks.
    """

    def __init__(
        self,
        reaction_time_seconds: int = 30,
        major_news_keywords: Optional[List[str]] = None,
        auto_close_minutes: int = 5,
        **kwargs
    ):
        """
        Initialize breaking news strategy.

        Args:
            reaction_time_seconds: Time window to react
            major_news_keywords: Keywords that trigger immediate action
            auto_close_minutes: Auto-close position after this time
            **kwargs: Base news strategy parameters
        """
        super().__init__(**kwargs)
        self.reaction_time_seconds = reaction_time_seconds
        self.major_news_keywords = major_news_keywords or [
            'injured', 'out', 'suspended', 'ejected',
            'benched', 'inactive', 'ruled out'
        ]
        self.auto_close_minutes = auto_close_minutes
        self.news_positions: Dict[str, datetime] = {}

    def analyze(
        self,
        market_data: pd.DataFrame,
        espn_data: Optional[Dict] = None,
        news_data: Optional[Dict] = None,
        **kwargs
    ) -> Signal:
        """
        React immediately to breaking news.

        Args:
            market_data: Market data
            espn_data: ESPN data
            news_data: Breaking news feed
            **kwargs: Additional parameters

        Returns:
            Immediate trading signal
        """
        # Check for positions to auto-close
        signal = self._check_auto_close(market_data)
        if signal.type == SignalType.CLOSE:
            return signal

        if not news_data:
            return self.hold()

        # Check for breaking news
        breaking = news_data.get('breaking', [])

        for news in breaking:
            if not self._is_breaking_fresh(news.get('timestamp')):
                continue

            # Check for major keywords
            headline = news.get('headline', '').lower()
            if not any(keyword in headline for keyword in self.major_news_keywords):
                continue

            # Immediate reaction
            ticker = market_data.iloc[-1]['ticker']
            if not self._is_relevant_to_ticker(
                news.get('team', ''), ticker
            ):
                continue

            # Trade immediately with high confidence
            latest = market_data.iloc[-1]
            impact = self._assess_breaking_impact(news)

            if abs(impact) < 0.05:
                continue

            size = int(self.current_capital * 0.2)  # 20% position

            if impact < 0:
                signal = self.buy_no(
                    ticker=ticker,
                    size=size,
                    confidence=0.9,
                    entry_price=latest['no_ask'],
                    breaking_news=headline,
                    immediate=True
                )
            else:
                signal = self.buy_yes(
                    ticker=ticker,
                    size=size,
                    confidence=0.9,
                    entry_price=latest['yes_ask'],
                    breaking_news=headline,
                    immediate=True
                )

            # Track for auto-close
            self.news_positions[ticker] = datetime.now()
            return signal

        return super().analyze(market_data, espn_data, news_data, **kwargs)

    def _is_breaking_fresh(self, timestamp: Optional[str]) -> bool:
        """Check if breaking news is within reaction window"""
        if not timestamp:
            return False

        try:
            news_time = datetime.fromisoformat(timestamp)
            age = datetime.now() - news_time
            return age.total_seconds() < self.reaction_time_seconds
        except:
            return False

    def _assess_breaking_impact(self, news: Dict) -> float:
        """Quick assessment of breaking news impact"""
        headline = news.get('headline', '').lower()

        # High impact keywords
        if any(word in headline for word in ['ruled out', 'ejected', 'suspended']):
            return -0.15

        # Medium impact
        if any(word in headline for word in ['injured', 'questionable', 'benched']):
            return -0.08

        # Low impact
        return -0.03

    def _check_auto_close(self, market_data: pd.DataFrame) -> Signal:
        """Check if any positions should auto-close"""
        if not self.news_positions:
            return self.hold()

        current_time = datetime.now()
        ticker = market_data.iloc[-1]['ticker']

        if ticker in self.news_positions:
            entry_time = self.news_positions[ticker]
            age = (current_time - entry_time).total_seconds() / 60

            if age >= self.auto_close_minutes:
                del self.news_positions[ticker]
                return self.close(
                    ticker=ticker,
                    reason='auto_close_timeout'
                )

        return self.hold(ticker)