"""
Momentum Trading Strategy

Trades based on momentum indicators and trend following.
Particularly effective during game events when markets trend strongly.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from .base import Strategy, Signal, SignalType


class MomentumStrategy(Strategy):
    """
    Momentum strategy that follows strong price trends.

    Works best during live games when markets trend with game events.
    """

    def __init__(
        self,
        lookback_periods: int = 10,
        momentum_threshold: float = 0.1,  # 10% momentum to trigger
        volume_confirmation: bool = True,
        use_rsi: bool = True,
        rsi_overbought: float = 70,
        rsi_oversold: float = 30,
        trend_strength_min: float = 0.6,  # R-squared of trend
        **kwargs
    ):
        """
        Initialize momentum strategy.

        Args:
            lookback_periods: Periods for momentum calculation
            momentum_threshold: Minimum momentum to trade
            volume_confirmation: Require volume increase
            use_rsi: Use RSI indicator
            rsi_overbought: RSI overbought level
            rsi_oversold: RSI oversold level
            trend_strength_min: Minimum trend strength (R²)
            **kwargs: Base strategy parameters
        """
        super().__init__(**kwargs)
        self.lookback_periods = lookback_periods
        self.momentum_threshold = momentum_threshold
        self.volume_confirmation = volume_confirmation
        self.use_rsi = use_rsi
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.trend_strength_min = trend_strength_min

    def analyze(
        self,
        market_data: pd.DataFrame,
        espn_data: Optional[Dict] = None,
        **kwargs
    ) -> Signal:
        """
        Analyze market for momentum opportunities.

        Args:
            market_data: DataFrame with columns: ticker, yes_ask, no_ask, volume
            espn_data: Optional ESPN play-by-play data
            **kwargs: Additional parameters

        Returns:
            Trading signal
        """
        if len(market_data) < self.lookback_periods:
            return self.hold()

        latest = market_data.iloc[-1]
        ticker = latest['ticker']

        # Calculate momentum indicators
        momentum = self._calculate_momentum(market_data)
        if momentum is None:
            return self.hold(ticker)

        # Check volume confirmation
        if self.volume_confirmation:
            if not self._check_volume_trend(market_data):
                return self.hold(ticker)

        # Calculate RSI if enabled
        if self.use_rsi:
            rsi = self._calculate_rsi(market_data)
            if rsi is None:
                return self.hold(ticker)
        else:
            rsi = 50  # Neutral

        # Check trend strength
        trend_strength = self._calculate_trend_strength(market_data)
        if trend_strength < self.trend_strength_min:
            return self.hold(ticker)

        # Generate signal based on momentum
        if momentum > self.momentum_threshold and rsi < self.rsi_overbought:
            # Strong upward momentum, buy YES
            edge = momentum * trend_strength
            confidence = self._calculate_confidence(momentum, trend_strength, rsi, espn_data)
            size = self.calculate_position_size(edge, 1.0, confidence)

            if size > 0:
                return self.buy_yes(
                    ticker=ticker,
                    size=size,
                    confidence=confidence,
                    entry_price=latest['yes_ask'],
                    momentum=momentum,
                    rsi=rsi,
                    trend_strength=trend_strength
                )

        elif momentum < -self.momentum_threshold and rsi > self.rsi_oversold:
            # Strong downward momentum, buy NO
            edge = abs(momentum) * trend_strength
            confidence = self._calculate_confidence(momentum, trend_strength, rsi, espn_data)
            size = self.calculate_position_size(edge, 1.0, confidence)

            if size > 0:
                return self.buy_no(
                    ticker=ticker,
                    size=size,
                    confidence=confidence,
                    entry_price=latest['no_ask'],
                    momentum=momentum,
                    rsi=rsi,
                    trend_strength=trend_strength
                )

        return self.hold(ticker)

    def _calculate_momentum(self, market_data: pd.DataFrame) -> Optional[float]:
        """Calculate price momentum"""
        if 'yes_ask' not in market_data.columns:
            return None

        prices = market_data['yes_ask'].tail(self.lookback_periods + 1).values
        if len(prices) < 2:
            return None

        # Simple momentum: (current - past) / past
        momentum = (prices[-1] - prices[0]) / prices[0] if prices[0] != 0 else 0
        return momentum

    def _calculate_rsi(self, market_data: pd.DataFrame, periods: int = 14) -> Optional[float]:
        """Calculate Relative Strength Index"""
        if 'yes_ask' not in market_data.columns or len(market_data) < periods + 1:
            return None

        prices = market_data['yes_ask'].tail(periods + 1).values
        deltas = np.diff(prices)

        gains = deltas[deltas > 0].sum() / periods if len(deltas[deltas > 0]) > 0 else 0
        losses = -deltas[deltas < 0].sum() / periods if len(deltas[deltas < 0]) > 0 else 0

        if losses == 0:
            return 100

        rs = gains / losses if losses != 0 else 100
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_trend_strength(self, market_data: pd.DataFrame) -> float:
        """Calculate trend strength using R-squared"""
        if 'yes_ask' not in market_data.columns:
            return 0

        prices = market_data['yes_ask'].tail(self.lookback_periods).values
        if len(prices) < 3:
            return 0

        # Linear regression
        x = np.arange(len(prices))
        coeffs = np.polyfit(x, prices, 1)
        predicted = np.poly1d(coeffs)(x)

        # Calculate R-squared
        ss_res = np.sum((prices - predicted) ** 2)
        ss_tot = np.sum((prices - np.mean(prices)) ** 2)

        if ss_tot == 0:
            return 0

        r_squared = 1 - (ss_res / ss_tot)
        return max(0, r_squared)

    def _check_volume_trend(self, market_data: pd.DataFrame) -> bool:
        """Check if volume is increasing with price movement"""
        if 'volume' not in market_data.columns:
            return True  # Don't block if no volume data

        recent = market_data.tail(self.lookback_periods)
        if len(recent) < 2:
            return True

        # Check if volume is trending up
        volumes = recent['volume'].values
        avg_early = np.mean(volumes[:len(volumes)//2])
        avg_late = np.mean(volumes[len(volumes)//2:])

        return avg_late > avg_early * 1.2  # 20% increase

    def _calculate_confidence(
        self,
        momentum: float,
        trend_strength: float,
        rsi: float,
        espn_data: Optional[Dict]
    ) -> float:
        """Calculate confidence based on multiple factors"""
        confidence = 1.0

        # Momentum strength factor
        momentum_factor = min(abs(momentum) / self.momentum_threshold, 2.0) / 2.0
        confidence *= momentum_factor

        # Trend strength factor
        confidence *= trend_strength

        # RSI extremes reduce confidence
        if rsi > 70 or rsi < 30:
            confidence *= 0.8

        # ESPN data confirmation
        if espn_data and self.use_espn:
            if 'scoring_drive' in espn_data:
                # Boost confidence during scoring drives
                confidence *= 1.2
            if 'red_zone' in espn_data and espn_data['red_zone']:
                # High confidence in red zone
                confidence *= 1.3

        return min(confidence, 1.0)


class GameMomentumStrategy(MomentumStrategy):
    """
    Specialized momentum strategy for in-game trading.

    Trades on game events like touchdowns, turnovers, injuries.
    """

    def __init__(
        self,
        event_window: int = 5,  # Minutes after event
        event_multipliers: Optional[Dict[str, float]] = None,
        fade_blowouts: bool = True,
        blowout_threshold: float = 0.8,  # 80% probability
        **kwargs
    ):
        """
        Initialize game momentum strategy.

        Args:
            event_window: Minutes to trade after major event
            event_multipliers: Confidence multipliers for events
            fade_blowouts: Fade extreme prices
            blowout_threshold: Threshold for fading
            **kwargs: Base momentum parameters
        """
        super().__init__(**kwargs)
        self.event_window = event_window
        self.event_multipliers = event_multipliers or {
            'touchdown': 1.5,
            'field_goal': 1.2,
            'turnover': 1.4,
            'injury_star': 1.6,
            'red_zone': 1.3,
            'two_minute': 1.4
        }
        self.fade_blowouts = fade_blowouts
        self.blowout_threshold = blowout_threshold
        self.recent_events: List[Dict] = []

    def analyze(
        self,
        market_data: pd.DataFrame,
        espn_data: Optional[Dict] = None,
        **kwargs
    ) -> Signal:
        """
        Analyze for game-specific momentum.

        Args:
            market_data: Market prices
            espn_data: ESPN play-by-play data
            **kwargs: Additional parameters

        Returns:
            Trading signal
        """
        if not espn_data:
            return super().analyze(market_data, espn_data, **kwargs)

        latest = market_data.iloc[-1]
        ticker = latest['ticker']
        yes_price = latest['yes_ask']

        # Check for blowout fade opportunity
        if self.fade_blowouts:
            if yes_price > self.blowout_threshold:
                # Fade the favorite
                size = self.calculate_position_size(0.05, 1.0, 0.7)
                return self.buy_no(
                    ticker=ticker,
                    size=size,
                    confidence=0.7,
                    entry_price=latest['no_ask'],
                    strategy='fade_blowout',
                    yes_price=yes_price
                )
            elif yes_price < (1 - self.blowout_threshold):
                # Fade the underdog being written off
                size = self.calculate_position_size(0.05, 1.0, 0.7)
                return self.buy_yes(
                    ticker=ticker,
                    size=size,
                    confidence=0.7,
                    entry_price=yes_price,
                    strategy='fade_blowout'
                )

        # Check for recent game events
        event_signal = self._check_game_events(espn_data, market_data)
        if event_signal.type != SignalType.HOLD:
            return event_signal

        # Fall back to regular momentum
        return super().analyze(market_data, espn_data, **kwargs)

    def _check_game_events(
        self,
        espn_data: Dict,
        market_data: pd.DataFrame
    ) -> Signal:
        """Check for tradeable game events"""
        ticker = market_data.iloc[-1]['ticker']

        # Check for touchdown
        if espn_data.get('last_play', {}).get('touchdown'):
            team = espn_data['last_play'].get('team')
            if self._is_home_team(team, ticker):
                # Home team scored, momentum up
                return self._create_event_signal(
                    'touchdown', True, market_data, espn_data
                )
            else:
                # Away team scored, momentum down
                return self._create_event_signal(
                    'touchdown', False, market_data, espn_data
                )

        # Check for turnover
        if espn_data.get('last_play', {}).get('turnover'):
            team = espn_data['last_play'].get('team')
            if self._is_home_team(team, ticker):
                # Home team turned it over, bad
                return self._create_event_signal(
                    'turnover', False, market_data, espn_data
                )
            else:
                # Away team turned it over, good for home
                return self._create_event_signal(
                    'turnover', True, market_data, espn_data
                )

        # Check for red zone
        if espn_data.get('red_zone'):
            return self._create_event_signal(
                'red_zone', True, market_data, espn_data
            )

        return self.hold(ticker)

    def _create_event_signal(
        self,
        event_type: str,
        bullish: bool,
        market_data: pd.DataFrame,
        espn_data: Dict
    ) -> Signal:
        """Create signal based on game event"""
        latest = market_data.iloc[-1]
        ticker = latest['ticker']

        multiplier = self.event_multipliers.get(event_type, 1.0)
        confidence = 0.6 * multiplier  # Base confidence times multiplier

        # Calculate position size
        edge = 0.05 * multiplier  # Assumed edge from event
        size = self.calculate_position_size(edge, 1.0, confidence)

        if size == 0:
            return self.hold(ticker)

        if bullish:
            return self.buy_yes(
                ticker=ticker,
                size=size,
                confidence=confidence,
                entry_price=latest['yes_ask'],
                event_type=event_type,
                game_time=espn_data.get('game_clock')
            )
        else:
            return self.buy_no(
                ticker=ticker,
                size=size,
                confidence=confidence,
                entry_price=latest['no_ask'],
                event_type=event_type,
                game_time=espn_data.get('game_clock')
            )

    def _is_home_team(self, team: str, ticker: str) -> bool:
        """Check if team is home team based on ticker"""
        # Ticker format: KXNFLGAME-25SEP22DETBAL
        # Home team is typically listed second
        parts = ticker.split('-')
        if len(parts) > 1:
            teams = parts[-1]
            # Last 3 chars are typically home team
            return team.upper()[:3] == teams[-3:]
        return False