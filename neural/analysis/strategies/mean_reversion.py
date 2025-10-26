"""
Mean Reversion Strategy

Trades when Kalshi prices diverge significantly from sportsbook consensus
or historical averages.
"""

import numpy as np
import pandas as pd

from .base import Signal, Strategy


class MeanReversionStrategy(Strategy):
    """
    Mean reversion strategy that identifies price divergences from consensus.

    This strategy assumes prices will revert to their mean (consensus) value
    and trades when significant divergences occur.
    """

    def __init__(
        self,
        divergence_threshold: float = 0.05,  # 5% divergence triggers signal
        reversion_target: float = 0.5,  # Target 50% reversion
        use_sportsbook: bool = True,
        lookback_periods: int = 20,
        confidence_decay: float = 0.95,  # Confidence decreases with time
        **kwargs,
    ):
        """
        Initialize mean reversion strategy.

        Args:
            divergence_threshold: Minimum price divergence to trigger signal
            reversion_target: Expected reversion percentage (0-1)
            use_sportsbook: Use sportsbook consensus if available
            lookback_periods: Number of periods for moving average
            confidence_decay: How confidence decreases over time
            **kwargs: Additional base strategy parameters
        """
        super().__init__(**kwargs)
        self.divergence_threshold = divergence_threshold
        self.reversion_target = reversion_target
        self.use_sportsbook = use_sportsbook
        self.lookback_periods = lookback_periods
        self.confidence_decay = confidence_decay
        self.price_history: dict[str, list[float]] = {}

    def analyze(self, market_data: pd.DataFrame, espn_data: dict | None = None, **kwargs) -> Signal:
        """
        Analyze market for mean reversion opportunities.

        Args:
            market_data: DataFrame with columns: ticker, yes_ask, no_ask, volume
            espn_data: Optional ESPN data for context
            **kwargs: Additional parameters

        Returns:
            Trading signal
        """
        if market_data.empty:
            return self.hold()

        # Get the latest market
        latest = market_data.iloc[-1]
        ticker = latest["ticker"]
        yes_price = latest["yes_ask"]
        no_price = latest["no_ask"]

        # Calculate mean price (fair value)
        fair_value = self._calculate_fair_value(ticker, yes_price, market_data)

        if fair_value is None:
            return self.hold(ticker)

        # Calculate divergence
        divergence = yes_price - fair_value

        # Check if divergence exceeds threshold
        if abs(divergence) < self.divergence_threshold:
            return self.hold(ticker)

        # Calculate expected edge
        edge = abs(divergence) * self.reversion_target

        # Adjust confidence based on various factors
        confidence = self._calculate_confidence(divergence, market_data, espn_data)

        # Calculate position size
        size = self.calculate_position_size(edge, 1.0, confidence)

        if size == 0:
            return self.hold(ticker)

        # Generate signal based on divergence direction
        if divergence > 0:  # Price too high, expect it to fall
            return self.buy_no(
                ticker=ticker,
                size=size,
                confidence=confidence,
                entry_price=no_price,
                target_price=1 - fair_value,
                stop_loss=no_price * 1.2,  # 20% stop loss
                divergence=divergence,
                fair_value=fair_value,
            )
        else:  # Price too low, expect it to rise
            return self.buy_yes(
                ticker=ticker,
                size=size,
                confidence=confidence,
                entry_price=yes_price,
                target_price=fair_value,
                stop_loss=yes_price * 0.8,  # 20% stop loss
                divergence=divergence,
                fair_value=fair_value,
            )

    def _calculate_fair_value(
        self, ticker: str, current_price: float, market_data: pd.DataFrame
    ) -> float | None:
        """
        Calculate fair value using multiple methods.

        Args:
            ticker: Market ticker
            current_price: Current market price
            market_data: Historical market data

        Returns:
            Estimated fair value or None
        """
        fair_values = []
        weights = []

        # Method 1: Sportsbook consensus
        if self.use_sportsbook:
            consensus = self.get_sportsbook_consensus(ticker)
            if consensus is not None:
                fair_values.append(consensus)
                weights.append(2.0)  # Higher weight for sportsbook

        # Method 2: Historical moving average
        if ticker in self.price_history:
            history = self.price_history[ticker]
            if len(history) >= self.lookback_periods:
                ma = np.mean(history[-self.lookback_periods :])
                fair_values.append(ma)
                weights.append(1.0)
        else:
            self.price_history[ticker] = []

        # Update price history
        self.price_history[ticker].append(current_price)
        if len(self.price_history[ticker]) > self.lookback_periods * 2:
            self.price_history[ticker] = self.price_history[ticker][-self.lookback_periods * 2 :]

        # Method 3: Volume-weighted average price (VWAP)
        if "volume" in market_data.columns and len(market_data) > 1:
            vwap = self._calculate_vwap(market_data)
            if vwap is not None:
                fair_values.append(vwap)
                weights.append(1.5)

        # Method 4: Bid-ask midpoint
        if "yes_bid" in market_data.columns:
            latest = market_data.iloc[-1]
            midpoint = (latest["yes_bid"] + latest["yes_ask"]) / 2
            fair_values.append(midpoint)
            weights.append(0.5)

        # Calculate weighted average
        if fair_values:
            return float(np.average(fair_values, weights=weights))

        return None

    def _calculate_vwap(self, market_data: pd.DataFrame) -> float | None:
        """Calculate volume-weighted average price"""
        if "volume" not in market_data.columns or market_data["volume"].sum() == 0:
            return None

        # Use last N periods
        recent = market_data.tail(self.lookback_periods)
        if "yes_ask" in recent.columns and "volume" in recent.columns:
            prices = recent["yes_ask"].values.astype(float)
            volumes = recent["volume"].values.astype(float)
            if float(np.sum(volumes)) > 0:
                return float(np.sum(prices * volumes) / np.sum(volumes))

        return None

    def _calculate_confidence(
        self, divergence: float, market_data: pd.DataFrame, espn_data: dict | None
    ) -> float:
        """
        Calculate confidence level for the trade.

        Args:
            divergence: Price divergence from fair value
            market_data: Market data
            espn_data: ESPN data if available

        Returns:
            Confidence level (0-1)
        """
        confidence = 1.0

        # Factor 1: Divergence strength
        divergence_factor = min(abs(divergence) / self.divergence_threshold, 2.0) / 2.0
        confidence *= divergence_factor

        # Factor 2: Volume confirmation
        if "volume" in market_data.columns:
            latest_volume = market_data.iloc[-1]["volume"]
            avg_volume = market_data["volume"].mean()
            if avg_volume > 0:
                volume_factor = min(latest_volume / avg_volume, 1.5) / 1.5
                confidence *= volume_factor

        # Factor 3: Time decay (less confident as event approaches)
        if "close_time" in market_data.columns:
            # Implement time decay logic
            confidence *= self.confidence_decay

        # Factor 4: ESPN data confirmation
        if espn_data and self.use_espn:
            # Check if ESPN data supports our thesis
            if "momentum" in espn_data:
                if (divergence > 0 and espn_data["momentum"] < 0) or (
                    divergence < 0 and espn_data["momentum"] > 0
                ):
                    confidence *= 1.2  # Boost confidence if ESPN agrees
                else:
                    confidence *= 0.8  # Reduce if ESPN disagrees

        return min(confidence, 1.0)

    def should_exit_position(self, position, current_price: float, fair_value: float) -> bool:
        """
        Determine if we should exit a mean reversion position.

        Args:
            position: Current position
            current_price: Current market price
            fair_value: Current fair value estimate

        Returns:
            True if position should be closed
        """
        # Check standard stop loss and take profit
        if super().should_close_position(position):
            return True

        # Check if reversion is complete
        if position.side == "yes":
            price_diff = abs(current_price - fair_value)
            initial_diff = abs(position.metadata.get("divergence", 0))
            if price_diff < initial_diff * (1 - self.reversion_target):
                return True
        else:  # "no" position
            price_diff = abs((1 - current_price) - (1 - fair_value))
            initial_diff = abs(position.metadata.get("divergence", 0))
            if price_diff < initial_diff * (1 - self.reversion_target):
                return True

        return False


class SportsbookArbitrageStrategy(MeanReversionStrategy):
    """
    Specialized mean reversion for sportsbook arbitrage opportunities.

    Trades when Kalshi prices significantly diverge from sportsbook consensus.
    """

    def __init__(
        self,
        min_sportsbook_sources: int = 3,
        max_line_age_seconds: int = 60,
        arbitrage_threshold: float = 0.03,  # 3% minimum arbitrage
        **kwargs,
    ):
        """
        Initialize sportsbook arbitrage strategy.

        Args:
            min_sportsbook_sources: Minimum number of sportsbooks for consensus
            max_line_age_seconds: Maximum age of sportsbook data
            arbitrage_threshold: Minimum profit threshold
            **kwargs: Base strategy parameters
        """
        super().__init__(use_sportsbook=True, **kwargs)
        self.min_sportsbook_sources = min_sportsbook_sources
        self.max_line_age_seconds = max_line_age_seconds
        self.arbitrage_threshold = arbitrage_threshold

    def analyze(
        self,
        market_data: pd.DataFrame,
        espn_data: dict | None = None,
        sportsbook_data: dict | None = None,
        **kwargs,
    ) -> Signal:
        """
        Analyze for sportsbook arbitrage opportunities.

        Args:
            market_data: Kalshi market data
            espn_data: ESPN play-by-play data
            sportsbook_data: Dictionary of sportsbook lines
            **kwargs: Additional parameters

        Returns:
            Trading signal
        """
        if not sportsbook_data or market_data.empty:
            return self.hold()

        latest = market_data.iloc[-1]
        ticker = latest["ticker"]
        kalshi_yes = latest["yes_ask"]
        kalshi_no = latest["no_ask"]

        # Calculate sportsbook consensus
        consensus = self._calculate_sportsbook_consensus(sportsbook_data)
        if consensus is None:
            return self.hold(ticker)

        # Check for arbitrage opportunity
        if kalshi_yes < consensus - self.arbitrage_threshold:
            # Kalshi YES is cheap relative to sportsbooks
            edge = consensus - kalshi_yes
            size = self.calculate_position_size(edge, 1.0, 0.9)  # High confidence
            return self.buy_yes(
                ticker=ticker,
                size=size,
                confidence=0.9,
                entry_price=kalshi_yes,
                sportsbook_consensus=consensus,
                arbitrage_profit=edge,
            )
        elif kalshi_no < (1 - consensus) - self.arbitrage_threshold:
            # Kalshi NO is cheap relative to sportsbooks
            edge = (1 - consensus) - kalshi_no
            size = self.calculate_position_size(edge, 1.0, 0.9)
            return self.buy_no(
                ticker=ticker,
                size=size,
                confidence=0.9,
                entry_price=kalshi_no,
                sportsbook_consensus=consensus,
                arbitrage_profit=edge,
            )

        return self.hold(ticker)

    def _calculate_sportsbook_consensus(self, sportsbook_data: dict) -> float | None:
        """Calculate consensus probability from multiple sportsbooks"""
        if not sportsbook_data:
            return None

        valid_lines = []
        current_time = pd.Timestamp.now()

        for _book, data in sportsbook_data.items():
            # Check data freshness
            if "timestamp" in data:
                age = (current_time - data["timestamp"]).seconds
                if age > self.max_line_age_seconds:
                    continue

            # Extract probability
            if "implied_probability" in data:
                valid_lines.append(data["implied_probability"])
            elif "moneyline" in data:
                # Convert moneyline to probability
                ml = data["moneyline"]
                if ml > 0:
                    prob = 100 / (ml + 100)
                else:
                    prob = abs(ml) / (abs(ml) + 100)
                valid_lines.append(prob)

        if len(valid_lines) >= self.min_sportsbook_sources:
            return float(np.median(valid_lines))  # Use median to reduce outlier impact

        return None
