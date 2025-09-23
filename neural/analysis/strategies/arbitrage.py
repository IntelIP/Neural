"""
Arbitrage Detection Strategy

Identifies and trades risk-free arbitrage opportunities in Kalshi markets.
Detects when YES + NO prices < $1.00 or cross-market discrepancies.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from .base import Strategy, Signal, SignalType


class ArbitrageStrategy(Strategy):
    """
    Arbitrage strategy for risk-free profit opportunities.

    Detects:
    - YES + NO < $1.00 arbitrage
    - Cross-market arbitrage (same event, different markets)
    - Sportsbook vs Kalshi arbitrage
    """

    def __init__(
        self,
        min_arbitrage_profit: float = 0.01,  # $0.01 minimum profit
        execution_confidence: float = 0.95,  # High confidence for arb
        max_exposure_per_arb: float = 0.3,  # 30% of capital per arb
        include_fees: bool = True,
        speed_priority: bool = True,  # Prioritize speed over size
        **kwargs
    ):
        """
        Initialize arbitrage strategy.

        Args:
            min_arbitrage_profit: Minimum profit per contract
            execution_confidence: Required confidence in execution
            max_exposure_per_arb: Maximum capital per arbitrage
            include_fees: Include Kalshi fees in calculation
            speed_priority: Prioritize execution speed
            **kwargs: Base strategy parameters
        """
        super().__init__(**kwargs)
        self.min_arbitrage_profit = min_arbitrage_profit
        self.execution_confidence = execution_confidence
        self.max_exposure_per_arb = max_exposure_per_arb
        self.include_fees = include_fees
        self.speed_priority = speed_priority
        self.active_arbitrages: List[Dict] = []

    def analyze(
        self,
        market_data: pd.DataFrame,
        espn_data: Optional[Dict] = None,
        cross_markets: Optional[Dict[str, pd.DataFrame]] = None,
        **kwargs
    ) -> Signal:
        """
        Analyze for arbitrage opportunities.

        Args:
            market_data: Primary market data
            espn_data: ESPN data (not used for arbitrage)
            cross_markets: Other markets for cross-market arbitrage
            **kwargs: Additional parameters

        Returns:
            Trading signal (immediate execution for arbitrage)
        """
        if market_data.empty:
            return self.hold()

        latest = market_data.iloc[-1]
        ticker = latest['ticker']

        # Check for YES+NO arbitrage
        yes_no_arb = self._check_yes_no_arbitrage(latest)
        if yes_no_arb:
            return yes_no_arb

        # Check for cross-market arbitrage
        if cross_markets:
            cross_arb = self._check_cross_market_arbitrage(
                latest, cross_markets
            )
            if cross_arb:
                return cross_arb

        # Check for sportsbook arbitrage
        if 'sportsbook_data' in kwargs:
            sb_arb = self._check_sportsbook_arbitrage(
                latest, kwargs['sportsbook_data']
            )
            if sb_arb:
                return sb_arb

        return self.hold(ticker)

    def _check_yes_no_arbitrage(self, market: pd.Series) -> Optional[Signal]:
        """
        Check for YES + NO < $1.00 arbitrage.

        Args:
            market: Current market data

        Returns:
            Signal if arbitrage exists
        """
        if 'yes_ask' not in market or 'no_ask' not in market:
            return None

        yes_price = market['yes_ask']
        no_price = market['no_ask']
        ticker = market['ticker']

        # Calculate total cost
        total_cost = yes_price + no_price

        # Include fees if enabled
        if self.include_fees:
            yes_fee = self.calculate_fees(yes_price, 1)
            no_fee = self.calculate_fees(no_price, 1)
            total_cost += yes_fee + no_fee

        # Check for arbitrage
        if total_cost < 1.0 - self.min_arbitrage_profit:
            profit_per_contract = 1.0 - total_cost

            # Calculate optimal size
            available_capital = self.current_capital * self.max_exposure_per_arb
            max_contracts = int(available_capital / total_cost)

            if self.speed_priority:
                # Smaller size for faster execution
                size = min(max_contracts, 100)
            else:
                size = max_contracts

            if size > 0:
                # Return composite signal for both sides
                return Signal(
                    type=SignalType.BUY_YES,  # Special handling needed
                    ticker=ticker,
                    size=size,
                    confidence=self.execution_confidence,
                    entry_price=yes_price,
                    metadata={
                        'strategy': 'yes_no_arbitrage',
                        'also_buy': 'no',
                        'no_price': no_price,
                        'no_size': size,
                        'total_cost': total_cost,
                        'expected_profit': profit_per_contract * size,
                        'profit_per_contract': profit_per_contract
                    }
                )

        return None

    def _check_cross_market_arbitrage(
        self,
        primary_market: pd.Series,
        cross_markets: Dict[str, pd.DataFrame]
    ) -> Optional[Signal]:
        """
        Check for arbitrage across related markets.

        Example: Team to win game vs Team to win by 7+
        If win by 7+ is cheaper than win, there's arbitrage.

        Args:
            primary_market: Primary market data
            cross_markets: Related markets data

        Returns:
            Signal if arbitrage exists
        """
        ticker = primary_market['ticker']

        # Find related markets
        related = self._find_related_markets(ticker, cross_markets)

        for related_ticker, related_data in related.items():
            if related_data.empty:
                continue

            related_latest = related_data.iloc[-1]

            # Check for logical arbitrage
            arb_opportunity = self._check_logical_arbitrage(
                primary_market, related_latest
            )

            if arb_opportunity:
                return arb_opportunity

        return None

    def _find_related_markets(
        self,
        ticker: str,
        cross_markets: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """Find markets related to the primary ticker"""
        related = {}

        # Extract team codes from ticker
        if '-' in ticker:
            parts = ticker.split('-')
            if len(parts) > 1:
                teams = parts[-1]  # e.g., "DETBAL"

                # Find markets with same teams
                for market_ticker, market_data in cross_markets.items():
                    if teams in market_ticker and market_ticker != ticker:
                        related[market_ticker] = market_data

        return related

    def _check_logical_arbitrage(
        self,
        market1: pd.Series,
        market2: pd.Series
    ) -> Optional[Signal]:
        """
        Check for logical arbitrage between two markets.

        Example: If "Team A wins" = 0.60 and "Team A wins by 7+" = 0.65,
        there's an arbitrage since winning by 7+ implies winning.

        Args:
            market1: First market
            market2: Second market

        Returns:
            Signal if logical arbitrage exists
        """
        # This requires specific market relationship logic
        # Implement based on actual Kalshi market structures

        # Example: Check if one market implies another
        ticker1 = market1['ticker']
        ticker2 = market2['ticker']

        # Check for spread markets
        if 'SPREAD' in ticker2 and 'SPREAD' not in ticker1:
            # market1 is win/loss, market2 is spread
            yes_price1 = market1['yes_ask']
            yes_price2 = market2['yes_ask']

            # If spread YES is cheaper than outright WIN
            if yes_price2 < yes_price1 - self.min_arbitrage_profit:
                size = self.calculate_position_size(
                    yes_price1 - yes_price2, 1.0, self.execution_confidence
                )
                if size > 0:
                    return Signal(
                        type=SignalType.BUY_YES,
                        ticker=ticker2,  # Buy the cheaper implied bet
                        size=size,
                        confidence=self.execution_confidence,
                        entry_price=yes_price2,
                        metadata={
                            'strategy': 'cross_market_arbitrage',
                            'hedge_market': ticker1,
                            'hedge_price': yes_price1,
                            'arbitrage_profit': yes_price1 - yes_price2
                        }
                    )

        return None

    def _check_sportsbook_arbitrage(
        self,
        market: pd.Series,
        sportsbook_data: Dict
    ) -> Optional[Signal]:
        """
        Check for arbitrage between Kalshi and sportsbooks.

        Args:
            market: Kalshi market data
            sportsbook_data: Sportsbook odds

        Returns:
            Signal if arbitrage exists
        """
        ticker = market['ticker']
        kalshi_yes = market['yes_ask']
        kalshi_no = market['no_ask']

        # Get sportsbook consensus
        sb_prob = self._get_sportsbook_probability(ticker, sportsbook_data)
        if sb_prob is None:
            return None

        # Check for arbitrage opportunities
        # Buy YES on Kalshi if significantly cheaper than sportsbook
        if kalshi_yes < sb_prob - self.min_arbitrage_profit:
            if self.include_fees:
                fee = self.calculate_fees(kalshi_yes, 1)
                if kalshi_yes + fee >= sb_prob:
                    return None

            profit = sb_prob - kalshi_yes
            size = self.calculate_position_size(
                profit, 1.0, self.execution_confidence
            )

            if size > 0:
                return self.buy_yes(
                    ticker=ticker,
                    size=size,
                    confidence=self.execution_confidence,
                    entry_price=kalshi_yes,
                    strategy='sportsbook_arbitrage',
                    sportsbook_prob=sb_prob,
                    expected_profit=profit * size
                )

        # Buy NO on Kalshi if significantly cheaper
        if kalshi_no < (1 - sb_prob) - self.min_arbitrage_profit:
            if self.include_fees:
                fee = self.calculate_fees(kalshi_no, 1)
                if kalshi_no + fee >= (1 - sb_prob):
                    return None

            profit = (1 - sb_prob) - kalshi_no
            size = self.calculate_position_size(
                profit, 1.0, self.execution_confidence
            )

            if size > 0:
                return self.buy_no(
                    ticker=ticker,
                    size=size,
                    confidence=self.execution_confidence,
                    entry_price=kalshi_no,
                    strategy='sportsbook_arbitrage',
                    sportsbook_prob=sb_prob,
                    expected_profit=profit * size
                )

        return None

    def _get_sportsbook_probability(
        self,
        ticker: str,
        sportsbook_data: Dict
    ) -> Optional[float]:
        """Extract sportsbook implied probability"""
        if not sportsbook_data:
            return None

        # Extract team from ticker
        if '-' in ticker:
            parts = ticker.split('-')
            teams = parts[-1] if len(parts) > 1 else ""

            # Look for matching game in sportsbook data
            for game, odds in sportsbook_data.items():
                if teams[:3] in game or teams[-3:] in game:
                    # Convert odds to probability
                    if 'moneyline' in odds:
                        return self._moneyline_to_probability(odds['moneyline'])
                    elif 'decimal' in odds:
                        return 1 / odds['decimal']

        return None

    def _moneyline_to_probability(self, moneyline: float) -> float:
        """Convert moneyline odds to probability"""
        if moneyline > 0:
            return 100 / (moneyline + 100)
        else:
            return abs(moneyline) / (abs(moneyline) + 100)


class HighSpeedArbitrageStrategy(ArbitrageStrategy):
    """
    High-speed arbitrage for immediate execution.

    Optimized for speed with pre-calculated sizes and instant execution.
    """

    def __init__(
        self,
        pre_calculate_size: bool = True,
        fixed_size: int = 100,  # Fixed size for speed
        latency_threshold_ms: float = 50,  # Max acceptable latency
        **kwargs
    ):
        """
        Initialize high-speed arbitrage.

        Args:
            pre_calculate_size: Use fixed pre-calculated sizes
            fixed_size: Fixed contract size for speed
            latency_threshold_ms: Maximum latency tolerance
            **kwargs: Base arbitrage parameters
        """
        super().__init__(speed_priority=True, **kwargs)
        self.pre_calculate_size = pre_calculate_size
        self.fixed_size = fixed_size
        self.latency_threshold_ms = latency_threshold_ms

    def analyze(
        self,
        market_data: pd.DataFrame,
        espn_data: Optional[Dict] = None,
        **kwargs
    ) -> Signal:
        """
        Fast arbitrage detection with pre-calculated parameters.

        Args:
            market_data: Market data
            espn_data: Not used
            **kwargs: Additional parameters

        Returns:
            Immediate execution signal
        """
        if market_data.empty:
            return self.hold()

        latest = market_data.iloc[-1]

        # Quick YES+NO check (most common arbitrage)
        if 'yes_ask' in latest and 'no_ask' in latest:
            total = latest['yes_ask'] + latest['no_ask']

            # No fee calculation for speed
            if total < 0.99:  # Quick threshold
                # Use fixed size for speed
                size = self.fixed_size if self.pre_calculate_size else 100

                return Signal(
                    type=SignalType.BUY_YES,
                    ticker=latest['ticker'],
                    size=size,
                    confidence=1.0,
                    entry_price=latest['yes_ask'],
                    metadata={
                        'strategy': 'high_speed_arbitrage',
                        'also_buy': 'no',
                        'no_price': latest['no_ask'],
                        'no_size': size,
                        'total_cost': total,
                        'immediate': True
                    }
                )

        return self.hold(latest['ticker'])