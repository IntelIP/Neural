"""
Position Sizing Methods for Risk Management

Implements various position sizing algorithms including Kelly Criterion,
fixed percentage, and edge-proportional sizing.
"""

import numpy as np
from typing import Optional, Dict, Tuple


def kelly_criterion(
    edge: float,
    odds: float,
    kelly_fraction: float = 0.25,
    max_position: float = 0.3
) -> float:
    """
    Calculate position size using Kelly Criterion.

    Kelly formula: f = (p*b - q) / b
    Where:
    - f = fraction of capital to bet
    - p = probability of winning
    - q = probability of losing (1-p)
    - b = odds received on the bet

    Args:
        edge: Your edge (your_prob - market_prob)
        odds: Potential payout odds
        kelly_fraction: Fraction of Kelly to use (safety)
        max_position: Maximum position as fraction of capital

    Returns:
        Fraction of capital to bet

    Example:
        >>> size = kelly_criterion(edge=0.05, odds=1.0, kelly_fraction=0.25)
        >>> position_value = capital * size
    """
    if edge <= 0 or odds <= 0:
        return 0.0

    # Estimate true probability from edge
    # edge = true_prob - market_prob
    # Assuming market is efficient, market_prob ≈ 0.5
    true_prob = 0.5 + edge

    # Kelly formula
    kelly_full = (true_prob * odds - (1 - true_prob)) / odds

    # Apply safety fraction
    kelly_size = kelly_full * kelly_fraction

    # Apply maximum limit
    return min(max(kelly_size, 0), max_position)


def fixed_percentage(
    capital: float,
    percentage: float = 0.02,
    min_contracts: int = 10,
    max_contracts: Optional[int] = None
) -> int:
    """
    Fixed percentage position sizing.

    Always risks a fixed percentage of capital.

    Args:
        capital: Current capital
        percentage: Percentage to risk (0.02 = 2%)
        min_contracts: Minimum contracts to trade
        max_contracts: Maximum contracts limit

    Returns:
        Number of contracts to trade

    Example:
        >>> contracts = fixed_percentage(capital=10000, percentage=0.02)
    """
    position_value = capital * percentage
    contracts = int(position_value)

    # Apply minimum
    contracts = max(contracts, min_contracts)

    # Apply maximum if set
    if max_contracts:
        contracts = min(contracts, max_contracts)

    return contracts


def edge_proportional(
    edge: float,
    capital: float,
    base_percentage: float = 0.01,
    edge_multiplier: float = 2.0,
    min_edge: float = 0.02,
    max_percentage: float = 0.1
) -> int:
    """
    Position size proportional to edge.

    Larger positions for stronger edges.

    Args:
        edge: Expected edge
        capital: Current capital
        base_percentage: Base position size
        edge_multiplier: How much to scale with edge
        min_edge: Minimum edge to trade
        max_percentage: Maximum position size

    Returns:
        Number of contracts

    Example:
        >>> contracts = edge_proportional(edge=0.05, capital=10000)
    """
    if edge < min_edge:
        return 0

    # Scale position with edge
    edge_factor = edge / min_edge
    position_percentage = base_percentage * min(edge_factor * edge_multiplier, max_percentage / base_percentage)

    # Cap at maximum
    position_percentage = min(position_percentage, max_percentage)

    position_value = capital * position_percentage
    return int(position_value)


def martingale(
    capital: float,
    consecutive_losses: int,
    base_size: float = 0.01,
    multiplier: float = 2.0,
    max_size: float = 0.16
) -> int:
    """
    Martingale position sizing (USE WITH CAUTION).

    Doubles position after each loss.
    WARNING: Can lead to rapid account depletion.

    Args:
        capital: Current capital
        consecutive_losses: Number of consecutive losses
        base_size: Base position size
        multiplier: Size multiplier after loss
        max_size: Maximum position size

    Returns:
        Number of contracts

    Example:
        >>> # After 3 losses, position = base * 2^3 = 8x base
        >>> contracts = martingale(capital=10000, consecutive_losses=3)
    """
    # Calculate current size
    current_size = base_size * (multiplier ** consecutive_losses)

    # Cap at maximum
    current_size = min(current_size, max_size)

    position_value = capital * current_size
    return int(position_value)


def anti_martingale(
    capital: float,
    consecutive_wins: int,
    base_size: float = 0.01,
    multiplier: float = 1.5,
    max_size: float = 0.1
) -> int:
    """
    Anti-Martingale (Paroli) position sizing.

    Increases position after wins, resets after loss.

    Args:
        capital: Current capital
        consecutive_wins: Number of consecutive wins
        base_size: Base position size
        multiplier: Size multiplier after win
        max_size: Maximum position size

    Returns:
        Number of contracts

    Example:
        >>> # After 2 wins, position = base * 1.5^2 = 2.25x base
        >>> contracts = anti_martingale(capital=10000, consecutive_wins=2)
    """
    # Calculate current size
    current_size = base_size * (multiplier ** consecutive_wins)

    # Cap at maximum
    current_size = min(current_size, max_size)

    position_value = capital * current_size
    return int(position_value)


def volatility_adjusted(
    capital: float,
    current_volatility: float,
    target_volatility: float = 0.15,
    base_size: float = 0.02,
    min_size: float = 0.005,
    max_size: float = 0.1
) -> int:
    """
    Adjust position size based on market volatility.

    Smaller positions in high volatility, larger in low volatility.

    Args:
        capital: Current capital
        current_volatility: Current market volatility
        target_volatility: Target volatility level
        base_size: Base position size
        min_size: Minimum position size
        max_size: Maximum position size

    Returns:
        Number of contracts

    Example:
        >>> contracts = volatility_adjusted(
        ...     capital=10000,
        ...     current_volatility=0.3,  # 30% volatility
        ...     target_volatility=0.15    # 15% target
        ... )
    """
    if current_volatility <= 0:
        return 0

    # Adjust size inversely to volatility
    volatility_adjustment = target_volatility / current_volatility
    adjusted_size = base_size * volatility_adjustment

    # Apply limits
    adjusted_size = max(min(adjusted_size, max_size), min_size)

    position_value = capital * adjusted_size
    return int(position_value)


def confidence_weighted(
    capital: float,
    confidence: float,
    max_size: float = 0.1,
    min_confidence: float = 0.5,
    confidence_power: float = 2.0
) -> int:
    """
    Size position based on confidence level.

    Higher confidence = larger position.

    Args:
        capital: Current capital
        confidence: Confidence level (0-1)
        max_size: Maximum position at full confidence
        min_confidence: Minimum confidence to trade
        confidence_power: Exponential scaling factor

    Returns:
        Number of contracts

    Example:
        >>> contracts = confidence_weighted(
        ...     capital=10000,
        ...     confidence=0.8  # 80% confidence
        ... )
    """
    if confidence < min_confidence:
        return 0

    # Scale position with confidence
    confidence_factor = (confidence - min_confidence) / (1 - min_confidence)
    confidence_factor = confidence_factor ** confidence_power

    position_size = max_size * confidence_factor
    position_value = capital * position_size

    return int(position_value)


def optimal_f(
    capital: float,
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    safety_factor: float = 0.5
) -> int:
    """
    Ralph Vince's Optimal F position sizing.

    Maximizes geometric growth rate based on historical performance.

    Args:
        capital: Current capital
        win_rate: Historical win rate (0-1)
        avg_win: Average winning trade profit
        avg_loss: Average losing trade loss (positive number)
        safety_factor: Fraction of optimal F to use

    Returns:
        Number of contracts

    Example:
        >>> contracts = optimal_f(
        ...     capital=10000,
        ...     win_rate=0.6,
        ...     avg_win=100,
        ...     avg_loss=80
        ... )
    """
    if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
        return 0

    # Calculate optimal F
    # f = (p * W - (1-p) * L) / W * L
    # Where p = win rate, W = avg win, L = avg loss
    numerator = win_rate * avg_win - (1 - win_rate) * avg_loss
    denominator = avg_win * avg_loss

    if denominator <= 0:
        return 0

    optimal_fraction = numerator / denominator

    # Apply safety factor
    safe_fraction = optimal_fraction * safety_factor

    # Ensure positive and reasonable
    safe_fraction = max(0, min(safe_fraction, 0.25))

    position_value = capital * safe_fraction
    return int(position_value)


def risk_parity(
    capital: float,
    positions: Dict[str, Dict],
    target_risk: float = 0.02,
    correlation_matrix: Optional[np.ndarray] = None
) -> Dict[str, int]:
    """
    Risk parity position sizing across multiple positions.

    Equalizes risk contribution from each position.

    Args:
        capital: Total capital
        positions: Dictionary of positions with volatilities
        target_risk: Target risk per position
        correlation_matrix: Correlation between positions

    Returns:
        Dictionary of position sizes

    Example:
        >>> positions = {
        ...     'KXNFLGAME-A': {'volatility': 0.2, 'price': 0.5},
        ...     'KXNFLGAME-B': {'volatility': 0.3, 'price': 0.6}
        ... }
        >>> sizes = risk_parity(capital=10000, positions=positions)
    """
    if not positions:
        return {}

    n_positions = len(positions)
    sizes = {}

    # Simple risk parity without correlations
    if correlation_matrix is None:
        total_inv_vol = sum(1 / p['volatility'] for p in positions.values())

        for ticker, pos_data in positions.items():
            # Weight inversely proportional to volatility
            weight = (1 / pos_data['volatility']) / total_inv_vol
            position_value = capital * weight * target_risk
            sizes[ticker] = int(position_value / pos_data.get('price', 1))
    else:
        # TODO: Implement with correlation matrix
        # Requires optimization to find weights where each position
        # contributes equally to portfolio risk
        pass

    return sizes


class PositionSizer:
    """
    Position sizing manager with state tracking.

    Tracks performance and adapts sizing methods.
    """

    def __init__(
        self,
        initial_capital: float,
        default_method: str = "kelly",
        track_performance: bool = True
    ):
        """
        Initialize position sizer.

        Args:
            initial_capital: Starting capital
            default_method: Default sizing method
            track_performance: Track and adapt to performance
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.default_method = default_method
        self.track_performance = track_performance

        # Performance tracking
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.total_profit = 0
        self.total_loss = 0

    def calculate_size(
        self,
        method: Optional[str] = None,
        **kwargs
    ) -> int:
        """
        Calculate position size using specified method.

        Args:
            method: Sizing method to use
            **kwargs: Method-specific parameters

        Returns:
            Number of contracts
        """
        method = method or self.default_method

        if method == "kelly":
            return int(self.current_capital * kelly_criterion(**kwargs))
        elif method == "fixed":
            return fixed_percentage(self.current_capital, **kwargs)
        elif method == "edge":
            return edge_proportional(capital=self.current_capital, **kwargs)
        elif method == "martingale":
            return martingale(
                self.current_capital,
                self.consecutive_losses,
                **kwargs
            )
        elif method == "anti_martingale":
            return anti_martingale(
                self.current_capital,
                self.consecutive_wins,
                **kwargs
            )
        elif method == "volatility":
            return volatility_adjusted(self.current_capital, **kwargs)
        elif method == "confidence":
            return confidence_weighted(self.current_capital, **kwargs)
        elif method == "optimal_f":
            # Calculate from tracked performance
            if self.winning_trades > 0 and self.total_trades > 0:
                win_rate = self.winning_trades / self.total_trades
                avg_win = self.total_profit / max(self.winning_trades, 1)
                avg_loss = self.total_loss / max(self.total_trades - self.winning_trades, 1)
                return optimal_f(
                    self.current_capital,
                    win_rate,
                    avg_win,
                    avg_loss,
                    **kwargs
                )
            else:
                # Fall back to fixed percentage
                return fixed_percentage(self.current_capital, **kwargs)
        else:
            raise ValueError(f"Unknown sizing method: {method}")

    def update_performance(self, pnl: float):
        """Update performance tracking after trade"""
        self.total_trades += 1

        if pnl > 0:
            self.winning_trades += 1
            self.total_profit += pnl
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.total_loss += abs(pnl)
            self.consecutive_losses += 1
            self.consecutive_wins = 0

        self.current_capital += pnl

    def get_stats(self) -> Dict:
        """Get current performance statistics"""
        return {
            'current_capital': self.current_capital,
            'total_return': (self.current_capital / self.initial_capital - 1) * 100,
            'total_trades': self.total_trades,
            'win_rate': self.winning_trades / max(self.total_trades, 1),
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses,
            'avg_win': self.total_profit / max(self.winning_trades, 1),
            'avg_loss': self.total_loss / max(self.total_trades - self.winning_trades, 1)
        }