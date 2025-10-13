"""
Base Strategy Class for Neural Analysis Stack

Provides the foundation for all trading strategies with built-in risk management,
backtesting support, and seamless integration with the trading stack.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class StrategyConfig:
    """Configuration for strategy parameters"""

    max_position_size: float = 0.1  # 10% of capital default
    min_edge: float = 0.03  # 3% minimum edge
    use_kelly: bool = False
    kelly_fraction: float = 0.25
    stop_loss: float | None = None
    take_profit: float | None = None
    max_positions: int = 10
    fee_rate: float = 0.0


class SignalType(Enum):
    """Trading signal types"""

    BUY_YES = "buy_yes"
    BUY_NO = "buy_no"
    SELL_YES = "sell_yes"
    SELL_NO = "sell_no"
    HOLD = "hold"
    CLOSE = "close"


@dataclass
class Signal:
    """Trading signal with metadata"""

    signal_type: SignalType  # Changed from 'type' for clarity
    market_id: str  # Market identifier (ticker)
    recommended_size: float  # Position size as a fraction
    confidence: float
    edge: float | None = None
    expected_value: float | None = None
    max_contracts: int | None = None
    stop_loss_price: float | None = None
    take_profit_price: float | None = None
    metadata: dict[str, Any] | None = None
    timestamp: datetime | None = None

    # Backward compatibility properties
    @property
    def type(self) -> SignalType:
        return self.signal_type

    @property
    def ticker(self) -> str:
        return self.market_id

    @property
    def size(self) -> float:
        return self.recommended_size

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


# Define Strategy class first (needed by BaseStrategy)
class Strategy(ABC):
    """
    Base class for all trading strategies.

    Subclasses must implement the analyze() method.
    """

    def __init__(
        self,
        name: str | None = None,
        initial_capital: float = 1000.0,
        max_position_size: float = 0.1,  # 10% of capital
        min_edge: float = 0.03,  # 3% minimum edge
        use_kelly: bool = False,
        kelly_fraction: float = 0.25,  # Conservative Kelly
        stop_loss: float | None = None,
        take_profit: float | None = None,
        max_positions: int = 10,
        fee_rate: float = 0.0,
    ):
        """
        Initialize strategy with risk parameters.

        Args:
            name: Strategy name
            initial_capital: Starting capital
            max_position_size: Maximum position as fraction of capital
            min_edge: Minimum edge required to trade
            use_kelly: Use Kelly Criterion for sizing
            kelly_fraction: Fraction of Kelly to use (safety)
            stop_loss: Stop loss percentage (e.g., 0.5 = 50%)
            take_profit: Take profit percentage (e.g., 2.0 = 200%)
            max_positions: Maximum concurrent positions
            fee_rate: Fee rate for calculations
        """
        self.name = name or self.__class__.__name__
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.capital = initial_capital  # Alias for compatibility
        self.max_position_size = max_position_size
        self.min_edge = min_edge
        self.use_kelly = use_kelly
        self.kelly_fraction = kelly_fraction
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_positions = max_positions
        self.fee_rate = fee_rate

        # State tracking
        self.positions: list[Position] = []
        self.closed_positions: list[Position] = []
        self.signals: list[Signal] = []
        self.trade_history: list[dict[str, Any]] = []

        # ESPN data integration
        self.espn_data: dict | None = None
        self.use_espn = False

        # Sportsbook consensus
        self.sportsbook_data: dict | None = None

    @abstractmethod
    def analyze(self, market_data: pd.DataFrame, espn_data: dict | None = None, **kwargs) -> Signal:
        """
        Analyze market and generate trading signal.

        Args:
            market_data: DataFrame with market prices and metrics
            espn_data: Optional ESPN play-by-play data
            **kwargs: Additional strategy-specific parameters

        Returns:
            Signal object with trade decision
        """
        pass

    def calculate_position_size(self, edge: float, odds: float, confidence: float = 1.0) -> int:
        """
        Calculate optimal position size based on edge and risk parameters.

        Args:
            edge: Expected edge (your_prob - market_prob)
            odds: Potential payout odds
            confidence: Confidence level (0-1)

        Returns:
            Position size in number of contracts
        """
        if edge <= self.min_edge:
            return 0

        available_capital = self.current_capital * (1 - self.get_exposure_ratio())

        if self.use_kelly:
            # Kelly Criterion with safety factor
            kelly_fraction = (edge * odds - (1 - edge)) / odds
            kelly_fraction = min(kelly_fraction, self.kelly_fraction)
            position_value = available_capital * kelly_fraction * confidence
        else:
            # Fixed percentage based on edge strength
            edge_multiplier = min(edge / self.min_edge, 3.0)  # Cap at 3x
            position_value = (
                available_capital * self.max_position_size * confidence * edge_multiplier
            )

        # Convert to number of contracts (assuming $1 per contract)
        contracts = int(position_value)

        # Apply maximum position size limit
        max_contracts = int(available_capital * self.max_position_size)
        return min(contracts, max_contracts)

    def calculate_edge(
        self, true_probability: float, market_price: float, confidence: float = 1.0
    ) -> float:
        """
        Calculate trading edge.

        Args:
            true_probability: Your estimated true probability
            market_price: Current market price (implied probability)
            confidence: Confidence in your estimate

        Returns:
            Adjusted edge
        """
        raw_edge = true_probability - market_price
        return raw_edge * confidence

    def calculate_fees(self, price: float, size: int) -> float:
        """
        Calculate Kalshi fees for a trade.

        Args:
            price: Contract price
            size: Number of contracts

        Returns:
            Total fees
        """
        if self.fee_rate > 0:
            return self.fee_rate * size
        # Kalshi fee formula: 0.07 × P × (1 - P) × contracts
        return 0.07 * price * (1 - price) * size

    def should_close_position(self, position: "Position") -> bool:
        """
        Check if position should be closed based on stop loss or take profit.

        Args:
            position: Current position

        Returns:
            True if position should be closed
        """
        if not position:
            return False

        pnl_pct = position.pnl_percentage / 100

        # Check stop loss
        if self.stop_loss and pnl_pct <= -self.stop_loss:
            return True

        # Check take profit
        if self.take_profit and pnl_pct >= self.take_profit:
            return True

        return False

    def get_exposure_ratio(self) -> float:
        """
        Calculate current exposure as percentage of capital.

        Returns:
            Exposure ratio (0-1)
        """
        if not self.positions:
            return 0.0

        total_exposure = sum(pos.size * pos.entry_price for pos in self.positions)
        return total_exposure / self.current_capital

    def can_open_position(self) -> bool:
        """
        Check if we can open a new position based on risk limits.

        Returns:
            True if new position is allowed
        """
        # Check max positions limit
        if len(self.positions) >= self.max_positions:
            return False

        # Check exposure limit
        if self.get_exposure_ratio() >= 0.8:  # 80% max exposure
            return False

        return True

    def buy_yes(
        self, ticker: str, size: int | None = None, confidence: float = 1.0, **kwargs
    ) -> Signal:
        """Generate BUY_YES signal"""
        return Signal(
            signal_type=SignalType.BUY_YES,
            market_id=ticker,
            recommended_size=(size or 100) / 1000.0,  # Convert to fraction
            confidence=confidence,
            metadata=kwargs,
        )

    def buy_no(
        self, ticker: str, size: int | None = None, confidence: float = 1.0, **kwargs
    ) -> Signal:
        """Generate BUY_NO signal"""
        return Signal(
            signal_type=SignalType.BUY_NO,
            market_id=ticker,
            recommended_size=(size or 100) / 1000.0,  # Convert to fraction
            confidence=confidence,
            metadata=kwargs,
        )

    def hold(self, ticker: str = "") -> Signal:
        """Generate HOLD signal"""
        return Signal(
            signal_type=SignalType.HOLD, market_id=ticker, recommended_size=0, confidence=0.0
        )

    def close(self, ticker: str, **kwargs) -> Signal:
        """Generate CLOSE signal"""
        return Signal(
            signal_type=SignalType.CLOSE,
            market_id=ticker,
            recommended_size=0,
            confidence=1.0,
            metadata=kwargs,
        )

    def set_espn_data(self, data: dict):
        """Set ESPN play-by-play data"""
        self.espn_data = data
        self.use_espn = True

    def set_sportsbook_consensus(self, data: dict):
        """Set sportsbook consensus data"""
        self.sportsbook_data = data

    def get_sportsbook_consensus(self, event: str) -> float | None:
        """Get consensus probability from sportsbooks"""
        if self.sportsbook_data and event in self.sportsbook_data:
            value = self.sportsbook_data[event]
            if isinstance(value, (int, float)):
                return float(value)
        return None

    def kelly_size(self, edge: float, odds: float) -> int:
        """Calculate Kelly Criterion position size"""
        if edge <= 0:
            return 0

        kelly_fraction = (edge * odds - (1 - edge)) / odds
        kelly_fraction = min(kelly_fraction, self.kelly_fraction)
        position_value = self.current_capital * kelly_fraction

        return int(position_value)

    def update_capital(self, pnl: float):
        """Update current capital after trade"""
        self.current_capital += pnl

    def get_performance_metrics(self) -> dict[str, float]:
        """
        Calculate performance metrics.

        Returns:
            Dictionary of performance metrics
        """
        if not self.trade_history:
            return {}

        trades = pd.DataFrame(self.trade_history)
        returns = np.array(trades["pnl"].values)

        metrics = {
            "total_trades": len(trades),
            "win_rate": len(trades[trades["pnl"] > 0]) / len(trades) if len(trades) > 0 else 0,
            "total_pnl": float(returns.sum()),
            "avg_pnl": float(returns.mean()),
            "total_return": (self.current_capital / self.initial_capital - 1) * 100,
            "max_win": float(returns.max()),
            "max_loss": float(returns.min()),
            "sharpe_ratio": (
                float(returns.mean() / returns.std())
                if len(returns) > 1 and returns.std() > 0
                else 0
            ),
            "max_drawdown": self._calculate_max_drawdown(returns),
        }

        return metrics

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        return abs(drawdown.min()) if len(drawdown) > 0 else 0

    def reset(self):
        """Reset strategy state"""
        self.current_capital = self.initial_capital
        self.positions = []
        self.closed_positions = []
        self.signals = []
        self.trade_history = []

    def __str__(self) -> str:
        return (
            f"{self.name} (Capital: ${self.current_capital:.2f}, Positions: {len(self.positions)})"
        )


@dataclass
class Position:
    """Represents a trading position"""

    ticker: str
    side: str  # "yes" or "no"
    size: int
    entry_price: float
    current_price: float
    entry_time: datetime
    metadata: dict[str, Any] | None = None

    @property
    def pnl(self) -> float:
        """Calculate current P&L"""
        if self.side == "yes":
            return (self.current_price - self.entry_price) * self.size
        else:
            return (self.entry_price - self.current_price) * self.size

    @property
    def pnl_percentage(self) -> float:
        """Calculate P&L percentage"""
        return (self.pnl / (self.entry_price * self.size)) * 100


class BaseStrategy(Strategy):
    """
    Modern base class using StrategyConfig.
    This is an alias/wrapper for compatibility.
    """

    def __init__(self, name: str | None = None, config: StrategyConfig | None = None):
        """Initialize with StrategyConfig"""
        if config is None:
            config = StrategyConfig()

        # Call the parent Strategy init with config values
        super().__init__(
            name=name,
            max_position_size=config.max_position_size,
            min_edge=config.min_edge,
            use_kelly=config.use_kelly,
            kelly_fraction=config.kelly_fraction,
            stop_loss=config.stop_loss,
            take_profit=config.take_profit,
            max_positions=config.max_positions,
            fee_rate=config.fee_rate,
        )
        self.config = config
