"""
Risk Management Framework for Neural Trading

Provides comprehensive risk monitoring, stop-loss management, and automated risk controls
for real-time trading operations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

_LOG = logging.getLogger(__name__)


class StopLossType(Enum):
    """Types of stop-loss orders supported."""

    PERCENTAGE = "percentage"  # Stop at % loss from entry
    ABSOLUTE = "absolute"  # Stop at absolute price level
    TRAILING = "trailing"  # Trailing stop that follows price


class RiskEvent(Enum):
    """Types of risk events that can trigger actions."""

    STOP_LOSS_TRIGGERED = "stop_loss_triggered"
    MAX_DRAWDOWN_EXCEEDED = "max_drawdown_exceeded"
    DAILY_LOSS_LIMIT_EXCEEDED = "daily_loss_limit_exceeded"
    POSITION_SIZE_EXCEEDED = "position_size_exceeded"


@dataclass
class StopLossConfig:
    """Configuration for stop-loss orders."""

    type: StopLossType
    value: float  # Percentage (0-1), absolute price, or trailing amount
    enabled: bool = True


@dataclass
class RiskLimits:
    """Risk limits configuration."""

    max_drawdown_pct: float = 0.10  # 10% max drawdown
    max_position_size_pct: float = 0.05  # 5% of portfolio per position
    daily_loss_limit_pct: float = 0.05  # 5% daily loss limit
    max_positions: int = 10  # Maximum open positions


@dataclass
class Position:
    """Represents a trading position with risk management data."""

    market_id: str
    side: str  # "yes" or "no"
    quantity: int
    entry_price: float
    current_price: float = 0.0
    entry_time: float = 0.0
    stop_loss: StopLossConfig | None = None
    trailing_high: float = 0.0  # For trailing stops

    @property
    def current_value(self) -> float:
        """Calculate current position value."""
        return self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L."""
        if self.side == "yes":
            return self.quantity * (self.current_price - self.entry_price)
        else:  # "no"
            return self.quantity * (self.entry_price - self.current_price)

    @property
    def pnl_percentage(self) -> float:
        """Calculate P&L as percentage of entry value."""
        entry_value = self.quantity * self.entry_price
        if entry_value == 0:
            return 0.0
        return self.unrealized_pnl / entry_value


class RiskEventHandler(Protocol):
    """Protocol for handling risk events."""

    def on_risk_event(self, event: RiskEvent, position: Position, data: dict[str, Any]) -> None:
        """Handle a risk event."""
        ...


@dataclass
class RiskManager:
    """
    Core risk management system for monitoring positions and enforcing risk controls.

    Provides real-time risk monitoring, stop-loss management, and automated risk responses.
    """

    limits: RiskLimits = field(default_factory=RiskLimits)
    event_handler: RiskEventHandler | None = None

    # Runtime state
    positions: dict[str, Position] = field(default_factory=dict)
    daily_pnl: float = 0.0
    peak_portfolio_value: float = 0.0
    portfolio_value: float = 0.0

    def __post_init__(self) -> None:
        """Initialize risk manager state."""
        self.peak_portfolio_value = self.portfolio_value

    def add_position(self, position: Position) -> None:
        """Add a position to risk monitoring."""
        self.positions[position.market_id] = position
        _LOG.info(f"Added position monitoring: {position.market_id}, quantity: {position.quantity}")

    def remove_position(self, market_id: str) -> None:
        """Remove a position from risk monitoring."""
        if market_id in self.positions:
            position = self.positions.pop(market_id)
            self.daily_pnl += position.unrealized_pnl
            _LOG.info(
                f"Removed position monitoring: {market_id}, realized P&L: {position.unrealized_pnl}"
            )

    def update_position_price(self, market_id: str, current_price: float) -> list[RiskEvent]:
        """
        Update position price and check for risk events.

        Returns list of triggered risk events.
        """
        if market_id not in self.positions:
            return []

        position = self.positions[market_id]
        old_price = position.current_price
        position.current_price = current_price

        # Update trailing stop high if applicable
        if position.stop_loss and position.stop_loss.type == StopLossType.TRAILING:
            if current_price > position.trailing_high:
                position.trailing_high = current_price

        events = []

        # Check stop-loss conditions
        if self._check_stop_loss(position):
            events.append(RiskEvent.STOP_LOSS_TRIGGERED)

        # Check position size limit
        if self._check_position_size_limit(position):
            events.append(RiskEvent.POSITION_SIZE_EXCEEDED)

        # Update portfolio value and check drawdown
        self._update_portfolio_value()
        if self._check_drawdown_limit():
            events.append(RiskEvent.MAX_DRAWDOWN_EXCEEDED)

        # Check daily loss limit
        if self._check_daily_loss_limit():
            events.append(RiskEvent.DAILY_LOSS_LIMIT_EXCEEDED)

        # Notify event handler
        for event in events:
            if self.event_handler:
                self.event_handler.on_risk_event(
                    event,
                    position,
                    {
                        "old_price": old_price,
                        "new_price": current_price,
                        "portfolio_value": self.portfolio_value,
                        "daily_pnl": self.daily_pnl,
                    },
                )

        return events

    def _check_stop_loss(self, position: Position) -> bool:
        """Check if stop-loss should be triggered."""
        if not position.stop_loss or not position.stop_loss.enabled:
            return False

        stop_loss = position.stop_loss

        if stop_loss.type == StopLossType.PERCENTAGE:
            if position.pnl_percentage <= -stop_loss.value:
                _LOG.warning(
                    f"Stop-loss triggered for {position.market_id}: "
                    f"{position.pnl_percentage:.2%} <= -{stop_loss.value:.2%}"
                )
                return True

        elif stop_loss.type == StopLossType.ABSOLUTE:
            stop_price = stop_loss.value
            if position.side == "yes" and position.current_price <= stop_price:
                _LOG.warning(
                    f"Stop-loss triggered for {position.market_id}: "
                    f"price {position.current_price} <= {stop_price}"
                )
                return True
            elif position.side == "no" and position.current_price >= stop_price:
                _LOG.warning(
                    f"Stop-loss triggered for {position.market_id}: "
                    f"price {position.current_price} >= {stop_price}"
                )
                return True

        elif stop_loss.type == StopLossType.TRAILING:
            stop_price = position.trailing_high * (1 - stop_loss.value)
            if position.current_price <= stop_price:
                _LOG.warning(
                    f"Trailing stop-loss triggered for {position.market_id}: "
                    f"price {position.current_price} <= {stop_price}"
                )
                return True

        return False

    def _check_position_size_limit(self, position: Position) -> bool:
        """Check if position exceeds size limits."""
        if self.portfolio_value == 0:
            return False

        position_pct = position.current_value / self.portfolio_value
        if position_pct > self.limits.max_position_size_pct:
            _LOG.warning(
                f"Position size limit exceeded for {position.market_id}: "
                f"{position_pct:.2%} > {self.limits.max_position_size_pct:.2%}"
            )
            return True

        return False

    def _check_drawdown_limit(self) -> bool:
        """Check if portfolio drawdown exceeds limit."""
        if self.peak_portfolio_value == 0:
            return False

        drawdown = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value
        if drawdown > self.limits.max_drawdown_pct:
            _LOG.warning(
                f"Drawdown limit exceeded: {drawdown:.2%} > {self.limits.max_drawdown_pct:.2%}"
            )
            return True

        return False

    def _check_daily_loss_limit(self) -> bool:
        """Check if daily loss exceeds limit."""
        if self.portfolio_value == 0:
            return False

        daily_loss_pct = -self.daily_pnl / self.portfolio_value
        if daily_loss_pct > self.limits.daily_loss_limit_pct:
            _LOG.warning(
                f"Daily loss limit exceeded: {daily_loss_pct:.2%} > {self.limits.daily_loss_limit_pct:.2%}"
            )
            return True

        return False

    def _update_portfolio_value(self) -> None:
        """Update total portfolio value from positions."""
        total_value = 0.0
        for position in self.positions.values():
            total_value += position.current_value

        self.portfolio_value = total_value
        self.peak_portfolio_value = max(self.peak_portfolio_value, self.portfolio_value)

    def get_risk_metrics(self) -> dict[str, Any]:
        """Get current risk metrics."""
        total_pnl = sum(pos.unrealized_pnl for pos in self.positions.values()) + self.daily_pnl

        return {
            "portfolio_value": self.portfolio_value,
            "peak_portfolio_value": self.peak_portfolio_value,
            "drawdown_pct": (
                (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value
                if self.peak_portfolio_value > 0
                else 0
            ),
            "daily_pnl": self.daily_pnl,
            "total_pnl": total_pnl,
            "open_positions": len(self.positions),
            "risk_limits": {
                "max_drawdown_pct": self.limits.max_drawdown_pct,
                "max_position_size_pct": self.limits.max_position_size_pct,
                "daily_loss_limit_pct": self.limits.daily_loss_limit_pct,
                "max_positions": self.limits.max_positions,
            },
        }

    def reset_daily_pnl(self) -> None:
        """Reset daily P&L counter (call at start of new trading day)."""
        self.daily_pnl = 0.0
        _LOG.info("Daily P&L reset to 0.0")


@dataclass
class StopLossEngine:
    """
    Advanced stop-loss engine with multiple strategies and dynamic adjustments.

    Provides sophisticated stop-loss management including volatility-adjusted,
    time-based, and adaptive strategies.
    """

    def calculate_stop_price(
        self, entry_price: float, current_price: float, side: str, strategy: str = "fixed", **kwargs
    ) -> float:
        """
        Calculate stop price using specified strategy.

        Args:
            entry_price: Position entry price
            current_price: Current market price
            side: "yes" or "no"
            strategy: Stop-loss strategy to use
            **kwargs: Strategy-specific parameters

        Returns:
            Stop price level
        """
        if strategy == "fixed":
            return self._fixed_stop(entry_price, side, **kwargs)
        elif strategy == "trailing":
            return self._trailing_stop(entry_price, current_price, side, **kwargs)
        elif strategy == "volatility":
            return self._volatility_adjusted_stop(entry_price, current_price, side, **kwargs)
        elif strategy == "time_based":
            return self._time_based_stop(entry_price, current_price, side, **kwargs)
        else:
            raise ValueError(f"Unknown stop-loss strategy: {strategy}")

    def _fixed_stop(self, entry_price: float, side: str, stop_pct: float = 0.05) -> float:
        """Fixed percentage stop-loss."""
        if side == "yes":
            return entry_price * (1 - stop_pct)
        else:  # "no"
            return entry_price * (1 + stop_pct)

    def _trailing_stop(
        self, entry_price: float, current_price: float, side: str, trail_pct: float = 0.03
    ) -> float:
        """Trailing stop-loss that follows favorable price movement."""
        if side == "yes":
            # For long positions, trail below the highest price
            trail_amount = current_price * trail_pct
            return current_price - trail_amount
        else:  # "no"
            # For short positions, trail above the lowest price
            trail_amount = current_price * trail_pct
            return current_price + trail_amount

    def _volatility_adjusted_stop(
        self,
        entry_price: float,
        current_price: float,
        side: str,
        volatility: float = 0.02,
        multiplier: float = 2.0,
    ) -> float:
        """Stop-loss adjusted for market volatility."""
        # Use volatility as base, multiply for safety
        stop_distance = volatility * multiplier

        if side == "yes":
            return current_price * (1 - stop_distance)
        else:
            return current_price * (1 + stop_distance)

    def _time_based_stop(
        self,
        entry_price: float,
        current_price: float,
        side: str,
        time_held: float,
        max_time: float = 86400,  # 24 hours in seconds
        time_factor: float = 0.1,
    ) -> float:
        """Time-based stop that widens as position ages."""
        # Stop gets wider as time passes
        time_ratio = min(time_held / max_time, 1.0)
        time_adjustment = time_factor * time_ratio

        if side == "yes":
            return entry_price * (1 - time_adjustment)
        else:
            return entry_price * (1 + time_adjustment)

    def should_exit_position(
        self, position: Position, current_time: float, market_volatility: float = 0.0
    ) -> tuple[bool, str]:
        """
        Determine if position should be exited based on stop-loss conditions.

        Returns:
            (should_exit, reason)
        """
        if not position.stop_loss or not position.stop_loss.enabled:
            return False, ""

        stop_price = self.calculate_stop_price(
            position.entry_price,
            position.current_price,
            position.side,
            strategy=self._map_stop_type_to_strategy(position.stop_loss.type),
            stop_pct=(
                position.stop_loss.value
                if position.stop_loss.type == StopLossType.PERCENTAGE
                else 0.05
            ),
            trail_pct=(
                position.stop_loss.value
                if position.stop_loss.type == StopLossType.TRAILING
                else 0.03
            ),
            volatility=market_volatility,
            time_held=current_time - position.entry_time,
        )

        # Check if stop price is breached
        if position.side == "yes" and position.current_price <= stop_price:
            return True, f"Stop-loss triggered at {stop_price:.4f}"
        elif position.side == "no" and position.current_price >= stop_price:
            return True, f"Stop-loss triggered at {stop_price:.4f}"

        return False, ""

    def _map_stop_type_to_strategy(self, stop_type: StopLossType) -> str:
        """Map StopLossType to strategy name."""
        mapping = {
            StopLossType.PERCENTAGE: "fixed",
            StopLossType.ABSOLUTE: "fixed",  # Absolute is handled separately
            StopLossType.TRAILING: "trailing",
        }
        return mapping.get(stop_type, "fixed")
