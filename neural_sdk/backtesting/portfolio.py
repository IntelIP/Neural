"""
Portfolio Management for Backtesting

Realistic portfolio simulation with:
- Position tracking
- Cash management
- Trade execution
- P&L calculation
- Risk metrics
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents a position in a specific market."""

    symbol: str
    quantity: int
    avg_cost: float
    current_price: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def market_value(self) -> float:
        """Current market value of position."""
        return self.quantity * self.current_price

    @property
    def cost_basis(self) -> float:
        """Total cost basis of position."""
        return self.quantity * self.avg_cost

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss."""
        return self.market_value - self.cost_basis

    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized P&L as percentage."""
        if self.cost_basis == 0:
            return 0.0
        return (self.unrealized_pnl / self.cost_basis) * 100

    def update_price(self, new_price: float):
        """Update current market price."""
        self.current_price = new_price


@dataclass
class Trade:
    """Represents a completed trade."""

    timestamp: datetime
    symbol: str
    action: str  # BUY, SELL
    quantity: int
    price: float
    commission: float = 0.0
    value: float = field(init=False)
    pnl: Optional[float] = None  # Realized P&L for closing trades

    def __post_init__(self):
        """Calculate trade value."""
        self.value = self.quantity * self.price

    @property
    def net_value(self) -> float:
        """Trade value after commission."""
        return self.value - self.commission


class Portfolio:
    """
    Portfolio manager for backtesting.

    Tracks cash, positions, trades, and performance metrics.
    Implements realistic order execution with commissions and slippage.
    """

    def __init__(self, initial_capital: float):
        """
        Initialize portfolio.

        Args:
            initial_capital: Starting cash amount
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Trade] = []

        # Performance tracking
        self._daily_values = []
        self._high_water_mark = initial_capital

        logger.info(f"Initialized portfolio with ${initial_capital:,.2f}")

    @property
    def market_value(self) -> float:
        """Total market value of all positions."""
        return sum(pos.market_value for pos in self.positions.values())

    @property
    def total_value(self, market_data: Optional[Dict[str, Any]] = None) -> float:
        """Total portfolio value (cash + positions)."""
        if market_data and "prices" in market_data:
            # Update position prices from market data
            for symbol, price in market_data["prices"].items():
                if symbol in self.positions:
                    self.positions[symbol].update_price(price)

        return self.cash + self.market_value

    @property
    def unrealized_pnl(self) -> float:
        """Total unrealized profit/loss."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())

    @property
    def realized_pnl(self) -> float:
        """Total realized profit/loss from closed trades."""
        return sum(
            trade.pnl or 0 for trade in self.trade_history if trade.pnl is not None
        )

    @property
    def total_pnl(self) -> float:
        """Total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl

    @property
    def total_return_pct(self) -> float:
        """Total return as percentage of initial capital."""
        return (self.total_pnl / self.initial_capital) * 100

    @property
    def buying_power(self) -> float:
        """Available cash for new positions."""
        return self.cash

    def buy(
        self,
        symbol: str,
        quantity: int,
        price: float,
        timestamp: datetime,
        commission: float = 0.0,
    ) -> bool:
        """
        Execute buy order.

        Args:
            symbol: Market symbol
            quantity: Number of contracts
            price: Execution price
            timestamp: Trade timestamp
            commission: Commission fee

        Returns:
            True if trade executed successfully
        """
        total_cost = quantity * price + commission

        # Check if sufficient cash
        if total_cost > self.cash:
            logger.warning(
                f"Insufficient cash for {symbol}: need ${total_cost:,.2f}, have ${self.cash:,.2f}"
            )
            return False

        # Update cash
        self.cash -= total_cost

        # Update or create position
        if symbol in self.positions:
            pos = self.positions[symbol]
            # Calculate new average cost
            total_quantity = pos.quantity + quantity
            total_cost_basis = pos.cost_basis + (quantity * price)
            new_avg_cost = total_cost_basis / total_quantity

            pos.quantity = total_quantity
            pos.avg_cost = new_avg_cost
            pos.current_price = price
            pos.timestamp = timestamp
        else:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                avg_cost=price,
                current_price=price,
                timestamp=timestamp,
            )

        # Record trade
        trade = Trade(
            timestamp=timestamp,
            symbol=symbol,
            action="BUY",
            quantity=quantity,
            price=price,
            commission=commission,
        )
        self.trade_history.append(trade)

        logger.debug(
            f"BUY: {quantity} {symbol} @ ${price:.2f} (commission: ${commission:.2f})"
        )
        return True

    def sell(
        self,
        symbol: str,
        quantity: int,
        price: float,
        timestamp: datetime,
        commission: float = 0.0,
    ) -> bool:
        """
        Execute sell order.

        Args:
            symbol: Market symbol
            quantity: Number of contracts to sell
            price: Execution price
            timestamp: Trade timestamp
            commission: Commission fee

        Returns:
            True if trade executed successfully
        """
        # Check if position exists and has sufficient quantity
        if symbol not in self.positions:
            logger.warning(f"Cannot sell {symbol}: no position")
            return False

        pos = self.positions[symbol]
        if pos.quantity < quantity:
            logger.warning(f"Cannot sell {quantity} {symbol}: only have {pos.quantity}")
            return False

        # Calculate proceeds and P&L
        gross_proceeds = quantity * price
        net_proceeds = gross_proceeds - commission
        cost_basis = quantity * pos.avg_cost
        realized_pnl = net_proceeds - cost_basis

        # Update cash
        self.cash += net_proceeds

        # Update position
        pos.quantity -= quantity

        # Remove position if fully closed
        if pos.quantity == 0:
            del self.positions[symbol]
        else:
            pos.current_price = price
            pos.timestamp = timestamp

        # Record trade with realized P&L
        trade = Trade(
            timestamp=timestamp,
            symbol=symbol,
            action="SELL",
            quantity=quantity,
            price=price,
            commission=commission,
            pnl=realized_pnl,
        )
        self.trade_history.append(trade)

        logger.debug(
            f"SELL: {quantity} {symbol} @ ${price:.2f} (P&L: ${realized_pnl:+.2f})"
        )
        return True

    def close_position(
        self, symbol: str, price: float, timestamp: datetime, commission: float = 0.0
    ) -> bool:
        """
        Close entire position in symbol.

        Args:
            symbol: Market symbol
            price: Execution price
            timestamp: Trade timestamp
            commission: Commission fee

        Returns:
            True if position closed successfully
        """
        if symbol not in self.positions:
            logger.warning(f"No position to close in {symbol}")
            return False

        quantity = self.positions[symbol].quantity
        return self.sell(symbol, quantity, price, timestamp, commission)

    def close_all_positions(
        self,
        market_data: Dict[str, Any],
        timestamp: datetime,
        commission_rate: float = 0.02,
    ):
        """
        Close all open positions at market prices.

        Args:
            market_data: Current market prices
            timestamp: Trade timestamp
            commission_rate: Commission rate (default 2%)
        """
        positions_to_close = list(self.positions.keys())

        for symbol in positions_to_close:
            if symbol in market_data.get("prices", {}):
                price = market_data["prices"][symbol]
                quantity = self.positions[symbol].quantity
                commission = quantity * price * commission_rate

                self.close_position(symbol, price, timestamp, commission)

        logger.info(f"Closed {len(positions_to_close)} positions")

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol."""
        return self.positions.get(symbol)

    def get_position_size(self, symbol: str) -> int:
        """Get position size for symbol."""
        pos = self.positions.get(symbol)
        return pos.quantity if pos else 0

    def is_long(self, symbol: str) -> bool:
        """Check if long position exists."""
        pos = self.positions.get(symbol)
        return pos is not None and pos.quantity > 0

    def positions_df(self) -> pd.DataFrame:
        """Get positions as DataFrame."""
        if not self.positions:
            return pd.DataFrame()

        positions_data = []
        for pos in self.positions.values():
            positions_data.append(
                {
                    "symbol": pos.symbol,
                    "quantity": pos.quantity,
                    "avg_cost": pos.avg_cost,
                    "current_price": pos.current_price,
                    "market_value": pos.market_value,
                    "cost_basis": pos.cost_basis,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "unrealized_pnl_pct": pos.unrealized_pnl_pct,
                    "timestamp": pos.timestamp,
                }
            )

        return pd.DataFrame(positions_data)

    def trades_df(self) -> pd.DataFrame:
        """Get trade history as DataFrame."""
        if not self.trade_history:
            return pd.DataFrame()

        trades_data = []
        for trade in self.trade_history:
            trades_data.append(
                {
                    "timestamp": trade.timestamp,
                    "symbol": trade.symbol,
                    "action": trade.action,
                    "quantity": trade.quantity,
                    "price": trade.price,
                    "value": trade.value,
                    "commission": trade.commission,
                    "net_value": trade.net_value,
                    "realized_pnl": trade.pnl,
                }
            )

        return pd.DataFrame(trades_data)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        return {
            "initial_capital": self.initial_capital,
            "current_cash": self.cash,
            "market_value": self.market_value,
            "total_value": self.total_value(),
            "total_return": self.total_value() - self.initial_capital,
            "total_return_pct": self.total_return_pct,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "total_pnl": self.total_pnl,
            "num_positions": len(self.positions),
            "num_trades": len(self.trade_history),
            "positions": (
                self.positions_df().to_dict("records") if self.positions else []
            ),
        }

    def calculate_drawdown(self, portfolio_values: pd.Series) -> pd.Series:
        """
        Calculate drawdown series.

        Args:
            portfolio_values: Series of portfolio values over time

        Returns:
            Series of drawdown values (negative percentages)
        """
        rolling_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - rolling_max) / rolling_max
        return drawdown

    def calculate_max_drawdown(self, portfolio_values: pd.Series) -> float:
        """Calculate maximum drawdown percentage."""
        drawdown = self.calculate_drawdown(portfolio_values)
        return drawdown.min() * 100  # Convert to percentage

    def get_trade_statistics(self) -> Dict[str, Any]:
        """Get detailed trade statistics."""
        if not self.trade_history:
            return {}

        trades_df = self.trades_df()

        # Separate buy and sell trades
        buys = trades_df[trades_df["action"] == "BUY"]
        sells = trades_df[trades_df["action"] == "SELL"]

        # Realized P&L from sell trades
        realized_trades = sells[sells["realized_pnl"].notna()]

        if len(realized_trades) == 0:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            largest_win = 0
            largest_loss = 0
        else:
            winning_trades = realized_trades[realized_trades["realized_pnl"] > 0]
            losing_trades = realized_trades[realized_trades["realized_pnl"] < 0]

            win_rate = len(winning_trades) / len(realized_trades) * 100
            avg_win = (
                winning_trades["realized_pnl"].mean() if len(winning_trades) > 0 else 0
            )
            avg_loss = (
                losing_trades["realized_pnl"].mean() if len(losing_trades) > 0 else 0
            )
            largest_win = (
                winning_trades["realized_pnl"].max() if len(winning_trades) > 0 else 0
            )
            largest_loss = (
                losing_trades["realized_pnl"].min() if len(losing_trades) > 0 else 0
            )

        return {
            "total_trades": len(self.trade_history),
            "buy_trades": len(buys),
            "sell_trades": len(sells),
            "realized_trades": len(realized_trades),
            "win_rate": win_rate,
            "winning_trades": (
                len(realized_trades[realized_trades["realized_pnl"] > 0])
                if len(realized_trades) > 0
                else 0
            ),
            "losing_trades": (
                len(realized_trades[realized_trades["realized_pnl"] < 0])
                if len(realized_trades) > 0
                else 0
            ),
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "largest_win": largest_win,
            "largest_loss": largest_loss,
            "total_commissions": trades_df["commission"].sum(),
        }
