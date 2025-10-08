"""
Paper Trading Portfolio Management

Realistic portfolio simulation for paper trading with:
- Position tracking and management
- Cash balance management
- Trade execution and P&L calculation
- Performance metrics and risk analysis
- Trade history and reporting
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents a position in a specific market."""

    symbol: str
    market_id: str
    quantity: int
    avg_cost: float
    current_price: float = 0.0
    side: str = "long"  # "long" or "short"
    timestamp: datetime = field(default_factory=datetime.now)
    market_name: Optional[str] = None

    @property
    def market_value(self) -> float:
        """Current market value of position."""
        return abs(self.quantity) * self.current_price

    @property
    def cost_basis(self) -> float:
        """Total cost basis of position."""
        return abs(self.quantity) * self.avg_cost

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss."""
        if self.side == "long":
            return (self.current_price - self.avg_cost) * self.quantity
        else:  # short position
            return (self.avg_cost - self.current_price) * abs(self.quantity)

    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized P&L as percentage."""
        if self.cost_basis == 0:
            return 0.0
        return (self.unrealized_pnl / self.cost_basis) * 100

    def update_price(self, new_price: float) -> None:
        """Update current market price."""
        self.current_price = new_price

    def add_quantity(self, quantity: int, price: float) -> None:
        """Add to position (averaging down/up)."""
        if self.quantity == 0:
            self.avg_cost = price
            self.quantity = quantity
        else:
            total_cost = (self.cost_basis + abs(quantity) * price)
            total_quantity = abs(self.quantity) + abs(quantity)
            self.avg_cost = total_cost / total_quantity
            self.quantity += quantity

    def reduce_quantity(self, quantity: int) -> float:
        """Reduce position and return realized P&L."""
        if abs(quantity) > abs(self.quantity):
            raise ValueError("Cannot reduce position by more than current quantity")

        # Calculate realized P&L for the portion being closed
        if self.side == "long":
            realized_pnl = (self.current_price - self.avg_cost) * abs(quantity)
        else:
            realized_pnl = (self.avg_cost - self.current_price) * abs(quantity)

        self.quantity -= quantity
        return realized_pnl


@dataclass
class Trade:
    """Represents a completed trade."""

    timestamp: datetime
    market_id: str
    symbol: str
    market_name: str
    action: str  # "BUY", "SELL"
    side: str   # "yes", "no"
    quantity: int
    price: float
    commission: float = 0.0
    slippage: float = 0.0
    value: float = field(init=False)
    realized_pnl: Optional[float] = None
    sentiment_score: Optional[float] = None
    confidence: Optional[float] = None
    strategy: Optional[str] = None

    def __post_init__(self):
        """Calculate trade value."""
        self.value = abs(self.quantity) * self.price

    @property
    def net_value(self) -> float:
        """Trade value after commission and slippage."""
        return self.value - self.commission - self.slippage

    @property
    def total_cost(self) -> float:
        """Total cost including fees."""
        return self.value + self.commission + self.slippage


class PaperPortfolio:
    """
    Paper trading portfolio manager.

    Tracks cash, positions, trades, and performance metrics.
    Implements realistic order execution with commissions and slippage.
    """

    def __init__(
        self,
        initial_capital: float,
        commission_per_trade: float = 0.50,
        default_slippage_pct: float = 0.002
    ):
        """
        Initialize paper trading portfolio.

        Args:
            initial_capital: Starting cash amount
            commission_per_trade: Fixed commission per trade
            default_slippage_pct: Default slippage percentage
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission_per_trade = commission_per_trade
        self.default_slippage_pct = default_slippage_pct

        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Trade] = []
        self.daily_portfolio_values: List[Tuple[datetime, float]] = []

        # Performance tracking
        self.total_commission_paid = 0.0
        self.total_slippage_paid = 0.0
        self.max_portfolio_value = initial_capital
        self.max_drawdown = 0.0

        # Add initial portfolio value point
        self.daily_portfolio_values.append((datetime.now(), initial_capital))

    @property
    def total_position_value(self) -> float:
        """Total market value of all positions."""
        return sum(pos.market_value for pos in self.positions.values())

    @property
    def total_portfolio_value(self) -> float:
        """Total portfolio value (cash + positions)."""
        return self.cash + self.total_position_value

    @property
    def unrealized_pnl(self) -> float:
        """Total unrealized P&L across all positions."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())

    @property
    def realized_pnl(self) -> float:
        """Total realized P&L from completed trades."""
        return sum(trade.realized_pnl or 0.0 for trade in self.trade_history)

    @property
    def total_pnl(self) -> float:
        """Total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl

    @property
    def total_return_pct(self) -> float:
        """Total return percentage."""
        if self.initial_capital == 0:
            return 0.0
        return ((self.total_portfolio_value - self.initial_capital) / self.initial_capital) * 100

    @property
    def position_count(self) -> int:
        """Number of open positions."""
        return len([pos for pos in self.positions.values() if pos.quantity != 0])

    def update_position_price(self, symbol: str, new_price: float) -> None:
        """Update the current price for a position."""
        if symbol in self.positions:
            self.positions[symbol].update_price(new_price)
            self._update_max_drawdown()

    def update_all_position_prices(self, price_updates: Dict[str, float]) -> None:
        """Update prices for multiple positions."""
        for symbol, price in price_updates.items():
            self.update_position_price(symbol, price)

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol."""
        return self.positions.get(symbol)

    def has_position(self, symbol: str) -> bool:
        """Check if portfolio has a position in symbol."""
        position = self.positions.get(symbol)
        return position is not None and position.quantity != 0

    def can_afford_trade(self, quantity: int, price: float, commission: float = None) -> bool:
        """Check if portfolio has enough cash for a trade."""
        if commission is None:
            commission = self.commission_per_trade

        trade_cost = abs(quantity) * price + commission
        slippage = trade_cost * self.default_slippage_pct
        total_cost = trade_cost + slippage

        return self.cash >= total_cost

    def execute_trade(
        self,
        market_id: str,
        symbol: str,
        market_name: str,
        action: str,
        side: str,
        quantity: int,
        price: float,
        sentiment_score: Optional[float] = None,
        confidence: Optional[float] = None,
        strategy: Optional[str] = None
    ) -> Tuple[bool, str, Optional[Trade]]:
        """
        Execute a paper trade.

        Args:
            market_id: Market identifier
            symbol: Trading symbol
            market_name: Human readable market name
            action: "BUY" or "SELL"
            side: "yes" or "no"
            quantity: Number of contracts
            price: Price per contract
            sentiment_score: Sentiment score that triggered trade
            confidence: Confidence in the signal
            strategy: Strategy name

        Returns:
            Tuple of (success, message, trade_object)
        """
        try:
            # Calculate costs
            commission = self.commission_per_trade
            trade_value = abs(quantity) * price
            slippage = trade_value * self.default_slippage_pct

            # For buying, check if we have enough cash
            if action == "BUY":
                total_cost = trade_value + commission + slippage
                if self.cash < total_cost:
                    return False, f"Insufficient cash: ${self.cash:.2f} < ${total_cost:.2f}", None

                # Deduct cash
                self.cash -= total_cost

                # Add or update position
                if symbol not in self.positions:
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        market_id=market_id,
                        quantity=0,
                        avg_cost=0.0,
                        current_price=price,
                        side="long",
                        market_name=market_name
                    )

                self.positions[symbol].add_quantity(quantity, price)
                realized_pnl = None

            else:  # SELL
                # Check if we have the position to sell
                if symbol not in self.positions or self.positions[symbol].quantity < quantity:
                    available = self.positions.get(symbol, Position("", "", 0, 0.0)).quantity
                    return False, f"Insufficient position: {available} < {quantity}", None

                # Calculate realized P&L and reduce position
                realized_pnl = self.positions[symbol].reduce_quantity(quantity)

                # Add cash from sale (minus costs)
                net_proceeds = trade_value - commission - slippage
                self.cash += net_proceeds

                # Remove position if quantity is now zero
                if self.positions[symbol].quantity == 0:
                    del self.positions[symbol]

            # Create trade record
            trade = Trade(
                timestamp=datetime.now(),
                market_id=market_id,
                symbol=symbol,
                market_name=market_name,
                action=action,
                side=side,
                quantity=quantity,
                price=price,
                commission=commission,
                slippage=slippage,
                realized_pnl=realized_pnl,
                sentiment_score=sentiment_score,
                confidence=confidence,
                strategy=strategy
            )

            self.trade_history.append(trade)
            self.total_commission_paid += commission
            self.total_slippage_paid += slippage

            # Update performance tracking
            self._update_max_drawdown()

            return True, f"{action} {quantity} {symbol} @ ${price:.3f}", trade

        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False, f"Trade execution error: {str(e)}", None

    def _update_max_drawdown(self) -> None:
        """Update maximum drawdown tracking."""
        current_value = self.total_portfolio_value

        if current_value > self.max_portfolio_value:
            self.max_portfolio_value = current_value

        current_drawdown = (self.max_portfolio_value - current_value) / self.max_portfolio_value
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        total_trades = len(self.trade_history)
        winning_trades = len([t for t in self.trade_history if (t.realized_pnl or 0) > 0])
        losing_trades = len([t for t in self.trade_history if (t.realized_pnl or 0) < 0])

        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        # Calculate average win/loss
        wins = [t.realized_pnl for t in self.trade_history if (t.realized_pnl or 0) > 0]
        losses = [t.realized_pnl for t in self.trade_history if (t.realized_pnl or 0) < 0]

        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0

        # Profit factor
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        return {
            'total_portfolio_value': self.total_portfolio_value,
            'cash': self.cash,
            'position_value': self.total_position_value,
            'total_pnl': self.total_pnl,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'total_return_pct': self.total_return_pct,
            'max_drawdown': self.max_drawdown * 100,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_commission_paid': self.total_commission_paid,
            'total_slippage_paid': self.total_slippage_paid,
            'position_count': self.position_count,
            'max_portfolio_value': self.max_portfolio_value
        }

    def get_positions_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all positions."""
        return [
            {
                'symbol': pos.symbol,
                'market_name': pos.market_name,
                'quantity': pos.quantity,
                'avg_cost': pos.avg_cost,
                'current_price': pos.current_price,
                'market_value': pos.market_value,
                'unrealized_pnl': pos.unrealized_pnl,
                'unrealized_pnl_pct': pos.unrealized_pnl_pct,
                'side': pos.side
            }
            for pos in self.positions.values()
            if pos.quantity != 0
        ]

    def save_to_file(self, file_path: str) -> None:
        """Save portfolio state to JSON file."""
        try:
            data = {
                'timestamp': datetime.now().isoformat(),
                'initial_capital': self.initial_capital,
                'current_cash': self.cash,
                'performance_metrics': self.get_performance_metrics(),
                'positions': [asdict(pos) for pos in self.positions.values()],
                'trade_history': [asdict(trade) for trade in self.trade_history],
                'daily_values': [(dt.isoformat(), val) for dt, val in self.daily_portfolio_values]
            }

            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            logger.info(f"Portfolio saved to {file_path}")

        except Exception as e:
            logger.error(f"Error saving portfolio: {e}")

    def load_from_file(self, file_path: str) -> bool:
        """Load portfolio state from JSON file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            self.initial_capital = data['initial_capital']
            self.cash = data['current_cash']

            # Restore positions
            self.positions = {}
            for pos_data in data['positions']:
                pos_data['timestamp'] = datetime.fromisoformat(pos_data['timestamp'])
                pos = Position(**pos_data)
                self.positions[pos.symbol] = pos

            # Restore trade history
            self.trade_history = []
            for trade_data in data['trade_history']:
                trade_data['timestamp'] = datetime.fromisoformat(trade_data['timestamp'])
                trade = Trade(**trade_data)
                self.trade_history.append(trade)

            # Restore daily values
            self.daily_portfolio_values = [
                (datetime.fromisoformat(dt), val)
                for dt, val in data['daily_values']
            ]

            logger.info(f"Portfolio loaded from {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error loading portfolio: {e}")
            return False

    def add_daily_value_snapshot(self) -> None:
        """Add current portfolio value to daily tracking."""
        self.daily_portfolio_values.append((datetime.now(), self.total_portfolio_value))

    def __str__(self) -> str:
        """String representation of portfolio."""
        metrics = self.get_performance_metrics()
        return (
            f"Paper Portfolio Summary:\n"
            f"  Total Value: ${metrics['total_portfolio_value']:,.2f}\n"
            f"  Cash: ${metrics['cash']:,.2f}\n"
            f"  Positions: ${metrics['position_value']:,.2f}\n"
            f"  Total P&L: ${metrics['total_pnl']:,.2f} ({metrics['total_return_pct']:.2f}%)\n"
            f"  Open Positions: {metrics['position_count']}\n"
            f"  Total Trades: {metrics['total_trades']}\n"
            f"  Win Rate: {metrics['win_rate']:.1f}%\n"
            f"  Max Drawdown: {metrics['max_drawdown']:.2f}%"
        )
    def get_portfolio_metrics(self) -> Dict[str, float]:
        """Return high-level portfolio metrics for quick inspection."""
        return {"cash": self.cash, "total_value": self.total_portfolio_value, "unrealized_pnl": self.unrealized_pnl, "realized_pnl": self.realized_pnl, "total_pnl": self.total_pnl, "total_return_pct": self.total_return_pct, "open_positions": self.position_count}
