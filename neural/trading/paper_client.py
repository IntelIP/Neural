"""
Paper Trading Client

A mock implementation of the TradingClient interface that simulates
trading without real money. Maintains portfolio state, executes
paper trades, and tracks performance.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from .paper_portfolio import PaperPortfolio, Position, Trade

logger = logging.getLogger(__name__)


@dataclass
class PaperOrder:
    """Represents a paper trading order."""

    order_id: str
    market_id: str
    symbol: str
    market_name: str
    side: str  # "yes" or "no"
    action: str  # "buy" or "sell"
    quantity: int
    order_type: str  # "market", "limit"
    price: Optional[float] = None  # For limit orders
    status: str = "pending"  # "pending", "filled", "cancelled"
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    filled_price: Optional[float] = None
    filled_quantity: int = 0


class PaperTradingClient:
    """
    Paper trading client that mimics the TradingClient interface.

    Simulates order execution, maintains portfolio state, and provides
    the same interface as the real trading client for seamless switching.
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission_per_trade: float = 0.50,
        slippage_pct: float = 0.002,
        save_trades: bool = True,
        data_dir: str = "paper_trading_data"
    ):
        """
        Initialize paper trading client.

        Args:
            initial_capital: Starting capital for paper trading
            commission_per_trade: Commission per trade
            slippage_pct: Slippage percentage for market orders
            save_trades: Whether to save trade data to files
            data_dir: Directory to save trade data
        """
        self.portfolio = PaperPortfolio(
            initial_capital=initial_capital,
            commission_per_trade=commission_per_trade,
            default_slippage_pct=slippage_pct
        )

        self.save_trades = save_trades
        self.data_dir = Path(data_dir)
        self.market_prices: Dict[str, float] = {}  # Cache for market prices
        self.pending_orders: Dict[str, PaperOrder] = {}
        self.order_counter = 0

        # Create data directory if saving trades
        if self.save_trades:
            self.data_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Paper trading client initialized with ${initial_capital:,.2f}")

    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        self.order_counter += 1
        return f"PAPER_{datetime.now().strftime('%Y%m%d')}_{self.order_counter:06d}"

    def _get_market_price(self, market_id: str, side: str) -> Optional[float]:
        """
        Get current market price for a market/side.

        In a real implementation, this would fetch from Kalshi API.
        For paper trading, we'll use cached prices or reasonable defaults.
        """
        # Try to get cached price
        price_key = f"{market_id}_{side}"
        if price_key in self.market_prices:
            return self.market_prices[price_key]

        # Default prices based on side (these would come from real market data)
        if side.lower() == "yes":
            return 0.50  # 50 cents default for yes side
        else:
            return 0.50  # 50 cents default for no side

    def update_market_price(self, market_id: str, side: str, price: float) -> None:
        """Update cached market price."""
        price_key = f"{market_id}_{side}"
        self.market_prices[price_key] = price

        # Update position prices if we have them
        symbol = f"{market_id}_{side}"
        self.portfolio.update_position_price(symbol, price)

    def update_market_prices(self, price_updates: Dict[str, Dict[str, float]]) -> None:
        """
        Update multiple market prices.

        Args:
            price_updates: Dict of {market_id: {"yes": price, "no": price}}
        """
        for market_id, sides in price_updates.items():
            for side, price in sides.items():
                self.update_market_price(market_id, side, price)

    async def place_order(
        self,
        market_id: str,
        side: str,
        quantity: int,
        order_type: str = "market",
        price: Optional[float] = None,
        market_name: Optional[str] = None,
        sentiment_score: Optional[float] = None,
        confidence: Optional[float] = None,
        strategy: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Place a paper trading order.

        Args:
            market_id: Market identifier
            side: "yes" or "no"
            quantity: Number of contracts
            order_type: "market" or "limit"
            price: Price for limit orders
            market_name: Human readable market name
            sentiment_score: Sentiment score that triggered this trade
            confidence: Confidence in the signal
            strategy: Strategy name

        Returns:
            Order result dictionary
        """
        try:
            order_id = self._generate_order_id()
            symbol = f"{market_id}_{side}"

            if market_name is None:
                market_name = f"Market {market_id}"

            # Create order object
            order = PaperOrder(
                order_id=order_id,
                market_id=market_id,
                symbol=symbol,
                market_name=market_name,
                side=side,
                action="buy",  # For now, all orders are buys
                quantity=quantity,
                order_type=order_type,
                price=price
            )

            # For market orders, execute immediately
            if order_type == "market":
                fill_price = self._get_market_price(market_id, side)
                if fill_price is None:
                    return {
                        "success": False,
                        "message": f"No market price available for {market_id} {side}",
                        "order_id": order_id
                    }

                # Apply slippage for market orders
                slippage_adjustment = fill_price * self.portfolio.default_slippage_pct
                if quantity > 0:  # Buying
                    fill_price += slippage_adjustment
                else:  # Selling
                    fill_price -= slippage_adjustment

                # Execute the trade
                success, message, trade = self.portfolio.execute_trade(
                    market_id=market_id,
                    symbol=symbol,
                    market_name=market_name,
                    action="BUY" if quantity > 0 else "SELL",
                    side=side,
                    quantity=abs(quantity),
                    price=fill_price,
                    sentiment_score=sentiment_score,
                    confidence=confidence,
                    strategy=strategy
                )

                if success:
                    order.status = "filled"
                    order.filled_at = datetime.now()
                    order.filled_price = fill_price
                    order.filled_quantity = abs(quantity)

                    # Save trade data if enabled
                    if self.save_trades and trade:
                        self._save_trade_data(trade)

                    return {
                        "success": True,
                        "message": message,
                        "order_id": order_id,
                        "filled_price": fill_price,
                        "filled_quantity": abs(quantity),
                        "trade": trade
                    }
                else:
                    order.status = "cancelled"
                    return {
                        "success": False,
                        "message": message,
                        "order_id": order_id
                    }

            else:  # Limit order
                self.pending_orders[order_id] = order
                return {
                    "success": True,
                    "message": f"Limit order placed: {quantity} {symbol} @ ${price:.3f}",
                    "order_id": order_id,
                    "status": "pending"
                }

        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return {
                "success": False,
                "message": f"Order failed: {str(e)}",
                "order_id": None
            }

    def close_position(
        self,
        market_id: str,
        side: str,
        quantity: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Close a position (sell all or partial).

        Args:
            market_id: Market identifier
            side: "yes" or "no"
            quantity: Quantity to close (None for full position)

        Returns:
            Close result dictionary
        """
        symbol = f"{market_id}_{side}"
        position = self.portfolio.get_position(symbol)

        if not position or position.quantity == 0:
            return {
                "success": False,
                "message": f"No position to close for {symbol}"
            }

        close_quantity = quantity if quantity is not None else position.quantity
        if close_quantity > position.quantity:
            return {
                "success": False,
                "message": f"Cannot close {close_quantity}, only have {position.quantity}"
            }

        # Get current market price
        current_price = self._get_market_price(market_id, side)
        if current_price is None:
            return {
                "success": False,
                "message": f"No market price available for {symbol}"
            }

        # Execute the closing trade
        success, message, trade = self.portfolio.execute_trade(
            market_id=market_id,
            symbol=symbol,
            market_name=position.market_name or f"Market {market_id}",
            action="SELL",
            side=side,
            quantity=close_quantity,
            price=current_price,
            strategy="position_close"
        )

        if success and self.save_trades and trade:
            self._save_trade_data(trade)

        return {
            "success": success,
            "message": message,
            "realized_pnl": trade.realized_pnl if trade else None
        }

    def get_portfolio(self) -> Dict[str, Any]:
        """Get current portfolio status."""
        return self.portfolio.get_performance_metrics()

    def get_positions(self) -> List[Dict[str, Any]]:
        """Get all current positions."""
        return self.portfolio.get_positions_summary()

    def get_position(self, market_id: str, side: str) -> Optional[Dict[str, Any]]:
        """Get specific position."""
        symbol = f"{market_id}_{side}"
        position = self.portfolio.get_position(symbol)

        if position and position.quantity != 0:
            return {
                'symbol': position.symbol,
                'market_id': market_id,
                'side': side,
                'market_name': position.market_name,
                'quantity': position.quantity,
                'avg_cost': position.avg_cost,
                'current_price': position.current_price,
                'market_value': position.market_value,
                'unrealized_pnl': position.unrealized_pnl,
                'unrealized_pnl_pct': position.unrealized_pnl_pct
            }
        return None

    def get_trade_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get trade history."""
        trades = self.portfolio.trade_history
        if limit:
            trades = trades[-limit:]

        return [
            {
                'timestamp': trade.timestamp.isoformat(),
                'market_id': trade.market_id,
                'symbol': trade.symbol,
                'market_name': trade.market_name,
                'action': trade.action,
                'side': trade.side,
                'quantity': trade.quantity,
                'price': trade.price,
                'value': trade.value,
                'commission': trade.commission,
                'slippage': trade.slippage,
                'realized_pnl': trade.realized_pnl,
                'sentiment_score': trade.sentiment_score,
                'confidence': trade.confidence,
                'strategy': trade.strategy
            }
            for trade in trades
        ]

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        metrics = self.portfolio.get_performance_metrics()
        positions = self.portfolio.get_positions_summary()
        recent_trades = self.get_trade_history(limit=10)

        return {
            'timestamp': datetime.now().isoformat(),
            'portfolio_metrics': metrics,
            'current_positions': positions,
            'recent_trades': recent_trades,
            'pending_orders': len(self.pending_orders),
            'data_directory': str(self.data_dir) if self.save_trades else None
        }

    def _save_trade_data(self, trade: Trade) -> None:
        """Save individual trade data."""
        if not self.save_trades:
            return

        try:
            # Create daily trade file
            date_str = trade.timestamp.strftime('%Y%m%d')
            trade_file = self.data_dir / f"trades_{date_str}.jsonl"

            # Append trade to daily file (JSON Lines format)
            trade_data = {
                'timestamp': trade.timestamp.isoformat(),
                'market_id': trade.market_id,
                'symbol': trade.symbol,
                'market_name': trade.market_name,
                'action': trade.action,
                'side': trade.side,
                'quantity': trade.quantity,
                'price': trade.price,
                'value': trade.value,
                'commission': trade.commission,
                'slippage': trade.slippage,
                'realized_pnl': trade.realized_pnl,
                'sentiment_score': trade.sentiment_score,
                'confidence': trade.confidence,
                'strategy': trade.strategy
            }

            with open(trade_file, 'a') as f:
                f.write(json.dumps(trade_data, default=str) + '\n')

        except Exception as e:
            logger.error(f"Error saving trade data: {e}")

    def save_portfolio_snapshot(self) -> None:
        """Save current portfolio state."""
        if not self.save_trades:
            return

        try:
            date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            portfolio_file = self.data_dir / f"portfolio_snapshot_{date_str}.json"
            self.portfolio.save_to_file(str(portfolio_file))

        except Exception as e:
            logger.error(f"Error saving portfolio snapshot: {e}")

    def reset_portfolio(self, new_initial_capital: Optional[float] = None) -> None:
        """Reset portfolio to initial state."""
        initial_capital = new_initial_capital or self.portfolio.initial_capital

        logger.info(f"Resetting paper portfolio to ${initial_capital:,.2f}")

        self.portfolio = PaperPortfolio(
            initial_capital=initial_capital,
            commission_per_trade=self.portfolio.commission_per_trade,
            default_slippage_pct=self.portfolio.default_slippage_pct
        )

        self.pending_orders.clear()
        self.market_prices.clear()
        self.order_counter = 0

    def close(self) -> None:
        """Close the paper trading client and save final state."""
        if self.save_trades:
            logger.info("Saving final portfolio snapshot...")
            self.save_portfolio_snapshot()

        logger.info("Paper trading client closed")

    def __enter__(self) -> "PaperTradingClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __str__(self) -> str:
        """String representation showing portfolio summary."""
        return f"PaperTradingClient:\n{self.portfolio}"


# Factory function for easy creation
def create_paper_trading_client(
    initial_capital: float = 10000.0,
    commission: float = 0.50,
    slippage_pct: float = 0.002,
    save_data: bool = True,
    data_dir: str = "paper_trading_data"
) -> PaperTradingClient:
    """Create a paper trading client with default settings."""
    return PaperTradingClient(
        initial_capital=initial_capital,
        commission_per_trade=commission,
        slippage_pct=slippage_pct,
        save_trades=save_data,
        data_dir=data_dir
    )