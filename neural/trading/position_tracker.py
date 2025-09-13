"""
Position Tracking System

This module provides real-time position and P&L tracking for Kalshi trading,
including position reconciliation, performance metrics, and risk monitoring.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio

from .kalshi_client import KalshiClient, MarketData
from .order_manager import OrderManager, Order, Fill

logger = logging.getLogger(__name__)


class PositionSide(Enum):
    """Position side."""
    LONG = "long"    # Holding YES contracts or short NO contracts
    SHORT = "short"  # Holding NO contracts or short YES contracts
    FLAT = "flat"    # No position


@dataclass
class Position:
    """Position tracking for a single market."""
    ticker: str
    side: PositionSide = PositionSide.FLAT
    
    # YES side position
    yes_long: int = 0      # Long YES contracts
    yes_short: int = 0     # Short YES contracts
    yes_net: int = field(init=False)
    yes_avg_cost: float = 0.0
    
    # NO side position  
    no_long: int = 0       # Long NO contracts
    no_short: int = 0      # Short NO contracts
    no_net: int = field(init=False)
    no_avg_cost: float = 0.0
    
    # P&L tracking
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_pnl: float = field(init=False)
    
    # Cost basis and exposure
    total_cost_basis: float = 0.0
    market_value: float = 0.0
    max_exposure: float = 0.0
    
    # Metadata
    first_trade_time: Optional[datetime] = None
    last_trade_time: Optional[datetime] = None
    strategy_id: Optional[str] = None
    
    def __post_init__(self):
        """Initialize computed fields."""
        self._update_computed_fields()
    
    def _update_computed_fields(self):
        """Update computed fields after position changes."""
        # Calculate net positions
        self.yes_net = self.yes_long - self.yes_short
        self.no_net = self.no_long - self.no_short
        
        # Determine overall position side
        if self.yes_net > 0 or self.no_net < 0:  # Long YES or short NO = bullish
            self.side = PositionSide.LONG
        elif self.yes_net < 0 or self.no_net > 0:  # Short YES or long NO = bearish
            self.side = PositionSide.SHORT
        else:
            self.side = PositionSide.FLAT
        
        # Calculate total P&L
        self.total_pnl = self.realized_pnl + self.unrealized_pnl
    
    @property
    def net_contracts(self) -> int:
        """Get net contract position (positive = bullish, negative = bearish)."""
        return self.yes_net - self.no_net
    
    @property
    def gross_contracts(self) -> int:
        """Get gross contract position (total exposure)."""
        return abs(self.yes_net) + abs(self.no_net)
    
    @property
    def is_flat(self) -> bool:
        """Check if position is flat."""
        return self.side == PositionSide.FLAT and self.net_contracts == 0
    
    @property
    def is_long(self) -> bool:
        """Check if position is net long."""
        return self.side == PositionSide.LONG
    
    @property
    def is_short(self) -> bool:
        """Check if position is net short."""
        return self.side == PositionSide.SHORT
    
    def add_trade(self, fill: Fill, market_price: Optional[float] = None):
        """
        Add a trade/fill to the position.
        
        Args:
            fill: Trade fill to add
            market_price: Current market price for unrealized P&L calculation
        """
        trade_value = (fill.count * fill.price) / 100.0  # Convert cents to dollars
        
        # Update first/last trade times
        if self.first_trade_time is None:
            self.first_trade_time = fill.timestamp
        self.last_trade_time = fill.timestamp
        
        # Process based on fill details
        if fill.side.lower() == "yes":
            if fill.action.lower() == "buy":
                # Bought YES contracts (going long YES)
                self._update_yes_position(fill.count, fill.price, True)
            else:
                # Sold YES contracts (going short YES)  
                self._update_yes_position(-fill.count, fill.price, False)
        else:  # NO side
            if fill.action.lower() == "buy":
                # Bought NO contracts (going long NO)
                self._update_no_position(fill.count, fill.price, True)
            else:
                # Sold NO contracts (going short NO)
                self._update_no_position(-fill.count, fill.price, False)
        
        # Update unrealized P&L if market price provided
        if market_price is not None:
            self.update_unrealized_pnl(market_price)
        
        self._update_computed_fields()
        logger.debug(f"Trade added to {self.ticker}: {fill.action} {fill.count} {fill.side} @ {fill.price}")
    
    def _update_yes_position(self, count: int, price: int, is_buy: bool):
        """Update YES side position."""
        if count > 0:  # Adding to position
            if is_buy:
                # Buying YES contracts
                old_cost = self.yes_long * self.yes_avg_cost
                new_cost = (count * price) / 100.0
                
                self.yes_long += count
                if self.yes_long > 0:
                    self.yes_avg_cost = (old_cost + new_cost) / self.yes_long
                
                self.total_cost_basis += new_cost
            else:
                # Selling YES contracts (shorting)
                self.yes_short += count
        else:  # Reducing position (count is negative)
            count = abs(count)
            if is_buy:
                # Covering YES short
                if self.yes_short >= count:
                    self.yes_short -= count
                    # Calculate realized P&L from covering short
                    cover_cost = (count * price) / 100.0
                    # P&L = (short_price - cover_price) * count
                    # For simplicity, assume average short price
                    self.realized_pnl += (self.yes_avg_cost - price/100.0) * count
                else:
                    # More complex case - partially covering, then going long
                    remaining = count - self.yes_short
                    self.yes_short = 0
                    self.yes_long += remaining
            else:
                # Selling existing YES longs
                if self.yes_long >= count:
                    # Calculate realized P&L
                    sell_value = (count * price) / 100.0
                    cost_basis = count * self.yes_avg_cost
                    self.realized_pnl += sell_value - cost_basis
                    
                    self.yes_long -= count
                    self.total_cost_basis -= cost_basis
                else:
                    # More complex case
                    remaining = count - self.yes_long
                    self.yes_long = 0
                    self.yes_short += remaining
    
    def _update_no_position(self, count: int, price: int, is_buy: bool):
        """Update NO side position."""
        if count > 0:  # Adding to position
            if is_buy:
                # Buying NO contracts
                old_cost = self.no_long * self.no_avg_cost
                new_cost = (count * price) / 100.0
                
                self.no_long += count
                if self.no_long > 0:
                    self.no_avg_cost = (old_cost + new_cost) / self.no_long
                
                self.total_cost_basis += new_cost
            else:
                # Selling NO contracts (shorting)
                self.no_short += count
        else:  # Reducing position
            count = abs(count)
            if is_buy:
                # Covering NO short
                if self.no_short >= count:
                    self.no_short -= count
                    cover_cost = (count * price) / 100.0
                    self.realized_pnl += (self.no_avg_cost - price/100.0) * count
                else:
                    remaining = count - self.no_short
                    self.no_short = 0
                    self.no_long += remaining
            else:
                # Selling existing NO longs
                if self.no_long >= count:
                    sell_value = (count * price) / 100.0
                    cost_basis = count * self.no_avg_cost
                    self.realized_pnl += sell_value - cost_basis
                    
                    self.no_long -= count
                    self.total_cost_basis -= cost_basis
                else:
                    remaining = count - self.no_long
                    self.no_long = 0
                    self.no_short += remaining
    
    def update_unrealized_pnl(self, current_yes_price: float, current_no_price: Optional[float] = None):
        """
        Update unrealized P&L based on current market prices.
        
        Args:
            current_yes_price: Current YES price (as probability 0-1)
            current_no_price: Current NO price (as probability 0-1, or derived from YES)
        """
        if current_no_price is None:
            current_no_price = 1.0 - current_yes_price
        
        unrealized = 0.0
        
        # Calculate unrealized P&L for YES positions
        if self.yes_net != 0:
            current_yes_value = self.yes_net * current_yes_price
            yes_cost_basis = self.yes_net * self.yes_avg_cost if self.yes_net > 0 else 0
            unrealized += current_yes_value - yes_cost_basis
        
        # Calculate unrealized P&L for NO positions
        if self.no_net != 0:
            current_no_value = self.no_net * current_no_price
            no_cost_basis = self.no_net * self.no_avg_cost if self.no_net > 0 else 0
            unrealized += current_no_value - no_cost_basis
        
        self.unrealized_pnl = unrealized
        self.market_value = abs(self.net_contracts) * max(current_yes_price, current_no_price)
        self._update_computed_fields()
    
    def close_position(self, settlement_price: float, settlement_side: str):
        """
        Close position at settlement.
        
        Args:
            settlement_price: Settlement price (0 or 1)
            settlement_side: Which side won ("yes" or "no")
        """
        if self.is_flat:
            return
        
        # Calculate final P&L based on settlement
        if settlement_side.lower() == "yes":
            # YES side wins (settles to $1), NO side loses (settles to $0)
            final_pnl = self.yes_net * (1.0 - self.yes_avg_cost) + self.no_net * (0.0 - self.no_avg_cost)
        else:
            # NO side wins (settles to $1), YES side loses (settles to $0)
            final_pnl = self.yes_net * (0.0 - self.yes_avg_cost) + self.no_net * (1.0 - self.no_avg_cost)
        
        # Move unrealized to realized
        self.realized_pnl += self.unrealized_pnl + (final_pnl - self.total_pnl)
        self.unrealized_pnl = 0.0
        
        # Reset positions
        self.yes_long = 0
        self.yes_short = 0
        self.no_long = 0
        self.no_short = 0
        self.total_cost_basis = 0.0
        self.market_value = 0.0
        
        self._update_computed_fields()
        logger.info(f"Position closed for {self.ticker}: Final P&L = ${self.realized_pnl:.2f}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary."""
        return {
            "ticker": self.ticker,
            "side": self.side.value,
            "yes_long": self.yes_long,
            "yes_short": self.yes_short,
            "yes_net": self.yes_net,
            "yes_avg_cost": self.yes_avg_cost,
            "no_long": self.no_long,
            "no_short": self.no_short,
            "no_net": self.no_net,
            "no_avg_cost": self.no_avg_cost,
            "net_contracts": self.net_contracts,
            "gross_contracts": self.gross_contracts,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "total_pnl": self.total_pnl,
            "total_cost_basis": self.total_cost_basis,
            "market_value": self.market_value,
            "max_exposure": self.max_exposure,
            "first_trade_time": self.first_trade_time.isoformat() if self.first_trade_time else None,
            "last_trade_time": self.last_trade_time.isoformat() if self.last_trade_time else None,
            "strategy_id": self.strategy_id
        }


class PositionTracker:
    """
    Real-time position and P&L tracking system.
    
    Handles:
    - Position updates from order fills
    - Real-time P&L calculation
    - Position reconciliation with exchange
    - Risk metrics calculation
    - Performance attribution
    """
    
    def __init__(
        self,
        kalshi_client: KalshiClient,
        order_manager: OrderManager,
        auto_update_prices: bool = True,
        price_update_interval: float = 30.0
    ):
        """
        Initialize position tracker.
        
        Args:
            kalshi_client: Kalshi client for market data
            order_manager: Order manager for fill updates
            auto_update_prices: Automatically update market prices
            price_update_interval: Price update interval in seconds
        """
        self.client = kalshi_client
        self.order_manager = order_manager
        self.auto_update_prices = auto_update_prices
        self.price_update_interval = price_update_interval
        
        # Position tracking
        self.positions: Dict[str, Position] = {}  # ticker -> Position
        self.market_prices: Dict[str, float] = {}  # ticker -> current_price
        
        # Event handlers
        self.position_handlers: List[Callable[[Position], None]] = []
        
        # Performance metrics
        self.total_realized_pnl = 0.0
        self.total_unrealized_pnl = 0.0
        self.total_trades = 0
        self.win_count = 0
        self.loss_count = 0
        
        # Auto-update task
        self._price_update_task = None
        
        # Setup fill handler
        self.order_manager.add_fill_handler(self._handle_fill)
        
        logger.info("PositionTracker initialized")
    
    async def start(self):
        """Start position tracking."""
        # Start price update task if enabled
        if self.auto_update_prices:
            self._price_update_task = asyncio.create_task(self._price_update_loop())
        
        # Load existing positions from exchange
        await self.reconcile_positions()
        
        logger.info("PositionTracker started")
    
    async def stop(self):
        """Stop position tracking."""
        if self._price_update_task:
            self._price_update_task.cancel()
            try:
                await self._price_update_task
            except asyncio.CancelledError:
                pass
        
        logger.info("PositionTracker stopped")
    
    async def _price_update_loop(self):
        """Continuously update market prices."""
        while True:
            try:
                await self.update_all_prices()
                await asyncio.sleep(self.price_update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in price update loop: {e}")
                await asyncio.sleep(5)  # Brief delay before retry
    
    async def _handle_fill(self, fill: Fill):
        """Handle fill from order manager."""
        try:
            # Get or create position
            position = self.get_or_create_position(fill.ticker, fill.order_id)
            
            # Add trade to position
            market_price = self.market_prices.get(fill.ticker, 0.5)
            position.add_trade(fill, market_price)
            
            # Update totals
            self.total_trades += 1
            self._update_totals()
            
            # Notify handlers
            await self._notify_position_handlers(position)
            
            logger.info(f"Fill processed for {fill.ticker}: {fill.count} @ {fill.price}")
            
        except Exception as e:
            logger.error(f"Error handling fill: {e}")
    
    def get_or_create_position(self, ticker: str, order_id: str) -> Position:
        """Get existing position or create new one."""
        if ticker not in self.positions:
            # Determine strategy ID from order
            strategy_id = None
            order = self.order_manager.get_order(order_id)
            if order:
                strategy_id = order.strategy_id
            
            self.positions[ticker] = Position(
                ticker=ticker,
                strategy_id=strategy_id
            )
            logger.info(f"Created new position for {ticker}")
        
        return self.positions[ticker]
    
    async def _notify_position_handlers(self, position: Position):
        """Notify all position event handlers."""
        for handler in self.position_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(position)
                else:
                    handler(position)
            except Exception as e:
                logger.error(f"Error in position handler: {e}")
    
    def _update_totals(self):
        """Update total P&L and statistics."""
        self.total_realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
        self.total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        
        # Update win/loss counts
        self.win_count = sum(1 for pos in self.positions.values() if pos.total_pnl > 0)
        self.loss_count = sum(1 for pos in self.positions.values() if pos.total_pnl < 0)
    
    # Position Management
    
    async def update_all_prices(self):
        """Update market prices for all positions."""
        tickers = list(self.positions.keys())
        if not tickers:
            return
        
        try:
            # Get current market data
            for ticker in tickers:
                market_data = await self.client.get_market(ticker)
                
                if market_data:
                    # Calculate mid price
                    yes_bid = market_data.get("yes_bid", 0)
                    yes_ask = market_data.get("yes_ask", 0)
                    
                    if yes_bid and yes_ask:
                        mid_price = (yes_bid + yes_ask) / 200.0  # Convert cents to probability
                        self.market_prices[ticker] = mid_price
                        
                        # Update position P&L
                        position = self.positions[ticker]
                        position.update_unrealized_pnl(mid_price)
            
            self._update_totals()
            logger.debug(f"Updated prices for {len(tickers)} positions")
            
        except Exception as e:
            logger.error(f"Error updating prices: {e}")
    
    async def update_position_price(self, ticker: str):
        """Update price for a specific position."""
        if ticker not in self.positions:
            return
        
        try:
            market_data = await self.client.get_market(ticker)
            
            if market_data:
                yes_bid = market_data.get("yes_bid", 0)
                yes_ask = market_data.get("yes_ask", 0)
                
                if yes_bid and yes_ask:
                    mid_price = (yes_bid + yes_ask) / 200.0
                    self.market_prices[ticker] = mid_price
                    
                    position = self.positions[ticker]
                    position.update_unrealized_pnl(mid_price)
                    
                    await self._notify_position_handlers(position)
                    
        except Exception as e:
            logger.error(f"Error updating price for {ticker}: {e}")
    
    async def reconcile_positions(self):
        """Reconcile positions with exchange."""
        try:
            # Get positions from exchange
            response = await self.client.get_positions(limit=1000)
            
            if response and "positions" in response:
                for exchange_position in response["positions"]:
                    ticker = exchange_position.get("ticker")
                    
                    if not ticker:
                        continue
                    
                    # Create or update local position
                    if ticker not in self.positions:
                        self.positions[ticker] = Position(ticker=ticker)
                    
                    position = self.positions[ticker]
                    
                    # Sync position data (this would need to be expanded based on Kalshi's position format)
                    # For now, just log the reconciliation
                    logger.info(f"Reconciled position for {ticker}")
            
            logger.info("Position reconciliation completed")
            
        except Exception as e:
            logger.error(f"Error reconciling positions: {e}")
    
    # Query Methods
    
    def get_position(self, ticker: str) -> Optional[Position]:
        """Get position for a ticker."""
        return self.positions.get(ticker)
    
    def get_all_positions(self) -> List[Position]:
        """Get all positions."""
        return list(self.positions.values())
    
    def get_active_positions(self) -> List[Position]:
        """Get positions that are not flat."""
        return [pos for pos in self.positions.values() if not pos.is_flat]
    
    def get_positions_by_strategy(self, strategy_id: str) -> List[Position]:
        """Get positions for a specific strategy."""
        return [pos for pos in self.positions.values() if pos.strategy_id == strategy_id]
    
    def get_long_positions(self) -> List[Position]:
        """Get all long positions."""
        return [pos for pos in self.positions.values() if pos.is_long]
    
    def get_short_positions(self) -> List[Position]:
        """Get all short positions."""
        return [pos for pos in self.positions.values() if pos.is_short]
    
    # Portfolio Metrics
    
    def get_portfolio_value(self) -> float:
        """Get total portfolio market value."""
        return sum(abs(pos.market_value) for pos in self.positions.values())
    
    def get_total_pnl(self) -> float:
        """Get total P&L (realized + unrealized)."""
        return self.total_realized_pnl + self.total_unrealized_pnl
    
    def get_portfolio_stats(self) -> Dict[str, Any]:
        """Get comprehensive portfolio statistics."""
        active_positions = self.get_active_positions()
        
        return {
            "total_positions": len(self.positions),
            "active_positions": len(active_positions),
            "long_positions": len(self.get_long_positions()),
            "short_positions": len(self.get_short_positions()),
            "total_realized_pnl": self.total_realized_pnl,
            "total_unrealized_pnl": self.total_unrealized_pnl,
            "total_pnl": self.get_total_pnl(),
            "portfolio_value": self.get_portfolio_value(),
            "total_trades": self.total_trades,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "win_rate": self.win_count / max(1, self.win_count + self.loss_count),
            "avg_win": sum(pos.total_pnl for pos in self.positions.values() if pos.total_pnl > 0) / max(1, self.win_count),
            "avg_loss": sum(pos.total_pnl for pos in self.positions.values() if pos.total_pnl < 0) / max(1, self.loss_count)
        }
    
    def get_strategy_performance(self, strategy_id: str) -> Dict[str, Any]:
        """Get performance metrics for a specific strategy."""
        strategy_positions = self.get_positions_by_strategy(strategy_id)
        
        if not strategy_positions:
            return {}
        
        total_pnl = sum(pos.total_pnl for pos in strategy_positions)
        realized_pnl = sum(pos.realized_pnl for pos in strategy_positions)
        unrealized_pnl = sum(pos.unrealized_pnl for pos in strategy_positions)
        
        wins = [pos for pos in strategy_positions if pos.total_pnl > 0]
        losses = [pos for pos in strategy_positions if pos.total_pnl < 0]
        
        return {
            "strategy_id": strategy_id,
            "total_positions": len(strategy_positions),
            "active_positions": len([pos for pos in strategy_positions if not pos.is_flat]),
            "total_pnl": total_pnl,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "win_count": len(wins),
            "loss_count": len(losses),
            "win_rate": len(wins) / max(1, len(strategy_positions)),
            "avg_win": sum(pos.total_pnl for pos in wins) / max(1, len(wins)),
            "avg_loss": sum(pos.total_pnl for pos in losses) / max(1, len(losses)),
            "profit_factor": abs(sum(pos.total_pnl for pos in wins) / sum(pos.total_pnl for pos in losses)) if losses else float('inf')
        }
    
    # Event Handler Management
    
    def add_position_handler(self, handler: Callable[[Position], None]):
        """Add position event handler."""
        self.position_handlers.append(handler)
        logger.info("Position handler added")
    
    def remove_position_handler(self, handler: Callable[[Position], None]):
        """Remove position event handler."""
        if handler in self.position_handlers:
            self.position_handlers.remove(handler)
            logger.info("Position handler removed")
    
    # Settlement Handling
    
    async def handle_settlement(self, ticker: str, settlement_side: str):
        """
        Handle market settlement.
        
        Args:
            ticker: Market ticker that settled
            settlement_side: Winning side ("yes" or "no")
        """
        if ticker in self.positions:
            position = self.positions[ticker]
            position.close_position(1.0, settlement_side)
            
            self._update_totals()
            await self._notify_position_handlers(position)
            
            logger.info(f"Settlement processed for {ticker}: {settlement_side} won")
    
    # Risk Metrics
    
    def calculate_var(self, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk (simplified)."""
        # This is a simplified VaR calculation
        # In practice, you'd use historical returns and proper statistical methods
        unrealized_pnls = [pos.unrealized_pnl for pos in self.positions.values() if not pos.is_flat]
        
        if not unrealized_pnls:
            return 0.0
        
        # Simple percentile-based VaR
        import numpy as np
        return float(np.percentile(unrealized_pnls, (1 - confidence_level) * 100))
    
    def get_largest_positions(self, limit: int = 10) -> List[Position]:
        """Get largest positions by market value."""
        return sorted(
            self.get_active_positions(),
            key=lambda pos: abs(pos.market_value),
            reverse=True
        )[:limit]
