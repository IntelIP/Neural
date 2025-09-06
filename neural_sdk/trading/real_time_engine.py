"""
Real-Time Trading Engine

Core trading engine for real-time market operations.
Handles signal generation, order management, and risk controls.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import statistics

from neural_sdk.data_sources.unified.stream_manager import (
    UnifiedStreamManager, EventType, UnifiedMarketData, StreamConfig
)
from neural_sdk.data_pipeline.data_sources.kalshi.client import KalshiClient

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Trading signal types."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class TradingSignal:
    """Trading signal."""
    signal_id: str
    timestamp: datetime
    market_ticker: str
    signal_type: SignalType
    confidence: float  # 0-1
    size: Optional[int] = None
    price_limit: Optional[float] = None
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Order:
    """Trading order."""
    order_id: str
    signal_id: str
    market_ticker: str
    side: str  # yes/no
    order_type: str  # market/limit
    size: int
    price: Optional[float]
    status: OrderStatus
    created_at: datetime
    filled_size: int = 0
    average_fill_price: Optional[float] = None
    commission: float = 0.0


@dataclass
class Position:
    """Market position."""
    market_ticker: str
    side: str  # yes/no
    size: int
    average_price: float
    current_price: Optional[float] = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    opened_at: datetime = field(default_factory=datetime.utcnow)
    
    def update_pnl(self, current_price: float):
        """Update P&L calculation."""
        self.current_price = current_price
        if self.side == "yes":
            self.unrealized_pnl = (current_price - self.average_price) * self.size
        else:
            self.unrealized_pnl = (self.average_price - current_price) * self.size


@dataclass
class RiskLimits:
    """Risk management limits."""
    max_position_size: int = 1000
    max_order_size: int = 100
    max_daily_loss: float = 1000.0
    max_daily_trades: int = 100
    max_open_positions: int = 10
    position_limit_per_market: int = 500
    stop_loss_percentage: float = 0.10  # 10%
    take_profit_percentage: float = 0.20  # 20%


@dataclass
class TradingStats:
    """Trading statistics."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    daily_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: Optional[float] = None
    win_rate: Optional[float] = None
    average_win: Optional[float] = None
    average_loss: Optional[float] = None
    
    def update(self):
        """Update computed statistics."""
        if self.total_trades > 0:
            self.win_rate = self.winning_trades / self.total_trades
        
        # Additional calculations would go here


class RealTimeTradingEngine:
    """
    Real-time trading engine for automated market operations.
    
    Features:
    - Signal generation from multiple strategies
    - Order management and execution
    - Position tracking
    - Risk management
    - Performance analytics
    """
    
    def __init__(
        self,
        stream_manager: UnifiedStreamManager,
        kalshi_client: Optional[KalshiClient] = None,
        risk_limits: Optional[RiskLimits] = None
    ):
        """
        Initialize trading engine.
        
        Args:
            stream_manager: Unified stream manager
            kalshi_client: Kalshi client for order execution
            risk_limits: Risk management limits
        """
        self.stream_manager = stream_manager
        self.kalshi_client = kalshi_client or KalshiClient()
        self.risk_limits = risk_limits or RiskLimits()
        
        # Trading state
        self._running = False
        self._positions: Dict[str, Position] = {}
        self._orders: Dict[str, Order] = {}
        self._signals: deque = deque(maxlen=1000)
        
        # Statistics
        self.stats = TradingStats()
        self._pnl_history: List[float] = []
        self._daily_trades = 0
        self._daily_loss = 0.0
        self._last_reset = datetime.utcnow()
        
        # Strategy functions
        self._strategies: List[Callable] = []
        
        # Event handlers
        self._signal_handlers: List[Callable] = []
        self._order_handlers: List[Callable] = []
        
        # Tasks
        self._tasks: List[asyncio.Task] = []
        
        # Circuit breaker
        self._circuit_breaker_active = False
        
        logger.info("Real-time trading engine initialized")
    
    async def start(self):
        """Start trading engine."""
        if self._running:
            logger.warning("Trading engine already running")
            return
        
        logger.info("Starting trading engine")
        self._running = True
        
        # Register stream event handlers
        self.stream_manager.on(EventType.PRICE_UPDATE, self._handle_price_update)
        self.stream_manager.on(EventType.ARBITRAGE_OPPORTUNITY, self._handle_arbitrage)
        self.stream_manager.on(EventType.DIVERGENCE_DETECTED, self._handle_divergence)
        
        # Start background tasks
        self._tasks.append(asyncio.create_task(self._signal_processing_loop()))
        self._tasks.append(asyncio.create_task(self._risk_monitoring_loop()))
        self._tasks.append(asyncio.create_task(self._position_monitoring_loop()))
        self._tasks.append(asyncio.create_task(self._daily_reset_loop()))
        
        logger.info("Trading engine started")
    
    async def stop(self):
        """Stop trading engine."""
        if not self._running:
            return
        
        logger.info("Stopping trading engine")
        self._running = False
        
        # Cancel all pending orders
        await self._cancel_all_orders()
        
        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
        
        logger.info("Trading engine stopped")
    
    # Strategy registration
    
    def add_strategy(self, strategy_func: Callable):
        """
        Add trading strategy.
        
        Args:
            strategy_func: Strategy function that generates signals
        """
        self._strategies.append(strategy_func)
        logger.info(f"Added strategy: {strategy_func.__name__}")
    
    # Event handlers
    
    async def _handle_price_update(self, event: Dict[str, Any]):
        """Handle price update event."""
        if not self._running or self._circuit_breaker_active:
            return
        
        ticker = event.get("ticker")
        market_data = event.get("data")
        
        # Update positions with current price
        if ticker in self._positions:
            position = self._positions[ticker]
            if market_data.kalshi_yes_price:
                position.update_pnl(market_data.kalshi_yes_price)
        
        # Run strategies
        for strategy in self._strategies:
            try:
                signal = await self._run_strategy(strategy, market_data)
                if signal:
                    await self._process_signal(signal)
            except Exception as e:
                logger.error(f"Strategy error: {e}")
    
    async def _handle_arbitrage(self, event: Dict[str, Any]):
        """Handle arbitrage opportunity."""
        if not self._running or self._circuit_breaker_active:
            return
        
        ticker = event.get("ticker")
        data = event.get("data")
        
        # Generate arbitrage signal
        signal = TradingSignal(
            signal_id=f"arb_{datetime.utcnow().timestamp()}",
            timestamp=datetime.utcnow(),
            market_ticker=ticker,
            signal_type=SignalType.BUY,
            confidence=0.9,
            reason="Arbitrage opportunity detected",
            metadata={"type": "arbitrage", "divergence": data.divergence_score}
        )
        
        await self._process_signal(signal)
    
    async def _handle_divergence(self, event: Dict[str, Any]):
        """Handle price divergence."""
        ticker = event.get("ticker")
        divergence = event.get("divergence")
        
        logger.info(f"Divergence detected for {ticker}: {divergence:.2%}")
    
    # Signal processing
    
    async def _run_strategy(
        self,
        strategy: Callable,
        market_data: UnifiedMarketData
    ) -> Optional[TradingSignal]:
        """
        Run strategy to generate signal.
        
        Args:
            strategy: Strategy function
            market_data: Current market data
            
        Returns:
            Trading signal or None
        """
        try:
            if asyncio.iscoroutinefunction(strategy):
                return await strategy(market_data, self)
            else:
                return strategy(market_data, self)
        except Exception as e:
            logger.error(f"Strategy execution error: {e}")
            return None
    
    async def _process_signal(self, signal: TradingSignal):
        """Process trading signal."""
        # Store signal
        self._signals.append(signal)
        
        # Emit to handlers
        for handler in self._signal_handlers:
            try:
                await handler(signal)
            except Exception as e:
                logger.error(f"Signal handler error: {e}")
        
        # Check risk limits
        if not self._check_risk_limits(signal):
            logger.warning(f"Signal rejected by risk limits: {signal.signal_id}")
            return
        
        # Execute signal
        await self._execute_signal(signal)
    
    async def _execute_signal(self, signal: TradingSignal):
        """Execute trading signal."""
        if signal.signal_type == SignalType.BUY:
            await self._place_buy_order(signal)
        elif signal.signal_type == SignalType.SELL:
            await self._place_sell_order(signal)
        elif signal.signal_type == SignalType.CLOSE:
            await self._close_position(signal.market_ticker)
    
    # Order management
    
    async def _place_buy_order(self, signal: TradingSignal):
        """Place buy order."""
        # Determine order size
        size = signal.size or self._calculate_position_size(signal)
        
        # Create order
        order = Order(
            order_id=f"order_{datetime.utcnow().timestamp()}",
            signal_id=signal.signal_id,
            market_ticker=signal.market_ticker,
            side="yes",  # Buying yes shares
            order_type="limit" if signal.price_limit else "market",
            size=size,
            price=signal.price_limit,
            status=OrderStatus.PENDING,
            created_at=datetime.utcnow()
        )
        
        # Submit order
        await self._submit_order(order)
    
    async def _place_sell_order(self, signal: TradingSignal):
        """Place sell order."""
        position = self._positions.get(signal.market_ticker)
        if not position:
            logger.warning(f"No position to sell: {signal.market_ticker}")
            return
        
        # Create order
        order = Order(
            order_id=f"order_{datetime.utcnow().timestamp()}",
            signal_id=signal.signal_id,
            market_ticker=signal.market_ticker,
            side="no" if position.side == "yes" else "yes",  # Opposite side to close
            order_type="limit" if signal.price_limit else "market",
            size=position.size,
            price=signal.price_limit,
            status=OrderStatus.PENDING,
            created_at=datetime.utcnow()
        )
        
        # Submit order
        await self._submit_order(order)
    
    async def _submit_order(self, order: Order):
        """Submit order to exchange."""
        try:
            # Store order
            self._orders[order.order_id] = order
            
            # Submit to Kalshi (would implement actual API call)
            # response = await self.kalshi_client.place_order(...)
            
            # For now, simulate order fill
            order.status = OrderStatus.FILLED
            order.filled_size = order.size
            order.average_fill_price = order.price or 0.5
            
            # Update position
            await self._update_position(order)
            
            # Update stats
            self._daily_trades += 1
            self.stats.total_trades += 1
            
            # Emit to handlers
            for handler in self._order_handlers:
                await handler(order)
            
            logger.info(f"Order submitted: {order.order_id}")
            
        except Exception as e:
            logger.error(f"Order submission error: {e}")
            order.status = OrderStatus.REJECTED
    
    async def _update_position(self, order: Order):
        """Update position after order fill."""
        if order.status != OrderStatus.FILLED:
            return
        
        ticker = order.market_ticker
        
        if ticker in self._positions:
            # Update existing position
            position = self._positions[ticker]
            if order.side == position.side:
                # Adding to position
                total_cost = (position.average_price * position.size + 
                            order.average_fill_price * order.filled_size)
                position.size += order.filled_size
                position.average_price = total_cost / position.size
            else:
                # Reducing position
                position.size -= order.filled_size
                if position.size <= 0:
                    # Position closed
                    self._positions.pop(ticker)
                    # Calculate realized P&L
                    pnl = (order.average_fill_price - position.average_price) * order.filled_size
                    self.stats.total_pnl += pnl
                    self._daily_loss -= pnl  # Negative because loss is positive
        else:
            # New position
            self._positions[ticker] = Position(
                market_ticker=ticker,
                side=order.side,
                size=order.filled_size,
                average_price=order.average_fill_price
            )
    
    async def _close_position(self, ticker: str):
        """Close position for market."""
        position = self._positions.get(ticker)
        if not position:
            return
        
        signal = TradingSignal(
            signal_id=f"close_{datetime.utcnow().timestamp()}",
            timestamp=datetime.utcnow(),
            market_ticker=ticker,
            signal_type=SignalType.SELL,
            confidence=1.0,
            reason="Position close requested"
        )
        
        await self._place_sell_order(signal)
    
    async def _cancel_all_orders(self):
        """Cancel all pending orders."""
        for order_id, order in self._orders.items():
            if order.status == OrderStatus.PENDING:
                order.status = OrderStatus.CANCELLED
                logger.info(f"Cancelled order: {order_id}")
    
    # Risk management
    
    def _check_risk_limits(self, signal: TradingSignal) -> bool:
        """Check if signal passes risk limits."""
        # Check circuit breaker
        if self._circuit_breaker_active:
            return False
        
        # Check daily trade limit
        if self._daily_trades >= self.risk_limits.max_daily_trades:
            logger.warning("Daily trade limit reached")
            return False
        
        # Check daily loss limit
        if self._daily_loss >= self.risk_limits.max_daily_loss:
            logger.warning("Daily loss limit reached")
            self._activate_circuit_breaker()
            return False
        
        # Check position limits
        if len(self._positions) >= self.risk_limits.max_open_positions:
            logger.warning("Max open positions reached")
            return False
        
        # Check position size for market
        if signal.market_ticker in self._positions:
            position = self._positions[signal.market_ticker]
            if position.size >= self.risk_limits.position_limit_per_market:
                logger.warning(f"Position limit reached for {signal.market_ticker}")
                return False
        
        return True
    
    def _calculate_position_size(self, signal: TradingSignal) -> int:
        """Calculate appropriate position size."""
        # Kelly Criterion or fixed fractional sizing could be implemented
        base_size = min(100, self.risk_limits.max_order_size)
        
        # Adjust by confidence
        size = int(base_size * signal.confidence)
        
        return max(1, size)
    
    def _activate_circuit_breaker(self):
        """Activate circuit breaker to stop trading."""
        logger.warning("Circuit breaker activated!")
        self._circuit_breaker_active = True
    
    async def _risk_monitoring_loop(self):
        """Monitor risk metrics continuously."""
        while self._running:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                
                # Calculate current risk metrics
                total_exposure = sum(p.size * p.average_price for p in self._positions.values())
                
                # Check drawdown
                if self._pnl_history:
                    peak = max(self._pnl_history)
                    current = self.stats.total_pnl
                    drawdown = (peak - current) / peak if peak > 0 else 0
                    
                    if drawdown > 0.20:  # 20% drawdown
                        logger.warning(f"High drawdown: {drawdown:.1%}")
                        self._activate_circuit_breaker()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Risk monitoring error: {e}")
    
    async def _position_monitoring_loop(self):
        """Monitor positions for stop-loss and take-profit."""
        while self._running:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds
                
                for ticker, position in list(self._positions.items()):
                    if not position.current_price:
                        continue
                    
                    # Calculate P&L percentage
                    pnl_pct = position.unrealized_pnl / (position.average_price * position.size)
                    
                    # Check stop-loss
                    if pnl_pct <= -self.risk_limits.stop_loss_percentage:
                        logger.info(f"Stop-loss triggered for {ticker}")
                        await self._close_position(ticker)
                    
                    # Check take-profit
                    elif pnl_pct >= self.risk_limits.take_profit_percentage:
                        logger.info(f"Take-profit triggered for {ticker}")
                        await self._close_position(ticker)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Position monitoring error: {e}")
    
    async def _daily_reset_loop(self):
        """Reset daily counters."""
        while self._running:
            try:
                # Wait until next day
                now = datetime.utcnow()
                tomorrow = now.replace(hour=0, minute=0, second=0) + timedelta(days=1)
                wait_seconds = (tomorrow - now).total_seconds()
                await asyncio.sleep(wait_seconds)
                
                # Reset daily counters
                self._daily_trades = 0
                self._daily_loss = 0.0
                self.stats.daily_pnl = 0.0
                self._circuit_breaker_active = False
                self._last_reset = datetime.utcnow()
                
                logger.info("Daily counters reset")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Daily reset error: {e}")
    
    async def _signal_processing_loop(self):
        """Process queued signals."""
        # This could be used for batch processing or delayed execution
        while self._running:
            try:
                await asyncio.sleep(1)
                # Process any queued signals
            except asyncio.CancelledError:
                break
    
    # Public methods
    
    def get_positions(self) -> Dict[str, Position]:
        """Get current positions."""
        return self._positions.copy()
    
    def get_position(self, ticker: str) -> Optional[Position]:
        """Get position for specific market."""
        return self._positions.get(ticker)
    
    def get_stats(self) -> TradingStats:
        """Get trading statistics."""
        self.stats.update()
        return self.stats
    
    def get_recent_signals(self, limit: int = 10) -> List[TradingSignal]:
        """Get recent trading signals."""
        return list(self._signals)[-limit:]
    
    def on_signal(self, handler: Callable):
        """Register signal handler."""
        self._signal_handlers.append(handler)
    
    def on_order(self, handler: Callable):
        """Register order handler."""
        self._order_handlers.append(handler)
    
    def is_running(self) -> bool:
        """Check if engine is running."""
        return self._running
    
    def reset_circuit_breaker(self):
        """Manually reset circuit breaker."""
        self._circuit_breaker_active = False
        logger.info("Circuit breaker reset")