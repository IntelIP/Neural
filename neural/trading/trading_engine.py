"""
Trading Engine

The main orchestrator that connects the analysis stack to the trading infrastructure,
converting signals into executable trades with comprehensive risk management.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid

from neural.strategy.base import BaseStrategy, Signal, SignalType, StrategyResult
from neural.analysis.base import AnalysisResult
from .kalshi_client import KalshiClient, KalshiConfig
from .websocket_manager import WebSocketManager
from .order_manager import OrderManager, Order, OrderSide, OrderAction, OrderType
from .position_tracker import PositionTracker, Position
from .risk_manager import TradingRiskManager, RiskRule
from .portfolio_manager import PortfolioManager

logger = logging.getLogger(__name__)


class TradingMode(Enum):
    """Trading modes."""
    LIVE = "live"           # Live trading with real money
    PAPER = "paper"         # Paper trading simulation
    BACKTEST = "backtest"   # Historical backtesting


class ExecutionMode(Enum):
    """Order execution modes."""
    MARKET = "market"       # Market orders for immediate execution
    LIMIT = "limit"         # Limit orders at specific prices
    ADAPTIVE = "adaptive"   # Adaptive execution based on conditions


@dataclass
class TradingConfig:
    """Trading engine configuration."""
    # Environment
    trading_mode: TradingMode = TradingMode.PAPER
    execution_mode: ExecutionMode = ExecutionMode.ADAPTIVE
    
    # Execution settings
    max_position_size: float = 0.10      # Max position as % of capital
    max_orders_per_minute: int = 10      # Rate limiting
    min_edge_threshold: float = 0.03     # Minimum edge to trade
    min_confidence_threshold: float = 0.6 # Minimum signal confidence
    
    # Risk management
    max_portfolio_exposure: float = 0.50  # Max total exposure
    stop_loss_enabled: bool = True
    take_profit_enabled: bool = True
    daily_loss_limit: float = 0.05       # 5% daily loss limit
    
    # Strategy management
    enable_multi_strategy: bool = True
    strategy_allocation_method: str = "equal_weight"  # or "risk_parity", "kelly"
    max_concurrent_strategies: int = 5
    
    # Execution optimization
    market_impact_threshold: float = 0.02  # Consider market impact above this size
    slice_large_orders: bool = True
    order_timeout_minutes: int = 30
    
    # Monitoring
    enable_performance_tracking: bool = True
    enable_real_time_alerts: bool = True
    log_all_decisions: bool = True


@dataclass
class TradeDecision:
    """Represents a trading decision made by the engine."""
    decision_id: str
    timestamp: datetime
    signal: Signal
    strategy_name: str
    
    # Decision details
    action: str  # "BUY", "SELL", "HOLD", "CLOSE"
    ticker: str
    side: Optional[OrderSide] = None
    size: int = 0  # Number of contracts
    price: Optional[int] = None  # Price in cents
    
    # Reasoning
    edge: float = 0.0
    confidence: float = 0.0
    risk_score: float = 0.0
    decision_reason: str = ""
    
    # Execution
    approved: bool = False
    order_id: Optional[str] = None
    rejection_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "decision_id": self.decision_id,
            "timestamp": self.timestamp.isoformat(),
            "signal": self.signal.to_dict(),
            "strategy_name": self.strategy_name,
            "action": self.action,
            "ticker": self.ticker,
            "side": self.side.value if self.side else None,
            "size": self.size,
            "price": self.price,
            "edge": self.edge,
            "confidence": self.confidence,
            "risk_score": self.risk_score,
            "decision_reason": self.decision_reason,
            "approved": self.approved,
            "order_id": self.order_id,
            "rejection_reason": self.rejection_reason
        }


class TradingEngine:
    """
    Main trading engine that orchestrates all trading operations.
    
    The engine:
    1. Receives signals from strategies
    2. Applies risk management and filtering
    3. Converts signals to executable orders
    4. Manages position lifecycle
    5. Provides performance monitoring
    """
    
    def __init__(
        self,
        config: Optional[TradingConfig] = None,
        kalshi_config: Optional[KalshiConfig] = None
    ):
        """
        Initialize trading engine.
        
        Args:
            config: Trading engine configuration
            kalshi_config: Kalshi API configuration
        """
        self.config = config or TradingConfig()
        self.kalshi_config = kalshi_config or KalshiConfig()
        
        # Core components (will be initialized in start())
        self.kalshi_client: Optional[KalshiClient] = None
        self.websocket_manager: Optional[WebSocketManager] = None
        self.order_manager: Optional[OrderManager] = None
        self.position_tracker: Optional[PositionTracker] = None
        self.risk_manager: Optional[TradingRiskManager] = None
        self.portfolio_manager: Optional[PortfolioManager] = None
        
        # Strategy management
        self.strategies: Dict[str, BaseStrategy] = {}
        self.strategy_allocations: Dict[str, float] = {}
        
        # State tracking
        self.running = False
        self.total_capital = 100000.0  # Default capital
        self.available_capital = self.total_capital
        
        # Decision tracking
        self.decisions: List[TradeDecision] = []
        self.recent_orders: List[str] = []
        self.order_count_last_minute = 0
        self.last_order_time = datetime.now(timezone.utc)
        
        # Event handlers
        self.decision_handlers: List[Callable[[TradeDecision], None]] = []
        self.trade_handlers: List[Callable[[Order], None]] = []
        
        # Performance tracking
        self.start_time: Optional[datetime] = None
        self.total_signals_processed = 0
        self.total_trades_executed = 0
        self.daily_pnl = 0.0
        
        logger.info(f"TradingEngine initialized in {self.config.trading_mode.value} mode")
    
    async def start(self):
        """Start the trading engine."""
        if self.running:
            logger.warning("Trading engine is already running")
            return
        
        logger.info("Starting trading engine...")
        
        try:
            # Initialize Kalshi client
            self.kalshi_client = KalshiClient(self.kalshi_config)
            await self.kalshi_client.connect()
            
            # Initialize WebSocket manager
            self.websocket_manager = WebSocketManager(self.kalshi_client)
            await self.websocket_manager.connect()
            
            # Initialize order manager
            self.order_manager = OrderManager(
                self.kalshi_client,
                self.websocket_manager
            )
            
            # Initialize position tracker
            self.position_tracker = PositionTracker(
                self.kalshi_client,
                self.order_manager
            )
            await self.position_tracker.start()
            
            # Initialize risk manager
            self.risk_manager = TradingRiskManager(
                initial_capital=self.total_capital,
                config=self.config
            )
            
            # Initialize portfolio manager
            self.portfolio_manager = PortfolioManager(
                self.position_tracker,
                self.risk_manager
            )
            
            # Set up event handlers
            self._setup_event_handlers()
            
            # Subscribe to necessary WebSocket feeds
            if self.websocket_manager.is_connected():
                await self.websocket_manager.subscribe_orders()
                await self.websocket_manager.subscribe_fills()
            
            self.running = True
            self.start_time = datetime.now(timezone.utc)
            
            logger.info("✅ Trading engine started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start trading engine: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the trading engine."""
        if not self.running:
            return
        
        logger.info("Stopping trading engine...")
        
        self.running = False
        
        # Stop components
        if self.position_tracker:
            await self.position_tracker.stop()
        
        if self.websocket_manager:
            await self.websocket_manager.disconnect()
        
        if self.kalshi_client:
            await self.kalshi_client.disconnect()
        
        # Cancel any pending orders if in live mode
        if self.config.trading_mode == TradingMode.LIVE and self.order_manager:
            await self._cancel_all_pending_orders()
        
        logger.info("✅ Trading engine stopped")
    
    def _setup_event_handlers(self):
        """Setup event handlers for components."""
        if self.order_manager:
            self.order_manager.add_order_handler(self._handle_order_update)
            self.order_manager.add_fill_handler(self._handle_fill_update)
        
        if self.position_tracker:
            self.position_tracker.add_position_handler(self._handle_position_update)
    
    async def _handle_order_update(self, order: Order):
        """Handle order status updates."""
        logger.info(f"Order update: {order.order_id} -> {order.status.value}")
        
        # Notify trade handlers
        for handler in self.trade_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(order)
                else:
                    handler(order)
            except Exception as e:
                logger.error(f"Error in trade handler: {e}")
    
    async def _handle_fill_update(self, fill):
        """Handle trade fill updates."""
        logger.info(f"Fill update: {fill.ticker} {fill.count}@{fill.price}")
        self.total_trades_executed += 1
    
    async def _handle_position_update(self, position: Position):
        """Handle position updates."""
        logger.debug(f"Position update: {position.ticker} P&L={position.total_pnl:.2f}")
        
        # Update daily P&L
        self._update_daily_pnl()
        
        # Check risk limits
        if self.risk_manager:
            portfolio_state = await self._get_portfolio_state()
            violations = self.risk_manager.check_all_limits(portfolio_state)
            
            if violations[1]:  # If there are violations
                logger.warning(f"Risk violations detected: {len(violations[1])}")
                await self._handle_risk_violations(violations[1])
    
    # Strategy Management
    
    def add_strategy(
        self, 
        strategy: BaseStrategy, 
        allocation: float = 1.0,
        max_position_size: Optional[float] = None
    ):
        """
        Add a trading strategy.
        
        Args:
            strategy: Strategy instance
            allocation: Capital allocation (0.0 to 1.0)
            max_position_size: Override max position size for this strategy
        """
        if strategy.name in self.strategies:
            logger.warning(f"Strategy {strategy.name} already exists, replacing")
        
        self.strategies[strategy.name] = strategy
        self.strategy_allocations[strategy.name] = allocation
        
        # Set strategy-specific limits if provided
        if max_position_size:
            strategy.config.max_position_size = max_position_size
        
        logger.info(f"Added strategy: {strategy.name} (allocation: {allocation:.1%})")
    
    def remove_strategy(self, strategy_name: str):
        """Remove a strategy."""
        if strategy_name in self.strategies:
            del self.strategies[strategy_name]
            del self.strategy_allocations[strategy_name]
            logger.info(f"Removed strategy: {strategy_name}")
    
    async def process_signal(self, signal: Signal, strategy_name: str) -> TradeDecision:
        """
        Process a trading signal and make a decision.
        
        Args:
            signal: Trading signal from strategy
            strategy_name: Name of the strategy that generated the signal
            
        Returns:
            Trading decision
        """
        decision_id = str(uuid.uuid4())
        decision = TradeDecision(
            decision_id=decision_id,
            timestamp=datetime.now(timezone.utc),
            signal=signal,
            strategy_name=strategy_name,
            action="HOLD",  # Default to hold
            ticker=signal.market_id,
            edge=signal.edge,
            confidence=signal.confidence
        )
        
        try:
            self.total_signals_processed += 1
            
            # Step 1: Basic signal validation
            if not self._validate_signal(signal, decision):
                decision.decision_reason = "Signal validation failed"
                self.decisions.append(decision)
                await self._notify_decision_handlers(decision)
                return decision
            
            # Step 2: Rate limiting check
            if not self._check_rate_limits(decision):
                decision.decision_reason = "Rate limit exceeded"
                self.decisions.append(decision)
                await self._notify_decision_handlers(decision)
                return decision
            
            # Step 3: Convert signal to trading action
            self._convert_signal_to_action(signal, decision)
            
            # Step 4: Position sizing
            self._calculate_position_size(signal, decision)
            
            # Step 5: Risk management check
            if not await self._check_risk_limits(decision):
                decision.approved = False
                self.decisions.append(decision)
                await self._notify_decision_handlers(decision)
                return decision
            
            # Step 6: Execute if approved
            if decision.approved and decision.action != "HOLD":
                await self._execute_decision(decision)
            
            self.decisions.append(decision)
            await self._notify_decision_handlers(decision)
            
            if self.config.log_all_decisions:
                logger.info(f"Decision: {decision.action} {decision.ticker} "
                           f"size={decision.size} confidence={decision.confidence:.1%}")
            
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
            decision.rejection_reason = f"Processing error: {str(e)}"
        
        return decision
    
    def _validate_signal(self, signal: Signal, decision: TradeDecision) -> bool:
        """Validate signal meets basic requirements."""
        # Check minimum confidence
        if signal.confidence < self.config.min_confidence_threshold:
            decision.rejection_reason = f"Low confidence: {signal.confidence:.1%} < {self.config.min_confidence_threshold:.1%}"
            return False
        
        # Check minimum edge
        if signal.edge < self.config.min_edge_threshold:
            decision.rejection_reason = f"Low edge: {signal.edge:.1%} < {self.config.min_edge_threshold:.1%}"
            return False
        
        # Check if signal is actionable
        if signal.signal_type == SignalType.HOLD:
            decision.rejection_reason = "Signal is HOLD"
            return False
        
        return True
    
    def _check_rate_limits(self, decision: TradeDecision) -> bool:
        """Check if we're within rate limits."""
        now = datetime.now(timezone.utc)
        
        # Check orders per minute limit
        if (now - self.last_order_time).total_seconds() < 60:
            if self.order_count_last_minute >= self.config.max_orders_per_minute:
                decision.rejection_reason = "Rate limit exceeded"
                return False
        else:
            # Reset counter if minute has passed
            self.order_count_last_minute = 0
        
        return True
    
    def _convert_signal_to_action(self, signal: Signal, decision: TradeDecision):
        """Convert signal to specific trading action."""
        if signal.signal_type == SignalType.BUY_YES:
            decision.action = "BUY"
            decision.side = OrderSide.YES
        elif signal.signal_type == SignalType.BUY_NO:
            decision.action = "BUY"
            decision.side = OrderSide.NO
        elif signal.signal_type == SignalType.SELL_YES:
            decision.action = "SELL"
            decision.side = OrderSide.YES
        elif signal.signal_type == SignalType.SELL_NO:
            decision.action = "SELL"
            decision.side = OrderSide.NO
        else:
            decision.action = "HOLD"
    
    def _calculate_position_size(self, signal: Signal, decision: TradeDecision):
        """Calculate position size based on signal and risk management."""
        if decision.action == "HOLD":
            return
        
        # Get strategy allocation
        strategy_allocation = self.strategy_allocations.get(decision.strategy_name, 1.0)
        
        # Calculate base size from signal
        signal_size = signal.recommended_size
        
        # Apply strategy allocation
        allocated_size = signal_size * strategy_allocation
        
        # Apply maximum position size limit
        max_size = min(allocated_size, self.config.max_position_size)
        
        # Convert to contract count
        # This is simplified - in practice you'd get current market price
        estimated_price = 0.50  # Assume 50 cents average
        max_dollars = self.available_capital * max_size
        decision.size = int(max_dollars / estimated_price)
        
        # Set price based on execution mode
        if self.config.execution_mode == ExecutionMode.MARKET:
            decision.price = None  # Market order
        else:
            # For limit orders, use signal metadata or market data
            decision.price = int(estimated_price * 100)  # Convert to cents
    
    async def _check_risk_limits(self, decision: TradeDecision) -> bool:
        """Check if decision passes risk management."""
        if not self.risk_manager:
            decision.approved = True
            return True
        
        # Get current portfolio state
        portfolio_state = await self._get_portfolio_state()
        
        # Create a mock order for risk checking
        mock_signal = decision.signal
        
        # Check if trade is allowed
        allowed, reasons = self.risk_manager.check_trade_allowed(mock_signal, portfolio_state)
        
        if allowed:
            decision.approved = True
            decision.decision_reason = "Approved by risk management"
            return True
        else:
            decision.approved = False
            decision.rejection_reason = f"Risk management rejection: {', '.join(reasons)}"
            return False
    
    async def _execute_decision(self, decision: TradeDecision):
        """Execute approved trading decision."""
        if not self.order_manager:
            logger.error("Order manager not initialized")
            return
        
        try:
            # Determine order type
            order_type = OrderType.MARKET if self.config.execution_mode == ExecutionMode.MARKET else OrderType.LIMIT
            
            # Create and submit order
            order = await self.order_manager.create_and_submit_order(
                ticker=decision.ticker,
                side=decision.side,
                action=OrderAction.BUY if decision.action == "BUY" else OrderAction.SELL,
                count=decision.size,
                order_type=order_type,
                yes_price=decision.price if decision.side == OrderSide.YES else None,
                no_price=decision.price if decision.side == OrderSide.NO else None,
                strategy_id=decision.strategy_name
            )
            
            if order:
                decision.order_id = order.order_id
                self.order_count_last_minute += 1
                self.last_order_time = datetime.now(timezone.utc)
                logger.info(f"Order executed: {order.order_id}")
            else:
                decision.rejection_reason = "Order execution failed"
                
        except Exception as e:
            logger.error(f"Error executing decision: {e}")
            decision.rejection_reason = f"Execution error: {str(e)}"
    
    async def _get_portfolio_state(self) -> Dict[str, Any]:
        """Get current portfolio state for risk management."""
        portfolio_stats = {}
        
        if self.position_tracker:
            portfolio_stats = self.position_tracker.get_portfolio_stats()
        
        return {
            "total_capital": self.total_capital,
            "available_capital": self.available_capital,
            "daily_pnl": self.daily_pnl,
            "peak_capital": self.total_capital,  # Simplified
            "positions": {},  # Would include current positions
            **portfolio_stats
        }
    
    def _update_daily_pnl(self):
        """Update daily P&L from position tracker."""
        if self.position_tracker:
            self.daily_pnl = self.position_tracker.get_total_pnl()
    
    async def _handle_risk_violations(self, violations: List[Any]):
        """Handle risk limit violations."""
        logger.warning(f"Handling {len(violations)} risk violations")
        
        # In a real system, you might:
        # 1. Send alerts
        # 2. Reduce positions
        # 3. Stop trading temporarily
        # 4. Notify operators
        
        for violation in violations:
            logger.warning(f"Risk violation: {violation}")
    
    async def _cancel_all_pending_orders(self):
        """Cancel all pending orders (emergency stop)."""
        if not self.order_manager:
            return
        
        active_orders = self.order_manager.get_active_orders()
        logger.info(f"Cancelling {len(active_orders)} active orders")
        
        for order in active_orders:
            try:
                await self.order_manager.cancel_order(order.order_id)
            except Exception as e:
                logger.error(f"Error cancelling order {order.order_id}: {e}")
    
    # Event Handler Management
    
    async def _notify_decision_handlers(self, decision: TradeDecision):
        """Notify decision event handlers."""
        for handler in self.decision_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(decision)
                else:
                    handler(decision)
            except Exception as e:
                logger.error(f"Error in decision handler: {e}")
    
    def add_decision_handler(self, handler: Callable[[TradeDecision], None]):
        """Add decision event handler."""
        self.decision_handlers.append(handler)
    
    def add_trade_handler(self, handler: Callable[[Order], None]):
        """Add trade event handler."""
        self.trade_handlers.append(handler)
    
    # Status and Monitoring
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get comprehensive engine status."""
        return {
            "running": self.running,
            "trading_mode": self.config.trading_mode.value,
            "uptime_seconds": (datetime.now(timezone.utc) - self.start_time).total_seconds() if self.start_time else 0,
            "strategies": len(self.strategies),
            "total_capital": self.total_capital,
            "available_capital": self.available_capital,
            "daily_pnl": self.daily_pnl,
            "signals_processed": self.total_signals_processed,
            "trades_executed": self.total_trades_executed,
            "active_orders": len(self.order_manager.get_active_orders()) if self.order_manager else 0,
            "active_positions": len(self.position_tracker.get_active_positions()) if self.position_tracker else 0,
            "recent_decisions": len([d for d in self.decisions if (datetime.now(timezone.utc) - d.timestamp).total_seconds() < 3600])
        }
    
    def get_strategy_performance(self) -> Dict[str, Any]:
        """Get performance by strategy."""
        performance = {}
        
        if self.position_tracker:
            for strategy_name in self.strategies.keys():
                perf = self.position_tracker.get_strategy_performance(strategy_name)
                performance[strategy_name] = perf
        
        return performance
    
    # Context Manager Support
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()
