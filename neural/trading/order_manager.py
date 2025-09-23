"""
Order Management System

This module provides comprehensive order lifecycle management for Kalshi trading,
including order creation, tracking, modification, and execution monitoring.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

from .kalshi_client import KalshiClient, OrderRequest
from .websocket_manager import WebSocketManager, WebSocketMessage, SubscriptionType

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order status types."""
    PENDING = "pending"           # Order created but not yet sent
    SUBMITTED = "submitted"       # Order sent to exchange
    OPEN = "open"                # Order active on exchange
    PARTIALLY_FILLED = "partially_filled"  # Order partially executed
    FILLED = "filled"            # Order fully executed
    CANCELLED = "cancelled"      # Order cancelled
    REJECTED = "rejected"        # Order rejected by exchange
    EXPIRED = "expired"          # Order expired
    FAILED = "failed"            # Order failed to submit


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"


class OrderSide(Enum):
    """Order sides."""
    YES = "yes"
    NO = "no"


class OrderAction(Enum):
    """Order actions."""
    BUY = "buy"
    SELL = "sell"


@dataclass
class Fill:
    """Trade fill information."""
    fill_id: str
    order_id: str
    ticker: str
    side: str
    action: str
    count: int
    price: int  # Price in cents
    timestamp: datetime
    trade_id: Optional[str] = None
    
    @property
    def value(self) -> float:
        """Calculate fill value in dollars."""
        return (self.count * self.price) / 100.0


@dataclass
class Order:
    """Order representation."""
    order_id: str
    client_order_id: str
    ticker: str
    side: OrderSide
    action: OrderAction
    order_type: OrderType
    count: int
    status: OrderStatus = OrderStatus.PENDING
    
    # Pricing
    yes_price: Optional[int] = None
    no_price: Optional[int] = None
    buy_max_cost: Optional[int] = None
    
    # Execution details
    filled_count: int = 0
    remaining_count: int = field(init=False)
    avg_fill_price: Optional[float] = None
    total_fill_value: float = 0.0
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    submitted_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    # Metadata
    strategy_id: Optional[str] = None
    fills: List[Fill] = field(default_factory=list)
    rejection_reason: Optional[str] = None
    
    def __post_init__(self):
        """Initialize computed fields."""
        self.remaining_count = self.count
    
    @property
    def is_active(self) -> bool:
        """Check if order is still active."""
        return self.status in [OrderStatus.SUBMITTED, OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]
    
    @property
    def is_complete(self) -> bool:
        """Check if order is complete (filled or cancelled)."""
        return self.status in [
            OrderStatus.FILLED, OrderStatus.CANCELLED, 
            OrderStatus.REJECTED, OrderStatus.EXPIRED, OrderStatus.FAILED
        ]
    
    @property
    def fill_percentage(self) -> float:
        """Calculate fill percentage."""
        if self.count == 0:
            return 0.0
        return (self.filled_count / self.count) * 100.0
    
    def add_fill(self, fill: Fill):
        """Add a fill to the order."""
        self.fills.append(fill)
        self.filled_count += fill.count
        self.remaining_count = max(0, self.count - self.filled_count)
        self.total_fill_value += fill.value
        
        # Calculate average fill price
        if self.filled_count > 0:
            self.avg_fill_price = self.total_fill_value / self.filled_count
        
        # Update status based on fill
        if self.filled_count >= self.count:
            self.status = OrderStatus.FILLED
        elif self.filled_count > 0:
            self.status = OrderStatus.PARTIALLY_FILLED
        
        self.updated_at = datetime.now(timezone.utc)
        logger.info(f"Fill added to order {self.order_id}: {fill.count} @ {fill.price}")
    
    def update_status(self, new_status: OrderStatus, reason: Optional[str] = None):
        """Update order status."""
        old_status = self.status
        self.status = new_status
        self.updated_at = datetime.now(timezone.utc)
        
        if new_status == OrderStatus.REJECTED and reason:
            self.rejection_reason = reason
        
        logger.info(f"Order {self.order_id} status: {old_status.value} -> {new_status.value}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary."""
        return {
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "ticker": self.ticker,
            "side": self.side.value,
            "action": self.action.value,
            "type": self.order_type.value,
            "count": self.count,
            "status": self.status.value,
            "yes_price": self.yes_price,
            "no_price": self.no_price,
            "buy_max_cost": self.buy_max_cost,
            "filled_count": self.filled_count,
            "remaining_count": self.remaining_count,
            "avg_fill_price": self.avg_fill_price,
            "total_fill_value": self.total_fill_value,
            "fill_percentage": self.fill_percentage,
            "created_at": self.created_at.isoformat(),
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "strategy_id": self.strategy_id,
            "fills": [fill.__dict__ for fill in self.fills],
            "rejection_reason": self.rejection_reason
        }


class OrderManager:
    """
    Order lifecycle management system.
    
    Handles:
    - Order creation and validation
    - Order submission to exchange
    - Real-time order status tracking
    - Fill processing and aggregation
    - Order modification and cancellation
    """
    
    def __init__(
        self,
        kalshi_client: KalshiClient,
        websocket_manager: Optional[WebSocketManager] = None,
        enable_real_time_updates: bool = True
    ):
        """
        Initialize order manager.
        
        Args:
            kalshi_client: Authenticated Kalshi client
            websocket_manager: WebSocket manager for real-time updates
            enable_real_time_updates: Enable real-time order/fill tracking
        """
        self.client = kalshi_client
        self.websocket = websocket_manager
        self.enable_real_time_updates = enable_real_time_updates
        
        # Order tracking
        self.orders: Dict[str, Order] = {}  # order_id -> Order
        self.client_order_map: Dict[str, str] = {}  # client_order_id -> order_id
        
        # Event handlers
        self.order_handlers: List[Callable[[Order], None]] = []
        self.fill_handlers: List[Callable[[Fill], None]] = []
        
        # Statistics
        self.total_orders = 0
        self.total_fills = 0
        self.total_volume = 0.0
        
        logger.info("OrderManager initialized")
        
        # Set up real-time updates if WebSocket is available
        if self.websocket and self.enable_real_time_updates:
            self._setup_real_time_handlers()
    
    def _setup_real_time_handlers(self):
        """Setup WebSocket handlers for real-time updates."""
        if not self.websocket:
            return
        
        # Add handlers for order and fill updates
        self.websocket.add_handler(SubscriptionType.ORDERS, self._handle_order_update)
        self.websocket.add_handler(SubscriptionType.FILLS, self._handle_fill_update)
        
        logger.info("Real-time order/fill handlers configured")
    
    async def _handle_order_update(self, message: WebSocketMessage):
        """Handle real-time order updates."""
        try:
            order_data = message.msg
            order_id = order_data.get("order_id")
            
            if order_id and order_id in self.orders:
                order = self.orders[order_id]
                
                # Update order status
                new_status = order_data.get("status")
                if new_status:
                    status_enum = OrderStatus(new_status.lower())
                    reason = order_data.get("rejection_reason")
                    order.update_status(status_enum, reason)
                
                # Notify handlers
                await self._notify_order_handlers(order)
                
        except Exception as e:
            logger.error(f"Error handling order update: {e}")
    
    async def _handle_fill_update(self, message: WebSocketMessage):
        """Handle real-time fill updates."""
        try:
            fill_data = message.msg
            order_id = fill_data.get("order_id")
            
            if order_id and order_id in self.orders:
                # Create Fill object
                fill = Fill(
                    fill_id=fill_data.get("fill_id", str(uuid.uuid4())),
                    order_id=order_id,
                    ticker=fill_data.get("ticker", ""),
                    side=fill_data.get("side", ""),
                    action=fill_data.get("action", ""),
                    count=fill_data.get("count", 0),
                    price=fill_data.get("price", 0),
                    timestamp=datetime.now(timezone.utc),
                    trade_id=fill_data.get("trade_id")
                )
                
                # Add fill to order
                order = self.orders[order_id]
                order.add_fill(fill)
                
                # Update statistics
                self.total_fills += 1
                self.total_volume += fill.value
                
                # Notify handlers
                await self._notify_fill_handlers(fill)
                await self._notify_order_handlers(order)
                
        except Exception as e:
            logger.error(f"Error handling fill update: {e}")
    
    async def _notify_order_handlers(self, order: Order):
        """Notify all registered order handlers."""
        for handler in self.order_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(order)
                else:
                    handler(order)
            except Exception as e:
                logger.error(f"Error in order handler: {e}")
    
    async def _notify_fill_handlers(self, fill: Fill):
        """Notify all registered fill handlers."""
        for handler in self.fill_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(fill)
                else:
                    handler(fill)
            except Exception as e:
                logger.error(f"Error in fill handler: {e}")
    
    # Order Creation and Management
    
    def create_order(
        self,
        ticker: str,
        side: OrderSide,
        action: OrderAction,
        count: int,
        order_type: OrderType = OrderType.LIMIT,
        yes_price: Optional[int] = None,
        no_price: Optional[int] = None,
        buy_max_cost: Optional[int] = None,
        strategy_id: Optional[str] = None
    ) -> Order:
        """
        Create a new order (does not submit to exchange).
        
        Args:
            ticker: Market ticker
            side: Order side (YES or NO)
            action: Order action (BUY or SELL)
            count: Number of contracts
            order_type: Order type (MARKET or LIMIT)
            yes_price: YES side price in cents (1-99)
            no_price: NO side price in cents (1-99)
            buy_max_cost: Maximum cost for buy orders
            strategy_id: Associated strategy identifier
            
        Returns:
            Created Order object
        """
        # Generate IDs
        client_order_id = f"order_{uuid.uuid4().hex[:12]}_{int(datetime.now().timestamp())}"
        order_id = client_order_id  # Will be updated with exchange order ID when submitted
        
        # Validate order
        self._validate_order_params(ticker, side, action, count, order_type, yes_price, no_price)
        
        # Create order
        order = Order(
            order_id=order_id,
            client_order_id=client_order_id,
            ticker=ticker,
            side=side,
            action=action,
            order_type=order_type,
            count=count,
            yes_price=yes_price,
            no_price=no_price,
            buy_max_cost=buy_max_cost,
            strategy_id=strategy_id
        )
        
        # Store order
        self.orders[order_id] = order
        self.client_order_map[client_order_id] = order_id
        self.total_orders += 1
        
        logger.info(f"Order created: {ticker} {action.value} {count} {side.value} @ {yes_price or no_price}")
        return order
    
    def _validate_order_params(
        self, 
        ticker: str, 
        side: OrderSide, 
        action: OrderAction,
        count: int, 
        order_type: OrderType,
        yes_price: Optional[int],
        no_price: Optional[int]
    ):
        """Validate order parameters."""
        if not ticker:
            raise ValueError("Ticker is required")
        
        if count <= 0:
            raise ValueError("Count must be positive")
        
        if order_type == OrderType.LIMIT:
            if side == OrderSide.YES and yes_price is None:
                raise ValueError("YES price required for YES side limit orders")
            if side == OrderSide.NO and no_price is None:
                raise ValueError("NO price required for NO side limit orders")
            
            # Validate price ranges
            if yes_price is not None and not (1 <= yes_price <= 99):
                raise ValueError("YES price must be between 1 and 99 cents")
            if no_price is not None and not (1 <= no_price <= 99):
                raise ValueError("NO price must be between 1 and 99 cents")
    
    async def submit_order(self, order: Order) -> bool:
        """
        Submit order to exchange.
        
        Args:
            order: Order to submit
            
        Returns:
            True if submission successful
        """
        try:
            # Create OrderRequest
            request = OrderRequest(
                ticker=order.ticker,
                client_order_id=order.client_order_id,
                side=order.side.value,
                action=order.action.value,
                type=order.order_type.value,
                count=order.count,
                yes_price=order.yes_price,
                no_price=order.no_price,
                buy_max_cost=order.buy_max_cost
            )
            
            # Submit to exchange
            order.update_status(OrderStatus.SUBMITTED)
            response = await self.client.create_order(request)
            
            if response:
                # Update order with exchange response
                exchange_order_id = response.get("order_id")
                if exchange_order_id:
                    # Update order ID mapping
                    del self.orders[order.order_id]
                    order.order_id = exchange_order_id
                    self.orders[exchange_order_id] = order
                
                order.submitted_at = datetime.now(timezone.utc)
                order.update_status(OrderStatus.OPEN)
                
                # Subscribe to real-time updates if WebSocket available
                if self.websocket and self.websocket.is_connected():
                    await self.websocket.subscribe_orders()
                    await self.websocket.subscribe_fills()
                
                logger.info(f"Order submitted successfully: {order.order_id}")
                await self._notify_order_handlers(order)
                return True
            else:
                order.update_status(OrderStatus.FAILED, "No response from exchange")
                await self._notify_order_handlers(order)
                return False
                
        except Exception as e:
            logger.error(f"Error submitting order: {e}")
            order.update_status(OrderStatus.FAILED, str(e))
            await self._notify_order_handlers(order)
            return False
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an active order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancellation successful
        """
        if order_id not in self.orders:
            logger.error(f"Order not found: {order_id}")
            return False
        
        order = self.orders[order_id]
        
        if not order.is_active:
            logger.warning(f"Order {order_id} is not active (status: {order.status.value})")
            return False
        
        try:
            response = await self.client.cancel_order(order_id)
            
            if response:
                order.update_status(OrderStatus.CANCELLED)
                await self._notify_order_handlers(order)
                logger.info(f"Order cancelled: {order_id}")
                return True
            else:
                logger.error(f"Failed to cancel order: {order_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    async def modify_order(self, order_id: str, new_count: int) -> bool:
        """
        Modify order size (decrease only for most exchanges).
        
        Args:
            order_id: Order ID to modify
            new_count: New order count (must be less than current)
            
        Returns:
            True if modification successful
        """
        if order_id not in self.orders:
            logger.error(f"Order not found: {order_id}")
            return False
        
        order = self.orders[order_id]
        
        if not order.is_active:
            logger.warning(f"Order {order_id} is not active")
            return False
        
        if new_count >= order.remaining_count:
            logger.error("New count must be less than remaining count")
            return False
        
        try:
            reduce_by = order.remaining_count - new_count
            response = await self.client.decrease_order(order_id, reduce_by)
            
            if response:
                # Update order (this will be confirmed by WebSocket updates)
                logger.info(f"Order modified: {order_id} reduced by {reduce_by}")
                return True
            else:
                logger.error(f"Failed to modify order: {order_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error modifying order {order_id}: {e}")
            return False
    
    # Convenience Methods
    
    async def create_and_submit_order(
        self,
        ticker: str,
        side: OrderSide,
        action: OrderAction,
        count: int,
        order_type: OrderType = OrderType.LIMIT,
        yes_price: Optional[int] = None,
        no_price: Optional[int] = None,
        buy_max_cost: Optional[int] = None,
        strategy_id: Optional[str] = None
    ) -> Optional[Order]:
        """Create and immediately submit an order."""
        order = self.create_order(
            ticker=ticker,
            side=side,
            action=action,
            count=count,
            order_type=order_type,
            yes_price=yes_price,
            no_price=no_price,
            buy_max_cost=buy_max_cost,
            strategy_id=strategy_id
        )
        
        success = await self.submit_order(order)
        return order if success else None
    
    # Query Methods
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self.orders.get(order_id)
    
    def get_order_by_client_id(self, client_order_id: str) -> Optional[Order]:
        """Get order by client order ID."""
        order_id = self.client_order_map.get(client_order_id)
        return self.orders.get(order_id) if order_id else None
    
    def get_active_orders(self) -> List[Order]:
        """Get all active orders."""
        return [order for order in self.orders.values() if order.is_active]
    
    def get_orders_by_ticker(self, ticker: str) -> List[Order]:
        """Get all orders for a specific ticker."""
        return [order for order in self.orders.values() if order.ticker == ticker]
    
    def get_orders_by_strategy(self, strategy_id: str) -> List[Order]:
        """Get all orders for a specific strategy."""
        return [order for order in self.orders.values() if order.strategy_id == strategy_id]
    
    def get_orders_by_status(self, status: OrderStatus) -> List[Order]:
        """Get orders by status."""
        return [order for order in self.orders.values() if order.status == status]
    
    # Event Handler Registration
    
    def add_order_handler(self, handler: Callable[[Order], None]):
        """Add order event handler."""
        self.order_handlers.append(handler)
        logger.info("Order handler added")
    
    def add_fill_handler(self, handler: Callable[[Fill], None]):
        """Add fill event handler."""
        self.fill_handlers.append(handler)
        logger.info("Fill handler added")
    
    def remove_order_handler(self, handler: Callable[[Order], None]):
        """Remove order event handler."""
        if handler in self.order_handlers:
            self.order_handlers.remove(handler)
            logger.info("Order handler removed")
    
    def remove_fill_handler(self, handler: Callable[[Fill], None]):
        """Remove fill event handler."""
        if handler in self.fill_handlers:
            self.fill_handlers.remove(handler)
            logger.info("Fill handler removed")
    
    # Statistics and Monitoring
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get order management statistics."""
        active_orders = self.get_active_orders()
        
        return {
            "total_orders": self.total_orders,
            "active_orders": len(active_orders),
            "total_fills": self.total_fills,
            "total_volume": self.total_volume,
            "status_breakdown": self._get_status_breakdown(),
            "avg_fill_rate": self.total_fills / max(1, self.total_orders)
        }
    
    def _get_status_breakdown(self) -> Dict[str, int]:
        """Get breakdown of orders by status."""
        breakdown = {}
        for status in OrderStatus:
            breakdown[status.value] = len(self.get_orders_by_status(status))
        return breakdown
    
    async def sync_with_exchange(self):
        """Sync local order state with exchange."""
        try:
            # Get orders from exchange
            response = await self.client.get_orders(limit=1000)
            
            if response and "orders" in response:
                for exchange_order in response["orders"]:
                    order_id = exchange_order.get("order_id")
                    
                    if order_id in self.orders:
                        # Update existing order
                        local_order = self.orders[order_id]
                        
                        # Sync status
                        exchange_status = exchange_order.get("status", "").lower()
                        if exchange_status:
                            try:
                                status_enum = OrderStatus(exchange_status)
                                if status_enum != local_order.status:
                                    local_order.update_status(status_enum)
                            except ValueError:
                                logger.warning(f"Unknown order status: {exchange_status}")
                
            logger.info("Order sync with exchange completed")
            
        except Exception as e:
            logger.error(f"Error syncing orders with exchange: {e}")
    
    async def cleanup(self):
        """Cleanup order manager resources."""
        logger.info("OrderManager cleanup initiated")
        
        # Cancel all active orders if requested
        active_orders = self.get_active_orders()
        if active_orders:
            logger.info(f"Cancelling {len(active_orders)} active orders")
            for order in active_orders:
                try:
                    await self.cancel_order(order.order_id)
                except Exception as e:
                    logger.error(f"Error cancelling order {order.order_id}: {e}")
        
        logger.info("OrderManager cleanup completed")
