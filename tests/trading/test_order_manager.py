"""
Unit tests for Order Management System

Tests cover order creation, lifecycle management, fill processing,
and real-time tracking capabilities.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone
import uuid

from neural.trading.order_manager import (
    OrderManager, Order, OrderStatus, OrderType, OrderSide, OrderAction, Fill
)
from neural.trading.kalshi_client import KalshiClient, KalshiConfig
from neural.trading.websocket_manager import WebSocketManager


class TestFill:
    """Test Fill data class."""
    
    def test_fill_creation(self):
        """Test creating a Fill object."""
        fill = Fill(
            fill_id="fill_123",
            order_id="order_123", 
            ticker="TEST-MARKET",
            side="yes",
            action="buy",
            count=50,
            price=52,
            timestamp=datetime.now(timezone.utc)
        )
        
        assert fill.fill_id == "fill_123"
        assert fill.count == 50
        assert fill.price == 52
        assert fill.value == 26.0  # (50 * 52) / 100
    
    def test_fill_value_calculation(self):
        """Test fill value calculation."""
        fill = Fill(
            fill_id="test",
            order_id="test",
            ticker="TEST",
            side="yes", 
            action="buy",
            count=100,
            price=48,
            timestamp=datetime.now(timezone.utc)
        )
        
        assert fill.value == 48.0  # (100 * 48) / 100


class TestOrder:
    """Test Order data class."""
    
    def test_order_creation(self):
        """Test creating an Order object."""
        order = Order(
            order_id="order_123",
            client_order_id="client_123", 
            ticker="TEST-MARKET",
            side=OrderSide.YES,
            action=OrderAction.BUY,
            order_type=OrderType.LIMIT,
            count=100,
            yes_price=50
        )
        
        assert order.order_id == "order_123"
        assert order.side == OrderSide.YES
        assert order.action == OrderAction.BUY
        assert order.count == 100
        assert order.remaining_count == 100
        assert order.status == OrderStatus.PENDING
    
    def test_order_properties(self):
        """Test order computed properties."""
        order = Order(
            order_id="test",
            client_order_id="test",
            ticker="TEST",
            side=OrderSide.YES,
            action=OrderAction.BUY,
            order_type=OrderType.LIMIT,
            count=100
        )
        
        # Test initial state
        assert order.is_active is False  # PENDING is not active
        assert order.is_complete is False
        assert order.fill_percentage == 0.0
        
        # Update to active status
        order.update_status(OrderStatus.OPEN)
        assert order.is_active is True
        assert order.is_complete is False
        
        # Update to complete status
        order.update_status(OrderStatus.FILLED)
        assert order.is_active is False
        assert order.is_complete is True
    
    def test_add_fill(self):
        """Test adding fills to order."""
        order = Order(
            order_id="order_123",
            client_order_id="client_123",
            ticker="TEST-MARKET",
            side=OrderSide.YES,
            action=OrderAction.BUY,
            order_type=OrderType.LIMIT,
            count=100
        )
        
        # Add partial fill
        fill1 = Fill(
            fill_id="fill_1",
            order_id="order_123",
            ticker="TEST-MARKET",
            side="yes",
            action="buy",
            count=30,
            price=50,
            timestamp=datetime.now(timezone.utc)
        )
        
        order.add_fill(fill1)
        
        assert order.filled_count == 30
        assert order.remaining_count == 70
        assert order.status == OrderStatus.PARTIALLY_FILLED
        assert order.fill_percentage == 30.0
        assert order.avg_fill_price == 15.0  # 15.0 / 30 = 0.5, but calculated as 15/30
        
        # Add completing fill
        fill2 = Fill(
            fill_id="fill_2",
            order_id="order_123", 
            ticker="TEST-MARKET",
            side="yes",
            action="buy",
            count=70,
            price=52,
            timestamp=datetime.now(timezone.utc)
        )
        
        order.add_fill(fill2)
        
        assert order.filled_count == 100
        assert order.remaining_count == 0
        assert order.status == OrderStatus.FILLED
        assert order.fill_percentage == 100.0
        assert len(order.fills) == 2
    
    def test_update_status(self):
        """Test updating order status."""
        order = Order(
            order_id="test",
            client_order_id="test",
            ticker="TEST",
            side=OrderSide.YES,
            action=OrderAction.BUY,
            order_type=OrderType.LIMIT,
            count=100
        )
        
        initial_time = order.created_at
        
        # Update status
        order.update_status(OrderStatus.SUBMITTED)
        
        assert order.status == OrderStatus.SUBMITTED
        assert order.updated_at > initial_time
        
        # Update with rejection reason
        order.update_status(OrderStatus.REJECTED, "Insufficient funds")
        
        assert order.status == OrderStatus.REJECTED
        assert order.rejection_reason == "Insufficient funds"
    
    def test_to_dict(self):
        """Test converting order to dictionary."""
        order = Order(
            order_id="order_123",
            client_order_id="client_123",
            ticker="TEST-MARKET", 
            side=OrderSide.YES,
            action=OrderAction.BUY,
            order_type=OrderType.LIMIT,
            count=100,
            yes_price=50,
            strategy_id="test_strategy"
        )
        
        order_dict = order.to_dict()
        
        assert order_dict["order_id"] == "order_123"
        assert order_dict["side"] == "yes"
        assert order_dict["action"] == "buy"
        assert order_dict["count"] == 100
        assert order_dict["yes_price"] == 50
        assert order_dict["strategy_id"] == "test_strategy"


class TestOrderManager:
    """Test OrderManager class."""
    
    @pytest.fixture
    def mock_kalshi_client(self):
        """Create mock Kalshi client."""
        client = Mock(spec=KalshiClient)
        client.authenticated = True
        return client
    
    @pytest.fixture
    def mock_websocket_manager(self):
        """Create mock WebSocket manager."""
        return Mock(spec=WebSocketManager)
    
    @pytest.fixture
    def order_manager(self, mock_kalshi_client, mock_websocket_manager):
        """Create OrderManager instance."""
        return OrderManager(
            mock_kalshi_client,
            mock_websocket_manager,
            enable_real_time_updates=True
        )
    
    def test_initialization(self, order_manager, mock_kalshi_client, mock_websocket_manager):
        """Test OrderManager initialization."""
        assert order_manager.client == mock_kalshi_client
        assert order_manager.websocket == mock_websocket_manager
        assert order_manager.enable_real_time_updates is True
        assert len(order_manager.orders) == 0
        assert order_manager.total_orders == 0
    
    def test_create_order(self, order_manager):
        """Test creating an order."""
        order = order_manager.create_order(
            ticker="TEST-MARKET",
            side=OrderSide.YES,
            action=OrderAction.BUY,
            count=100,
            order_type=OrderType.LIMIT,
            yes_price=50,
            strategy_id="test_strategy"
        )
        
        assert isinstance(order, Order)
        assert order.ticker == "TEST-MARKET"
        assert order.side == OrderSide.YES
        assert order.count == 100
        assert order.strategy_id == "test_strategy"
        
        # Check it was stored
        assert len(order_manager.orders) == 1
        assert order_manager.total_orders == 1
        assert order.order_id in order_manager.orders
        assert order.client_order_id in order_manager.client_order_map
    
    def test_create_order_validation_errors(self, order_manager):
        """Test order creation validation errors."""
        # Empty ticker
        with pytest.raises(ValueError, match="Ticker is required"):
            order_manager.create_order(
                ticker="",
                side=OrderSide.YES,
                action=OrderAction.BUY,
                count=100
            )
        
        # Zero count
        with pytest.raises(ValueError, match="Count must be positive"):
            order_manager.create_order(
                ticker="TEST",
                side=OrderSide.YES,
                action=OrderAction.BUY,
                count=0
            )
        
        # Limit order without price
        with pytest.raises(ValueError, match="YES price required"):
            order_manager.create_order(
                ticker="TEST",
                side=OrderSide.YES,
                action=OrderAction.BUY,
                count=100,
                order_type=OrderType.LIMIT
            )
        
        # Invalid price range
        with pytest.raises(ValueError, match="YES price must be between 1 and 99"):
            order_manager.create_order(
                ticker="TEST",
                side=OrderSide.YES,
                action=OrderAction.BUY,
                count=100,
                order_type=OrderType.LIMIT,
                yes_price=150
            )
    
    @pytest.mark.asyncio
    async def test_submit_order_success(self, order_manager, mock_kalshi_client):
        """Test successful order submission."""
        # Create order
        order = order_manager.create_order(
            ticker="TEST-MARKET",
            side=OrderSide.YES,
            action=OrderAction.BUY,
            count=100,
            order_type=OrderType.LIMIT,
            yes_price=50
        )
        
        # Mock successful submission
        mock_kalshi_client.create_order = AsyncMock(return_value={
            "order_id": "exchange_order_123",
            "status": "submitted"
        })
        
        result = await order_manager.submit_order(order)
        
        assert result is True
        assert order.status == OrderStatus.OPEN
        assert order.order_id == "exchange_order_123"
        assert order.submitted_at is not None
        
        # Check order ID mapping was updated
        assert "exchange_order_123" in order_manager.orders
        assert order_manager.orders["exchange_order_123"] == order
    
    @pytest.mark.asyncio
    async def test_submit_order_failure(self, order_manager, mock_kalshi_client):
        """Test order submission failure."""
        # Create order
        order = order_manager.create_order(
            ticker="TEST-MARKET",
            side=OrderSide.YES,
            action=OrderAction.BUY,
            count=100
        )
        
        # Mock failed submission
        mock_kalshi_client.create_order = AsyncMock(return_value=None)
        
        result = await order_manager.submit_order(order)
        
        assert result is False
        assert order.status == OrderStatus.FAILED
    
    @pytest.mark.asyncio 
    async def test_cancel_order_success(self, order_manager, mock_kalshi_client):
        """Test successful order cancellation."""
        # Create and add active order
        order = order_manager.create_order(
            ticker="TEST-MARKET",
            side=OrderSide.YES,
            action=OrderAction.BUY,
            count=100
        )
        order.update_status(OrderStatus.OPEN)
        
        # Mock successful cancellation
        mock_kalshi_client.cancel_order = AsyncMock(return_value={
            "order_id": order.order_id,
            "status": "cancelled"
        })
        
        result = await order_manager.cancel_order(order.order_id)
        
        assert result is True
        assert order.status == OrderStatus.CANCELLED
    
    @pytest.mark.asyncio
    async def test_cancel_nonexistent_order(self, order_manager):
        """Test cancelling non-existent order."""
        result = await order_manager.cancel_order("nonexistent_order")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_cancel_inactive_order(self, order_manager):
        """Test cancelling inactive order."""
        # Create completed order
        order = order_manager.create_order(
            ticker="TEST-MARKET",
            side=OrderSide.YES,
            action=OrderAction.BUY,
            count=100
        )
        order.update_status(OrderStatus.FILLED)
        
        result = await order_manager.cancel_order(order.order_id)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_modify_order_success(self, order_manager, mock_kalshi_client):
        """Test successful order modification."""
        # Create active order
        order = order_manager.create_order(
            ticker="TEST-MARKET",
            side=OrderSide.YES,
            action=OrderAction.BUY,
            count=100
        )
        order.update_status(OrderStatus.OPEN)
        
        # Mock successful modification
        mock_kalshi_client.decrease_order = AsyncMock(return_value={
            "order_id": order.order_id,
            "reduce_by": 20
        })
        
        result = await order_manager.modify_order(order.order_id, 80)
        
        assert result is True
        mock_kalshi_client.decrease_order.assert_called_once_with(order.order_id, 20)
    
    @pytest.mark.asyncio
    async def test_create_and_submit_order(self, order_manager, mock_kalshi_client):
        """Test create and submit order convenience method."""
        # Mock successful submission
        mock_kalshi_client.create_order = AsyncMock(return_value={
            "order_id": "exchange_order_123",
            "status": "submitted"
        })
        
        order = await order_manager.create_and_submit_order(
            ticker="TEST-MARKET",
            side=OrderSide.YES,
            action=OrderAction.BUY,
            count=100,
            order_type=OrderType.LIMIT,
            yes_price=50
        )
        
        assert order is not None
        assert order.status == OrderStatus.OPEN
        assert order.order_id == "exchange_order_123"
    
    def test_get_order_methods(self, order_manager):
        """Test various order retrieval methods."""
        # Create test orders
        order1 = order_manager.create_order(
            ticker="TEST1", side=OrderSide.YES, action=OrderAction.BUY, count=100,
            strategy_id="strategy1"
        )
        order1.update_status(OrderStatus.OPEN)
        
        order2 = order_manager.create_order(
            ticker="TEST1", side=OrderSide.NO, action=OrderAction.SELL, count=50,
            strategy_id="strategy2"
        )
        order2.update_status(OrderStatus.FILLED)
        
        order3 = order_manager.create_order(
            ticker="TEST2", side=OrderSide.YES, action=OrderAction.BUY, count=75,
            strategy_id="strategy1"
        )
        order3.update_status(OrderStatus.CANCELLED)
        
        # Test get_order
        retrieved = order_manager.get_order(order1.order_id)
        assert retrieved == order1
        
        # Test get_order_by_client_id
        retrieved = order_manager.get_order_by_client_id(order2.client_order_id)
        assert retrieved == order2
        
        # Test get_active_orders
        active = order_manager.get_active_orders()
        assert len(active) == 1
        assert active[0] == order1
        
        # Test get_orders_by_ticker
        test1_orders = order_manager.get_orders_by_ticker("TEST1")
        assert len(test1_orders) == 2
        
        # Test get_orders_by_strategy
        strategy1_orders = order_manager.get_orders_by_strategy("strategy1")
        assert len(strategy1_orders) == 2
        
        # Test get_orders_by_status
        filled_orders = order_manager.get_orders_by_status(OrderStatus.FILLED)
        assert len(filled_orders) == 1
        assert filled_orders[0] == order2
    
    def test_event_handlers(self, order_manager):
        """Test event handler management."""
        order_handler_calls = []
        fill_handler_calls = []
        
        def order_handler(order):
            order_handler_calls.append(order)
        
        def fill_handler(fill):
            fill_handler_calls.append(fill)
        
        # Add handlers
        order_manager.add_order_handler(order_handler)
        order_manager.add_fill_handler(fill_handler)
        
        assert len(order_manager.order_handlers) == 1
        assert len(order_manager.fill_handlers) == 1
        
        # Remove handlers
        order_manager.remove_order_handler(order_handler)
        order_manager.remove_fill_handler(fill_handler)
        
        assert len(order_manager.order_handlers) == 0
        assert len(order_manager.fill_handlers) == 0
    
    def test_statistics(self, order_manager):
        """Test order statistics."""
        # Create orders with different statuses
        order1 = order_manager.create_order(
            ticker="TEST1", side=OrderSide.YES, action=OrderAction.BUY, count=100
        )
        order1.update_status(OrderStatus.FILLED)
        
        order2 = order_manager.create_order(
            ticker="TEST2", side=OrderSide.NO, action=OrderAction.BUY, count=50
        )
        order2.update_status(OrderStatus.OPEN)
        
        order3 = order_manager.create_order(
            ticker="TEST3", side=OrderSide.YES, action=OrderAction.SELL, count=25
        )
        order3.update_status(OrderStatus.CANCELLED)
        
        # Add fills to increase stats
        order_manager.total_fills = 5
        order_manager.total_volume = 1000.0
        
        stats = order_manager.get_statistics()
        
        assert stats["total_orders"] == 3
        assert stats["active_orders"] == 1
        assert stats["total_fills"] == 5
        assert stats["total_volume"] == 1000.0
        assert stats["avg_fill_rate"] == 5/3  # fills/orders
        assert "status_breakdown" in stats
        assert stats["status_breakdown"]["filled"] == 1
        assert stats["status_breakdown"]["open"] == 1
        assert stats["status_breakdown"]["cancelled"] == 1
    
    @pytest.mark.asyncio
    async def test_handle_order_update(self, order_manager):
        """Test handling order updates from WebSocket."""
        # Create order
        order = order_manager.create_order(
            ticker="TEST", side=OrderSide.YES, action=OrderAction.BUY, count=100
        )
        
        # Mock WebSocket message
        from neural.trading.websocket_manager import WebSocketMessage
        message = WebSocketMessage(
            id="msg_1",
            type="order",
            seq=1,
            msg={
                "order_id": order.order_id,
                "status": "filled"
            },
            timestamp=datetime.now()
        )
        
        await order_manager._handle_order_update(message)
        
        assert order.status == OrderStatus.FILLED
    
    @pytest.mark.asyncio
    async def test_handle_fill_update(self, order_manager):
        """Test handling fill updates from WebSocket."""
        # Create order
        order = order_manager.create_order(
            ticker="TEST-MARKET", side=OrderSide.YES, action=OrderAction.BUY, count=100
        )
        
        # Mock WebSocket message
        from neural.trading.websocket_manager import WebSocketMessage
        message = WebSocketMessage(
            id="msg_1", 
            type="fill",
            seq=1,
            msg={
                "fill_id": "fill_123",
                "order_id": order.order_id,
                "ticker": "TEST-MARKET",
                "side": "yes",
                "action": "buy", 
                "count": 50,
                "price": 52,
                "trade_id": "trade_123"
            },
            timestamp=datetime.now()
        )
        
        await order_manager._handle_fill_update(message)
        
        assert order.filled_count == 50
        assert order.status == OrderStatus.PARTIALLY_FILLED
        assert len(order.fills) == 1
        assert order_manager.total_fills == 1
    
    @pytest.mark.asyncio
    async def test_sync_with_exchange(self, order_manager, mock_kalshi_client):
        """Test syncing orders with exchange."""
        # Create local order
        order = order_manager.create_order(
            ticker="TEST", side=OrderSide.YES, action=OrderAction.BUY, count=100
        )
        order.update_status(OrderStatus.OPEN)
        
        # Mock exchange response
        mock_kalshi_client.get_orders = AsyncMock(return_value={
            "orders": [{
                "order_id": order.order_id,
                "status": "filled"
            }]
        })
        
        await order_manager.sync_with_exchange()
        
        assert order.status == OrderStatus.FILLED
    
    @pytest.mark.asyncio
    async def test_cleanup(self, order_manager, mock_kalshi_client):
        """Test order manager cleanup."""
        # Create active orders
        order1 = order_manager.create_order(
            ticker="TEST1", side=OrderSide.YES, action=OrderAction.BUY, count=100
        )
        order1.update_status(OrderStatus.OPEN)
        
        order2 = order_manager.create_order(
            ticker="TEST2", side=OrderSide.NO, action=OrderAction.BUY, count=50
        )
        order2.update_status(OrderStatus.OPEN)
        
        # Mock cancellation
        mock_kalshi_client.cancel_order = AsyncMock(return_value={"status": "cancelled"})
        
        await order_manager.cleanup()
        
        # Should attempt to cancel both orders
        assert mock_kalshi_client.cancel_order.call_count == 2
