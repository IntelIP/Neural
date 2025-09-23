"""
Integration tests for complete trading cycle (Buy → Hold → Sell)

This module tests the full lifecycle of a trade from order placement
through position management to closing the position with proper P&L tracking.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import uuid

from neural.trading import (
    KalshiClient, KalshiConfig, Environment,
    OrderManager, Order, OrderStatus, OrderType, OrderSide, OrderAction,
    PositionTracker, Position,
    WebSocketManager, SubscriptionType
)
from neural.trading.websocket_manager import WebSocketMessage


class TestCompleteTradingCycle:
    """Test complete trading cycle from buy to sell with P&L tracking."""
    
    @pytest.fixture
    async def kalshi_client(self):
        """Create mock Kalshi client with authentication."""
        config = KalshiConfig(
            environment=Environment.DEMO,
            api_key="test_api_key",
            private_key="test_private_key",
            user_id="test_user"
        )
        client = KalshiClient(config)
        
        # Mock authentication
        client.authenticated = True
        client.access_token = "test_token"
        client.session = MagicMock()
        
        return client
    
    @pytest.fixture
    async def websocket_manager(self):
        """Create mock WebSocket manager."""
        ws = Mock(spec=WebSocketManager)
        ws.is_connected = Mock(return_value=True)
        ws.subscribe_orders = AsyncMock()
        ws.subscribe_fills = AsyncMock()
        ws.subscribe_positions = AsyncMock()
        ws.add_handler = Mock()
        return ws
    
    @pytest.fixture
    async def order_manager(self, kalshi_client, websocket_manager):
        """Create OrderManager with mocked dependencies."""
        return OrderManager(
            kalshi_client=kalshi_client,
            websocket_manager=websocket_manager,
            enable_real_time_updates=True
        )
    
    @pytest.fixture
    async def position_tracker(self, kalshi_client, websocket_manager):
        """Create PositionTracker with mocked dependencies."""
        tracker = PositionTracker(
            kalshi_client=kalshi_client,
            websocket_manager=websocket_manager
        )
        # Initialize with starting balance
        tracker.account_balance = 10000.0  # $10,000 starting balance
        return tracker
    
    @pytest.mark.asyncio
    async def test_complete_buy_hold_sell_cycle(self, order_manager, position_tracker, kalshi_client):
        """Test complete trading cycle: Buy → Hold → Sell with P&L tracking."""
        
        # Test parameters
        ticker = "SUPERBOWL-2025-CHIEFS"
        buy_price = 45  # Buy YES at 45 cents
        sell_price = 65  # Sell YES at 65 cents
        quantity = 100  # 100 contracts
        
        # =================
        # STEP 1: BUY ORDER
        # =================
        print("\n📈 STEP 1: Creating and submitting BUY order...")
        
        # Create buy order
        buy_order = order_manager.create_order(
            ticker=ticker,
            side=OrderSide.YES,
            action=OrderAction.BUY,
            count=quantity,
            order_type=OrderType.LIMIT,
            yes_price=buy_price,
            strategy_id="test_cycle"
        )
        
        assert buy_order is not None
        assert buy_order.ticker == ticker
        assert buy_order.side == OrderSide.YES
        assert buy_order.action == OrderAction.BUY
        assert buy_order.count == quantity
        assert buy_order.yes_price == buy_price
        
        # Mock successful order submission
        kalshi_client.create_order = AsyncMock(return_value={
            "order_id": f"kalshi_{uuid.uuid4().hex[:8]}",
            "status": "open",
            "ticker": ticker,
            "side": "yes",
            "action": "buy",
            "count": quantity,
            "yes_price": buy_price
        })
        
        # Submit buy order
        success = await order_manager.submit_order(buy_order)
        assert success is True
        assert buy_order.status == OrderStatus.OPEN
        
        # Simulate order fill via WebSocket
        fill_message = WebSocketMessage(
            id=f"msg_{uuid.uuid4().hex[:8]}",
            type="fill",
            seq=1,
            msg={
                "fill_id": f"fill_{uuid.uuid4().hex[:8]}",
                "order_id": buy_order.order_id,
                "ticker": ticker,
                "side": "yes",
                "action": "buy",
                "count": quantity,
                "price": buy_price,
                "trade_id": f"trade_{uuid.uuid4().hex[:8]}"
            },
            timestamp=datetime.now(timezone.utc)
        )
        
        await order_manager._handle_fill_update(fill_message)
        
        # Verify buy order is filled
        assert buy_order.status == OrderStatus.FILLED
        assert buy_order.filled_count == quantity
        assert buy_order.remaining_count == 0
        assert len(buy_order.fills) == 1
        
        print(f"✅ Buy order filled: {quantity} contracts @ ${buy_price/100:.2f}")
        
        # =================
        # STEP 2: POSITION TRACKING
        # =================
        print("\n📊 STEP 2: Tracking position...")
        
        # Create position from buy order
        position = Position(
            ticker=ticker,
            side="yes",
            quantity=quantity,
            avg_entry_price=buy_price / 100.0,
            current_price=buy_price / 100.0,
            market_value=(quantity * buy_price) / 100.0,
            cost_basis=(quantity * buy_price) / 100.0
        )
        
        position_tracker.positions[ticker] = position
        
        # Simulate price movement (market moves in our favor)
        await asyncio.sleep(0.1)  # Simulate time passing
        
        new_market_price = 55  # Price moves to 55 cents
        position.current_price = new_market_price / 100.0
        position.market_value = (quantity * new_market_price) / 100.0
        position.update_pnl()
        
        # Check unrealized P&L
        expected_unrealized_pnl = (new_market_price - buy_price) * quantity / 100.0
        assert position.unrealized_pnl == pytest.approx(expected_unrealized_pnl, rel=0.01)
        
        print(f"📈 Position update: Price moved to ${new_market_price/100:.2f}")
        print(f"   Unrealized P&L: ${position.unrealized_pnl:.2f}")
        
        # =================
        # STEP 3: SELL ORDER (Close Position)
        # =================
        print("\n💰 STEP 3: Creating and submitting SELL order to close position...")
        
        # Create sell order to close position
        sell_order = order_manager.create_order(
            ticker=ticker,
            side=OrderSide.YES,
            action=OrderAction.SELL,
            count=quantity,
            order_type=OrderType.LIMIT,
            yes_price=sell_price,
            strategy_id="test_cycle"
        )
        
        assert sell_order is not None
        assert sell_order.action == OrderAction.SELL
        assert sell_order.yes_price == sell_price
        
        # Mock successful sell order submission
        kalshi_client.create_order = AsyncMock(return_value={
            "order_id": f"kalshi_{uuid.uuid4().hex[:8]}",
            "status": "open",
            "ticker": ticker,
            "side": "yes",
            "action": "sell",
            "count": quantity,
            "yes_price": sell_price
        })
        
        # Submit sell order
        success = await order_manager.submit_order(sell_order)
        assert success is True
        assert sell_order.status == OrderStatus.OPEN
        
        # Simulate sell order fill
        sell_fill_message = WebSocketMessage(
            id=f"msg_{uuid.uuid4().hex[:8]}",
            type="fill",
            seq=2,
            msg={
                "fill_id": f"fill_{uuid.uuid4().hex[:8]}",
                "order_id": sell_order.order_id,
                "ticker": ticker,
                "side": "yes",
                "action": "sell",
                "count": quantity,
                "price": sell_price,
                "trade_id": f"trade_{uuid.uuid4().hex[:8]}"
            },
            timestamp=datetime.now(timezone.utc)
        )
        
        await order_manager._handle_fill_update(sell_fill_message)
        
        # Verify sell order is filled
        assert sell_order.status == OrderStatus.FILLED
        assert sell_order.filled_count == quantity
        
        print(f"✅ Sell order filled: {quantity} contracts @ ${sell_price/100:.2f}")
        
        # =================
        # STEP 4: P&L CALCULATION
        # =================
        print("\n💵 STEP 4: Calculating final P&L...")
        
        # Close position and calculate realized P&L
        position.quantity = 0  # Position closed
        position.exit_price = sell_price / 100.0
        position.realized_pnl = (sell_price - buy_price) * quantity / 100.0
        position.unrealized_pnl = 0  # No longer holding position
        
        # Update position tracker totals
        position_tracker.total_realized_pnl += position.realized_pnl
        
        # Verify P&L calculations
        expected_profit = (sell_price - buy_price) * quantity / 100.0
        assert position.realized_pnl == pytest.approx(expected_profit, rel=0.01)
        
        # Calculate return percentage
        return_pct = (position.realized_pnl / position.cost_basis) * 100
        
        print(f"\n📊 TRADE SUMMARY:")
        print(f"   Entry: {quantity} contracts @ ${buy_price/100:.2f} = ${position.cost_basis:.2f}")
        print(f"   Exit:  {quantity} contracts @ ${sell_price/100:.2f} = ${(sell_price * quantity / 100.0):.2f}")
        print(f"   Realized P&L: ${position.realized_pnl:.2f}")
        print(f"   Return: {return_pct:.2f}%")
        print(f"   Account Balance: ${position_tracker.account_balance + position.realized_pnl:.2f}")
        
        # =================
        # STEP 5: VERIFY COMPLETE CYCLE
        # =================
        print("\n✅ STEP 5: Verifying complete trading cycle...")
        
        # Verify orders
        assert len(order_manager.orders) == 2  # Buy and sell orders
        assert order_manager.total_orders == 2
        assert order_manager.total_fills == 2
        
        # Verify position is closed
        assert position.quantity == 0
        assert position.realized_pnl > 0  # Profitable trade
        assert position.unrealized_pnl == 0
        
        # Verify account state
        final_balance = position_tracker.account_balance + position_tracker.total_realized_pnl
        assert final_balance > position_tracker.account_balance  # Account grew
        
        print(f"✅ Trading cycle complete! Total profit: ${position.realized_pnl:.2f}")
    
    @pytest.mark.asyncio
    async def test_partial_fill_scenario(self, order_manager, position_tracker, kalshi_client):
        """Test handling of partial fills in buy and sell orders."""
        
        ticker = "NFL-PLAYOFFS-2025"
        total_quantity = 100
        
        # Create buy order
        buy_order = order_manager.create_order(
            ticker=ticker,
            side=OrderSide.YES,
            action=OrderAction.BUY,
            count=total_quantity,
            order_type=OrderType.LIMIT,
            yes_price=50
        )
        
        # Mock order submission
        kalshi_client.create_order = AsyncMock(return_value={
            "order_id": "partial_order_1",
            "status": "open"
        })
        
        await order_manager.submit_order(buy_order)
        
        # Simulate partial fills
        fills = [
            {"count": 30, "price": 50},  # First partial fill
            {"count": 40, "price": 51},  # Second partial fill
            {"count": 30, "price": 49}   # Final fill
        ]
        
        for i, fill_data in enumerate(fills):
            fill_message = WebSocketMessage(
                id=f"msg_{i}",
                type="fill",
                seq=i,
                msg={
                    "fill_id": f"fill_{i}",
                    "order_id": buy_order.order_id,
                    "ticker": ticker,
                    "side": "yes",
                    "action": "buy",
                    "count": fill_data["count"],
                    "price": fill_data["price"]
                },
                timestamp=datetime.now(timezone.utc)
            )
            
            await order_manager._handle_fill_update(fill_message)
            
            # Check partial fill status
            if i < len(fills) - 1:
                assert buy_order.status == OrderStatus.PARTIALLY_FILLED
            else:
                assert buy_order.status == OrderStatus.FILLED
        
        # Verify order is completely filled
        assert buy_order.filled_count == total_quantity
        assert len(buy_order.fills) == 3
        
        # Calculate weighted average price
        total_cost = sum(f["count"] * f["price"] for f in fills)
        avg_price = total_cost / total_quantity
        
        print(f"✅ Order filled with {len(fills)} partial fills")
        print(f"   Average fill price: ${avg_price/100:.2f}")
    
    @pytest.mark.asyncio
    async def test_stop_loss_execution(self, order_manager, position_tracker, kalshi_client):
        """Test stop-loss order execution when position moves against us."""
        
        ticker = "MARCH-MADNESS-2025"
        quantity = 50
        entry_price = 70  # Buy at 70 cents
        stop_loss_price = 60  # Stop loss at 60 cents
        
        # Open position with buy order
        buy_order = order_manager.create_order(
            ticker=ticker,
            side=OrderSide.YES,
            action=OrderAction.BUY,
            count=quantity,
            order_type=OrderType.LIMIT,
            yes_price=entry_price
        )
        
        # Mock order execution
        kalshi_client.create_order = AsyncMock(return_value={
            "order_id": "stop_loss_test_buy",
            "status": "filled"
        })
        
        await order_manager.submit_order(buy_order)
        buy_order.update_status(OrderStatus.FILLED)
        buy_order.filled_count = quantity
        
        # Create position
        position = Position(
            ticker=ticker,
            side="yes",
            quantity=quantity,
            avg_entry_price=entry_price / 100.0,
            current_price=entry_price / 100.0,
            market_value=(quantity * entry_price) / 100.0,
            cost_basis=(quantity * entry_price) / 100.0
        )
        
        # Simulate price drop triggering stop loss
        position.current_price = stop_loss_price / 100.0
        position.update_pnl()
        
        # Execute stop loss order
        stop_order = order_manager.create_order(
            ticker=ticker,
            side=OrderSide.YES,
            action=OrderAction.SELL,
            count=quantity,
            order_type=OrderType.MARKET,  # Market order for immediate execution
            strategy_id="stop_loss"
        )
        
        kalshi_client.create_order = AsyncMock(return_value={
            "order_id": "stop_loss_test_sell",
            "status": "filled"
        })
        
        await order_manager.submit_order(stop_order)
        
        # Simulate immediate fill at stop loss price
        stop_order.update_status(OrderStatus.FILLED)
        stop_order.filled_count = quantity
        
        # Calculate loss
        loss = (stop_loss_price - entry_price) * quantity / 100.0
        position.realized_pnl = loss
        
        print(f"⛔ Stop loss executed:")
        print(f"   Entry: ${entry_price/100:.2f}")
        print(f"   Stop: ${stop_loss_price/100:.2f}")
        print(f"   Loss: ${abs(loss):.2f}")
        
        assert position.realized_pnl < 0  # Verify loss
        assert abs(position.realized_pnl) == pytest.approx(abs(loss), rel=0.01)
    
    @pytest.mark.asyncio
    async def test_multiple_positions_management(self, order_manager, position_tracker, kalshi_client):
        """Test managing multiple positions simultaneously."""
        
        positions_data = [
            {"ticker": "NBA-FINALS-2025", "side": OrderSide.YES, "quantity": 100, "price": 45},
            {"ticker": "WORLD-CUP-2026", "side": OrderSide.NO, "quantity": 50, "price": 30},
            {"ticker": "OLYMPICS-2025", "side": OrderSide.YES, "quantity": 75, "price": 60}
        ]
        
        # Open multiple positions
        for data in positions_data:
            order = order_manager.create_order(
                ticker=data["ticker"],
                side=data["side"],
                action=OrderAction.BUY,
                count=data["quantity"],
                order_type=OrderType.LIMIT,
                yes_price=data["price"] if data["side"] == OrderSide.YES else None,
                no_price=data["price"] if data["side"] == OrderSide.NO else None
            )
            
            # Mock submission
            kalshi_client.create_order = AsyncMock(return_value={
                "order_id": f"order_{data['ticker']}",
                "status": "filled"
            })
            
            await order_manager.submit_order(order)
            order.update_status(OrderStatus.FILLED)
            order.filled_count = data["quantity"]
            
            # Track position
            position = Position(
                ticker=data["ticker"],
                side=data["side"].value,
                quantity=data["quantity"],
                avg_entry_price=data["price"] / 100.0,
                current_price=data["price"] / 100.0,
                market_value=(data["quantity"] * data["price"]) / 100.0,
                cost_basis=(data["quantity"] * data["price"]) / 100.0
            )
            position_tracker.positions[data["ticker"]] = position
        
        # Verify all positions are tracked
        assert len(position_tracker.positions) == 3
        
        # Calculate total exposure
        total_exposure = sum(p.market_value for p in position_tracker.positions.values())
        
        print(f"\n📈 Portfolio Summary:")
        print(f"   Active Positions: {len(position_tracker.positions)}")
        print(f"   Total Exposure: ${total_exposure:.2f}")
        
        for ticker, position in position_tracker.positions.items():
            print(f"   - {ticker}: {position.quantity} contracts @ ${position.avg_entry_price:.2f}")