"""
Unit tests for Position Tracking System

Tests cover position updates, P&L calculations, portfolio metrics,
and performance attribution.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone
import numpy as np

from neural.trading.position_tracker import (
    PositionTracker, Position, PositionSide
)
from neural.trading.order_manager import OrderManager, Fill
from neural.trading.kalshi_client import KalshiClient


class TestPosition:
    """Test Position data class."""
    
    def test_position_creation(self):
        """Test creating a Position object."""
        position = Position(
            ticker="TEST-MARKET",
            strategy_id="test_strategy"
        )
        
        assert position.ticker == "TEST-MARKET"
        assert position.side == PositionSide.FLAT
        assert position.yes_long == 0
        assert position.yes_short == 0
        assert position.no_long == 0
        assert position.no_short == 0
        assert position.yes_net == 0
        assert position.no_net == 0
        assert position.realized_pnl == 0.0
        assert position.unrealized_pnl == 0.0
        assert position.strategy_id == "test_strategy"
    
    def test_position_properties(self):
        """Test position computed properties."""
        position = Position("TEST-MARKET")
        
        # Test initial flat position
        assert position.net_contracts == 0
        assert position.gross_contracts == 0
        assert position.is_flat is True
        assert position.is_long is False
        assert position.is_short is False
        
        # Simulate YES long position
        position.yes_long = 100
        position._update_computed_fields()
        
        assert position.net_contracts == 100
        assert position.gross_contracts == 100
        assert position.is_flat is False
        assert position.is_long is True
        assert position.is_short is False
        assert position.side == PositionSide.LONG
    
    def test_add_trade_buy_yes(self):
        """Test adding a BUY YES trade."""
        position = Position("TEST-MARKET")
        
        fill = Fill(
            fill_id="fill_1",
            order_id="order_1",
            ticker="TEST-MARKET",
            side="yes",
            action="buy",
            count=100,
            price=50,  # 50 cents
            timestamp=datetime.now(timezone.utc)
        )
        
        position.add_trade(fill)
        
        assert position.yes_long == 100
        assert position.yes_short == 0
        assert position.yes_net == 100
        assert position.side == PositionSide.LONG
        assert position.first_trade_time is not None
        assert position.last_trade_time is not None
        assert position.total_cost_basis == 50.0  # (100 * 50) / 100
    
    def test_add_trade_sell_yes(self):
        """Test adding a SELL YES trade."""
        position = Position("TEST-MARKET")
        
        # First establish a long position
        buy_fill = Fill(
            fill_id="fill_1",
            order_id="order_1", 
            ticker="TEST-MARKET",
            side="yes",
            action="buy",
            count=100,
            price=48,
            timestamp=datetime.now(timezone.utc)
        )
        position.add_trade(buy_fill)
        
        # Now sell some
        sell_fill = Fill(
            fill_id="fill_2",
            order_id="order_2",
            ticker="TEST-MARKET", 
            side="yes",
            action="sell",
            count=30,
            price=52,
            timestamp=datetime.now(timezone.utc)
        )
        position.add_trade(sell_fill)
        
        assert position.yes_long == 70  # 100 - 30
        assert position.yes_net == 70
        assert position.realized_pnl > 0  # Sold at higher price than bought
    
    def test_add_trade_buy_no(self):
        """Test adding a BUY NO trade."""
        position = Position("TEST-MARKET")
        
        fill = Fill(
            fill_id="fill_1",
            order_id="order_1",
            ticker="TEST-MARKET",
            side="no", 
            action="buy",
            count=100,
            price=55,
            timestamp=datetime.now(timezone.utc)
        )
        
        position.add_trade(fill)
        
        assert position.no_long == 100
        assert position.no_short == 0
        assert position.no_net == 100
        assert position.side == PositionSide.SHORT  # NO long = bearish
        assert position.total_cost_basis == 55.0
    
    def test_update_unrealized_pnl(self):
        """Test updating unrealized P&L."""
        position = Position("TEST-MARKET")
        
        # Establish YES long position at 48 cents
        fill = Fill(
            fill_id="fill_1",
            order_id="order_1",
            ticker="TEST-MARKET",
            side="yes",
            action="buy", 
            count=100,
            price=48,
            timestamp=datetime.now(timezone.utc)
        )
        position.add_trade(fill)
        
        # Update with current market price of 52 cents (0.52 probability)
        position.update_unrealized_pnl(0.52)
        
        assert position.unrealized_pnl > 0  # Position gained value
        assert position.market_value == 52.0  # 100 contracts * 0.52
    
    def test_close_position_yes_wins(self):
        """Test closing position when YES side wins."""
        position = Position("TEST-MARKET")
        
        # Establish mixed position
        yes_fill = Fill(
            fill_id="fill_1", order_id="order_1", ticker="TEST-MARKET",
            side="yes", action="buy", count=100, price=48,
            timestamp=datetime.now(timezone.utc)
        )
        position.add_trade(yes_fill)
        
        no_fill = Fill(
            fill_id="fill_2", order_id="order_2", ticker="TEST-MARKET", 
            side="no", action="buy", count=50, price=52,
            timestamp=datetime.now(timezone.utc)
        )
        position.add_trade(no_fill)
        
        # Close position with YES winning (settles to $1)
        position.close_position(1.0, "yes")
        
        assert position.is_flat is True
        assert position.yes_long == 0
        assert position.no_long == 0
        assert position.unrealized_pnl == 0.0
        # YES position should profit, NO position should lose
    
    def test_to_dict(self):
        """Test converting position to dictionary."""
        position = Position("TEST-MARKET", strategy_id="test_strategy")
        
        # Add some activity
        fill = Fill(
            fill_id="fill_1", order_id="order_1", ticker="TEST-MARKET",
            side="yes", action="buy", count=100, price=50,
            timestamp=datetime.now(timezone.utc)
        )
        position.add_trade(fill)
        
        position_dict = position.to_dict()
        
        assert position_dict["ticker"] == "TEST-MARKET"
        assert position_dict["side"] == "long"
        assert position_dict["yes_long"] == 100
        assert position_dict["net_contracts"] == 100
        assert position_dict["strategy_id"] == "test_strategy"
        assert "first_trade_time" in position_dict
        assert "last_trade_time" in position_dict


class TestPositionTracker:
    """Test PositionTracker class."""
    
    @pytest.fixture
    def mock_kalshi_client(self):
        """Create mock Kalshi client."""
        client = Mock(spec=KalshiClient)
        client.authenticated = True
        return client
    
    @pytest.fixture
    def mock_order_manager(self):
        """Create mock OrderManager."""
        order_manager = Mock(spec=OrderManager)
        order_manager.add_fill_handler = Mock()
        return order_manager
    
    @pytest.fixture
    def position_tracker(self, mock_kalshi_client, mock_order_manager):
        """Create PositionTracker instance."""
        return PositionTracker(
            mock_kalshi_client,
            mock_order_manager,
            auto_update_prices=False  # Disable for testing
        )
    
    def test_initialization(self, position_tracker, mock_kalshi_client, mock_order_manager):
        """Test PositionTracker initialization."""
        assert position_tracker.client == mock_kalshi_client
        assert position_tracker.order_manager == mock_order_manager
        assert position_tracker.auto_update_prices is False
        assert len(position_tracker.positions) == 0
        assert position_tracker.total_realized_pnl == 0.0
        
        # Should register fill handler
        mock_order_manager.add_fill_handler.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_start_stop(self, position_tracker):
        """Test starting and stopping position tracker."""
        with patch.object(position_tracker, 'reconcile_positions') as mock_reconcile:
            await position_tracker.start()
            mock_reconcile.assert_called_once()
        
        await position_tracker.stop()
        # Should complete without error
    
    def test_get_or_create_position(self, position_tracker, mock_order_manager):
        """Test getting or creating positions."""
        # Mock order for strategy attribution
        mock_order = Mock()
        mock_order.strategy_id = "test_strategy"
        mock_order_manager.get_order.return_value = mock_order
        
        # First call should create position
        position = position_tracker.get_or_create_position("TEST-MARKET", "order_123")
        
        assert isinstance(position, Position)
        assert position.ticker == "TEST-MARKET"
        assert position.strategy_id == "test_strategy"
        assert len(position_tracker.positions) == 1
        
        # Second call should return existing
        same_position = position_tracker.get_or_create_position("TEST-MARKET", "order_456")
        assert same_position is position
    
    @pytest.mark.asyncio
    async def test_handle_fill(self, position_tracker):
        """Test handling fill notifications."""
        fill = Fill(
            fill_id="fill_123",
            order_id="order_123",
            ticker="TEST-MARKET",
            side="yes",
            action="buy",
            count=100,
            price=50,
            timestamp=datetime.now(timezone.utc)
        )
        
        await position_tracker._handle_fill(fill)
        
        assert len(position_tracker.positions) == 1
        assert position_tracker.total_trades == 1
        
        position = position_tracker.positions["TEST-MARKET"]
        assert position.yes_long == 100
    
    @pytest.mark.asyncio
    async def test_update_position_price(self, position_tracker, mock_kalshi_client):
        """Test updating price for specific position."""
        # Create position
        position = position_tracker.get_or_create_position("TEST-MARKET", "order_123")
        
        # Mock market data response
        mock_kalshi_client.get_market = AsyncMock(return_value={
            "ticker": "TEST-MARKET",
            "yes_bid": 48,
            "yes_ask": 52
        })
        
        await position_tracker.update_position_price("TEST-MARKET")
        
        assert "TEST-MARKET" in position_tracker.market_prices
        assert position_tracker.market_prices["TEST-MARKET"] == 0.50  # (48+52)/200
    
    @pytest.mark.asyncio  
    async def test_update_all_prices(self, position_tracker, mock_kalshi_client):
        """Test updating prices for all positions."""
        # Create multiple positions
        position_tracker.get_or_create_position("TEST1", "order_1")
        position_tracker.get_or_create_position("TEST2", "order_2")
        
        # Mock market data responses
        def mock_get_market(ticker):
            if ticker == "TEST1":
                return {"yes_bid": 45, "yes_ask": 47}
            elif ticker == "TEST2": 
                return {"yes_bid": 60, "yes_ask": 64}
            return None
        
        mock_kalshi_client.get_market = AsyncMock(side_effect=mock_get_market)
        
        await position_tracker.update_all_prices()
        
        assert len(position_tracker.market_prices) == 2
        assert position_tracker.market_prices["TEST1"] == 0.46  # (45+47)/200
        assert position_tracker.market_prices["TEST2"] == 0.62  # (60+64)/200
    
    @pytest.mark.asyncio
    async def test_reconcile_positions(self, position_tracker, mock_kalshi_client):
        """Test reconciling positions with exchange."""
        # Mock exchange positions response
        mock_kalshi_client.get_positions = AsyncMock(return_value={
            "positions": [
                {"ticker": "TEST1", "position": 100},
                {"ticker": "TEST2", "position": -50}
            ]
        })
        
        await position_tracker.reconcile_positions()
        
        # Should create positions for exchange data
        assert len(position_tracker.positions) == 2
        assert "TEST1" in position_tracker.positions
        assert "TEST2" in position_tracker.positions
    
    def test_query_methods(self, position_tracker):
        """Test various position query methods."""
        # Create test positions
        pos1 = position_tracker.get_or_create_position("TEST1", "order_1")
        pos1.strategy_id = "strategy_a"
        pos1.yes_long = 100
        pos1._update_computed_fields()
        
        pos2 = position_tracker.get_or_create_position("TEST2", "order_2")
        pos2.strategy_id = "strategy_b" 
        pos2.no_long = 50
        pos2._update_computed_fields()
        
        pos3 = position_tracker.get_or_create_position("TEST3", "order_3")  # Flat
        
        # Test get_position
        retrieved = position_tracker.get_position("TEST1")
        assert retrieved is pos1
        
        # Test get_all_positions
        all_positions = position_tracker.get_all_positions()
        assert len(all_positions) == 3
        
        # Test get_active_positions
        active = position_tracker.get_active_positions()
        assert len(active) == 2  # pos1 and pos2 are not flat
        
        # Test get_positions_by_strategy
        strategy_a_positions = position_tracker.get_positions_by_strategy("strategy_a")
        assert len(strategy_a_positions) == 1
        assert strategy_a_positions[0] is pos1
        
        # Test get_long_positions
        long_positions = position_tracker.get_long_positions()
        assert len(long_positions) == 1
        assert long_positions[0] is pos1
        
        # Test get_short_positions  
        short_positions = position_tracker.get_short_positions()
        assert len(short_positions) == 1
        assert short_positions[0] is pos2
    
    def test_portfolio_metrics(self, position_tracker):
        """Test portfolio-level metrics."""
        # Create positions with market values
        pos1 = position_tracker.get_or_create_position("TEST1", "order_1")
        pos1.market_value = 100.0
        pos1.total_pnl = 15.0
        pos1.realized_pnl = 10.0
        pos1.unrealized_pnl = 5.0
        
        pos2 = position_tracker.get_or_create_position("TEST2", "order_2")  
        pos2.market_value = 50.0
        pos2.total_pnl = -8.0
        pos2.realized_pnl = -3.0
        pos2.unrealized_pnl = -5.0
        
        # Update tracker totals
        position_tracker._update_totals()
        
        # Test portfolio value
        portfolio_value = position_tracker.get_portfolio_value()
        assert portfolio_value == 150.0  # 100 + 50
        
        # Test total P&L
        total_pnl = position_tracker.get_total_pnl()
        assert total_pnl == 7.0  # 15 + (-8)
        
        # Test portfolio stats
        stats = position_tracker.get_portfolio_stats()
        assert stats["total_positions"] == 2
        assert stats["total_realized_pnl"] == 7.0  # 10 + (-3)
        assert stats["total_unrealized_pnl"] == 0.0  # 5 + (-5)
        assert stats["portfolio_value"] == 150.0
    
    def test_strategy_performance(self, position_tracker):
        """Test strategy-specific performance metrics."""
        # Create positions for same strategy
        pos1 = position_tracker.get_or_create_position("TEST1", "order_1")
        pos1.strategy_id = "test_strategy"
        pos1.total_pnl = 25.0
        pos1.realized_pnl = 20.0
        pos1.unrealized_pnl = 5.0
        
        pos2 = position_tracker.get_or_create_position("TEST2", "order_2")
        pos2.strategy_id = "test_strategy"
        pos2.total_pnl = -10.0
        pos2.realized_pnl = -5.0
        pos2.unrealized_pnl = -5.0
        
        # Different strategy
        pos3 = position_tracker.get_or_create_position("TEST3", "order_3")
        pos3.strategy_id = "other_strategy"
        pos3.total_pnl = 5.0
        
        perf = position_tracker.get_strategy_performance("test_strategy")
        
        assert perf["strategy_id"] == "test_strategy"
        assert perf["total_positions"] == 2
        assert perf["total_pnl"] == 15.0  # 25 + (-10)
        assert perf["realized_pnl"] == 15.0  # 20 + (-5)
        assert perf["unrealized_pnl"] == 0.0  # 5 + (-5)
        assert perf["win_count"] == 1  # pos1
        assert perf["loss_count"] == 1  # pos2
        assert perf["win_rate"] == 0.5
    
    def test_event_handlers(self, position_tracker):
        """Test position event handler management."""
        handler_calls = []
        
        def position_handler(position):
            handler_calls.append(position)
        
        # Add handler
        position_tracker.add_position_handler(position_handler)
        assert len(position_tracker.position_handlers) == 1
        
        # Remove handler
        position_tracker.remove_position_handler(position_handler)
        assert len(position_tracker.position_handlers) == 0
    
    @pytest.mark.asyncio
    async def test_handle_settlement(self, position_tracker):
        """Test handling market settlement."""
        # Create position
        position = position_tracker.get_or_create_position("TEST-MARKET", "order_123")
        
        # Add some activity
        fill = Fill(
            fill_id="fill_1", order_id="order_123", ticker="TEST-MARKET",
            side="yes", action="buy", count=100, price=50,
            timestamp=datetime.now(timezone.utc)
        )
        position.add_trade(fill)
        
        # Settle market
        await position_tracker.handle_settlement("TEST-MARKET", "yes")
        
        assert position.is_flat is True
    
    def test_risk_metrics(self, position_tracker):
        """Test risk metric calculations."""
        # Create positions with different P&L
        positions_pnl = [10.0, 5.0, -8.0, 15.0, -12.0]
        
        for i, pnl in enumerate(positions_pnl):
            pos = position_tracker.get_or_create_position(f"TEST{i}", f"order_{i}")
            pos.unrealized_pnl = pnl
            if pnl != 0:  # Make non-flat
                pos.yes_long = 100
                pos._update_computed_fields()
        
        # Test VaR calculation (simplified)
        var_95 = position_tracker.calculate_var(0.95)
        assert isinstance(var_95, float)
        
        # Test largest positions
        largest = position_tracker.get_largest_positions(limit=3)
        assert len(largest) <= 3
        assert all(not pos.is_flat for pos in largest)
    
    def test_update_totals(self, position_tracker):
        """Test updating portfolio totals."""
        # Create positions with P&L
        pos1 = position_tracker.get_or_create_position("TEST1", "order_1")
        pos1.realized_pnl = 15.0
        pos1.unrealized_pnl = 5.0
        pos1.total_pnl = 20.0
        
        pos2 = position_tracker.get_or_create_position("TEST2", "order_2")
        pos2.realized_pnl = -8.0
        pos2.unrealized_pnl = 3.0
        pos2.total_pnl = -5.0
        
        position_tracker._update_totals()
        
        assert position_tracker.total_realized_pnl == 7.0  # 15 + (-8)
        assert position_tracker.total_unrealized_pnl == 8.0  # 5 + 3
        assert position_tracker.win_count == 1  # pos1 has positive P&L
        assert position_tracker.loss_count == 1  # pos2 has negative P&L
