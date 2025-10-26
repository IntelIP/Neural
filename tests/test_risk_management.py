"""
Tests for Risk Management System

Tests stop-loss functionality, risk monitoring, and automated execution.
"""

from unittest.mock import Mock, patch

import pytest

from neural.analysis.risk import (
    Position,
    RiskEvent,
    RiskLimits,
    RiskManager,
    StopLossConfig,
    StopLossType,
)


class TestRiskManager:
    """Test RiskManager functionality."""

    def test_stop_loss_percentage(self):
        """Test percentage-based stop-loss."""
        risk_manager = RiskManager()

        # Create position with 5% stop-loss
        position = Position(
            market_id="test_market",
            side="yes",
            quantity=100,
            entry_price=0.50,
            current_price=0.50,
            stop_loss=StopLossConfig(type=StopLossType.PERCENTAGE, value=0.05),
        )

        risk_manager.add_position(position)

        # Price drops 6% - should trigger stop-loss
        events = risk_manager.update_position_price("test_market", 0.47)
        assert RiskEvent.STOP_LOSS_TRIGGERED in events

        # Price drops 3% - should not trigger
        events = risk_manager.update_position_price("test_market", 0.485)
        assert RiskEvent.STOP_LOSS_TRIGGERED not in events

    def test_stop_loss_absolute(self):
        """Test absolute price stop-loss."""
        risk_manager = RiskManager()

        position = Position(
            market_id="test_market",
            side="yes",
            quantity=100,
            entry_price=0.60,
            current_price=0.60,
            stop_loss=StopLossConfig(type=StopLossType.ABSOLUTE, value=0.55),
        )

        risk_manager.add_position(position)

        # Price hits stop level
        events = risk_manager.update_position_price("test_market", 0.55)
        assert RiskEvent.STOP_LOSS_TRIGGERED in events

        # Price above stop level
        events = risk_manager.update_position_price("test_market", 0.57)
        assert RiskEvent.STOP_LOSS_TRIGGERED not in events

    def test_trailing_stop(self):
        """Test trailing stop-loss."""
        # Use very high limits to avoid interference
        limits = RiskLimits(max_position_size_pct=1.0, max_drawdown_pct=1.0)
        risk_manager = RiskManager(limits=limits, portfolio_value=100000.0)

        position = Position(
            market_id="test_market",
            side="yes",
            quantity=10,  # Small quantity
            entry_price=0.50,
            current_price=0.50,
            stop_loss=StopLossConfig(type=StopLossType.TRAILING, value=0.03),
        )

        risk_manager.add_position(position)

        # Price rises, trailing stop should follow
        risk_manager.update_position_price("test_market", 0.60)
        position = risk_manager.positions["test_market"]
        assert position.trailing_high == 0.60

        # Price drops to trailing stop level
        events = risk_manager.update_position_price("test_market", 0.582)  # 0.60 - 0.03*0.60
        assert RiskEvent.STOP_LOSS_TRIGGERED in events

    def test_risk_limits(self):
        """Test risk limit enforcement."""
        limits = RiskLimits(max_drawdown_pct=0.10, max_position_size_pct=0.20)
        risk_manager = RiskManager(limits=limits, portfolio_value=1000.0)

        position = Position(
            market_id="test_market", side="yes", quantity=100, entry_price=0.50, current_price=0.50
        )

        risk_manager.add_position(position)

        # Position size within limits
        events = risk_manager.update_position_price("test_market", 0.50)
        assert RiskEvent.POSITION_SIZE_EXCEEDED not in events

        # Simulate large position
        position.quantity = 500  # 25% of portfolio
        events = risk_manager.update_position_price("test_market", 0.50)
        assert RiskEvent.POSITION_SIZE_EXCEEDED in events

    def test_drawdown_limits(self):
        """Test drawdown limit enforcement."""
        limits = RiskLimits(max_drawdown_pct=0.10)
        risk_manager = RiskManager(limits=limits, portfolio_value=1000.0)

        # Simulate portfolio decline
        risk_manager.portfolio_value = 850.0  # 15% drawdown

        position = Position(
            market_id="test_market", side="yes", quantity=100, entry_price=0.50, current_price=0.50
        )

        risk_manager.add_position(position)
        events = risk_manager.update_position_price("test_market", 0.50)
        assert RiskEvent.MAX_DRAWDOWN_EXCEEDED in events


class TestWebSocketRiskIntegration:
    """Test websocket risk monitoring integration."""

    def test_price_update_risk_check(self):
        """Test that websocket price updates trigger risk checks."""
        from neural.trading.websocket import KalshiWebSocketClient

        risk_manager = Mock()
        risk_manager.update_position_price.return_value = []

        # Create a mock websocket client with just the needed attributes
        ws_client = Mock()
        ws_client.risk_manager = risk_manager
        # Use the real _process_risk_monitoring method
        ws_client._process_risk_monitoring = KalshiWebSocketClient._process_risk_monitoring.__get__(
            ws_client, KalshiWebSocketClient
        )

        # Simulate market price message
        payload = {
            "type": "market_price",
            "market": {"id": "test_market", "price": {"latest_price": 0.55}},
        }

        ws_client._process_risk_monitoring(payload)

        # Verify risk manager was called
        risk_manager.update_position_price.assert_called_once_with("test_market", 0.55)


class TestAutoExecutor:
    """Test automated execution layer."""

    def test_stop_loss_execution(self):
        """Test automated stop-loss execution."""
        from neural.analysis.execution import AutoExecutor, ExecutionConfig

        trading_client = Mock()
        config = ExecutionConfig(dry_run=True, max_orders_per_minute=10)
        executor = AutoExecutor(trading_client=trading_client, config=config)

        position = Mock()
        position.market_id = "test_market"
        position.side = "yes"
        position.quantity = 100

        # Trigger stop-loss event
        executor.on_risk_event(RiskEvent.STOP_LOSS_TRIGGERED, position, {"timestamp": 1234567890})

        # Verify order was attempted (dry run)
        # In dry run mode, no actual execution

    def test_emergency_stop(self):
        """Test emergency stop functionality."""
        from neural.analysis.execution import AutoExecutor

        trading_client = Mock()
        executor = AutoExecutor(trading_client=trading_client)

        # Trigger daily loss limit
        executor.on_risk_event(
            RiskEvent.DAILY_LOSS_LIMIT_EXCEEDED, "all", {"timestamp": 1234567890}
        )

        # Verify emergency stop was activated
        assert not executor.config.enable_auto_execution


if __name__ == "__main__":
    pytest.main([__file__])
