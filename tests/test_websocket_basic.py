#!/usr/bin/env python3
"""
Simple WebSocket infrastructure tests.
Tests core functionality without complex mocking.
"""

import pytest
import asyncio
from datetime import datetime
from typing import Dict, Any

from neural_sdk.data_sources.base.websocket_source import ConnectionConfig, ConnectionState
from neural_sdk.data_sources.kalshi.websocket_adapter import KalshiWebSocketAdapter, KalshiChannel
from neural_sdk.data_sources.unified.stream_manager import (
    UnifiedStreamManager, UnifiedMarketData, StreamConfig, EventType
)
from neural_sdk.trading.real_time_engine import (
    RealTimeTradingEngine, TradingSignal, SignalType, RiskLimits,
    Order, OrderStatus, Position, TradingStats
)


class TestDataModels:
    """Test data model creation and functionality."""
    
    def test_connection_config(self):
        """Test ConnectionConfig creation."""
        config = ConnectionConfig(
            url="wss://test.example.com",
            api_key="test_key",
            heartbeat_interval=30
        )
        
        assert config.url == "wss://test.example.com"
        assert config.api_key == "test_key"
        assert config.heartbeat_interval == 30
        assert config.max_reconnect_attempts == 10  # Default
    
    def test_unified_market_data(self):
        """Test UnifiedMarketData creation and methods."""
        data = UnifiedMarketData(
            ticker="TEST-MARKET",
            kalshi_yes_price=0.65,
            kalshi_volume=1000,
            odds_consensus_home=0.70,
            timestamp=datetime.utcnow()
        )
        
        assert data.ticker == "TEST-MARKET"
        assert data.kalshi_yes_price == 0.65
        
        # Test compute_metrics
        data.compute_metrics()
        assert data.divergence_score > 0
        assert data.arbitrage_exists is True  # Due to divergence
    
    def test_trading_signal(self):
        """Test TradingSignal creation."""
        signal = TradingSignal(
            signal_id="test_001",
            timestamp=datetime.utcnow(),
            market_ticker="TEST-MARKET",
            signal_type=SignalType.BUY,
            confidence=0.8,
            size=100,
            reason="Test signal"
        )
        
        assert signal.signal_id == "test_001"
        assert signal.signal_type == SignalType.BUY
        assert signal.confidence == 0.8
        assert signal.size == 100
    
    def test_risk_limits(self):
        """Test RiskLimits creation."""
        limits = RiskLimits(
            max_position_size=1000,
            max_order_size=100,
            max_daily_loss=500.0,
            max_daily_trades=50
        )
        
        assert limits.max_position_size == 1000
        assert limits.max_order_size == 100
        assert limits.max_daily_loss == 500.0
        assert limits.max_daily_trades == 50
        assert limits.max_open_positions == 10  # Default
    
    def test_position(self):
        """Test Position creation and P&L calculation."""
        position = Position(
            market_ticker="TEST-MARKET",
            side="yes",
            size=100,
            average_price=0.50
        )
        
        assert position.market_ticker == "TEST-MARKET"
        assert position.size == 100
        assert position.average_price == 0.50
        assert position.unrealized_pnl == 0.0
        
        # Update price and calculate P&L
        position.update_pnl(0.55)
        
        assert position.current_price == 0.55
        assert abs(position.unrealized_pnl - 5.0) < 0.01  # 100 * (0.55 - 0.50)
    
    def test_order(self):
        """Test Order creation and status updates."""
        order = Order(
            order_id="order_001",
            signal_id="signal_001",
            market_ticker="TEST-MARKET",
            side="yes",
            size=100,
            price=0.50,
            order_type="limit",
            status=OrderStatus.PENDING,
            created_at=datetime.utcnow()
        )
        
        assert order.order_id == "order_001"
        assert order.status == OrderStatus.PENDING
        assert order.side == "yes"
        assert order.size == 100
        
        # Update status
        order.status = OrderStatus.FILLED
        order.fill_price = 0.49
        order.filled_at = datetime.utcnow()
        
        assert order.status == OrderStatus.FILLED
        assert order.fill_price == 0.49


class TestKalshiChannels:
    """Test Kalshi channel enums."""
    
    def test_channel_values(self):
        """Test KalshiChannel enum values."""
        assert KalshiChannel.TICKER.value == "ticker"
        assert KalshiChannel.ORDERBOOK_DELTA.value == "orderbook_delta"
        assert KalshiChannel.TRADE.value == "trade"
        assert KalshiChannel.FILL.value == "fill"
        assert KalshiChannel.MARKET_POSITIONS.value == "market_positions"
        assert KalshiChannel.MARKET_LIFECYCLE.value == "market_lifecycle_v2"


class TestStreamConfig:
    """Test stream configuration."""
    
    def test_default_config(self):
        """Test default StreamConfig values."""
        config = StreamConfig()
        
        assert config.enable_kalshi is True
        assert config.enable_odds_polling is True  # Default is True
        assert config.odds_poll_interval == 30  # Default is 30
        assert config.correlation_window == 5
        assert config.divergence_threshold == 0.05
    
    def test_custom_config(self):
        """Test custom StreamConfig values."""
        config = StreamConfig(
            enable_kalshi=False,
            enable_odds_polling=True,
            odds_poll_interval=30,
            correlation_window=10,
            divergence_threshold=0.10
        )
        
        assert config.enable_kalshi == False
        assert config.enable_odds_polling == True
        assert config.odds_poll_interval == 30
        assert config.correlation_window == 10
        assert config.divergence_threshold == 0.10


class TestEventTypes:
    """Test event type enums."""
    
    def test_event_type_values(self):
        """Test EventType enum values."""
        assert EventType.PRICE_UPDATE.value == "price_update"
        assert EventType.ORDERBOOK_UPDATE.value == "orderbook_update"
        assert EventType.TRADE_EXECUTED.value == "trade_executed"
        assert EventType.ODDS_UPDATE.value == "odds_update"
        assert EventType.LINE_MOVEMENT.value == "line_movement"
        assert EventType.ARBITRAGE_OPPORTUNITY.value == "arbitrage_opportunity"
        assert EventType.DIVERGENCE_DETECTED.value == "divergence_detected"
        assert EventType.SIGNAL_GENERATED.value == "signal_generated"


class TestTradingStats:
    """Test trading statistics tracking."""
    
    def test_stats_initialization(self):
        """Test TradingStats initialization."""
        stats = TradingStats()
        
        assert stats.total_trades == 0
        assert stats.winning_trades == 0
        assert stats.losing_trades == 0
        assert stats.total_pnl == 0.0
        assert stats.daily_pnl == 0.0
        assert stats.win_rate is None
        assert stats.max_drawdown == 0.0
    
    def test_stats_calculation(self):
        """Test stats calculations."""
        stats = TradingStats()
        
        # Add trades
        stats.total_trades = 10
        stats.winning_trades = 6
        stats.losing_trades = 4
        stats.total_pnl = 150.0
        stats.daily_pnl = 50.0
        
        # Calculate win rate
        stats.win_rate = stats.winning_trades / stats.total_trades if stats.total_trades > 0 else None
        
        assert stats.win_rate == 0.6
        assert stats.total_pnl == 150.0


class TestConnectionStates:
    """Test connection state transitions."""
    
    def test_connection_states(self):
        """Test ConnectionState enum values."""
        assert ConnectionState.DISCONNECTED.value == "disconnected"
        assert ConnectionState.CONNECTING.value == "connecting"
        assert ConnectionState.CONNECTED.value == "connected"
        assert ConnectionState.RECONNECTING.value == "reconnecting"
        assert ConnectionState.CLOSING.value == "closing"
        assert ConnectionState.CLOSED.value == "closed"


class TestSignalTypes:
    """Test signal type enums."""
    
    def test_signal_types(self):
        """Test SignalType enum values."""
        assert SignalType.BUY.value == "buy"
        assert SignalType.SELL.value == "sell"
        assert SignalType.HOLD.value == "hold"
        assert SignalType.CLOSE.value == "close"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])