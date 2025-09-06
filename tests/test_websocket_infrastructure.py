#!/usr/bin/env python3
"""
Test suite for WebSocket infrastructure.
"""

import pytest
import asyncio
from datetime import datetime
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import json

from neural_sdk.data_sources.base.websocket_source import WebSocketDataSource, ConnectionConfig, ConnectionState
from neural_sdk.data_sources.kalshi.websocket_adapter import KalshiWebSocketAdapter, KalshiChannel
from neural_sdk.data_sources.unified.stream_manager import (
    UnifiedStreamManager, UnifiedMarketData, StreamConfig, EventType
)
from neural_sdk.trading.real_time_engine import (
    RealTimeTradingEngine, TradingSignal, SignalType, RiskLimits,
    Order, OrderStatus, Position
)


class TestWebSocketBase:
    """Test base WebSocket functionality."""
    
    @pytest.fixture
    def mock_websocket(self):
        """Create a mock WebSocket data source."""
        class MockWebSocket(WebSocketDataSource):
            async def _connect(self):
                return True
            
            async def _authenticate(self):
                return True
            
            async def _handle_message(self, message):
                pass
            
            async def _send_heartbeat(self):
                pass
            
            async def _subscribe_internal(self, subscription: str) -> bool:
                return True
            
            async def _unsubscribe_internal(self, subscription: str) -> bool:
                return True
            
            async def _process_message(self, message: Dict[str, Any]):
                pass
        
        config = ConnectionConfig(url="wss://test.example.com")
        return MockWebSocket(config, name="MockWebSocket")
    
    @pytest.mark.asyncio
    async def test_connection_lifecycle(self, mock_websocket):
        """Test WebSocket connection lifecycle."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_ws = AsyncMock()
            mock_session.ws_connect = AsyncMock(return_value=mock_ws)
            mock_session_class.return_value = mock_session
            
            # Test connect
            await mock_websocket.connect()
            assert mock_websocket.is_connected()
            mock_session.ws_connect.assert_called_once()
            
            # Test disconnect
            await mock_websocket.disconnect()
            assert not mock_websocket.is_connected()
            mock_ws.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_auto_reconnection(self, mock_websocket):
        """Test automatic reconnection on disconnect."""
        mock_websocket.config.reconnect_interval = 0.1
        mock_websocket.config.max_reconnect_attempts = 2
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_ws = AsyncMock()
            
            # First connection fails, second succeeds
            mock_session.ws_connect = AsyncMock(side_effect=[
                Exception("Connection failed"),
                mock_ws
            ])
            mock_session_class.return_value = mock_session
            
            await mock_websocket.connect()
            
            # Should have attempted twice
            assert mock_session.ws_connect.call_count == 2
            assert mock_websocket.is_connected()
    
    @pytest.mark.asyncio
    async def test_message_queue(self, mock_websocket):
        """Test message queuing during disconnection."""
        mock_websocket.state = ConnectionState.DISCONNECTED
        
        # Queue messages while disconnected
        await mock_websocket.send({"test": "message1"})
        await mock_websocket.send({"test": "message2"})
        
        assert len(mock_websocket._message_queue) == 2
        
        # Connect and verify queue is processed
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_ws = AsyncMock()
            mock_session.ws_connect = AsyncMock(return_value=mock_ws)
            mock_session_class.return_value = mock_session
            
            await mock_websocket.connect()
            await asyncio.sleep(0.1)  # Allow queue processing
            
            # Messages should be sent
            assert mock_ws.send.call_count >= 2


class TestKalshiWebSocket:
    """Test Kalshi WebSocket adapter."""
    
    @pytest.fixture
    def kalshi_ws(self):
        """Create Kalshi WebSocket adapter."""
        return KalshiWebSocketAdapter(api_key="test_key")
    
    @pytest.mark.asyncio
    async def test_subscribe_to_market(self, kalshi_ws):
        """Test market subscription."""
        with patch.object(kalshi_ws, 'send', new_callable=AsyncMock) as mock_send:
            kalshi_ws.state = ConnectionState.CONNECTED
            
            await kalshi_ws.subscribe_market(
                "TEST-MARKET",
                [KalshiChannel.TICKER, KalshiChannel.ORDERBOOK_DELTA]
            )
            
            # Verify subscription messages sent
            assert mock_send.call_count == 2
            
            # Check subscription format
            calls = mock_send.call_args_list
            for call in calls:
                msg = call[0][0]
                assert msg["type"] == "subscribe"
                assert msg["params"]["market_ticker"] == "TEST-MARKET"
    
    @pytest.mark.asyncio
    async def test_message_parsing(self, kalshi_ws):
        """Test parsing of different message types."""
        # Test ticker update
        ticker_msg = {
            "type": "ticker",
            "ticker": {
                "market_ticker": "TEST-MARKET",
                "yes_price": 0.65,
                "no_price": 0.35,
                "volume": 1000
            }
        }
        
        with patch.object(kalshi_ws, '_emit_event') as mock_emit:
            await kalshi_ws._handle_message(json.dumps(ticker_msg))
            
            mock_emit.assert_called_once()
            event_type, data = mock_emit.call_args[0]
            assert event_type == "ticker_update"
            assert data["market_ticker"] == "TEST-MARKET"
            assert data["yes_price"] == 0.65
        
        # Test orderbook update
        orderbook_msg = {
            "type": "orderbook_delta",
            "market_ticker": "TEST-MARKET",
            "orderbook_delta": {
                "yes": [[0.65, 100], [0.64, 200]],
                "no": [[0.35, 150], [0.36, 250]]
            }
        }
        
        with patch.object(kalshi_ws, '_emit_event') as mock_emit:
            await kalshi_ws._handle_message(json.dumps(orderbook_msg))
            
            mock_emit.assert_called_once()
            event_type, data = mock_emit.call_args[0]
            assert event_type == "orderbook_update"
            assert "orderbook_delta" in data


class TestUnifiedStreamManager:
    """Test unified stream manager."""
    
    @pytest.fixture
    def stream_manager(self):
        """Create stream manager."""
        config = StreamConfig(
            enable_kalshi=True,
            enable_odds_polling=True,
            odds_poll_interval=30,
            correlation_window=5,
            divergence_threshold=0.05
        )
        return UnifiedStreamManager(config)
    
    @pytest.mark.asyncio
    async def test_market_tracking(self, stream_manager):
        """Test market tracking and data aggregation."""
        # Mock Kalshi WebSocket
        stream_manager.kalshi_ws = AsyncMock()
        stream_manager.kalshi_ws.is_connected = Mock(return_value=True)
        
        # Track market
        await stream_manager.track_market(
            "TEST-MARKET",
            "game_001",
            [KalshiChannel.TICKER]
        )
        
        assert "TEST-MARKET" in stream_manager.market_mappings
        assert stream_manager.market_mappings["TEST-MARKET"] == "game_001"
        
        # Update market data via ticker handler
        ticker_data = {
            "market_ticker": "TEST-MARKET",
            "yes_price": 0.65,
            "no_price": 0.35,
            "volume": 1000
        }
        
        await stream_manager._handle_kalshi_ticker(ticker_data)
        
        # Verify data stored
        assert "TEST-MARKET" in stream_manager.market_data
        assert len(stream_manager.market_history["TEST-MARKET"]) == 1
    
    @pytest.mark.asyncio
    async def test_arbitrage_detection(self, stream_manager):
        """Test arbitrage opportunity detection."""
        event_fired = False
        arb_data = None
        
        async def on_arbitrage(event):
            nonlocal event_fired, arb_data
            event_fired = True
            arb_data = event
        
        stream_manager.on(EventType.ARBITRAGE_OPPORTUNITY, on_arbitrage)
        
        # First add the market
        stream_manager.market_mappings["TEST-MARKET"] = "game_001"
        
        # Create market data with arbitrage by simulating ticker update
        ticker_data = {
            "market_ticker": "TEST-MARKET",
            "yes_price": 0.60,
            "no_price": 0.40,
            "volume": 1000
        }
        
        # Manually inject odds data to create arbitrage
        stream_manager.market_data["TEST-MARKET"] = UnifiedMarketData(
            ticker="TEST-MARKET",
            game_id="game_001",
            kalshi_yes_price=0.60,
            kalshi_volume=1000,
            odds_implied_prob_home=0.70,  # 10% divergence
            arbitrage_exists=True,
            divergence_score=0.10,
            timestamp=datetime.utcnow()
        )
        
        # Trigger correlation check
        await stream_manager._correlate_markets()
        await asyncio.sleep(0.1)  # Allow event processing
        
        assert event_fired
        assert arb_data["ticker"] == "TEST-MARKET"
        assert arb_data["data"].arbitrage_exists
    
    @pytest.mark.asyncio
    async def test_volatility_calculation(self, stream_manager):
        """Test volatility calculation."""
        # Add price history
        prices = [0.50, 0.52, 0.51, 0.53, 0.52, 0.54, 0.53, 0.55]
        
        stream_manager.market_mappings["TEST-MARKET"] = "game_001"
        stream_manager.market_history["TEST-MARKET"] = []
        
        for price in prices:
            market_data = UnifiedMarketData(
                ticker="TEST-MARKET",
                game_id="game_001",
                kalshi_yes_price=price,
                timestamp=datetime.utcnow()
            )
            stream_manager.market_history["TEST-MARKET"].append(market_data)
        
        volatility = stream_manager.calculate_volatility("TEST-MARKET", window=5)
        
        assert volatility is not None
        assert volatility > 0


class TestRealTimeTradingEngine:
    """Test real-time trading engine."""
    
    @pytest.fixture
    def trading_engine(self):
        """Create trading engine."""
        stream_manager = Mock()
        risk_limits = RiskLimits(
            max_position_size=100,
            max_order_size=50,
            max_daily_loss=500.0,
            max_daily_trades=10,
            max_open_positions=3
        )
        return RealTimeTradingEngine(stream_manager, risk_limits)
    
    @pytest.mark.asyncio
    async def test_signal_generation(self, trading_engine):
        """Test trading signal generation."""
        signal_generated = False
        generated_signal = None
        
        async def strategy(market_data, engine):
            return TradingSignal(
                signal_id="test_001",
                timestamp=datetime.utcnow(),
                market_ticker="TEST-MARKET",
                signal_type=SignalType.BUY,
                confidence=0.8,
                size=10,
                reason="Test signal"
            )
        
        def on_signal(signal):
            nonlocal signal_generated, generated_signal
            signal_generated = True
            generated_signal = signal
        
        trading_engine.on_signal(on_signal)
        trading_engine.add_strategy(strategy)
        
        # Process market data
        market_data = UnifiedMarketData(
            ticker="TEST-MARKET",
            kalshi_yes_price=0.65,
            timestamp=datetime.utcnow()
        )
        
        await trading_engine._process_market_data(market_data)
        
        assert signal_generated
        assert generated_signal.market_ticker == "TEST-MARKET"
        assert generated_signal.signal_type == SignalType.BUY
    
    @pytest.mark.asyncio
    async def test_risk_management(self, trading_engine):
        """Test risk management limits."""
        # Exceed daily trade limit
        trading_engine.stats.daily_trades = 10  # At limit
        
        signal = TradingSignal(
            signal_id="test_001",
            timestamp=datetime.utcnow(),
            market_ticker="TEST-MARKET",
            signal_type=SignalType.BUY,
            confidence=0.8,
            size=10,
            reason="Test signal"
        )
        
        order = await trading_engine._execute_signal(signal)
        
        # Should be rejected due to daily limit
        assert order is None or order.status == OrderStatus.REJECTED
    
    @pytest.mark.asyncio
    async def test_position_tracking(self, trading_engine):
        """Test position tracking and P&L calculation."""
        # Create position
        position = Position(
            market_ticker="TEST-MARKET",
            size=100,
            average_price=0.50,
            timestamp=datetime.utcnow()
        )
        
        trading_engine.positions["TEST-MARKET"] = position
        
        # Update market price
        position.current_price = 0.55
        position.update_pnl()
        
        assert position.unrealized_pnl == 5.0  # 100 * (0.55 - 0.50)
        
        # Close position
        await trading_engine._close_position("TEST-MARKET", close_price=0.55)
        
        assert "TEST-MARKET" not in trading_engine.positions
        assert trading_engine.stats.total_pnl == 5.0
    
    @pytest.mark.asyncio
    async def test_stop_loss_trigger(self, trading_engine):
        """Test stop-loss trigger."""
        # Create losing position
        position = Position(
            market_ticker="TEST-MARKET",
            size=100,
            average_price=0.50,
            timestamp=datetime.utcnow()
        )
        
        trading_engine.positions["TEST-MARKET"] = position
        trading_engine.risk_limits.stop_loss_percentage = 0.10
        
        # Update with loss exceeding stop-loss
        position.current_price = 0.44  # 12% loss
        position.update_pnl()
        
        await trading_engine._check_stop_loss()
        
        # Position should be closed
        assert "TEST-MARKET" not in trading_engine.positions


@pytest.mark.asyncio
async def test_integration():
    """Test full integration of WebSocket components."""
    # Create integrated system
    config = StreamConfig(
        enable_kalshi=True,
        enable_odds_polling=False,  # Disable for testing
        correlation_window=5,
        divergence_threshold=0.05
    )
    
    stream_manager = UnifiedStreamManager(config)
    
    risk_limits = RiskLimits(
        max_position_size=100,
        max_order_size=50,
        max_daily_loss=500.0,
        max_daily_trades=10
    )
    
    trading_engine = RealTimeTradingEngine(stream_manager, risk_limits)
    
    # Add test strategy
    async def test_strategy(market_data, engine):
        if market_data.kalshi_yes_price and market_data.kalshi_yes_price > 0.60:
            return TradingSignal(
                signal_id=f"test_{datetime.utcnow().timestamp()}",
                timestamp=datetime.utcnow(),
                market_ticker=market_data.ticker,
                signal_type=SignalType.BUY,
                confidence=0.7,
                size=10,
                reason="Price above threshold"
            )
        return None
    
    trading_engine.add_strategy(test_strategy)
    
    # Mock WebSocket connection
    with patch('websockets.connect', new_callable=AsyncMock):
        # Start components
        await stream_manager.start()
        await trading_engine.start()
        
        # Simulate market data
        market_data = UnifiedMarketData(
            ticker="TEST-MARKET",
            kalshi_yes_price=0.65,
            timestamp=datetime.utcnow()
        )
        
        await stream_manager._update_market_data("TEST-MARKET", market_data)
        await asyncio.sleep(0.1)  # Allow processing
        
        # Verify signal generated and order created
        stats = trading_engine.get_stats()
        assert stats.total_signals > 0
        
        # Cleanup
        await trading_engine.stop()
        await stream_manager.stop()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])