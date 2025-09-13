"""
Unit tests for Trading Engine

Tests cover signal processing, risk management integration, 
strategy orchestration, and decision making.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone
import uuid

from neural.trading.trading_engine import (
    TradingEngine, TradingConfig, TradingMode, ExecutionMode, TradeDecision
)
from neural.trading.kalshi_client import KalshiConfig, Environment
from neural.strategy.base import BaseStrategy, Signal, SignalType, StrategyConfig
from neural.trading.order_manager import OrderSide, OrderAction


class TestTradingConfig:
    """Test TradingConfig class."""
    
    def test_default_config_creation(self):
        """Test creating config with defaults."""
        config = TradingConfig()
        
        assert config.trading_mode == TradingMode.PAPER
        assert config.execution_mode == ExecutionMode.ADAPTIVE
        assert config.max_position_size == 0.10
        assert config.min_edge_threshold == 0.03
        assert config.min_confidence_threshold == 0.6
        assert config.enable_multi_strategy is True
        assert config.log_all_decisions is True
    
    def test_custom_config_creation(self):
        """Test creating custom configuration."""
        config = TradingConfig(
            trading_mode=TradingMode.LIVE,
            execution_mode=ExecutionMode.MARKET,
            max_position_size=0.05,
            min_edge_threshold=0.05,
            min_confidence_threshold=0.7,
            daily_loss_limit=0.03
        )
        
        assert config.trading_mode == TradingMode.LIVE
        assert config.execution_mode == ExecutionMode.MARKET
        assert config.max_position_size == 0.05
        assert config.daily_loss_limit == 0.03


class TestTradeDecision:
    """Test TradeDecision class."""
    
    def test_decision_creation(self):
        """Test creating a TradeDecision."""
        signal = Signal(
            signal_type=SignalType.BUY_YES,
            market_id="TEST-MARKET",
            timestamp=datetime.now(timezone.utc),
            confidence=0.75,
            edge=0.05
        )
        
        decision = TradeDecision(
            decision_id="decision_123",
            timestamp=datetime.now(timezone.utc),
            signal=signal,
            strategy_name="test_strategy",
            action="BUY",
            ticker="TEST-MARKET",
            side=OrderSide.YES,
            size=100,
            price=50,
            edge=0.05,
            confidence=0.75
        )
        
        assert decision.decision_id == "decision_123"
        assert decision.action == "BUY"
        assert decision.side == OrderSide.YES
        assert decision.size == 100
        assert decision.edge == 0.05
    
    def test_decision_to_dict(self):
        """Test converting decision to dictionary."""
        signal = Signal(
            signal_type=SignalType.BUY_YES,
            market_id="TEST-MARKET", 
            timestamp=datetime.now(timezone.utc),
            confidence=0.75,
            edge=0.05
        )
        
        decision = TradeDecision(
            decision_id="decision_123",
            timestamp=datetime.now(timezone.utc),
            signal=signal,
            strategy_name="test_strategy",
            action="BUY",
            ticker="TEST-MARKET",
            approved=True
        )
        
        decision_dict = decision.to_dict()
        
        assert decision_dict["decision_id"] == "decision_123"
        assert decision_dict["action"] == "BUY"
        assert decision_dict["strategy_name"] == "test_strategy"
        assert decision_dict["approved"] is True


class TestTradingEngine:
    """Test TradingEngine class."""
    
    @pytest.fixture
    def trading_config(self):
        """Create test trading configuration."""
        return TradingConfig(
            trading_mode=TradingMode.PAPER,
            max_position_size=0.10,
            min_edge_threshold=0.03,
            min_confidence_threshold=0.6
        )
    
    @pytest.fixture
    def kalshi_config(self):
        """Create test Kalshi configuration."""
        return KalshiConfig(environment=Environment.DEMO)
    
    @pytest.fixture
    def trading_engine(self, trading_config, kalshi_config):
        """Create TradingEngine instance."""
        return TradingEngine(trading_config, kalshi_config)
    
    def test_initialization(self, trading_engine, trading_config, kalshi_config):
        """Test TradingEngine initialization."""
        assert trading_engine.config == trading_config
        assert trading_engine.kalshi_config == kalshi_config
        assert trading_engine.running is False
        assert trading_engine.total_capital == 100000.0
        assert len(trading_engine.strategies) == 0
        assert len(trading_engine.decisions) == 0
    
    @pytest.mark.asyncio
    async def test_start_success(self, trading_engine):
        """Test successful engine startup."""
        with patch('neural.trading.kalshi_client.KalshiClient') as mock_client_class, \
             patch('neural.trading.websocket_manager.WebSocketManager') as mock_ws_class, \
             patch('neural.trading.order_manager.OrderManager') as mock_order_class, \
             patch('neural.trading.position_tracker.PositionTracker') as mock_pos_class, \
             patch('neural.trading.risk_manager.TradingRiskManager') as mock_risk_class, \
             patch('neural.trading.portfolio_manager.PortfolioManager') as mock_portfolio_class:
            
            # Setup mocks
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            mock_ws = AsyncMock()
            mock_ws.is_connected.return_value = True
            mock_ws_class.return_value = mock_ws
            
            mock_order_manager = Mock()
            mock_order_class.return_value = mock_order_manager
            
            mock_pos_tracker = AsyncMock()
            mock_pos_class.return_value = mock_pos_tracker
            
            await trading_engine.start()
            
            assert trading_engine.running is True
            assert trading_engine.kalshi_client is not None
            assert trading_engine.websocket_manager is not None
            assert trading_engine.order_manager is not None
            assert trading_engine.position_tracker is not None
            assert trading_engine.start_time is not None
    
    @pytest.mark.asyncio
    async def test_start_failure(self, trading_engine):
        """Test engine startup failure."""
        with patch('neural.trading.kalshi_client.KalshiClient') as mock_client_class:
            # Mock client connection failure
            mock_client = AsyncMock()
            mock_client.connect.side_effect = Exception("Connection failed")
            mock_client_class.return_value = mock_client
            
            with pytest.raises(Exception, match="Connection failed"):
                await trading_engine.start()
            
            assert trading_engine.running is False
    
    @pytest.mark.asyncio
    async def test_stop(self, trading_engine):
        """Test engine shutdown."""
        # Mock components
        trading_engine.position_tracker = AsyncMock()
        trading_engine.websocket_manager = AsyncMock()
        trading_engine.kalshi_client = AsyncMock()
        trading_engine.order_manager = Mock()
        trading_engine.order_manager.get_active_orders.return_value = []
        trading_engine.running = True
        
        await trading_engine.stop()
        
        assert trading_engine.running is False
        trading_engine.position_tracker.stop.assert_called_once()
        trading_engine.websocket_manager.disconnect.assert_called_once()
        trading_engine.kalshi_client.disconnect.assert_called_once()
    
    def test_add_strategy(self, trading_engine):
        """Test adding a strategy."""
        # Create mock strategy
        strategy = Mock(spec=BaseStrategy)
        strategy.name = "test_strategy"
        strategy.config = StrategyConfig()
        
        trading_engine.add_strategy(strategy, allocation=0.5, max_position_size=0.08)
        
        assert "test_strategy" in trading_engine.strategies
        assert trading_engine.strategies["test_strategy"] == strategy
        assert trading_engine.strategy_allocations["test_strategy"] == 0.5
        assert strategy.config.max_position_size == 0.08
    
    def test_remove_strategy(self, trading_engine):
        """Test removing a strategy."""
        # Add strategy first
        strategy = Mock(spec=BaseStrategy)
        strategy.name = "test_strategy"
        strategy.config = StrategyConfig()
        trading_engine.add_strategy(strategy)
        
        # Remove strategy
        trading_engine.remove_strategy("test_strategy")
        
        assert "test_strategy" not in trading_engine.strategies
        assert "test_strategy" not in trading_engine.strategy_allocations
    
    def test_validate_signal_success(self, trading_engine):
        """Test successful signal validation."""
        signal = Signal(
            signal_type=SignalType.BUY_YES,
            market_id="TEST-MARKET",
            timestamp=datetime.now(timezone.utc),
            confidence=0.75,  # Above threshold
            edge=0.05         # Above threshold
        )
        
        decision = TradeDecision(
            decision_id="test",
            timestamp=datetime.now(timezone.utc),
            signal=signal,
            strategy_name="test",
            action="HOLD",
            ticker="TEST-MARKET"
        )
        
        result = trading_engine._validate_signal(signal, decision)
        
        assert result is True
        assert decision.rejection_reason is None
    
    def test_validate_signal_low_confidence(self, trading_engine):
        """Test signal validation with low confidence."""
        signal = Signal(
            signal_type=SignalType.BUY_YES,
            market_id="TEST-MARKET",
            timestamp=datetime.now(timezone.utc),
            confidence=0.5,   # Below threshold (0.6)
            edge=0.05
        )
        
        decision = TradeDecision(
            decision_id="test",
            timestamp=datetime.now(timezone.utc),
            signal=signal,
            strategy_name="test",
            action="HOLD",
            ticker="TEST-MARKET"
        )
        
        result = trading_engine._validate_signal(signal, decision)
        
        assert result is False
        assert "Low confidence" in decision.rejection_reason
    
    def test_validate_signal_low_edge(self, trading_engine):
        """Test signal validation with low edge."""
        signal = Signal(
            signal_type=SignalType.BUY_YES,
            market_id="TEST-MARKET",
            timestamp=datetime.now(timezone.utc),
            confidence=0.75,
            edge=0.01  # Below threshold (0.03)
        )
        
        decision = TradeDecision(
            decision_id="test",
            timestamp=datetime.now(timezone.utc),
            signal=signal,
            strategy_name="test",
            action="HOLD",
            ticker="TEST-MARKET"
        )
        
        result = trading_engine._validate_signal(signal, decision)
        
        assert result is False
        assert "Low edge" in decision.rejection_reason
    
    def test_check_rate_limits_success(self, trading_engine):
        """Test rate limit checking when within limits."""
        decision = TradeDecision(
            decision_id="test",
            timestamp=datetime.now(timezone.utc),
            signal=Signal(SignalType.BUY_YES, "TEST", datetime.now(timezone.utc), 0.7, 0.05),
            strategy_name="test",
            action="BUY",
            ticker="TEST"
        )
        
        result = trading_engine._check_rate_limits(decision)
        
        assert result is True
    
    def test_check_rate_limits_exceeded(self, trading_engine):
        """Test rate limit checking when limits exceeded."""
        # Simulate hitting rate limit
        trading_engine.order_count_last_minute = 10  # Assuming limit is 10
        trading_engine.last_order_time = datetime.now(timezone.utc)
        
        decision = TradeDecision(
            decision_id="test",
            timestamp=datetime.now(timezone.utc),
            signal=Signal(SignalType.BUY_YES, "TEST", datetime.now(timezone.utc), 0.7, 0.05),
            strategy_name="test",
            action="BUY",
            ticker="TEST"
        )
        
        result = trading_engine._check_rate_limits(decision)
        
        assert result is False
        assert "Rate limit exceeded" in decision.rejection_reason
    
    def test_convert_signal_to_action(self, trading_engine):
        """Test converting signals to trading actions."""
        decision = TradeDecision(
            decision_id="test",
            timestamp=datetime.now(timezone.utc),
            signal=Mock(),
            strategy_name="test",
            action="HOLD",
            ticker="TEST"
        )
        
        # Test BUY_YES
        signal_buy_yes = Signal(
            SignalType.BUY_YES, "TEST", datetime.now(timezone.utc), 0.7, 0.05
        )
        trading_engine._convert_signal_to_action(signal_buy_yes, decision)
        assert decision.action == "BUY"
        assert decision.side == OrderSide.YES
        
        # Test BUY_NO
        signal_buy_no = Signal(
            SignalType.BUY_NO, "TEST", datetime.now(timezone.utc), 0.7, 0.05
        )
        trading_engine._convert_signal_to_action(signal_buy_no, decision)
        assert decision.action == "BUY"
        assert decision.side == OrderSide.NO
        
        # Test SELL_YES
        signal_sell_yes = Signal(
            SignalType.SELL_YES, "TEST", datetime.now(timezone.utc), 0.7, 0.05
        )
        trading_engine._convert_signal_to_action(signal_sell_yes, decision)
        assert decision.action == "SELL"
        assert decision.side == OrderSide.YES
        
        # Test HOLD
        signal_hold = Signal(
            SignalType.HOLD, "TEST", datetime.now(timezone.utc), 0.7, 0.05
        )
        trading_engine._convert_signal_to_action(signal_hold, decision)
        assert decision.action == "HOLD"
    
    def test_calculate_position_size(self, trading_engine):
        """Test position size calculation."""
        # Add strategy with allocation
        trading_engine.strategy_allocations["test_strategy"] = 0.5
        trading_engine.available_capital = 100000.0
        
        signal = Signal(
            SignalType.BUY_YES, "TEST", datetime.now(timezone.utc), 0.7, 0.05,
            recommended_size=0.08  # 8% position
        )
        
        decision = TradeDecision(
            decision_id="test",
            timestamp=datetime.now(timezone.utc),
            signal=signal,
            strategy_name="test_strategy",
            action="BUY",
            ticker="TEST",
            side=OrderSide.YES
        )
        
        trading_engine._calculate_position_size(signal, decision)
        
        assert decision.size > 0
        # Size should be limited by max_position_size (10% = 0.1)
        # Available: 100k, max 10% = 10k, at 50 cents = 20k contracts max
        # But signal recommends 8% * 50% allocation = 4% = 8k contracts
        expected_size = int((100000.0 * min(0.08 * 0.5, 0.1)) / 0.5)
        assert decision.size == expected_size
    
    @pytest.mark.asyncio
    async def test_check_risk_limits_approved(self, trading_engine):
        """Test risk limit checking with approval."""
        # Mock risk manager
        trading_engine.risk_manager = Mock()
        trading_engine.risk_manager.check_trade_allowed.return_value = (True, [])
        
        # Mock portfolio state
        with patch.object(trading_engine, '_get_portfolio_state') as mock_portfolio:
            mock_portfolio.return_value = {"test": "data"}
            
            signal = Signal(
                SignalType.BUY_YES, "TEST", datetime.now(timezone.utc), 0.7, 0.05
            )
            decision = TradeDecision(
                decision_id="test",
                timestamp=datetime.now(timezone.utc),
                signal=signal,
                strategy_name="test",
                action="BUY",
                ticker="TEST"
            )
            
            result = await trading_engine._check_risk_limits(decision)
            
            assert result is True
            assert decision.approved is True
    
    @pytest.mark.asyncio
    async def test_check_risk_limits_rejected(self, trading_engine):
        """Test risk limit checking with rejection."""
        # Mock risk manager
        trading_engine.risk_manager = Mock()
        trading_engine.risk_manager.check_trade_allowed.return_value = (
            False, ["Position size too large"]
        )
        
        # Mock portfolio state
        with patch.object(trading_engine, '_get_portfolio_state') as mock_portfolio:
            mock_portfolio.return_value = {"test": "data"}
            
            signal = Signal(
                SignalType.BUY_YES, "TEST", datetime.now(timezone.utc), 0.7, 0.05
            )
            decision = TradeDecision(
                decision_id="test",
                timestamp=datetime.now(timezone.utc),
                signal=signal,
                strategy_name="test",
                action="BUY",
                ticker="TEST"
            )
            
            result = await trading_engine._check_risk_limits(decision)
            
            assert result is False
            assert decision.approved is False
            assert "Position size too large" in decision.rejection_reason
    
    @pytest.mark.asyncio
    async def test_execute_decision_success(self, trading_engine):
        """Test successful decision execution."""
        # Mock order manager
        trading_engine.order_manager = AsyncMock()
        mock_order = Mock()
        mock_order.order_id = "order_123"
        trading_engine.order_manager.create_and_submit_order.return_value = mock_order
        
        decision = TradeDecision(
            decision_id="test",
            timestamp=datetime.now(timezone.utc),
            signal=Signal(SignalType.BUY_YES, "TEST", datetime.now(timezone.utc), 0.7, 0.05),
            strategy_name="test",
            action="BUY",
            ticker="TEST",
            side=OrderSide.YES,
            size=100,
            price=50
        )
        
        await trading_engine._execute_decision(decision)
        
        assert decision.order_id == "order_123"
        assert trading_engine.order_count_last_minute == 1
    
    @pytest.mark.asyncio
    async def test_execute_decision_failure(self, trading_engine):
        """Test decision execution failure."""
        # Mock order manager to return None (failure)
        trading_engine.order_manager = AsyncMock()
        trading_engine.order_manager.create_and_submit_order.return_value = None
        
        decision = TradeDecision(
            decision_id="test",
            timestamp=datetime.now(timezone.utc),
            signal=Signal(SignalType.BUY_YES, "TEST", datetime.now(timezone.utc), 0.7, 0.05),
            strategy_name="test",
            action="BUY",
            ticker="TEST",
            side=OrderSide.YES,
            size=100
        )
        
        await trading_engine._execute_decision(decision)
        
        assert "Order execution failed" in decision.rejection_reason
    
    @pytest.mark.asyncio
    async def test_process_signal_full_flow_approved(self, trading_engine):
        """Test complete signal processing with approval."""
        # Setup mocks
        trading_engine.risk_manager = Mock()
        trading_engine.risk_manager.check_trade_allowed.return_value = (True, [])
        trading_engine.order_manager = AsyncMock()
        mock_order = Mock()
        mock_order.order_id = "order_123"
        trading_engine.order_manager.create_and_submit_order.return_value = mock_order
        
        # Add strategy
        trading_engine.strategy_allocations["test_strategy"] = 1.0
        
        with patch.object(trading_engine, '_get_portfolio_state') as mock_portfolio:
            mock_portfolio.return_value = {"test": "data"}
            
            signal = Signal(
                SignalType.BUY_YES, "TEST-MARKET", datetime.now(timezone.utc),
                confidence=0.75, edge=0.05, recommended_size=0.08
            )
            
            decision = await trading_engine.process_signal(signal, "test_strategy")
            
            assert decision.action == "BUY"
            assert decision.side == OrderSide.YES
            assert decision.approved is True
            assert decision.order_id == "order_123"
            assert len(trading_engine.decisions) == 1
    
    @pytest.mark.asyncio
    async def test_process_signal_rejected_low_confidence(self, trading_engine):
        """Test signal processing with low confidence rejection."""
        signal = Signal(
            SignalType.BUY_YES, "TEST-MARKET", datetime.now(timezone.utc),
            confidence=0.5, edge=0.05  # Low confidence
        )
        
        decision = await trading_engine.process_signal(signal, "test_strategy")
        
        assert decision.action == "HOLD"
        assert decision.approved is False
        assert "Low confidence" in decision.rejection_reason
    
    @pytest.mark.asyncio
    async def test_get_portfolio_state(self, trading_engine):
        """Test getting portfolio state."""
        # Mock position tracker
        trading_engine.position_tracker = Mock()
        trading_engine.position_tracker.get_portfolio_stats.return_value = {
            "total_pnl": 1500.0,
            "active_positions": 5
        }
        
        state = await trading_engine._get_portfolio_state()
        
        assert state["total_capital"] == 100000.0
        assert state["total_pnl"] == 1500.0
        assert state["active_positions"] == 5
    
    def test_get_engine_status(self, trading_engine):
        """Test getting engine status."""
        trading_engine.running = True
        trading_engine.start_time = datetime.now(timezone.utc)
        trading_engine.total_signals_processed = 10
        trading_engine.total_trades_executed = 8
        trading_engine.daily_pnl = 250.0
        
        # Mock components
        trading_engine.order_manager = Mock()
        trading_engine.order_manager.get_active_orders.return_value = [Mock(), Mock()]
        trading_engine.position_tracker = Mock()
        trading_engine.position_tracker.get_active_positions.return_value = [Mock()]
        
        # Add some decisions
        trading_engine.decisions = [Mock() for _ in range(5)]
        
        status = trading_engine.get_engine_status()
        
        assert status["running"] is True
        assert status["trading_mode"] == "paper"
        assert status["signals_processed"] == 10
        assert status["trades_executed"] == 8
        assert status["daily_pnl"] == 250.0
        assert status["active_orders"] == 2
        assert status["active_positions"] == 1
        assert status["recent_decisions"] == 5
    
    def test_get_strategy_performance(self, trading_engine):
        """Test getting strategy performance."""
        # Mock position tracker
        trading_engine.position_tracker = Mock()
        trading_engine.position_tracker.get_strategy_performance.return_value = {
            "total_pnl": 150.0,
            "win_rate": 0.6
        }
        
        # Add strategy
        trading_engine.strategies["test_strategy"] = Mock()
        
        performance = trading_engine.get_strategy_performance()
        
        assert "test_strategy" in performance
        assert performance["test_strategy"]["total_pnl"] == 150.0
    
    def test_event_handlers(self, trading_engine):
        """Test event handler management."""
        decision_handler_calls = []
        trade_handler_calls = []
        
        def decision_handler(decision):
            decision_handler_calls.append(decision)
        
        def trade_handler(order):
            trade_handler_calls.append(order)
        
        # Add handlers
        trading_engine.add_decision_handler(decision_handler)
        trading_engine.add_trade_handler(trade_handler)
        
        assert len(trading_engine.decision_handlers) == 1
        assert len(trading_engine.trade_handlers) == 1
    
    @pytest.mark.asyncio
    async def test_context_manager(self, trading_engine):
        """Test using engine as async context manager."""
        with patch.object(trading_engine, 'start') as mock_start, \
             patch.object(trading_engine, 'stop') as mock_stop:
            
            async with trading_engine:
                pass
            
            mock_start.assert_called_once()
            mock_stop.assert_called_once()


# Integration tests
class TestTradingEngineIntegration:
    """Integration tests for TradingEngine with multiple components."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_signal_processing(self):
        """Test complete end-to-end signal processing."""
        config = TradingConfig(
            trading_mode=TradingMode.PAPER,
            min_confidence_threshold=0.6,
            min_edge_threshold=0.03
        )
        engine = TradingEngine(config)
        
        # Mock all components
        with patch('neural.trading.kalshi_client.KalshiClient'), \
             patch('neural.trading.websocket_manager.WebSocketManager'), \
             patch('neural.trading.order_manager.OrderManager'), \
             patch('neural.trading.position_tracker.PositionTracker'), \
             patch('neural.trading.risk_manager.TradingRiskManager'), \
             patch('neural.trading.portfolio_manager.PortfolioManager'):
            
            # Mock successful startup
            engine.kalshi_client = AsyncMock()
            engine.websocket_manager = AsyncMock()
            engine.websocket_manager.is_connected.return_value = True
            engine.order_manager = AsyncMock()
            engine.position_tracker = AsyncMock()
            engine.risk_manager = Mock()
            engine.portfolio_manager = Mock()
            engine.running = True
            
            # Mock risk approval and order execution
            engine.risk_manager.check_trade_allowed.return_value = (True, [])
            mock_order = Mock()
            mock_order.order_id = "order_123"
            engine.order_manager.create_and_submit_order.return_value = mock_order
            
            # Add strategy
            engine.add_strategy(Mock(name="test_strategy"), allocation=1.0)
            
            # Process signal
            signal = Signal(
                SignalType.BUY_YES, "TEST-MARKET", datetime.now(timezone.utc),
                confidence=0.8, edge=0.06, recommended_size=0.05
            )
            
            decision = await engine.process_signal(signal, "test_strategy")
            
            # Verify complete flow
            assert decision.approved is True
            assert decision.action == "BUY"
            assert decision.side == OrderSide.YES
            assert decision.order_id == "order_123"
            assert len(engine.decisions) == 1
    
    @pytest.mark.asyncio
    async def test_multi_strategy_processing(self):
        """Test processing signals from multiple strategies."""
        config = TradingConfig(enable_multi_strategy=True)
        engine = TradingEngine(config)
        
        # Setup engine (simplified)
        engine.running = True
        engine.risk_manager = Mock()
        engine.risk_manager.check_trade_allowed.return_value = (True, [])
        engine.order_manager = AsyncMock()
        engine.order_manager.create_and_submit_order.return_value = Mock(order_id="order_123")
        
        with patch.object(engine, '_get_portfolio_state') as mock_portfolio:
            mock_portfolio.return_value = {"test": "data"}
            
            # Add multiple strategies
            engine.add_strategy(Mock(name="strategy_a"), allocation=0.6)
            engine.add_strategy(Mock(name="strategy_b"), allocation=0.4)
            
            # Process signals from different strategies
            signal_a = Signal(
                SignalType.BUY_YES, "MARKET-A", datetime.now(timezone.utc),
                confidence=0.7, edge=0.04, recommended_size=0.05
            )
            signal_b = Signal(
                SignalType.BUY_NO, "MARKET-B", datetime.now(timezone.utc),
                confidence=0.8, edge=0.05, recommended_size=0.06
            )
            
            decision_a = await engine.process_signal(signal_a, "strategy_a")
            decision_b = await engine.process_signal(signal_b, "strategy_b")
            
            # Verify both processed successfully
            assert decision_a.approved is True
            assert decision_b.approved is True
            assert decision_a.strategy_name == "strategy_a"
            assert decision_b.strategy_name == "strategy_b"
            assert len(engine.decisions) == 2
