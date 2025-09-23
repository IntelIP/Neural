"""
Unit tests for base strategy framework components.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from dataclasses import asdict

from neural.strategy.base import (
    BaseStrategy,
    Signal,
    StrategyConfig,
    StrategyResult,
    StrategyManager,
    SignalType,
    PositionAction
)


class MockStrategy(BaseStrategy):
    """Mock strategy implementation for testing."""
    
    def __init__(self, name: str, config: StrategyConfig = None):
        super().__init__(name, config)
        self.initialize_called = False
        self.analyze_called = False
        self.analysis_results = {}
        
    async def initialize(self) -> None:
        self.initialize_called = True
        self._initialized = True
        
    async def analyze_market(
        self,
        market_id: str,
        market_data: dict,
        context: dict = None
    ) -> StrategyResult:
        self.analyze_called = True
        
        # Return predefined result or create default
        if market_id in self.analysis_results:
            return self.analysis_results[market_id]
        
        # Default signal for testing
        signal = Signal(
            signal_type=SignalType.BUY_YES,
            market_id=market_id,
            timestamp=datetime.now(),
            confidence=0.75,
            edge=0.05,
            expected_value=25.0,
            recommended_size=0.02,
            max_contracts=50,
            reason="Mock strategy signal"
        )
        
        return StrategyResult(
            strategy_name=self.name,
            market_id=market_id,
            timestamp=datetime.now(),
            signal=signal,
            analysis_time_ms=10.5
        )
    
    def get_required_data(self) -> list:
        return ['market_price', 'volume']
    
    def set_analysis_result(self, market_id: str, result: StrategyResult):
        """Helper to set predefined analysis results for testing."""
        self.analysis_results[market_id] = result


class TestStrategyConfig:
    """Test StrategyConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = StrategyConfig()
        
        assert config.max_position_size == 0.10
        assert config.max_portfolio_exposure == 0.50
        assert config.stop_loss_pct == 0.20
        assert config.min_confidence == 0.60
        assert config.min_edge == 0.03
        assert config.use_kelly_criterion is True
        assert config.max_kelly_fraction == 0.25
        assert config.max_trades_per_day == 10
        assert len(config.markets_blacklist) == 0
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = StrategyConfig(
            max_position_size=0.05,
            min_confidence=0.80,
            min_edge=0.05,
            use_kelly_criterion=False,
            fixed_position_size=0.02,
            markets_blacklist=['BANNED_MARKET']
        )
        
        assert config.max_position_size == 0.05
        assert config.min_confidence == 0.80
        assert config.min_edge == 0.05
        assert config.use_kelly_criterion is False
        assert config.fixed_position_size == 0.02
        assert 'BANNED_MARKET' in config.markets_blacklist
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config should not raise
        valid_config = StrategyConfig()
        valid_config.validate()
        
        # Invalid max_position_size
        with pytest.raises(ValueError):
            invalid_config = StrategyConfig(max_position_size=1.5)  # > 1.0
            invalid_config.validate()
        
        with pytest.raises(ValueError):
            invalid_config = StrategyConfig(max_position_size=0)  # <= 0
            invalid_config.validate()
        
        # Invalid min_confidence
        with pytest.raises(ValueError):
            invalid_config = StrategyConfig(min_confidence=1.5)  # > 1.0
            invalid_config.validate()
        
        # Invalid min_edge
        with pytest.raises(ValueError):
            invalid_config = StrategyConfig(min_edge=-0.1)  # < 0
            invalid_config.validate()


class TestSignal:
    """Test Signal dataclass."""
    
    @pytest.fixture
    def sample_signal(self):
        return Signal(
            signal_type=SignalType.BUY_YES,
            market_id="TEST_MARKET",
            timestamp=datetime(2024, 1, 1, 12, 0),
            confidence=0.80,
            edge=0.06,
            expected_value=30.0,
            recommended_size=0.03,
            max_contracts=75,
            stop_loss_price=0.35,
            take_profit_price=0.65,
            max_hold_time=timedelta(hours=24),
            reason="Strong edge detected"
        )
    
    def test_signal_properties(self, sample_signal):
        """Test signal property methods."""
        assert sample_signal.is_actionable is True
        assert sample_signal.is_buy_signal is True
        assert sample_signal.is_sell_signal is False
        
        # Test with HOLD signal
        hold_signal = Signal(
            signal_type=SignalType.HOLD,
            market_id="TEST",
            timestamp=datetime.now(),
            confidence=0.5,
            edge=0,
            expected_value=0,
            recommended_size=0,
            max_contracts=0
        )
        
        assert hold_signal.is_actionable is False
        assert hold_signal.is_buy_signal is False
        assert hold_signal.is_sell_signal is False
    
    def test_signal_sell_properties(self):
        """Test signal properties for sell signals."""
        sell_signal = Signal(
            signal_type=SignalType.SELL_YES,
            market_id="TEST",
            timestamp=datetime.now(),
            confidence=0.7,
            edge=-0.04,
            expected_value=-20,
            recommended_size=0,
            max_contracts=0
        )
        
        assert sell_signal.is_actionable is True
        assert sell_signal.is_buy_signal is False
        assert sell_signal.is_sell_signal is True
    
    def test_signal_to_dict(self, sample_signal):
        """Test signal conversion to dictionary."""
        signal_dict = sample_signal.to_dict()
        
        assert signal_dict['signal_type'] == SignalType.BUY_YES.value
        assert signal_dict['market_id'] == "TEST_MARKET"
        assert signal_dict['confidence'] == 0.80
        assert signal_dict['edge'] == 0.06
        assert signal_dict['expected_value'] == 30.0
        assert signal_dict['stop_loss_price'] == 0.35
        assert signal_dict['take_profit_price'] == 0.65
        assert signal_dict['max_hold_time'] == 86400.0  # 24 hours in seconds
        assert signal_dict['reason'] == "Strong edge detected"


class TestStrategyResult:
    """Test StrategyResult dataclass."""
    
    def test_create_strategy_result(self):
        """Test creating strategy result."""
        signal = Signal(
            signal_type=SignalType.BUY_NO,
            market_id="TEST_MARKET",
            timestamp=datetime.now(),
            confidence=0.65,
            edge=0.04,
            expected_value=20.0,
            recommended_size=0.02,
            max_contracts=40
        )
        
        result = StrategyResult(
            strategy_name="Test Strategy",
            market_id="TEST_MARKET",
            timestamp=datetime.now(),
            signal=signal,
            analysis_time_ms=15.2,
            data_quality_score=0.9,
            warnings=["Low volume warning"],
            debug_info={"param1": "value1"}
        )
        
        assert result.strategy_name == "Test Strategy"
        assert result.market_id == "TEST_MARKET"
        assert result.signal == signal
        assert result.analysis_time_ms == 15.2
        assert result.data_quality_score == 0.9
        assert len(result.warnings) == 1
        assert result.debug_info["param1"] == "value1"
    
    def test_strategy_result_to_dict(self):
        """Test converting strategy result to dictionary."""
        signal = Signal(
            signal_type=SignalType.HOLD,
            market_id="TEST",
            timestamp=datetime(2024, 1, 1, 10, 0),
            confidence=0.3,
            edge=0.01,
            expected_value=0,
            recommended_size=0,
            max_contracts=0
        )
        
        result = StrategyResult(
            strategy_name="Test Strategy",
            market_id="TEST",
            timestamp=datetime(2024, 1, 1, 10, 0),
            signal=signal
        )
        
        result_dict = result.to_dict()
        
        assert result_dict['strategy_name'] == "Test Strategy"
        assert result_dict['market_id'] == "TEST"
        assert isinstance(result_dict['timestamp'], str)  # ISO format
        assert isinstance(result_dict['signal'], dict)
        assert result_dict['analysis_time_ms'] == 0
        assert result_dict['data_quality_score'] == 1.0


class TestBaseStrategy:
    """Test BaseStrategy abstract base class."""
    
    @pytest.fixture
    def config(self):
        return StrategyConfig(
            max_position_size=0.05,
            min_confidence=0.70,
            min_edge=0.04,
            max_trades_per_day=5
        )
    
    @pytest.fixture
    def strategy(self, config):
        return MockStrategy("Test Strategy", config)
    
    def test_strategy_initialization(self, strategy):
        """Test strategy initialization."""
        assert strategy.name == "Test Strategy"
        assert isinstance(strategy.config, StrategyConfig)
        assert strategy._initialized is False
        assert strategy._signals_generated == 0
        assert strategy._trade_count_today == 0
    
    @pytest.mark.asyncio
    async def test_strategy_initialize(self, strategy):
        """Test strategy initialization method."""
        await strategy.initialize()
        
        assert strategy.initialize_called is True
        assert strategy._initialized is True
    
    @pytest.mark.asyncio
    async def test_analyze_market(self, strategy):
        """Test market analysis."""
        market_data = {
            'market_price': 0.45,
            'volume': 5000,
            'status': 'open'
        }
        
        result = await strategy.analyze_market("TEST_MARKET", market_data)
        
        assert isinstance(result, StrategyResult)
        assert result.strategy_name == "Test Strategy"
        assert result.market_id == "TEST_MARKET"
        assert result.signal.market_id == "TEST_MARKET"
        assert strategy.analyze_called is True
    
    @pytest.mark.asyncio
    async def test_batch_analyze(self, strategy):
        """Test batch analysis of multiple markets."""
        markets_data = {
            "MARKET_1": {'market_price': 0.40, 'volume': 3000, 'status': 'open'},
            "MARKET_2": {'market_price': 0.60, 'volume': 4000, 'status': 'open'},
            "MARKET_3": {'market_price': 0.50, 'volume': 5000, 'status': 'open'}
        }
        
        results = await strategy.batch_analyze(markets_data)
        
        assert len(results) == 3
        assert all(isinstance(r, StrategyResult) for r in results)
        assert {r.market_id for r in results} == {"MARKET_1", "MARKET_2", "MARKET_3"}
        assert strategy._initialized is True  # Should auto-initialize
    
    def test_should_analyze_market(self, strategy):
        """Test market analysis filtering logic."""
        # Normal market should be analyzed
        normal_market = {'status': 'open', 'market_price': 0.45}
        assert strategy._should_analyze_market("NORMAL", normal_market) is True
        
        # Blacklisted market should not be analyzed
        strategy.config.markets_blacklist = ["BLACKLISTED"]
        assert strategy._should_analyze_market("BLACKLISTED", normal_market) is False
        
        # Inactive market should not be analyzed
        inactive_market = {'status': 'closed', 'market_price': 0.45}
        assert strategy._should_analyze_market("INACTIVE", inactive_market) is False
    
    def test_should_analyze_rate_limiting(self, strategy):
        """Test rate limiting logic."""
        market_data = {'status': 'open'}
        
        # First analysis should be allowed
        assert strategy._should_analyze_market("RATE_TEST", market_data) is True
        
        # Simulate trade to set last trade time
        strategy._last_trade_time["RATE_TEST"] = datetime.now()
        
        # Immediate re-analysis should be blocked
        assert strategy._should_analyze_market("RATE_TEST", market_data) is False
        
        # Analysis after sufficient time should be allowed
        past_time = datetime.now() - timedelta(seconds=strategy.config.min_time_between_trades + 1)
        strategy._last_trade_time["RATE_TEST"] = past_time
        assert strategy._should_analyze_market("RATE_TEST", market_data) is True
    
    def test_should_analyze_daily_limit(self, strategy):
        """Test daily trade limit logic."""
        market_data = {'status': 'open'}
        
        # Normal case should allow analysis
        assert strategy._should_analyze_market("DAILY_TEST", market_data) is True
        
        # Exceed daily limit
        strategy._trade_count_today = strategy.config.max_trades_per_day
        assert strategy._should_analyze_market("DAILY_TEST", market_data) is False
    
    def test_calculate_position_size_kelly(self, strategy):
        """Test Kelly criterion position sizing."""
        signal = Signal(
            signal_type=SignalType.BUY_YES,
            market_id="TEST",
            timestamp=datetime.now(),
            confidence=0.80,
            edge=0.08,  # 8% edge
            expected_value=40.0,
            recommended_size=0.04,
            max_contracts=100,
            metadata={'market_price': 0.40}
        )
        
        contracts = strategy.calculate_position_size(
            signal=signal,
            current_capital=10000,
            current_positions={}
        )
        
        assert contracts > 0
        assert contracts <= signal.max_contracts
        assert contracts <= (10000 * strategy.config.max_position_size / 0.40)
    
    def test_calculate_position_size_fixed(self, strategy):
        """Test fixed position sizing."""
        strategy.config.use_kelly_criterion = False
        strategy.config.fixed_position_size = 0.03
        
        signal = Signal(
            signal_type=SignalType.BUY_YES,
            market_id="TEST",
            timestamp=datetime.now(),
            confidence=0.75,
            edge=0.05,
            expected_value=25.0,
            recommended_size=0.02,  # Should be overridden by fixed size
            max_contracts=1000,  # High enough to not limit the test
            metadata={'market_price': 0.50}
        )
        
        contracts = strategy.calculate_position_size(
            signal=signal,
            current_capital=10000,
            current_positions={}
        )
        
        expected_contracts = int((10000 * 0.03) / 0.50)  # $300 / $0.50
        assert contracts == expected_contracts
    
    def test_calculate_position_size_portfolio_limit(self, strategy):
        """Test portfolio exposure limit."""
        signal = Signal(
            signal_type=SignalType.BUY_YES,
            market_id="TEST",
            timestamp=datetime.now(),
            confidence=0.90,
            edge=0.15,  # Very high edge
            expected_value=100.0,
            recommended_size=0.20,  # Would exceed individual limit
            max_contracts=1000,
            metadata={'market_price': 0.50}
        )
        
        # Portfolio at exposure limit
        current_positions = {
            "MARKET_1": 0.25,  # 25% exposure
            "MARKET_2": 0.25   # 25% exposure (total 50%)
        }
        
        contracts = strategy.calculate_position_size(
            signal=signal,
            current_capital=10000,
            current_positions=current_positions
        )
        
        assert contracts == 0  # Should not allow new positions
    
    def test_update_position_tracking(self, strategy):
        """Test position tracking updates."""
        # Open new position
        strategy.update_position(
            market_id="TRACK_TEST",
            action=PositionAction.OPEN_LONG,
            size=100,
            price=0.40
        )
        
        position = strategy.get_current_position("TRACK_TEST")
        assert position['size'] == 100
        assert position['avg_price'] == 0.40
        assert "TRACK_TEST" in strategy._last_trade_time
        assert strategy._trade_count_today == 1
        
        # Increase position
        strategy.update_position(
            market_id="TRACK_TEST",
            action=PositionAction.INCREASE_SIZE,
            size=50,
            price=0.50
        )
        
        updated_position = strategy.get_current_position("TRACK_TEST")
        assert updated_position['size'] == 150
        # Average price should be weighted: (100*0.40 + 50*0.50) / 150 = 0.433
        assert updated_position['avg_price'] == pytest.approx(0.433, rel=1e-2)
        
        # Close position
        strategy.update_position(
            market_id="TRACK_TEST",
            action=PositionAction.CLOSE_LONG,
            size=150,
            price=0.55
        )
        
        closed_position = strategy.get_current_position("TRACK_TEST")
        assert closed_position['size'] == 0
        assert closed_position['avg_price'] == 0
    
    def test_get_performance_stats(self, strategy):
        """Test performance statistics retrieval."""
        strategy._signals_generated = 20
        strategy._successful_trades = 12
        strategy._total_pnl = 150.0
        strategy._trade_count_today = 3
        
        stats = strategy.get_performance_stats()
        
        assert stats['strategy_name'] == "Test Strategy"
        assert stats['signals_generated'] == 20
        assert stats['successful_trades'] == 12
        assert stats['win_rate'] == 0.6  # 12/20
        assert stats['total_pnl'] == 150.0
        assert stats['trades_today'] == 3
        assert 'config' in stats
    
    def test_reset_daily_counters(self, strategy):
        """Test daily counter reset."""
        strategy._trade_count_today = 5
        strategy.reset_daily_counters()
        assert strategy._trade_count_today == 0
    
    @pytest.mark.asyncio
    async def test_cleanup(self, strategy):
        """Test strategy cleanup."""
        await strategy.initialize()
        assert strategy._initialized is True
        
        await strategy.cleanup()
        assert strategy._initialized is False


class TestStrategyManager:
    """Test StrategyManager class."""
    
    @pytest.fixture
    def manager(self):
        return StrategyManager()
    
    @pytest.fixture
    def sample_strategies(self):
        strategy1 = MockStrategy("Strategy A")
        strategy2 = MockStrategy("Strategy B") 
        return [strategy1, strategy2]
    
    def test_manager_initialization(self, manager):
        """Test manager initialization."""
        assert len(manager.strategies) == 0
        assert len(manager.active_signals) == 0
        assert len(manager.signal_history) == 0
        assert manager.max_history_size == 10000
    
    def test_add_remove_strategies(self, manager, sample_strategies):
        """Test adding and removing strategies."""
        strategy1, strategy2 = sample_strategies
        
        # Add strategies
        manager.add_strategy(strategy1)
        manager.add_strategy(strategy2)
        
        assert len(manager.strategies) == 2
        assert "Strategy A" in manager.strategies
        assert "Strategy B" in manager.strategies
        
        # Remove strategy
        manager.remove_strategy("Strategy A")
        
        assert len(manager.strategies) == 1
        assert "Strategy A" not in manager.strategies
        assert "Strategy B" in manager.strategies
    
    @pytest.mark.asyncio
    async def test_analyze_markets(self, manager, sample_strategies):
        """Test analyzing markets with multiple strategies."""
        strategy1, strategy2 = sample_strategies
        manager.add_strategy(strategy1)
        manager.add_strategy(strategy2)
        
        markets_data = {
            "MARKET_1": {'market_price': 0.40, 'volume': 3000, 'status': 'open'},
            "MARKET_2": {'market_price': 0.60, 'volume': 4000, 'status': 'open'}
        }
        
        results = await manager.analyze_markets(markets_data)
        
        assert len(results) == 2  # Two strategies
        assert "Strategy A" in results
        assert "Strategy B" in results
        assert len(results["Strategy A"]) == 2  # Two markets per strategy
        assert len(results["Strategy B"]) == 2
        
        # Check signal history was updated
        assert len(manager.signal_history) == 4  # 2 strategies × 2 markets
    
    def test_get_consolidated_signals(self, manager):
        """Test signal consolidation by market."""
        # Create mock results
        signal1 = Signal(
            signal_type=SignalType.BUY_YES,
            market_id="MARKET_1",
            timestamp=datetime.now(),
            confidence=0.8,
            edge=0.05,
            expected_value=25,
            recommended_size=0.02,
            max_contracts=50
        )
        
        signal2 = Signal(
            signal_type=SignalType.BUY_NO,
            market_id="MARKET_1",
            timestamp=datetime.now(),
            confidence=0.7,
            edge=0.04,
            expected_value=20,
            recommended_size=0.015,
            max_contracts=40
        )
        
        results = {
            "Strategy A": [
                StrategyResult("Strategy A", "MARKET_1", datetime.now(), signal1),
                StrategyResult("Strategy A", "MARKET_2", datetime.now(), 
                              Signal(SignalType.HOLD, "MARKET_2", datetime.now(), 0.3, 0, 0, 0, 0))
            ],
            "Strategy B": [
                StrategyResult("Strategy B", "MARKET_1", datetime.now(), signal2)
            ]
        }
        
        consolidated = manager.get_consolidated_signals(results)
        
        assert "MARKET_1" in consolidated
        assert "MARKET_2" not in consolidated  # HOLD signals filtered out
        assert len(consolidated["MARKET_1"]) == 2  # Two actionable signals for MARKET_1
    
    def test_get_manager_stats(self, manager, sample_strategies):
        """Test manager statistics."""
        strategy1, strategy2 = sample_strategies
        manager.add_strategy(strategy1)
        manager.add_strategy(strategy2)
        
        # Add some signal history
        test_signal = Signal(
            SignalType.BUY_YES, "TEST", datetime.now(), 0.7, 0.05, 20, 0.02, 40
        )
        manager.signal_history = [test_signal] * 5
        
        stats = manager.get_manager_stats()
        
        assert stats['num_strategies'] == 2
        assert set(stats['strategy_names']) == {"Strategy A", "Strategy B"}
        assert stats['signal_history_size'] == 5
        assert stats['active_signals'] == 0
        assert 'individual_stats' in stats
        assert 'Strategy A' in stats['individual_stats']
        assert 'Strategy B' in stats['individual_stats']
    
    def test_signal_history_limit(self, manager):
        """Test signal history size limiting."""
        manager.max_history_size = 3
        
        # Add signals beyond limit
        for i in range(5):
            signal = Signal(
                SignalType.BUY_YES, f"MARKET_{i}", datetime.now(), 0.7, 0.05, 20, 0.02, 40
            )
            manager.signal_history.append(signal)
        
        # Simulate trimming (would happen in analyze_markets)
        if len(manager.signal_history) > manager.max_history_size:
            manager.signal_history = manager.signal_history[-manager.max_history_size:]
        
        assert len(manager.signal_history) == 3
        # Should keep most recent signals
        assert manager.signal_history[0].market_id == "MARKET_2"
        assert manager.signal_history[-1].market_id == "MARKET_4"
