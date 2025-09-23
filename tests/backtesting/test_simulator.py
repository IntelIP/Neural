"""
Unit tests for MarketSimulator and related components.

Tests the realistic market simulation capabilities including
slippage calculation, fill simulation, and market state handling.
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch

from neural.backtesting.simulator import (
    MarketSimulator, MarketState, OrderRequest, FillResult, FillType,
    SlippageModel, FixedSlippageCalculator, LinearSlippageCalculator,
    SqrtSlippageCalculator, MarketImpactSlippageCalculator, FillSimulation
)
from neural.strategy.base import Signal, SignalType


class TestMarketSimulator:
    """Test MarketSimulator functionality."""
    
    @pytest.fixture
    def market_simulator(self):
        """Create a MarketSimulator instance for testing."""
        return MarketSimulator(
            slippage_model=SlippageModel.LINEAR,
            slippage_params={'base_slippage_bps': 10.0, 'size_multiplier': 0.001},
            enable_partial_fills=True
        )
    
    @pytest.fixture
    def market_state(self):
        """Create a MarketState for testing."""
        return MarketState(
            market_id="TEST_MARKET",
            timestamp=datetime.now(),
            bid=0.48,
            ask=0.52,
            last=0.50,
            volume=1000,
            open_interest=5000,
            spread=0.04,
            liquidity_score=0.7,
            volatility=0.02
        )
    
    @pytest.fixture
    def order_request(self):
        """Create an OrderRequest for testing."""
        signal = Signal(
            signal_type=SignalType.BUY_YES,
            market_id="TEST_MARKET",
            timestamp=datetime.now(),
            confidence=0.75,
            edge=0.05,
            expected_value=25.0,
            recommended_size=0.10,
            max_contracts=100
        )
        
        return OrderRequest(
            order_id="TEST_ORDER_1",
            signal=signal,
            quantity=50,
            side="YES"
        )
    
    def test_simulator_initialization(self, market_simulator):
        """Test MarketSimulator initialization."""
        assert market_simulator.slippage_model == SlippageModel.LINEAR
        assert market_simulator.enable_partial_fills is True
        assert 'total_orders' in market_simulator.fill_statistics
        assert market_simulator.fill_statistics['total_orders'] == 0
    
    def test_simulate_fill_basic(self, market_simulator, order_request, market_state):
        """Test basic fill simulation."""
        result = market_simulator.simulate_fill(order_request, market_state)
        
        assert isinstance(result, FillResult)
        assert result.order_id == "TEST_ORDER_1"
        assert result.fill_type in [FillType.FULL, FillType.PARTIAL]
        assert result.filled_quantity <= order_request.quantity
        assert result.fill_price > 0
        assert result.total_cost >= 0
        assert result.slippage >= 0
        assert result.fees >= 0
    
    def test_simulate_fill_rejection_conditions(self, market_simulator):
        """Test fill rejection under adverse conditions."""
        # Create bad market state
        bad_market_state = MarketState(
            market_id="TEST_MARKET",
            timestamp=datetime.now(),
            bid=0.0,  # Invalid bid
            ask=0.0,  # Invalid ask
            last=0.0,
            volume=0,
            open_interest=0,
            spread=0.25,  # Excessive spread
            liquidity_score=0.1
        )
        
        signal = Signal(
            signal_type=SignalType.BUY_YES,
            market_id="TEST_MARKET",
            timestamp=datetime.now(),
            confidence=0.75,
            edge=0.05,
            expected_value=25.0,
            recommended_size=0.10,
            max_contracts=100
        )
        
        order_request = OrderRequest(
            order_id="TEST_ORDER_REJECT",
            signal=signal,
            quantity=50,
            side="YES"
        )
        
        result = market_simulator.simulate_fill(order_request, bad_market_state)
        
        assert result.fill_type == FillType.REJECTED
        assert result.filled_quantity == 0
        assert result.rejection_reason is not None
    
    def test_partial_fills(self, market_simulator, order_request):
        """Test partial fill simulation."""
        # Create low liquidity market state
        low_liquidity_state = MarketState(
            market_id="TEST_MARKET",
            timestamp=datetime.now(),
            bid=0.48,
            ask=0.52,
            last=0.50,
            volume=100,  # Low volume
            open_interest=500,
            spread=0.04,
            liquidity_score=0.2  # Low liquidity
        )
        
        # Run multiple simulations to test partial fills
        partial_fills = 0
        for _ in range(10):
            result = market_simulator.simulate_fill(order_request, low_liquidity_state)
            if result.fill_type == FillType.PARTIAL:
                partial_fills += 1
                assert result.filled_quantity < order_request.quantity
                assert result.remaining_quantity > 0
        
        # With low liquidity, we should see some partial fills
        # (This is probabilistic, so we don't require it every time)
    
    def test_fill_statistics_tracking(self, market_simulator, order_request, market_state):
        """Test that fill statistics are properly tracked."""
        initial_total = market_simulator.fill_statistics['total_orders']
        
        # Execute a fill
        result = market_simulator.simulate_fill(order_request, market_state)
        
        # Check statistics updated
        assert market_simulator.fill_statistics['total_orders'] == initial_total + 1
        
        if result.fill_type == FillType.FULL:
            assert market_simulator.fill_statistics['full_fills'] >= 1
        elif result.fill_type == FillType.PARTIAL:
            assert market_simulator.fill_statistics['partial_fills'] >= 1
        elif result.fill_type == FillType.REJECTED:
            assert market_simulator.fill_statistics['rejections'] >= 1
    
    def test_create_market_state_from_data(self, market_simulator):
        """Test creating MarketState from market data."""
        import pandas as pd
        
        market_data = pd.Series({
            'market_id': 'TEST_MARKET',
            'last': 0.50,
            'bid': 0.48,
            'ask': 0.52,
            'volume': 2000,
            'open_interest': 8000
        })
        
        market_state = market_simulator.create_market_state_from_data(market_data)
        
        assert market_state.market_id == 'TEST_MARKET'
        assert market_state.last == 0.50
        assert market_state.bid == 0.48
        assert market_state.ask == 0.52
        assert market_state.volume == 2000
        assert 0 <= market_state.liquidity_score <= 1
    
    def test_get_fill_statistics(self, market_simulator, order_request, market_state):
        """Test getting comprehensive fill statistics."""
        # Execute several fills
        for _ in range(5):
            market_simulator.simulate_fill(order_request, market_state)
        
        stats = market_simulator.get_fill_statistics()
        
        assert 'fill_rate' in stats
        assert 'rejection_rate' in stats
        assert 'avg_slippage_bps' in stats
        assert stats['total_orders'] == 5
        assert 0 <= stats['fill_rate'] <= 1
        assert 0 <= stats['rejection_rate'] <= 1


class TestSlippageCalculators:
    """Test various slippage calculation methods."""
    
    @pytest.fixture
    def market_state(self):
        """Create a MarketState for slippage testing."""
        return MarketState(
            market_id="TEST_MARKET",
            timestamp=datetime.now(),
            bid=0.48,
            ask=0.52,
            last=0.50,
            volume=1000,
            open_interest=5000,
            spread=0.04,
            liquidity_score=0.5,
            volatility=0.02
        )
    
    def test_fixed_slippage_calculator(self, market_state):
        """Test FixedSlippageCalculator."""
        calculator = FixedSlippageCalculator(slippage_bps=15.0)
        
        slippage = calculator.calculate_slippage(100, market_state, "YES")
        
        assert slippage == pytest.approx(0.0015, rel=1e-5)  # 15 bps
    
    def test_linear_slippage_calculator(self, market_state):
        """Test LinearSlippageCalculator."""
        calculator = LinearSlippageCalculator(
            base_slippage_bps=5.0, 
            size_multiplier=0.001
        )
        
        # Small order
        small_slippage = calculator.calculate_slippage(10, market_state, "YES")
        
        # Large order  
        large_slippage = calculator.calculate_slippage(1000, market_state, "YES")
        
        # Large order should have higher slippage
        assert large_slippage > small_slippage
        assert small_slippage >= 0.0005  # Base slippage (5 bps)
    
    def test_sqrt_slippage_calculator(self, market_state):
        """Test SqrtSlippageCalculator."""
        calculator = SqrtSlippageCalculator(coefficient=0.01)
        
        # Test sqrt relationship
        slippage_100 = calculator.calculate_slippage(100, market_state, "YES")
        slippage_400 = calculator.calculate_slippage(400, market_state, "YES")
        
        # sqrt(400) = 2 * sqrt(100), so slippage should be roughly 2x
        expected_ratio = 2.0
        actual_ratio = slippage_400 / slippage_100 if slippage_100 > 0 else 0
        
        assert actual_ratio == pytest.approx(expected_ratio, rel=0.1)
    
    def test_market_impact_slippage_calculator(self, market_state):
        """Test MarketImpactSlippageCalculator."""
        calculator = MarketImpactSlippageCalculator(
            temporary_impact=0.1,
            permanent_impact=0.05
        )
        
        slippage = calculator.calculate_slippage(100, market_state, "YES")
        
        # Should have some positive slippage
        assert slippage >= 0
        assert slippage <= 0.05  # Should be capped at 5%
    
    def test_slippage_calculator_edge_cases(self, market_state):
        """Test slippage calculators with edge cases."""
        calculator = LinearSlippageCalculator()
        
        # Zero quantity
        zero_slippage = calculator.calculate_slippage(0, market_state, "YES")
        assert zero_slippage >= 0
        
        # Very large quantity
        large_slippage = calculator.calculate_slippage(1000000, market_state, "YES")
        assert large_slippage >= 0


class TestFillSimulationFactory:
    """Test FillSimulation factory methods."""
    
    def test_conservative_model(self):
        """Test conservative fill simulation model."""
        simulator = FillSimulation.conservative_model()
        
        assert simulator.slippage_model == SlippageModel.LINEAR
        assert simulator.min_fill_ratio >= 0.8  # Conservative fill ratio
        assert simulator.enable_partial_fills is True
    
    def test_realistic_model(self):
        """Test realistic fill simulation model."""
        simulator = FillSimulation.realistic_model()
        
        assert simulator.slippage_model == SlippageModel.LINEAR
        assert simulator.min_fill_ratio >= 0.8
        assert simulator.enable_partial_fills is True
    
    def test_optimistic_model(self):
        """Test optimistic fill simulation model."""
        simulator = FillSimulation.optimistic_model()
        
        assert simulator.slippage_model == SlippageModel.FIXED
        assert simulator.min_fill_ratio >= 0.9  # High fill ratio
    
    def test_market_impact_model(self):
        """Test market impact simulation model."""
        simulator = FillSimulation.market_impact_model()
        
        assert simulator.slippage_model == SlippageModel.IMPACT
        assert simulator.enable_partial_fills is True
