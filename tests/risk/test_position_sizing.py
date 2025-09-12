"""
Unit tests for Position Sizing algorithms.

Tests the various position sizing methods including Kelly Criterion,
fixed sizing, volatility-based sizing, and the orchestration layer.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, patch

from neural.risk.position_sizing import (
    PositionSizer, KellySizer, FixedSizer, VolatilitySizer,
    PositionSizingMethod, PositionSizeResult, BaseSizer
)
from neural.strategy.base import Signal, SignalType


class TestKellySizer:
    """Test Kelly Criterion position sizing."""
    
    @pytest.fixture
    def kelly_sizer(self):
        """Create KellySizer instance for testing."""
        return KellySizer(
            max_kelly_fraction=0.25,
            min_kelly_fraction=0.01,
            max_position_size=0.10,
            confidence_threshold=0.6,
            edge_threshold=0.02
        )
    
    @pytest.fixture
    def strong_signal(self):
        """Create a strong signal for testing."""
        return Signal(
            signal_type=SignalType.BUY_YES,
            market_id="STRONG_SIGNAL",
            timestamp=datetime.now(),
            confidence=0.80,
            edge=0.10,  # 10% edge
            expected_value=50.0,
            recommended_size=0.15,
            max_contracts=200,
            metadata={'market_price': 0.45}
        )
    
    @pytest.fixture
    def weak_signal(self):
        """Create a weak signal for testing."""
        return Signal(
            signal_type=SignalType.BUY_YES,
            market_id="WEAK_SIGNAL", 
            timestamp=datetime.now(),
            confidence=0.55,
            edge=0.01,  # 1% edge
            expected_value=10.0,
            recommended_size=0.05,
            max_contracts=50,
            metadata={'market_price': 0.60}
        )
    
    def test_kelly_sizer_initialization(self, kelly_sizer):
        """Test KellySizer initialization."""
        assert kelly_sizer.max_kelly_fraction == 0.25
        assert kelly_sizer.min_kelly_fraction == 0.01
        assert kelly_sizer.max_position_size == 0.10
        assert kelly_sizer.confidence_threshold == 0.6
        assert kelly_sizer.edge_threshold == 0.02
    
    def test_calculate_size_strong_signal(self, kelly_sizer, strong_signal):
        """Test Kelly sizing with a strong signal."""
        result = kelly_sizer.calculate_size(strong_signal, 10000, {})
        
        assert isinstance(result, PositionSizeResult)
        assert result.method == PositionSizingMethod.KELLY_CRITERION
        assert result.recommended_size > 0
        assert result.recommended_contracts > 0
        assert result.risk_percentage <= 0.10  # Capped by max_position_size
        assert result.confidence == 0.80
        assert 'kelly_fraction' in result.metadata
    
    def test_calculate_size_weak_signal(self, kelly_sizer, weak_signal):
        """Test Kelly sizing with a weak signal."""
        result = kelly_sizer.calculate_size(weak_signal, 10000, {})
        
        assert isinstance(result, PositionSizeResult)
        assert result.method == PositionSizingMethod.KELLY_CRITERION
        assert result.recommended_size > 0
        
        # Should have warnings due to low edge/confidence
        assert len(result.warnings) > 0
        assert any('Low' in warning for warning in result.warnings)
    
    def test_calculate_size_negative_edge(self, kelly_sizer):
        """Test Kelly sizing with negative edge."""
        negative_signal = Signal(
            signal_type=SignalType.BUY_YES,
            market_id="NEGATIVE_EDGE",
            timestamp=datetime.now(),
            confidence=0.70,
            edge=-0.05,  # Negative edge
            expected_value=-25.0,
            recommended_size=0.10,
            max_contracts=100,
            metadata={'market_price': 0.55}
        )
        
        result = kelly_sizer.calculate_size(negative_signal, 10000, {})
        
        # Should use minimum position size
        assert result.recommended_size > 0  # Still positive due to minimum
        assert len(result.warnings) > 0
        assert any('negative edge' in warning.lower() for warning in result.warnings)
    
    def test_calculate_size_position_capping(self, kelly_sizer):
        """Test position size capping."""
        huge_edge_signal = Signal(
            signal_type=SignalType.BUY_YES,
            market_id="HUGE_EDGE",
            timestamp=datetime.now(),
            confidence=0.95,
            edge=0.50,  # Unrealistic but huge edge
            expected_value=500.0,
            recommended_size=0.50,
            max_contracts=1000,
            metadata={'market_price': 0.30}
        )
        
        result = kelly_sizer.calculate_size(huge_edge_signal, 10000, {})
        
        # Should be capped at max_position_size
        assert result.risk_percentage <= kelly_sizer.max_position_size
        assert len(result.warnings) > 0
        assert any('capped' in warning.lower() for warning in result.warnings)


class TestFixedSizer:
    """Test fixed position sizing."""
    
    @pytest.fixture
    def percentage_sizer(self):
        """Create percentage-based FixedSizer."""
        return FixedSizer(
            sizing_type="percentage",
            size_value=0.03,  # 3% of capital
            confidence_scaling=True
        )
    
    @pytest.fixture
    def amount_sizer(self):
        """Create amount-based FixedSizer."""
        return FixedSizer(
            sizing_type="amount",
            size_value=500.0,  # $500 fixed
            confidence_scaling=False
        )
    
    @pytest.fixture
    def test_signal(self):
        """Create test signal."""
        return Signal(
            signal_type=SignalType.BUY_YES,
            market_id="TEST_MARKET",
            timestamp=datetime.now(),
            confidence=0.75,
            edge=0.05,
            expected_value=25.0,
            recommended_size=0.10,
            max_contracts=100,
            metadata={'market_price': 0.50}
        )
    
    def test_percentage_sizing(self, percentage_sizer, test_signal):
        """Test percentage-based sizing."""
        result = percentage_sizer.calculate_size(test_signal, 20000, {})
        
        assert isinstance(result, PositionSizeResult)
        assert result.method == PositionSizingMethod.FIXED_PERCENTAGE
        assert result.recommended_size > 0
        
        # Should be approximately 3% * confidence * capital
        expected_base = 20000 * 0.03 * 0.75  # 3% * 75% confidence
        assert result.recommended_size <= expected_base  # May be less due to contract calculation
        
        assert 'confidence_scaling' in result.metadata
        assert result.metadata['confidence_scaling'] is True
    
    def test_amount_sizing(self, amount_sizer, test_signal):
        """Test fixed amount sizing."""
        result = amount_sizer.calculate_size(test_signal, 20000, {})
        
        assert isinstance(result, PositionSizeResult)
        assert result.method == PositionSizingMethod.FIXED_AMOUNT
        assert result.recommended_size > 0
        
        # Should be close to $500 (adjusted for fees/contracts)
        assert result.recommended_size <= 500  # May be less due to fees
        assert result.metadata['confidence_scaling'] is False
    
    def test_minimum_position_size(self, percentage_sizer):
        """Test minimum position size enforcement."""
        low_confidence_signal = Signal(
            signal_type=SignalType.BUY_YES,
            market_id="LOW_CONFIDENCE",
            timestamp=datetime.now(),
            confidence=0.10,  # Very low confidence
            edge=0.02,
            expected_value=5.0,
            recommended_size=0.02,
            max_contracts=50,
            metadata={'market_price': 0.50}
        )
        
        result = percentage_sizer.calculate_size(low_confidence_signal, 1000, {})  # Small capital
        
        # Should enforce minimum position size (allow small tolerance for fees/rounding)
        assert result.recommended_size >= percentage_sizer.min_position_size * 0.95  # 95% tolerance
        assert len(result.warnings) > 0
        assert any('minimum' in warning.lower() for warning in result.warnings)


class TestVolatilitySizer:
    """Test volatility-based position sizing."""
    
    @pytest.fixture
    def volatility_sizer(self):
        """Create VolatilitySizer instance."""
        return VolatilitySizer(
            target_volatility=0.02,  # 2% daily target
            max_position_size=0.15,
            min_position_size=0.01
        )
    
    @pytest.fixture
    def test_signal(self):
        """Create test signal."""
        return Signal(
            signal_type=SignalType.BUY_YES,
            market_id="VOL_TEST",
            timestamp=datetime.now(),
            confidence=0.70,
            edge=0.04,
            expected_value=20.0,
            recommended_size=0.08,
            max_contracts=150,
            metadata={'market_price': 0.48}
        )
    
    def test_volatility_sizing_basic(self, volatility_sizer, test_signal):
        """Test basic volatility sizing."""
        result = volatility_sizer.calculate_size(test_signal, 15000, {})
        
        assert isinstance(result, PositionSizeResult)
        assert result.method == PositionSizingMethod.VOLATILITY_TARGET
        assert result.recommended_size > 0
        # Allow small tolerance for rounding/fees in risk percentage
        assert result.risk_percentage <= volatility_sizer.max_position_size
        assert result.risk_percentage >= volatility_sizer.min_position_size * 0.95  # 95% tolerance
        
        assert 'estimated_volatility' in result.metadata
        assert 'target_volatility' in result.metadata
        assert 'volatility_ratio' in result.metadata
    
    def test_volatility_estimation_with_data(self, volatility_sizer, test_signal):
        """Test volatility estimation with market data."""
        import pandas as pd
        
        # Create mock price history
        price_history = pd.Series([0.45, 0.47, 0.46, 0.49, 0.48, 0.50, 0.47])
        market_data = {'price_history': price_history}
        
        result = volatility_sizer.calculate_size(test_signal, 15000, {}, market_data)
        
        assert result.recommended_size > 0
        assert result.metadata['estimated_volatility'] > 0
    
    def test_volatility_bounds(self, volatility_sizer, test_signal):
        """Test volatility floor and cap enforcement."""
        # Test with extreme volatility (should be capped)
        very_volatile_data = {
            'price_history': pd.Series([0.10, 0.90, 0.20, 0.80, 0.15])  # Extreme swings
        }
        
        result = volatility_sizer.calculate_size(test_signal, 15000, {}, very_volatile_data)
        
        # Should be capped at volatility_cap
        assert result.metadata['estimated_volatility'] <= volatility_sizer.volatility_cap
        
        # Test with no data (should use floor)
        result_no_data = volatility_sizer.calculate_size(test_signal, 15000, {}, {})
        assert result_no_data.metadata['estimated_volatility'] >= volatility_sizer.volatility_floor


class TestPositionSizer:
    """Test the main PositionSizer orchestrator."""
    
    @pytest.fixture
    def position_sizer(self):
        """Create PositionSizer instance."""
        return PositionSizer(
            primary_method=PositionSizingMethod.KELLY_CRITERION,
            fallback_method=PositionSizingMethod.FIXED_PERCENTAGE
        )
    
    @pytest.fixture
    def test_signal(self):
        """Create test signal."""
        return Signal(
            signal_type=SignalType.BUY_NO,
            market_id="ORCHESTRATOR_TEST",
            timestamp=datetime.now(),
            confidence=0.85,
            edge=0.08,
            expected_value=40.0,
            recommended_size=0.12,
            max_contracts=200,
            metadata={'market_price': 0.42}
        )
    
    def test_position_sizer_initialization(self, position_sizer):
        """Test PositionSizer initialization."""
        assert position_sizer.primary_method == PositionSizingMethod.KELLY_CRITERION
        assert position_sizer.fallback_method == PositionSizingMethod.FIXED_PERCENTAGE
        assert PositionSizingMethod.KELLY_CRITERION in position_sizer.sizers
        assert PositionSizingMethod.FIXED_PERCENTAGE in position_sizer.sizers
    
    def test_calculate_position_size_primary_method(self, position_sizer, test_signal):
        """Test position sizing with primary method."""
        result = position_sizer.calculate_position_size(test_signal, 25000, {})
        
        assert isinstance(result, PositionSizeResult)
        assert result.method == PositionSizingMethod.KELLY_CRITERION  # Should use primary
        assert result.recommended_size > 0
        assert result.recommended_contracts >= 0
        assert result.confidence == 0.85
    
    def test_calculate_position_size_override_method(self, position_sizer, test_signal):
        """Test position sizing with method override."""
        result = position_sizer.calculate_position_size(
            signal=test_signal,
            current_capital=25000,
            current_positions={},
            override_method=PositionSizingMethod.FIXED_PERCENTAGE
        )
        
        assert result.method == PositionSizingMethod.FIXED_PERCENTAGE
    
    def test_calculate_position_size_with_existing_positions(self, position_sizer, test_signal):
        """Test position sizing with existing positions."""
        existing_positions = {
            'EXISTING_MARKET': {
                'market_value': 2500,  # $2500 existing position
                'contracts': 50
            }
        }
        
        result = position_sizer.calculate_position_size(test_signal, 25000, existing_positions)
        
        # Should still work with existing positions
        assert result.recommended_size > 0
    
    def test_emergency_fallback(self, position_sizer):
        """Test emergency fallback when all methods fail."""
        invalid_signal = Signal(
            signal_type=SignalType.BUY_YES,
            market_id="INVALID",
            timestamp=datetime.now(),
            confidence=-1.0,  # Invalid confidence
            edge=0.05,
            expected_value=25.0,
            recommended_size=0.10,
            max_contracts=100,
            metadata={'market_price': 0.50}
        )
        
        # This should trigger emergency fallback
        result = position_sizer.calculate_position_size(invalid_signal, 10000, {})
        
        assert result.recommended_size > 0  # Should still return something
        assert len(result.warnings) > 0
        assert any('emergency' in warning.lower() for warning in result.warnings)
    
    def test_get_sizing_statistics(self, position_sizer):
        """Test getting sizing statistics across multiple signals."""
        signals = [
            Signal(
                signal_type=SignalType.BUY_YES,
                market_id=f"STAT_TEST_{i}",
                timestamp=datetime.now(),
                confidence=0.60 + i * 0.1,
                edge=0.03 + i * 0.01,
                expected_value=15.0 + i * 5,
                recommended_size=0.05 + i * 0.02,
                max_contracts=50 + i * 10,
                metadata={'market_price': 0.45 + i * 0.05}
            ) for i in range(3)
        ]
        
        stats = position_sizer.get_sizing_statistics(signals, 20000)
        
        assert 'total_signals' in stats
        assert 'successful_sizing' in stats
        assert 'total_allocation' in stats
        assert 'allocation_percentage' in stats
        assert 'average_risk_per_position' in stats
        assert 'method_distribution' in stats
        
        assert stats['total_signals'] == 3
        assert stats['successful_sizing'] <= 3
        assert stats['total_allocation'] >= 0
        assert 0 <= stats['allocation_percentage'] <= 1
    
    def test_validation_edge_cases(self, position_sizer):
        """Test position sizer validation with edge cases."""
        valid_signal = Signal(
            signal_type=SignalType.BUY_YES,
            market_id="VALID",
            timestamp=datetime.now(),
            confidence=0.75,
            edge=0.05,
            expected_value=25.0,
            recommended_size=0.10,
            max_contracts=100,
            metadata={'market_price': 0.50}
        )
        
        # Test with zero capital (should use emergency fallback)
        result_zero = position_sizer.calculate_position_size(valid_signal, 0, {})
        assert result_zero.recommended_size >= 0
        assert 'emergency' in result_zero.warnings[0].lower()
        
        # Test with negative capital (should also use emergency fallback)
        result_negative = position_sizer.calculate_position_size(valid_signal, -1000, {})
        assert result_negative.recommended_size >= 0
        assert 'emergency' in result_negative.warnings[0].lower()
