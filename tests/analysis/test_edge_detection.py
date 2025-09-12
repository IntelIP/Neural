"""
Unit tests for edge detection and market inefficiency components.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from neural.analysis.edge_detection import (
    EdgeCalculator,
    MarketInefficiencyDetector
)
from neural.analysis.base import (
    AnalysisConfig,
    AnalysisType,
    SignalStrength
)


class TestEdgeCalculator:
    """Test EdgeCalculator class."""
    
    @pytest.fixture
    def config(self):
        return AnalysisConfig(
            min_edge_threshold=0.03,
            confidence_level=0.95,
            max_position_size=0.10
        )
    
    @pytest.fixture
    def edge_calculator(self, config):
        return EdgeCalculator(config)
    
    @pytest.mark.asyncio
    async def test_initialization(self, edge_calculator):
        await edge_calculator.initialize()
        assert edge_calculator._initialized is True
    
    @pytest.mark.asyncio
    async def test_analyze_basic_edge(self, edge_calculator):
        """Test basic edge calculation with valid data."""
        market_data = {
            'market_price': 0.40,
            'true_probability': 0.55,  # 15% edge
            'sportsbook_consensus': 0.52,
            'volume_data': {
                'volume_24h': 5000,
                'open_interest': 10000,
                'spread': 0.02
            }
        }
        
        result = await edge_calculator.analyze("TEST_MARKET", market_data)
        
        assert result.analysis_type == AnalysisType.EDGE_DETECTION
        assert result.market_id == "TEST_MARKET"
        assert result.is_valid
        assert result.value > 0  # Should have positive edge
        assert result.confidence > 0.5  # Should have reasonable confidence
        assert result.signal in ['BUY_YES', 'BUY_NO']
        assert result.signal_strength != SignalStrength.NEUTRAL
    
    @pytest.mark.asyncio
    async def test_analyze_negative_edge(self, edge_calculator):
        """Test analysis with negative edge (market overpricing)."""
        market_data = {
            'market_price': 0.70,
            'true_probability': 0.40,  # -30% edge (stronger signal)
            'sportsbook_consensus': 0.42,  # Closer agreement for higher confidence
            'volume_data': {
                'volume_24h': 8000,  # Higher volume for higher confidence
                'spread': 0.02
            }
        }
        
        result = await edge_calculator.analyze("TEST_MARKET", market_data)
        
        assert result.is_valid
        assert result.value < 0  # Should have negative edge
        assert result.signal == 'BUY_NO'  # Should recommend buying NO
    
    @pytest.mark.asyncio
    async def test_analyze_no_edge(self, edge_calculator):
        """Test analysis when no significant edge exists."""
        market_data = {
            'market_price': 0.50,
            'true_probability': 0.52,  # Only 2% edge (below threshold)
            'volume_data': {
                'volume_24h': 1000,
                'spread': 0.02
            }
        }
        
        result = await edge_calculator.analyze("TEST_MARKET", market_data)
        
        assert result.is_valid
        assert abs(result.value) < edge_calculator.config.min_edge_threshold
        assert result.signal == 'HOLD'
        assert result.signal_strength == SignalStrength.NEUTRAL
    
    @pytest.mark.asyncio
    async def test_analyze_missing_required_data(self, edge_calculator):
        """Test analysis with missing required data."""
        incomplete_data = {
            'market_price': 0.40
            # Missing true_probability
        }
        
        result = await edge_calculator.analyze("TEST_MARKET", incomplete_data)
        
        assert not result.is_valid
        assert len(result.errors) > 0
        assert result.confidence == 0
    
    @pytest.mark.asyncio
    async def test_batch_analyze(self, edge_calculator):
        """Test batch analysis of multiple markets."""
        markets = ["MARKET_1", "MARKET_2", "MARKET_3"]
        
        # Mock the analyze method to return valid results
        async def mock_analyze(market_id, data=None):
            from neural.analysis.base import AnalysisResult, AnalysisType, SignalStrength
            from datetime import datetime
            return AnalysisResult(
                analysis_type=AnalysisType.EDGE_DETECTION,
                timestamp=datetime.now(),
                market_id=market_id,
                value=0.05,
                confidence=0.7,
                signal='BUY_YES',
                signal_strength=SignalStrength.MODERATE,
                components={},
                metadata={}
            )
        
        with patch.object(edge_calculator, 'analyze', side_effect=mock_analyze):
            results = await edge_calculator.batch_analyze(markets)
        
        assert len(results) == 3
        assert all(result.is_valid for result in results)
        assert {result.market_id for result in results} == set(markets)
    
    def test_calculate_comprehensive_edge(self, edge_calculator):
        """Test comprehensive edge calculation with all factors."""
        edge_results = edge_calculator._calculate_comprehensive_edge(
            market_price=0.40,
            true_probability=0.60,  # 20% raw edge
            sportsbook_consensus=0.58,  # Close agreement
            volume_data={
                'volume_24h': 8000,
                'open_interest': 15000,
                'spread': 0.015
            }
        )
        
        assert 'raw_edge' in edge_results
        assert 'adjusted_edge' in edge_results
        assert 'confidence' in edge_results
        assert edge_results['raw_edge'] == pytest.approx(0.20, rel=1e-2)
        assert edge_results['adjusted_edge'] > 0
        assert edge_results['confidence'] > 0.5
    
    def test_confidence_factors_calculation(self, edge_calculator):
        """Test confidence factor calculations."""
        factors = edge_calculator._calculate_confidence_factors(
            market_price=0.45,
            sportsbook_consensus=0.47,  # Close to market
            volume_data={
                'volume_24h': 5000,
                'open_interest': 10000,
                'close_time': datetime.now() + timedelta(hours=48)
            }
        )
        
        assert 'sportsbook_agreement' in factors
        assert 'volume_confidence' in factors
        assert 'time_confidence' in factors
        assert 'overall_confidence' in factors
        assert 0 <= factors['overall_confidence'] <= 1
        assert factors['sportsbook_agreement'] > 0.5  # Should be high (close agreement)
    
    def test_liquidity_adjustment(self, edge_calculator):
        """Test liquidity adjustment calculations."""
        # High liquidity scenario
        high_liquidity = edge_calculator._calculate_liquidity_adjustment({
            'volume_24h': 10000,
            'spread': 0.01
        })
        
        # Low liquidity scenario
        low_liquidity = edge_calculator._calculate_liquidity_adjustment({
            'volume_24h': 500,
            'spread': 0.08
        })
        
        assert 0.5 <= high_liquidity <= 1.0
        assert 0.5 <= low_liquidity <= 1.0
        assert high_liquidity > low_liquidity  # Higher liquidity should have higher adjustment
    
    def test_signal_generation(self, edge_calculator):
        """Test signal generation logic."""
        # Strong positive edge
        signal, strength = edge_calculator._generate_signal(0.12, 0.85)
        assert signal == 'BUY_YES'
        assert strength in [SignalStrength.STRONG, SignalStrength.VERY_STRONG]
        
        # Strong negative edge
        signal, strength = edge_calculator._generate_signal(-0.08, 0.70)
        assert signal == 'BUY_NO'
        assert strength in [SignalStrength.MODERATE, SignalStrength.STRONG]
        
        # Weak edge
        signal, strength = edge_calculator._generate_signal(0.02, 0.40)
        assert signal == 'HOLD'
        assert strength == SignalStrength.NEUTRAL
    
    def test_position_size_recommendation(self, edge_calculator):
        """Test position sizing recommendations."""
        recommendation = edge_calculator.calculate_position_size_recommendation(
            edge=0.10,
            confidence=0.80,
            bankroll=10000,
            market_price=0.40
        )
        
        assert 'recommended_contracts' in recommendation
        assert 'recommended_dollars' in recommendation
        assert 'kelly_fraction' in recommendation
        assert recommendation['recommended_contracts'] >= 0
        assert recommendation['recommended_dollars'] >= 0
        assert 0 <= recommendation['kelly_fraction'] <= edge_calculator.config.max_position_size
    
    def test_position_size_no_edge(self, edge_calculator):
        """Test position sizing with no edge."""
        recommendation = edge_calculator.calculate_position_size_recommendation(
            edge=-0.05,  # Negative edge
            confidence=0.60,
            bankroll=10000,
            market_price=0.50
        )
        
        assert recommendation['recommended_contracts'] == 0
        assert recommendation['recommended_dollars'] == 0
        assert recommendation['kelly_fraction'] == 0
        assert 'No positive edge' in recommendation['reason']
    
    def test_required_fields(self, edge_calculator):
        """Test required fields specification."""
        required = edge_calculator.get_required_fields()
        assert 'market_price' in required
        assert 'true_probability' in required


class TestMarketInefficiencyDetector:
    """Test MarketInefficiencyDetector class."""
    
    @pytest.fixture
    def detector(self):
        return MarketInefficiencyDetector()
    
    def test_detect_price_jump_anomaly(self, detector):
        """Test detection of price jump anomalies."""
        # Create historical data with stable prices
        dates = pd.date_range('2024-01-01', periods=10, freq='H')
        stable_prices = [0.45, 0.46, 0.44, 0.45, 0.46, 0.45, 0.44, 0.46, 0.45, 0.44]
        
        historical_data = pd.DataFrame({
            'last': stable_prices,
            'volume': [1000] * 10
        }, index=dates)
        
        # Current data with price jump
        current_data = {
            'last_price': 0.60,  # Significant jump from ~0.45
            'volume': 1200
        }
        
        anomalies = detector.detect_anomalies(
            "TEST_MARKET", 
            current_data, 
            historical_data
        )
        
        assert anomalies['price_jump'] is True
        assert anomalies['severity_score'] > 0
    
    def test_detect_volume_spike(self, detector):
        """Test detection of volume spike anomalies."""
        dates = pd.date_range('2024-01-01', periods=10, freq='H')
        normal_volumes = [1000, 1200, 800, 1100, 900, 1000, 1300, 950, 1050, 1150]
        
        historical_data = pd.DataFrame({
            'last': [0.45] * 10,
            'volume': normal_volumes
        }, index=dates)
        
        # Current data with volume spike
        current_data = {
            'last_price': 0.46,
            'volume': 5000  # 5x normal volume
        }
        
        anomalies = detector.detect_anomalies(
            "TEST_MARKET",
            current_data,
            historical_data
        )
        
        assert anomalies['volume_spike'] is True
        assert anomalies['severity_score'] > 3  # Should reflect volume ratio
    
    def test_detect_spread_widening(self, detector):
        """Test detection of spread widening."""
        dates = pd.date_range('2024-01-01', periods=10, freq='H')
        
        historical_data = pd.DataFrame({
            'last': [0.45] * 10,
            'volume': [1000] * 10,
            'bid': [0.44] * 10,
            'ask': [0.46] * 10  # Normal 2 cent spread
        }, index=dates)
        
        # Current data with wide spread
        current_data = {
            'last_price': 0.45,
            'volume': 1000,
            'spread': 0.08  # 8 cent spread (4x normal)
        }
        
        anomalies = detector.detect_anomalies(
            "TEST_MARKET",
            current_data,
            historical_data
        )
        
        assert anomalies['spread_widening'] is True
    
    def test_no_anomalies_detected(self, detector):
        """Test when no anomalies are present."""
        dates = pd.date_range('2024-01-01', periods=10, freq='H')
        
        historical_data = pd.DataFrame({
            'last': [0.45, 0.46, 0.44, 0.45, 0.46, 0.45, 0.44, 0.46, 0.45, 0.44],
            'volume': [1000, 1200, 800, 1100, 900, 1000, 1300, 950, 1050, 1150]
        }, index=dates)
        
        # Normal current data
        current_data = {
            'last_price': 0.45,
            'volume': 1100,
            'spread': 0.02
        }
        
        anomalies = detector.detect_anomalies(
            "TEST_MARKET",
            current_data,
            historical_data
        )
        
        assert anomalies['price_jump'] is False
        assert anomalies['volume_spike'] is False
        assert anomalies['severity_score'] == 0
    
    def test_insufficient_historical_data(self, detector):
        """Test behavior with insufficient historical data."""
        # Very short history
        dates = pd.date_range('2024-01-01', periods=2, freq='H')
        historical_data = pd.DataFrame({
            'last': [0.45, 0.46],
            'volume': [1000, 1200]
        }, index=dates)
        
        current_data = {
            'last_price': 0.60,  # Would be anomaly with more data
            'volume': 5000
        }
        
        anomalies = detector.detect_anomalies(
            "TEST_MARKET",
            current_data,
            historical_data
        )
        
        # Should return default (no anomalies) due to insufficient data
        assert anomalies['price_jump'] is False
        assert anomalies['volume_spike'] is False
        assert anomalies['severity_score'] == 0
    
    def test_calculate_momentum_score(self, detector):
        """Test momentum score calculation."""
        # Upward trending prices
        upward_prices = pd.Series([0.40, 0.42, 0.44, 0.46, 0.48, 0.50])
        upward_momentum = detector.calculate_momentum_score(upward_prices)
        assert upward_momentum > 0
        
        # Downward trending prices  
        downward_prices = pd.Series([0.50, 0.48, 0.46, 0.44, 0.42, 0.40])
        downward_momentum = detector.calculate_momentum_score(downward_prices)
        assert downward_momentum < 0
        
        # Sideways prices
        sideways_prices = pd.Series([0.45, 0.46, 0.45, 0.44, 0.45, 0.46])
        sideways_momentum = detector.calculate_momentum_score(sideways_prices)
        assert abs(sideways_momentum) < 0.1  # Should be near zero
    
    def test_momentum_insufficient_data(self, detector):
        """Test momentum calculation with insufficient data."""
        short_series = pd.Series([0.45, 0.46])  # Only 2 points
        momentum = detector.calculate_momentum_score(short_series)
        assert momentum == 0  # Should return neutral
    
    def test_momentum_score_bounds(self, detector):
        """Test that momentum scores are properly bounded."""
        # Extreme upward movement
        extreme_up = pd.Series([0.01, 0.99, 0.99, 0.99, 0.99, 0.99])
        momentum = detector.calculate_momentum_score(extreme_up)
        assert -1 <= momentum <= 1  # Should be clipped
        
        # Extreme downward movement
        extreme_down = pd.Series([0.99, 0.01, 0.01, 0.01, 0.01, 0.01])
        momentum = detector.calculate_momentum_score(extreme_down)
        assert -1 <= momentum <= 1  # Should be clipped
