"""
Unit tests for probability calculation engines and odds conversion.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
import numpy as np

from neural.analysis.probability import (
    ProbabilityEngine,
    ProbabilityMethod,
    ProbabilityEstimate,
    SportsbookOdds,
    OddsConverter
)
from neural.analysis.base import (
    AnalysisConfig,
    AnalysisType,
    SignalStrength
)


class TestProbabilityEngine:
    """Test ProbabilityEngine class."""
    
    @pytest.fixture
    def config(self):
        return AnalysisConfig(
            confidence_level=0.95,
            min_edge_threshold=0.03
        )
    
    @pytest.fixture
    def probability_engine(self, config):
        return ProbabilityEngine(config)
    
    @pytest.mark.asyncio
    async def test_initialization(self, probability_engine):
        await probability_engine.initialize()
        assert probability_engine._initialized is True
    
    @pytest.mark.asyncio
    async def test_analyze_with_sportsbook_data(self, probability_engine):
        """Test analysis with sportsbook odds data."""
        market_data = {
            'sportsbook_odds': [
                {'source': 'pinnacle', 'yes_price': 0.55},
                {'source': 'bet365', 'yes_price': 0.53},
                {'source': 'draftkings', 'yes_price': 0.57},
            ]
        }
        
        result = await probability_engine.analyze("TEST_MARKET", market_data)
        
        assert result.analysis_type == AnalysisType.PROBABILITY_ESTIMATION
        assert result.is_valid
        assert 0.01 <= result.value <= 0.99
        assert result.confidence > 0
        assert 'individual_probabilities' in result.components
    
    @pytest.mark.asyncio
    async def test_analyze_with_historical_data(self, probability_engine):
        """Test analysis with historical outcomes."""
        market_data = {
            'historical_outcomes': [
                {'outcome': 1, 'similarity_score': 0.9, 'recency_weight': 1.0},
                {'outcome': 0, 'similarity_score': 0.8, 'recency_weight': 0.9},
                {'outcome': 1, 'similarity_score': 0.85, 'recency_weight': 0.8},
                {'outcome': 1, 'similarity_score': 0.75, 'recency_weight': 0.7},
            ]
        }
        
        result = await probability_engine.analyze("TEST_MARKET", market_data)
        
        assert result.is_valid
        assert result.value > 0.5  # Should lean toward YES (3/4 historical outcomes)
        assert result.components['method'] == ProbabilityMethod.BAYESIAN_UPDATE.value
    
    @pytest.mark.asyncio
    async def test_analyze_ensemble_method(self, probability_engine):
        """Test analysis using ensemble of multiple methods."""
        market_data = {
            'sportsbook_odds': [
                {'source': 'pinnacle', 'yes_price': 0.60},
                {'source': 'bet365', 'yes_price': 0.58},
            ],
            'historical_outcomes': [
                {'outcome': 1, 'similarity_score': 0.9},
                {'outcome': 1, 'similarity_score': 0.8},
                {'outcome': 0, 'similarity_score': 0.7},
            ],
            'market_context': {
                'home_team_strength': 0.7,
                'away_team_strength': 0.5,
                'weather_factor': 0.1,
                'injury_factor': -0.2,
                'momentum_factor': 0.3
            }
        }
        
        result = await probability_engine.analyze("TEST_MARKET", market_data)
        
        assert result.is_valid
        assert result.components['method'] == ProbabilityMethod.ENSEMBLE.value
        assert 'methods_used' in result.metadata
        assert len(result.metadata['methods_used']) > 1
    
    @pytest.mark.asyncio
    async def test_analyze_no_data(self, probability_engine):
        """Test analysis with no input data."""
        result = await probability_engine.analyze("TEST_MARKET", {})
        
        assert result.is_valid
        assert result.value == pytest.approx(0.5, rel=1e-3)  # Should default to neutral probability
        assert result.confidence == pytest.approx(0.1, rel=1e-3)  # Low confidence
        assert 'fallback' in result.components
    
    @pytest.mark.asyncio
    async def test_batch_analyze(self, probability_engine):
        """Test batch analysis of multiple markets."""
        markets = ["MARKET_1", "MARKET_2", "MARKET_3"]
        
        # Mock the analyze method
        async def mock_analyze(market_id, data=None):
            from neural.analysis.base import AnalysisResult, AnalysisType, SignalStrength
            from datetime import datetime
            return AnalysisResult(
                analysis_type=AnalysisType.PROBABILITY_ESTIMATION,
                timestamp=datetime.now(),
                market_id=market_id,
                value=0.55,
                confidence=0.8,
                signal=None,
                signal_strength=SignalStrength.STRONG,
                components={'method': 'sportsbook_consensus'},
                metadata={}
            )
        
        with patch.object(probability_engine, 'analyze', side_effect=mock_analyze):
            results = await probability_engine.batch_analyze(markets)
        
        assert len(results) == 3
        assert all(result.is_valid for result in results)
    
    def test_aggregate_sportsbook_odds_basic(self, probability_engine):
        """Test basic sportsbook odds aggregation."""
        odds_list = [
            {'source': 'pinnacle', 'yes_price': 0.55},
            {'source': 'bet365', 'yes_price': 0.53},
            {'source': 'draftkings', 'yes_price': 0.57},
        ]
        
        estimate = probability_engine.aggregate_sportsbook_odds(odds_list)
        
        assert isinstance(estimate, ProbabilityEstimate)
        assert estimate.method == ProbabilityMethod.SPORTSBOOK_CONSENSUS
        assert 0.53 <= estimate.probability <= 0.57  # Should be within range
        assert estimate.confidence > 0
    
    def test_aggregate_sportsbook_odds_weighted(self, probability_engine):
        """Test weighted sportsbook aggregation (sharp books weighted higher)."""
        odds_list = [
            {'source': 'pinnacle', 'yes_price': 0.50},  # Sharp book, high weight
            {'source': 'draftkings', 'yes_price': 0.60},  # Lower weight
        ]
        
        estimate = probability_engine.aggregate_sportsbook_odds(odds_list)
        
        # Result should be closer to Pinnacle due to higher weight
        assert estimate.probability < 0.55  # Closer to 0.50 than simple average (0.55)
    
    def test_aggregate_sportsbook_outlier_removal(self, probability_engine):
        """Test outlier removal in sportsbook aggregation."""
        odds_list = [
            {'source': 'pinnacle', 'yes_price': 0.50},
            {'source': 'bet365', 'yes_price': 0.52},
            {'source': 'draftkings', 'yes_price': 0.51},
            {'source': 'unknown', 'yes_price': 0.90},  # Outlier
        ]
        
        estimate = probability_engine.aggregate_sportsbook_odds(odds_list)
        
        # Should be close to consensus of non-outlier books
        assert 0.49 <= estimate.probability <= 0.53
        assert estimate.components['num_sources'] <= 3  # Outlier may be removed
    
    def test_odds_to_probability_conversion(self, probability_engine):
        """Test various odds format conversions."""
        # Decimal price (most direct)
        prob1 = probability_engine._odds_to_probability({'yes_price': 0.65})
        assert prob1 == pytest.approx(0.65, rel=1e-3)
        
        # American odds (positive)
        prob2 = probability_engine._odds_to_probability({'yes_odds': 150})
        assert prob2 == pytest.approx(0.40, rel=1e-2)  # 100/(150+100) = 0.40
        
        # American odds (negative)
        prob3 = probability_engine._odds_to_probability({'yes_odds': -200})
        assert prob3 == pytest.approx(0.67, rel=1e-2)  # 200/(200+100) = 0.67
        
        # Decimal odds (European)
        prob4 = probability_engine._odds_to_probability({'decimal_odds': 2.0})
        assert prob4 == pytest.approx(0.50, rel=1e-2)  # 1/2.0 = 0.50
    
    def test_calculate_historical_probability(self, probability_engine):
        """Test historical probability calculation with Bayesian updating."""
        historical_outcomes = [
            {'outcome': 1, 'similarity_score': 1.0, 'recency_weight': 1.0},
            {'outcome': 1, 'similarity_score': 0.9, 'recency_weight': 0.9},
            {'outcome': 0, 'similarity_score': 0.8, 'recency_weight': 0.8},
            {'outcome': 1, 'similarity_score': 0.7, 'recency_weight': 0.7},
        ]
        
        estimate = probability_engine._calculate_historical_probability(historical_outcomes)
        
        assert isinstance(estimate, ProbabilityEstimate)
        assert estimate.method == ProbabilityMethod.BAYESIAN_UPDATE
        assert estimate.probability > 0.5  # Should favor YES (3 of 4 outcomes)
        assert estimate.confidence > 0
        assert 'alpha_posterior' in estimate.components
        assert 'beta_posterior' in estimate.components
    
    def test_calculate_model_probability(self, probability_engine):
        """Test simple model-based probability calculation."""
        market_context = {
            'home_team_strength': 0.8,   # Strong home team
            'away_team_strength': 0.4,   # Weak away team  
            'weather_factor': 0.0,       # Neutral weather
            'injury_factor': -0.1,       # Minor injuries
            'momentum_factor': 0.2       # Positive momentum
        }
        
        estimate = probability_engine._calculate_model_probability(market_context)
        
        assert isinstance(estimate, ProbabilityEstimate)
        assert estimate.method == ProbabilityMethod.REGRESSION_MODEL
        assert estimate.probability > 0.5  # Should favor strong home team
        assert 'available_features' in estimate.components
    
    def test_ensemble_probability_combination(self, probability_engine):
        """Test ensemble combination of multiple estimates."""
        estimates = [
            ProbabilityEstimate(0.60, 0.8, ProbabilityMethod.SPORTSBOOK_CONSENSUS, {}, {}),
            ProbabilityEstimate(0.55, 0.7, ProbabilityMethod.BAYESIAN_UPDATE, {}, {}),
            ProbabilityEstimate(0.65, 0.6, ProbabilityMethod.REGRESSION_MODEL, {}, {}),
        ]
        
        ensemble = probability_engine._ensemble_probability(estimates)
        
        assert isinstance(ensemble, ProbabilityEstimate)
        assert ensemble.method == ProbabilityMethod.ENSEMBLE
        assert 0.55 <= ensemble.probability <= 0.65  # Should be weighted average
        assert ensemble.confidence > 0
        assert ensemble.components['num_estimates'] == 3
    
    def test_ensemble_agreement_confidence(self, probability_engine):
        """Test that ensemble confidence increases with agreement."""
        # High agreement estimates
        high_agreement = [
            ProbabilityEstimate(0.60, 0.8, ProbabilityMethod.SPORTSBOOK_CONSENSUS, {}, {}),
            ProbabilityEstimate(0.61, 0.8, ProbabilityMethod.BAYESIAN_UPDATE, {}, {}),
            ProbabilityEstimate(0.59, 0.8, ProbabilityMethod.REGRESSION_MODEL, {}, {}),
        ]
        
        # Low agreement estimates
        low_agreement = [
            ProbabilityEstimate(0.40, 0.8, ProbabilityMethod.SPORTSBOOK_CONSENSUS, {}, {}),
            ProbabilityEstimate(0.70, 0.8, ProbabilityMethod.BAYESIAN_UPDATE, {}, {}),
            ProbabilityEstimate(0.55, 0.8, ProbabilityMethod.REGRESSION_MODEL, {}, {}),
        ]
        
        high_ensemble = probability_engine._ensemble_probability(high_agreement)
        low_ensemble = probability_engine._ensemble_probability(low_agreement)
        
        assert high_ensemble.confidence > low_ensemble.confidence
    
    def test_calculate_confidence_interval(self, probability_engine):
        """Test confidence interval calculation."""
        lower, upper = probability_engine.calculate_confidence_interval(
            probability=0.60,
            confidence=0.80,
            confidence_level=0.95
        )
        
        assert 0.01 <= lower <= 0.60
        assert 0.60 <= upper <= 0.99
        assert upper > lower
    
    def test_confidence_to_signal_strength(self, probability_engine):
        """Test conversion from confidence to signal strength."""
        assert probability_engine._confidence_to_signal_strength(0.95) == SignalStrength.VERY_STRONG
        assert probability_engine._confidence_to_signal_strength(0.75) == SignalStrength.STRONG
        assert probability_engine._confidence_to_signal_strength(0.55) == SignalStrength.MODERATE
        assert probability_engine._confidence_to_signal_strength(0.35) == SignalStrength.WEAK
        assert probability_engine._confidence_to_signal_strength(0.15) == SignalStrength.VERY_WEAK
    
    def test_required_fields_empty(self, probability_engine):
        """Test that probability engine has no required fields (flexible)."""
        required = probability_engine.get_required_fields()
        assert required == []


class TestOddsConverter:
    """Test OddsConverter utility class."""
    
    def test_american_to_probability(self):
        """Test American odds to probability conversion."""
        # Positive odds
        prob1 = OddsConverter.american_to_probability(100)
        assert prob1 == pytest.approx(0.50, rel=1e-2)
        
        prob2 = OddsConverter.american_to_probability(200)
        assert prob2 == pytest.approx(0.333, rel=1e-2)
        
        # Negative odds
        prob3 = OddsConverter.american_to_probability(-200)
        assert prob3 == pytest.approx(0.667, rel=1e-2)
        
        prob4 = OddsConverter.american_to_probability(-100)
        assert prob4 == pytest.approx(0.50, rel=1e-2)
    
    def test_decimal_to_probability(self):
        """Test decimal odds to probability conversion."""
        assert OddsConverter.decimal_to_probability(2.0) == pytest.approx(0.50, rel=1e-2)
        assert OddsConverter.decimal_to_probability(1.5) == pytest.approx(0.667, rel=1e-2)
        assert OddsConverter.decimal_to_probability(3.0) == pytest.approx(0.333, rel=1e-2)
        assert OddsConverter.decimal_to_probability(0) == 0  # Edge case
    
    def test_fractional_to_probability(self):
        """Test fractional odds to probability conversion."""
        assert OddsConverter.fractional_to_probability(1, 1) == pytest.approx(0.50, rel=1e-2)  # 1/1 = even
        assert OddsConverter.fractional_to_probability(1, 2) == pytest.approx(0.667, rel=1e-2)  # 1/2
        assert OddsConverter.fractional_to_probability(2, 1) == pytest.approx(0.333, rel=1e-2)  # 2/1
    
    def test_probability_to_american(self):
        """Test probability to American odds conversion."""
        # Favorites (>50%)
        odds1 = OddsConverter.probability_to_american(0.667)
        assert odds1 == pytest.approx(-200, rel=1e-1)
        
        odds2 = OddsConverter.probability_to_american(0.50)
        assert odds2 == pytest.approx(-100, rel=1e-1)
        
        # Underdogs (<50%)
        odds3 = OddsConverter.probability_to_american(0.333)
        assert odds3 == pytest.approx(200, rel=1e-1)
        
        odds4 = OddsConverter.probability_to_american(0.25)
        assert odds4 == pytest.approx(300, rel=1e-1)
    
    def test_remove_vig(self):
        """Test vig removal from sportsbook odds."""
        # Normal case with vig
        fair_yes, fair_no = OddsConverter.remove_vig(0.52, 0.53)  # Adds to 1.05 (5% vig)
        
        assert fair_yes + fair_no == pytest.approx(1.0, rel=1e-2)
        assert fair_yes == pytest.approx(0.495, rel=1e-2)
        assert fair_no == pytest.approx(0.505, rel=1e-2)
        
        # No vig case
        fair_yes2, fair_no2 = OddsConverter.remove_vig(0.48, 0.52)  # Already adds to 1.0
        assert fair_yes2 == pytest.approx(0.48, rel=1e-3)
        assert fair_no2 == pytest.approx(0.52, rel=1e-3)
    
    def test_odds_conversion_roundtrip(self):
        """Test that odds conversions are reversible."""
        original_prob = 0.65
        
        # Probability -> American -> Probability
        american = OddsConverter.probability_to_american(original_prob)
        converted_prob = OddsConverter.american_to_probability(american)
        assert converted_prob == pytest.approx(original_prob, rel=1e-2)
        
        # Probability -> Decimal -> Probability
        decimal = 1 / original_prob
        converted_prob2 = OddsConverter.decimal_to_probability(decimal)
        assert converted_prob2 == pytest.approx(original_prob, rel=1e-2)


class TestProbabilityEstimate:
    """Test ProbabilityEstimate dataclass."""
    
    def test_create_estimate(self):
        """Test creating a probability estimate."""
        estimate = ProbabilityEstimate(
            probability=0.65,
            confidence=0.80,
            method=ProbabilityMethod.SPORTSBOOK_CONSENSUS,
            components={'test': 'data'},
            metadata={'source': 'test'}
        )
        
        assert estimate.probability == pytest.approx(0.65, rel=1e-3)
        assert estimate.confidence == pytest.approx(0.80, rel=1e-3)
        assert estimate.method == ProbabilityMethod.SPORTSBOOK_CONSENSUS
        assert estimate.components['test'] == 'data'
        assert estimate.metadata['source'] == 'test'


class TestSportsbookOdds:
    """Test SportsbookOdds dataclass."""
    
    def test_create_sportsbook_odds(self):
        """Test creating sportsbook odds."""
        odds = SportsbookOdds(
            source='pinnacle',
            yes_odds=120,
            yes_price=0.45,
            timestamp=datetime.now(),
            confidence=0.95
        )
        
        assert odds.source == 'pinnacle'
        assert odds.yes_odds == pytest.approx(120, rel=1e-3)
        assert odds.yes_price == pytest.approx(0.45, rel=1e-3)
        assert odds.confidence == pytest.approx(0.95, rel=1e-3)
