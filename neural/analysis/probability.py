"""
Probability calculation engines for Neural SDK Analysis Infrastructure.

This module provides sophisticated probability estimation methods including
sportsbook consensus aggregation, Bayesian updates, and confidence intervals.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum

from neural.analysis.base import (
    BaseAnalyzer, AnalysisResult, AnalysisConfig, AnalysisType, SignalStrength
)

logger = logging.getLogger(__name__)


class ProbabilityMethod(Enum):
    """Probability calculation methods."""
    SPORTSBOOK_CONSENSUS = "sportsbook_consensus"
    BAYESIAN_UPDATE = "bayesian_update"
    REGRESSION_MODEL = "regression_model"
    ENSEMBLE = "ensemble"


@dataclass
class SportsbookOdds:
    """Represents odds from a sportsbook."""
    source: str
    yes_odds: Optional[float] = None  # American odds for YES
    no_odds: Optional[float] = None   # American odds for NO
    yes_price: Optional[float] = None # Decimal price for YES
    no_price: Optional[float] = None  # Decimal price for NO
    timestamp: Optional[datetime] = None
    confidence: float = 1.0  # Source reliability weight


@dataclass
class ProbabilityEstimate:
    """Probability estimate with confidence metrics."""
    probability: float
    confidence: float
    method: ProbabilityMethod
    components: Dict[str, Any]
    metadata: Dict[str, Any]


class ProbabilityEngine(BaseAnalyzer):
    """
    Advanced probability calculation engine.
    
    Aggregates multiple data sources to produce robust probability estimates
    for Kalshi markets, with confidence intervals and uncertainty quantification.
    """
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        """Initialize probability engine."""
        super().__init__(config)
        self.sportsbook_weights = {
            'pinnacle': 1.0,      # Highest weight for sharp book
            'bet365': 0.9,
            'draftkings': 0.8,
            'fanduel': 0.8,
            'betmgm': 0.7,
            'caesars': 0.7,
            'default': 0.5
        }
        
    async def initialize(self) -> None:
        """Initialize the probability engine."""
        self._initialized = True
        logger.info("ProbabilityEngine initialized")
    
    async def analyze(
        self, 
        market_id: str,
        data: Optional[Dict[str, Any]] = None
    ) -> AnalysisResult:
        """
        Calculate probability estimate for a market.
        
        Args:
            market_id: Market identifier
            data: Optional data containing sportsbook odds, historical data, etc.
            
        Returns:
            AnalysisResult with probability estimate
        """
        if not self._initialized:
            await self.initialize()
        
        timestamp = datetime.now()
        
        try:
            # Validate input data
            is_valid, errors = await self.validate_data(data or {})
            if not is_valid:
                return AnalysisResult(
                    analysis_type=AnalysisType.PROBABILITY_ESTIMATION,
                    timestamp=timestamp,
                    market_id=market_id,
                    value=0.5,  # Default neutral probability
                    confidence=0,
                    errors=errors
                )
            
            # Extract data sources
            sportsbook_odds = data.get('sportsbook_odds', [])
            historical_outcomes = data.get('historical_outcomes', [])
            market_context = data.get('market_context', {})
            
            # Calculate probability using best available method
            probability_estimate = await self._calculate_best_probability(
                sportsbook_odds, historical_outcomes, market_context
            )
            
            # Determine signal strength based on confidence
            signal_strength = self._confidence_to_signal_strength(probability_estimate.confidence)
            
            # Ensure components includes method for test compatibility
            components = probability_estimate.components.copy()
            components['method'] = probability_estimate.method.value
            
            return AnalysisResult(
                analysis_type=AnalysisType.PROBABILITY_ESTIMATION,
                timestamp=timestamp,
                market_id=market_id,
                value=probability_estimate.probability,
                confidence=probability_estimate.confidence,
                signal=None,  # Probability engine doesn't generate trading signals
                signal_strength=signal_strength,
                components=components,
                metadata=probability_estimate.metadata
            )
            
        except Exception as e:
            logger.error(f"Error analyzing market {market_id}: {e}")
            return AnalysisResult(
                analysis_type=AnalysisType.PROBABILITY_ESTIMATION,
                timestamp=timestamp,
                market_id=market_id,
                value=0.5,
                confidence=0,
                errors=[str(e)]
            )
    
    async def batch_analyze(
        self,
        market_ids: List[str]
    ) -> List[AnalysisResult]:
        """
        Calculate probabilities for multiple markets.
        
        Args:
            market_ids: List of market identifiers
            
        Returns:
            List of AnalysisResults
        """
        tasks = [self.analyze(market_id) for market_id in market_ids]
        return await asyncio.gather(*tasks)
    
    async def _calculate_best_probability(
        self,
        sportsbook_odds: List[Dict[str, Any]],
        historical_outcomes: List[Dict[str, Any]],
        market_context: Dict[str, Any]
    ) -> ProbabilityEstimate:
        """
        Calculate best probability estimate using available data sources.
        
        Args:
            sportsbook_odds: List of sportsbook odds
            historical_outcomes: Historical similar outcomes
            market_context: Additional market context
            
        Returns:
            ProbabilityEstimate with best estimate
        """
        estimates = []
        
        # Method 1: Sportsbook consensus
        if sportsbook_odds:
            sportsbook_estimate = self.aggregate_sportsbook_odds(sportsbook_odds)
            estimates.append(sportsbook_estimate)
        
        # Method 2: Historical outcomes (Bayesian update)
        if historical_outcomes:
            historical_estimate = self._calculate_historical_probability(historical_outcomes)
            estimates.append(historical_estimate)
        
        # Method 3: Regression model (if enough context)
        if len(market_context) >= 5:
            model_estimate = self._calculate_model_probability(market_context)
            estimates.append(model_estimate)
        
        # Combine estimates using ensemble method
        if len(estimates) > 1:
            return self._ensemble_probability(estimates)
        elif estimates:
            return estimates[0]
        else:
            # Fallback to neutral probability
            return ProbabilityEstimate(
                probability=0.5,
                confidence=0.1,
                method=ProbabilityMethod.ENSEMBLE,
                components={'fallback': True},
                metadata={'reason': 'No data sources available'}
            )
    
    def aggregate_sportsbook_odds(self, odds_list: List[Dict[str, Any]]) -> ProbabilityEstimate:
        """
        Aggregate sportsbook odds into consensus probability.
        
        Uses weighted average with higher weights for sharper books
        and removes outliers to improve accuracy.
        
        Args:
            odds_list: List of sportsbook odds dictionaries
            
        Returns:
            ProbabilityEstimate from sportsbook consensus
        """
        if not odds_list:
            raise ValueError("No sportsbook odds provided")
        
        # Convert odds to probabilities
        probabilities = []
        weights = []
        
        for odds_data in odds_list:
            source = odds_data.get('source', 'unknown').lower()
            weight = self.sportsbook_weights.get(source, self.sportsbook_weights['default'])
            
            # Convert different odds formats to probability
            prob = self._odds_to_probability(odds_data)
            if prob is not None:
                probabilities.append(prob)
                weights.append(weight)
        
        if not probabilities:
            raise ValueError("No valid probabilities extracted from odds")
        
        probabilities = np.array(probabilities)
        weights = np.array(weights)
        
        # Remove outliers (more than 2 standard deviations from median)
        if len(probabilities) > 3:
            median_prob = np.median(probabilities)
            std_prob = np.std(probabilities)
            
            if std_prob > 0:
                outlier_mask = np.abs(probabilities - median_prob) < 2 * std_prob
                probabilities = probabilities[outlier_mask]
                weights = weights[outlier_mask]
        
        # Calculate weighted average
        weighted_prob = np.average(probabilities, weights=weights)
        
        # Calculate confidence based on agreement
        prob_std = np.std(probabilities)
        agreement_confidence = max(0.1, 1 - (prob_std / 0.1))  # Higher confidence when books agree
        
        # Adjust confidence based on number of sources
        source_confidence = min(1.0, len(probabilities) / 5)
        
        overall_confidence = (agreement_confidence * 0.7) + (source_confidence * 0.3)
        
        return ProbabilityEstimate(
            probability=float(np.clip(weighted_prob, 0.01, 0.99)),
            confidence=overall_confidence,
            method=ProbabilityMethod.SPORTSBOOK_CONSENSUS,
            components={
                'individual_probabilities': probabilities.tolist(),
                'weights': weights.tolist(),
                'agreement_std': float(prob_std),
                'num_sources': len(probabilities)
            },
            metadata={
                'sources': [odds.get('source') for odds in odds_list],
                'calculation_time': datetime.now().isoformat()
            }
        )
    
    def _odds_to_probability(self, odds_data: Dict[str, Any]) -> Optional[float]:
        """
        Convert various odds formats to probability.
        
        Args:
            odds_data: Dictionary with odds information
            
        Returns:
            Probability or None if conversion fails
        """
        # If decimal price is available (most direct)
        if 'yes_price' in odds_data and odds_data['yes_price']:
            return float(odds_data['yes_price'])
        
        # Convert from American odds
        if 'yes_odds' in odds_data and odds_data['yes_odds']:
            american_odds = odds_data['yes_odds']
            if american_odds > 0:
                # Positive odds: probability = 100 / (odds + 100)
                implied_prob = 100 / (american_odds + 100)
            else:
                # Negative odds: probability = -odds / (-odds + 100)
                implied_prob = (-american_odds) / (-american_odds + 100)
            
            return implied_prob
        
        # Convert from decimal odds (European format)
        if 'decimal_odds' in odds_data and odds_data['decimal_odds']:
            decimal_odds = odds_data['decimal_odds']
            return 1 / decimal_odds if decimal_odds > 0 else None
        
        return None
    
    def _calculate_historical_probability(
        self, 
        historical_outcomes: List[Dict[str, Any]]
    ) -> ProbabilityEstimate:
        """
        Calculate probability using historical outcomes with Bayesian updating.
        
        Args:
            historical_outcomes: List of similar historical events
            
        Returns:
            ProbabilityEstimate based on historical data
        """
        if not historical_outcomes:
            raise ValueError("No historical outcomes provided")
        
        # Extract outcomes and weights
        outcomes = []
        weights = []
        
        for outcome in historical_outcomes:
            result = outcome.get('outcome')  # 1 for YES, 0 for NO
            similarity = outcome.get('similarity_score', 1.0)
            recency_weight = outcome.get('recency_weight', 1.0)
            
            if result is not None:
                outcomes.append(result)
                weights.append(similarity * recency_weight)
        
        if not outcomes:
            raise ValueError("No valid outcomes in historical data")
        
        outcomes = np.array(outcomes)
        weights = np.array(weights)
        
        # Bayesian update with beta prior (slightly pessimistic)
        alpha_prior = 1.0  # Prior "YES" observations
        beta_prior = 1.0   # Prior "NO" observations
        
        # Weighted counts
        weighted_yes = np.sum(weights[outcomes == 1])
        weighted_no = np.sum(weights[outcomes == 0])
        
        # Posterior parameters
        alpha_post = alpha_prior + weighted_yes
        beta_post = beta_prior + weighted_no
        
        # Posterior mean (probability estimate)
        probability = alpha_post / (alpha_post + beta_post)
        
        # Confidence based on sample size and posterior variance
        total_weight = np.sum(weights)
        posterior_var = (alpha_post * beta_post) / ((alpha_post + beta_post)**2 * (alpha_post + beta_post + 1))
        
        # Higher confidence with more data and lower variance
        sample_confidence = min(1.0, total_weight / 20)  # Asymptote at 20 weighted samples
        variance_confidence = 1 - min(1.0, posterior_var * 10)  # Lower variance = higher confidence
        
        overall_confidence = (sample_confidence * 0.6) + (variance_confidence * 0.4)
        
        return ProbabilityEstimate(
            probability=float(np.clip(probability, 0.01, 0.99)),
            confidence=overall_confidence,
            method=ProbabilityMethod.BAYESIAN_UPDATE,
            components={
                'alpha_posterior': float(alpha_post),
                'beta_posterior': float(beta_post),
                'total_weight': float(total_weight),
                'posterior_variance': float(posterior_var),
                'num_outcomes': len(outcomes)
            },
            metadata={
                'historical_events': len(historical_outcomes),
                'weighted_yes_outcomes': float(weighted_yes),
                'weighted_no_outcomes': float(weighted_no)
            }
        )
    
    def _calculate_model_probability(self, market_context: Dict[str, Any]) -> ProbabilityEstimate:
        """
        Calculate probability using a simple regression model.
        
        This is a placeholder for more sophisticated ML models.
        
        Args:
            market_context: Market context features
            
        Returns:
            ProbabilityEstimate from model
        """
        # Simple logistic regression-style calculation
        features = []
        feature_weights = {
            'home_team_strength': 0.3,
            'away_team_strength': -0.3,
            'weather_factor': 0.1,
            'injury_factor': -0.2,
            'momentum_factor': 0.2
        }
        
        weighted_score = 0
        for feature, weight in feature_weights.items():
            if feature in market_context:
                value = market_context[feature]
                if isinstance(value, (int, float)):
                    weighted_score += weight * value
                    features.append(feature)
        
        # Simple sigmoid transformation
        probability = 1 / (1 + np.exp(-weighted_score))
        
        # Confidence based on number of available features
        confidence = len(features) / len(feature_weights)
        
        return ProbabilityEstimate(
            probability=float(np.clip(probability, 0.01, 0.99)),
            confidence=confidence,
            method=ProbabilityMethod.REGRESSION_MODEL,
            components={
                'weighted_score': float(weighted_score),
                'available_features': features,
                'total_features': len(feature_weights)
            },
            metadata={'model_type': 'simple_logistic'}
        )
    
    def _ensemble_probability(self, estimates: List[ProbabilityEstimate]) -> ProbabilityEstimate:
        """
        Combine multiple probability estimates using ensemble method.
        
        Args:
            estimates: List of probability estimates
            
        Returns:
            Combined ProbabilityEstimate
        """
        if not estimates:
            raise ValueError("No estimates provided for ensemble")
        
        if len(estimates) == 1:
            return estimates[0]
        
        # Weight estimates by their confidence
        probabilities = np.array([est.probability for est in estimates])
        confidences = np.array([est.confidence for est in estimates])
        
        # Weighted average by confidence
        ensemble_probability = np.average(probabilities, weights=confidences)
        
        # Ensemble confidence is higher when estimates agree
        prob_std = np.std(probabilities)
        agreement_factor = max(0.1, 1 - (prob_std / 0.1))
        
        # Base confidence is average of individual confidences
        base_confidence = np.mean(confidences)
        
        # Boost confidence for agreement, reduce for disagreement
        ensemble_confidence = base_confidence * agreement_factor
        
        # Combine components from all estimates
        combined_components = {}
        combined_metadata = {'methods_used': []}
        
        for i, est in enumerate(estimates):
            combined_components[f'estimate_{i+1}'] = {
                'probability': est.probability,
                'confidence': est.confidence,
                'method': est.method.value
            }
            combined_metadata['methods_used'].append(est.method.value)
        
        combined_components['agreement_std'] = float(prob_std)
        combined_components['num_estimates'] = len(estimates)
        
        return ProbabilityEstimate(
            probability=float(np.clip(ensemble_probability, 0.01, 0.99)),
            confidence=min(1.0, ensemble_confidence),
            method=ProbabilityMethod.ENSEMBLE,
            components=combined_components,
            metadata=combined_metadata
        )
    
    def calculate_confidence_interval(
        self, 
        probability: float, 
        confidence: float,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for probability estimate.
        
        Args:
            probability: Point estimate
            confidence: Confidence in estimate
            confidence_level: Desired confidence level
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        # Use beta distribution approximation
        # Higher confidence = tighter interval
        
        if confidence <= 0:
            return 0.01, 0.99  # Maximum uncertainty
        
        # Effective sample size based on confidence
        effective_n = confidence * 100  # Scale factor
        
        # Beta distribution parameters
        alpha = probability * effective_n
        beta = (1 - probability) * effective_n
        
        # Calculate percentiles for confidence interval
        from scipy import stats
        
        alpha_level = (1 - confidence_level) / 2
        lower = stats.beta.ppf(alpha_level, alpha, beta)
        upper = stats.beta.ppf(1 - alpha_level, alpha, beta)
        
        return (
            float(np.clip(lower, 0.01, probability)),
            float(np.clip(upper, probability, 0.99))
        )
    
    def _confidence_to_signal_strength(self, confidence: float) -> SignalStrength:
        """Convert confidence level to signal strength."""
        if confidence >= 0.9:
            return SignalStrength.VERY_STRONG
        elif confidence >= 0.7:
            return SignalStrength.STRONG
        elif confidence >= 0.5:
            return SignalStrength.MODERATE
        elif confidence >= 0.3:
            return SignalStrength.WEAK
        else:
            return SignalStrength.VERY_WEAK
    
    def get_required_fields(self) -> List[str]:
        """Get required data fields for probability analysis."""
        return []  # No required fields - can work with any available data


class OddsConverter:
    """
    Utility class for converting between different odds formats.
    """
    
    @staticmethod
    def american_to_probability(odds: float) -> float:
        """Convert American odds to probability."""
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return (-odds) / (-odds + 100)
    
    @staticmethod
    def decimal_to_probability(odds: float) -> float:
        """Convert decimal odds to probability."""
        return 1 / odds if odds > 0 else 0
    
    @staticmethod
    def fractional_to_probability(numerator: int, denominator: int) -> float:
        """Convert fractional odds to probability."""
        return denominator / (numerator + denominator)
    
    @staticmethod
    def probability_to_american(probability: float) -> float:
        """Convert probability to American odds."""
        if probability >= 0.5:
            return -100 * probability / (1 - probability)
        else:
            return 100 * (1 - probability) / probability
    
    @staticmethod
    def remove_vig(yes_prob: float, no_prob: float) -> Tuple[float, float]:
        """
        Remove sportsbook vig to get fair probabilities.
        
        Args:
            yes_prob: Implied probability for YES
            no_prob: Implied probability for NO
            
        Returns:
            Tuple of (fair_yes_prob, fair_no_prob)
        """
        total_prob = yes_prob + no_prob
        if total_prob <= 1:
            return yes_prob, no_prob  # No vig detected
        
        # Remove vig proportionally
        fair_yes = yes_prob / total_prob
        fair_no = no_prob / total_prob
        
        return fair_yes, fair_no
