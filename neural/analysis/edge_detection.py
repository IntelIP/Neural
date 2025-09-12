"""
Edge detection and market inefficiency identification for Kalshi trading.

This module provides the EdgeCalculator class and related utilities for identifying
trading opportunities by detecting when market prices diverge from fair value.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from neural.analysis.base import (
    BaseAnalyzer, AnalysisResult, AnalysisConfig, AnalysisType, SignalStrength
)
from neural.kalshi.fees import calculate_expected_value, calculate_edge, calculate_breakeven_probability
from neural.kalshi.markets import KalshiMarket, OrderSide

logger = logging.getLogger(__name__)


class EdgeCalculator(BaseAnalyzer):
    """
    Identifies trading edges by comparing market prices to estimated fair values.
    
    This analyzer detects market inefficiencies by calculating the difference
    between market-implied probabilities and estimated true probabilities.
    """
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        """Initialize edge calculator."""
        super().__init__(config)
        self.confidence_cache = {}  # Cache confidence calculations
        
    async def initialize(self) -> None:
        """Initialize the edge calculator."""
        self._initialized = True
        logger.info("EdgeCalculator initialized")
    
    async def analyze(
        self, 
        market_id: str,
        data: Optional[Dict[str, Any]] = None
    ) -> AnalysisResult:
        """
        Analyze market for trading edges.
        
        Args:
            market_id: Market identifier
            data: Optional data containing market info and external probabilities
            
        Returns:
            AnalysisResult with edge calculation
        """
        if not self._initialized:
            await self.initialize()
        
        timestamp = datetime.now()
        
        try:
            # Validate input data
            is_valid, errors = await self.validate_data(data or {})
            if not is_valid:
                return AnalysisResult(
                    analysis_type=AnalysisType.EDGE_DETECTION,
                    timestamp=timestamp,
                    market_id=market_id,
                    value=0,
                    confidence=0,
                    errors=errors
                )
            
            # Extract market data
            market_price = data.get('market_price')
            true_probability = data.get('true_probability')
            sportsbook_consensus = data.get('sportsbook_consensus')
            volume_data = data.get('volume_data', {})
            
            # Calculate edge components
            edge_results = self._calculate_comprehensive_edge(
                market_price, true_probability, sportsbook_consensus, volume_data
            )
            
            # Determine signal
            signal, signal_strength = self._generate_signal(
                edge_results['adjusted_edge'], 
                edge_results['confidence']
            )
            
            return AnalysisResult(
                analysis_type=AnalysisType.EDGE_DETECTION,
                timestamp=timestamp,
                market_id=market_id,
                value=edge_results['adjusted_edge'],
                confidence=edge_results['confidence'],
                signal=signal,
                signal_strength=signal_strength,
                components=edge_results,
                metadata={
                    'breakeven_probability': calculate_breakeven_probability(market_price),
                    'expected_value_yes': calculate_expected_value(true_probability, market_price),
                    'expected_value_no': calculate_expected_value(1 - true_probability, 1 - market_price)
                }
            )
            
        except Exception as e:
            logger.error(f"Error analyzing market {market_id}: {e}")
            return AnalysisResult(
                analysis_type=AnalysisType.EDGE_DETECTION,
                timestamp=timestamp,
                market_id=market_id,
                value=0,
                confidence=0,
                errors=[str(e)]
            )
    
    async def batch_analyze(
        self,
        market_ids: List[str]
    ) -> List[AnalysisResult]:
        """
        Analyze multiple markets for edges.
        
        Args:
            market_ids: List of market identifiers
            
        Returns:
            List of AnalysisResults
        """
        tasks = [self.analyze(market_id) for market_id in market_ids]
        return await asyncio.gather(*tasks)
    
    def _calculate_comprehensive_edge(
        self,
        market_price: float,
        true_probability: float,
        sportsbook_consensus: Optional[float] = None,
        volume_data: Dict[str, Any] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive edge with multiple factors.
        
        Args:
            market_price: Current Kalshi market price
            true_probability: Estimated true probability
            sportsbook_consensus: Consensus probability from sportsbooks
            volume_data: Volume and liquidity information
            
        Returns:
            Dictionary with edge components
        """
        volume_data = volume_data or {}
        
        # Basic edge calculation
        raw_edge = calculate_edge(true_probability, market_price)
        
        # Confidence adjustments
        confidence_factors = self._calculate_confidence_factors(
            market_price, sportsbook_consensus, volume_data
        )
        
        # Adjust edge for confidence
        confidence_weighted_edge = raw_edge * confidence_factors['overall_confidence']
        
        # Volume and liquidity adjustments
        liquidity_adjustment = self._calculate_liquidity_adjustment(volume_data)
        adjusted_edge = confidence_weighted_edge * liquidity_adjustment
        
        return {
            'raw_edge': raw_edge,
            'confidence_weighted_edge': confidence_weighted_edge,
            'adjusted_edge': adjusted_edge,
            'confidence': confidence_factors['overall_confidence'],
            'sportsbook_agreement': confidence_factors.get('sportsbook_agreement', 0),
            'volume_confidence': confidence_factors.get('volume_confidence', 0.5),
            'liquidity_adjustment': liquidity_adjustment
        }
    
    def _calculate_confidence_factors(
        self,
        market_price: float,
        sportsbook_consensus: Optional[float],
        volume_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate confidence factors for edge estimation.
        
        Args:
            market_price: Current market price
            sportsbook_consensus: Sportsbook consensus probability
            volume_data: Volume and trading data
            
        Returns:
            Dictionary with confidence components
        """
        factors = {}
        
        # Sportsbook agreement factor
        if sportsbook_consensus is not None:
            price_diff = abs(market_price - sportsbook_consensus)
            # Higher agreement when prices are closer (more forgiving threshold)
            factors['sportsbook_agreement'] = max(0.2, 1 - (price_diff / 0.25))
        else:
            factors['sportsbook_agreement'] = 0.5  # Neutral when no sportsbook data
        
        # Volume confidence factor
        volume_24h = volume_data.get('volume_24h', 0)
        open_interest = volume_data.get('open_interest', 0)
        
        # Higher volume = higher confidence
        if volume_24h > 1000:
            factors['volume_confidence'] = min(1.0, volume_24h / 10000)
        else:
            factors['volume_confidence'] = 0.3  # Low confidence for low volume
        
        # Market maturity factor (time to close)
        close_time = volume_data.get('close_time')
        if close_time:
            time_to_close = (close_time - datetime.now()).total_seconds() / 3600
            # Confidence decreases as we approach close time
            if time_to_close > 24:
                factors['time_confidence'] = 1.0
            elif time_to_close > 1:
                factors['time_confidence'] = 0.5 + 0.5 * (time_to_close / 24)
            else:
                factors['time_confidence'] = 0.1  # Very low confidence near close
        else:
            factors['time_confidence'] = 0.7  # Default
        
        # Overall confidence (weighted average)
        weights = {
            'sportsbook_agreement': 0.4,
            'volume_confidence': 0.3,
            'time_confidence': 0.3
        }
        
        factors['overall_confidence'] = sum(
            factors[factor] * weights[factor] 
            for factor in weights
        )
        
        return factors
    
    def _calculate_liquidity_adjustment(self, volume_data: Dict[str, Any]) -> float:
        """
        Calculate liquidity adjustment factor.
        
        Low liquidity markets may have higher apparent edges but be harder to trade.
        
        Args:
            volume_data: Volume and liquidity metrics
            
        Returns:
            Liquidity adjustment factor (0.5 to 1.0)
        """
        volume_24h = volume_data.get('volume_24h', 0)
        spread = volume_data.get('spread', 0.05)  # Default 5 cent spread
        
        # Volume adjustment
        if volume_24h >= 5000:
            volume_factor = 1.0
        elif volume_24h >= 1000:
            volume_factor = 0.8 + 0.2 * (volume_24h / 5000)
        else:
            volume_factor = 0.5 + 0.3 * (volume_24h / 1000)
        
        # Spread adjustment (wider spreads = harder to trade)
        if spread <= 0.02:
            spread_factor = 1.0
        elif spread <= 0.05:
            spread_factor = 0.8
        else:
            spread_factor = 0.6
        
        return max(0.5, min(volume_factor * spread_factor, 1.0))
    
    def _generate_signal(self, edge: float, confidence: float) -> Tuple[Optional[str], SignalStrength]:
        """
        Generate trading signal based on edge and confidence.
        
        Args:
            edge: Calculated trading edge
            confidence: Confidence in the edge estimate
            
        Returns:
            Tuple of (signal, signal_strength)
        """
        # Adjust edge threshold based on confidence
        min_edge = self.config.min_edge_threshold / confidence if confidence > 0 else 1.0
        
        if abs(edge) < min_edge:
            return 'HOLD', SignalStrength.NEUTRAL
        
        # Determine signal direction
        if edge > 0:
            signal = 'BUY_YES'
        else:
            signal = 'BUY_NO'
        
        # Determine signal strength
        edge_magnitude = abs(edge)
        
        if edge_magnitude >= 0.15 and confidence >= 0.8:
            strength = SignalStrength.VERY_STRONG
        elif edge_magnitude >= 0.10 and confidence >= 0.6:
            strength = SignalStrength.STRONG
        elif edge_magnitude >= 0.05 and confidence >= 0.5:
            strength = SignalStrength.MODERATE
        elif edge_magnitude >= 0.03:
            strength = SignalStrength.WEAK
        else:
            strength = SignalStrength.VERY_WEAK
        
        return signal, strength
    
    def get_required_fields(self) -> List[str]:
        """Get required data fields for edge analysis."""
        return ['market_price', 'true_probability']
    
    def calculate_position_size_recommendation(
        self,
        edge: float,
        confidence: float,
        bankroll: float,
        market_price: float
    ) -> Dict[str, Any]:
        """
        Calculate recommended position size based on edge and Kelly criterion.
        
        Args:
            edge: Calculated edge
            confidence: Confidence in edge estimate
            bankroll: Available capital
            market_price: Current market price
            
        Returns:
            Position sizing recommendation
        """
        if edge <= 0 or confidence <= 0:
            return {
                'recommended_contracts': 0,
                'recommended_dollars': 0,
                'kelly_fraction': 0,
                'reason': 'No positive edge detected'
            }
        
        # Conservative Kelly sizing with confidence adjustment
        from neural.kalshi.fees import calculate_kelly_fraction
        
        # Adjust probability for confidence
        adjusted_probability = market_price + (edge * confidence)
        kelly_fraction = calculate_kelly_fraction(
            adjusted_probability, 
            market_price, 
            max_fraction=self.config.max_position_size
        )
        
        recommended_dollars = bankroll * kelly_fraction
        recommended_contracts = int(recommended_dollars / market_price) if market_price > 0 else 0
        
        return {
            'recommended_contracts': recommended_contracts,
            'recommended_dollars': recommended_dollars,
            'kelly_fraction': kelly_fraction,
            'adjusted_probability': adjusted_probability,
            'reason': f'Kelly sizing with {confidence:.1%} confidence'
        }


class MarketInefficiencyDetector:
    """
    Detects various types of market inefficiencies and anomalies.
    
    This class identifies patterns that suggest mispricing:
    - Sudden price movements without news
    - Unusual volume patterns
    - Divergence from fundamentals
    - Arbitrage opportunities
    """
    
    def __init__(self):
        """Initialize inefficiency detector."""
        self.price_history = {}  # Cache recent price history
        
    def detect_anomalies(
        self,
        market_id: str,
        current_data: Dict[str, Any],
        historical_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Detect market anomalies that may indicate inefficiencies.
        
        Args:
            market_id: Market identifier
            current_data: Current market state
            historical_data: Historical price/volume data
            
        Returns:
            Dictionary with detected anomalies
        """
        anomalies = {
            'price_jump': False,
            'volume_spike': False,
            'spread_widening': False,
            'momentum_divergence': False,
            'severity_score': 0
        }
        
        if historical_data is None or len(historical_data) < 10:
            return anomalies
        
        # Detect price jumps
        current_price = current_data.get('last_price')
        if current_price is not None:
            recent_prices = historical_data['last'].tail(5)
            price_std = recent_prices.std()
            price_mean = recent_prices.mean()
            
            if price_std > 0:
                z_score = abs(current_price - price_mean) / price_std
                if z_score > 2:  # 2 standard deviation move
                    anomalies['price_jump'] = True
                    anomalies['severity_score'] += z_score
        
        # Detect volume spikes
        current_volume = current_data.get('volume', 0)
        if len(historical_data) >= 5:
            recent_volumes = historical_data['volume'].tail(10)
            volume_mean = recent_volumes.mean()
            
            if volume_mean > 0 and current_volume > volume_mean * 3:
                anomalies['volume_spike'] = True
                anomalies['severity_score'] += current_volume / volume_mean
        
        # Detect spread widening
        current_spread = current_data.get('spread')
        if current_spread is not None and len(historical_data) >= 5:
            # Calculate typical spread from bid-ask data if available
            if 'bid' in historical_data.columns and 'ask' in historical_data.columns:
                historical_spreads = historical_data['ask'] - historical_data['bid']
                mean_spread = historical_spreads.mean()
                
                if mean_spread > 0 and current_spread > mean_spread * 2:
                    anomalies['spread_widening'] = True
                    anomalies['severity_score'] += current_spread / mean_spread
        
        return anomalies
    
    def calculate_momentum_score(self, price_history: pd.Series) -> float:
        """
        Calculate momentum score for price series.
        
        Args:
            price_history: Historical price data
            
        Returns:
            Momentum score (-1 to 1, where 1 is strong upward momentum)
        """
        if len(price_history) < 5:
            return 0
        
        # Calculate different timeframe returns
        returns_1 = price_history.pct_change(1).tail(5).mean()
        returns_3 = price_history.pct_change(3).tail(3).mean()
        
        # Weight recent moves more heavily
        momentum = 0.7 * returns_1 + 0.3 * returns_3
        
        # Normalize to -1 to 1 range
        return np.clip(momentum * 10, -1, 1)
