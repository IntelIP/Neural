"""
Position Sizing Algorithms for Risk Management

This module implements various position sizing methods that are crucial
for risk management and capital preservation:

- Kelly Criterion: Optimal sizing based on edge and odds
- Fixed Percentage: Conservative fixed allocation
- Volatility Sizing: Size based on market volatility
- Risk Parity: Equal risk contribution across positions
- Target Volatility: Size to achieve target portfolio volatility

All methods include safeguards and caps to prevent over-leveraging
and ensure conservative risk management practices.
"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

from neural.strategy.base import Signal
from neural.kalshi.fees import calculate_kalshi_fee

logger = logging.getLogger(__name__)


class PositionSizingMethod(Enum):
    """Available position sizing methods."""
    KELLY_CRITERION = "kelly_criterion"
    FIXED_PERCENTAGE = "fixed_percentage"
    FIXED_AMOUNT = "fixed_amount"
    VOLATILITY_TARGET = "volatility_target"
    RISK_PARITY = "risk_parity"
    EQUAL_WEIGHT = "equal_weight"
    SIGNAL_WEIGHTED = "signal_weighted"


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation."""
    method: PositionSizingMethod
    recommended_size: float  # Dollar amount to allocate
    recommended_contracts: int  # Number of contracts
    risk_percentage: float  # Percentage of capital at risk
    rationale: str  # Explanation of sizing decision
    confidence: float  # Confidence in the sizing (0-1)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseSizer(ABC):
    """Abstract base class for position sizing algorithms."""
    
    def __init__(self, name: str):
        self.name = name
        logger.debug(f"Initialized {name} position sizer")
    
    @abstractmethod
    def calculate_size(
        self,
        signal: Signal,
        current_capital: float,
        current_positions: Dict[str, Any],
        market_data: Dict[str, Any] = None
    ) -> PositionSizeResult:
        """Calculate position size for given signal."""
        pass
    
    def _validate_inputs(self, signal: Signal, current_capital: float) -> None:
        """Validate inputs for position sizing."""
        if current_capital <= 0:
            raise ValueError("Current capital must be positive")
        
        if not hasattr(signal, 'confidence') or signal.confidence < 0:
            raise ValueError("Signal must have valid confidence score")
        
        if hasattr(signal, 'edge') and signal.edge is not None and signal.edge < 0:
            logger.warning(f"Negative edge detected: {signal.edge}")


class KellySizer(BaseSizer):
    """
    Kelly Criterion position sizing for optimal growth.
    
    The Kelly Criterion maximizes long-term growth by sizing positions
    based on edge and odds. We use a conservative approach with caps
    and fractional Kelly for safety.
    """
    
    def __init__(
        self,
        max_kelly_fraction: float = 0.25,  # Cap at 25% of Kelly
        min_kelly_fraction: float = 0.01,  # Minimum 1% position
        max_position_size: float = 0.10,   # Never exceed 10% of capital
        confidence_threshold: float = 0.6,  # Minimum confidence for Kelly
        edge_threshold: float = 0.02       # Minimum edge for Kelly
    ):
        super().__init__("Kelly Criterion")
        self.max_kelly_fraction = max_kelly_fraction
        self.min_kelly_fraction = min_kelly_fraction
        self.max_position_size = max_position_size
        self.confidence_threshold = confidence_threshold
        self.edge_threshold = edge_threshold
    
    def calculate_size(
        self,
        signal: Signal,
        current_capital: float,
        current_positions: Dict[str, Any],
        market_data: Dict[str, Any] = None
    ) -> PositionSizeResult:
        """Calculate Kelly Criterion position size."""
        self._validate_inputs(signal, current_capital)
        
        warnings = []
        
        # Get signal parameters
        edge = getattr(signal, 'edge', 0.0)
        confidence = signal.confidence
        market_price = getattr(signal, 'market_price', 0.5)
        
        # Check if we should use Kelly
        if confidence < self.confidence_threshold:
            warnings.append(f"Low confidence ({confidence:.2f}) for Kelly sizing")
        
        if edge < self.edge_threshold:
            warnings.append(f"Low edge ({edge:.3f}) for Kelly sizing")
        
        # Calculate Kelly fraction
        if edge > 0 and market_price > 0:
            # For Kalshi binary contracts: Kelly = edge / odds
            # Odds for YES: (1 - market_price) / market_price
            # Odds for NO: market_price / (1 - market_price)
            
            if signal.signal_type.value in ['BUY_YES', 'buy_yes']:
                odds = (1 - market_price) / market_price if market_price > 0 else 1.0
            else:  # BUY_NO
                odds = market_price / (1 - market_price) if market_price < 1 else 1.0
            
            kelly_fraction = edge / odds if odds > 0 else 0.0
            
            # Apply confidence adjustment
            kelly_fraction *= confidence
            
            # Apply caps and floors
            kelly_fraction = max(self.min_kelly_fraction, kelly_fraction)
            kelly_fraction = min(self.max_kelly_fraction, kelly_fraction)
            
        else:
            kelly_fraction = self.min_kelly_fraction
            warnings.append("Zero or negative edge, using minimum position size")
        
        # Calculate position size
        raw_position_size = current_capital * kelly_fraction
        
        # Apply maximum position size limit
        position_size = min(raw_position_size, current_capital * self.max_position_size)
        
        if position_size < raw_position_size:
            warnings.append(f"Position capped at {self.max_position_size*100:.1f}% of capital")
        
        # Calculate number of contracts
        # Account for fees in the calculation
        effective_price = market_price
        fee_estimate = calculate_kalshi_fee(effective_price, 1)  # Fee per contract
        cost_per_contract = effective_price + fee_estimate
        
        contracts = int(position_size / cost_per_contract) if cost_per_contract > 0 else 0
        actual_position_size = contracts * cost_per_contract
        
        # Calculate risk percentage
        risk_pct = actual_position_size / current_capital if current_capital > 0 else 0
        
        # Generate rationale
        rationale = (f"Kelly sizing: edge={edge:.3f}, odds={odds:.2f}, "
                   f"kelly={kelly_fraction:.3f}, confidence={confidence:.2f}")
        
        return PositionSizeResult(
            method=PositionSizingMethod.KELLY_CRITERION,
            recommended_size=actual_position_size,
            recommended_contracts=contracts,
            risk_percentage=risk_pct,
            rationale=rationale,
            confidence=confidence,
            warnings=warnings,
            metadata={
                'edge': edge,
                'odds': odds,
                'kelly_fraction': kelly_fraction,
                'cost_per_contract': cost_per_contract,
                'fee_estimate': fee_estimate
            }
        )


class FixedSizer(BaseSizer):
    """
    Fixed percentage or fixed amount position sizing.
    
    Simple and conservative approach that allocates a fixed
    percentage of capital to each position.
    """
    
    def __init__(
        self,
        sizing_type: str = "percentage",  # "percentage" or "amount"
        size_value: float = 0.02,        # 2% of capital or $100
        min_position_size: float = 10.0,  # Minimum $10 position
        max_position_size: float = None,  # Optional maximum
        confidence_scaling: bool = True   # Scale by confidence
    ):
        super().__init__("Fixed Sizing")
        self.sizing_type = sizing_type
        self.size_value = size_value
        self.min_position_size = min_position_size
        self.max_position_size = max_position_size
        self.confidence_scaling = confidence_scaling
    
    def calculate_size(
        self,
        signal: Signal,
        current_capital: float,
        current_positions: Dict[str, Any],
        market_data: Dict[str, Any] = None
    ) -> PositionSizeResult:
        """Calculate fixed position size."""
        self._validate_inputs(signal, current_capital)
        
        warnings = []
        confidence = signal.confidence
        market_price = getattr(signal, 'market_price', 0.5)
        
        # Calculate base position size
        if self.sizing_type == "percentage":
            base_size = current_capital * self.size_value
            rationale = f"Fixed {self.size_value*100:.1f}% of capital"
        else:  # amount
            base_size = self.size_value
            rationale = f"Fixed ${self.size_value:.2f} amount"
        
        # Apply confidence scaling if enabled
        if self.confidence_scaling:
            position_size = base_size * confidence
            rationale += f" scaled by confidence ({confidence:.2f})"
        else:
            position_size = base_size
        
        # Apply minimum position size
        if position_size < self.min_position_size:
            position_size = self.min_position_size
            warnings.append(f"Position increased to minimum ${self.min_position_size:.2f}")
        
        # Apply maximum position size if set
        if self.max_position_size and position_size > self.max_position_size:
            position_size = self.max_position_size
            warnings.append(f"Position capped at ${self.max_position_size:.2f}")
        
        # Calculate contracts
        fee_estimate = calculate_kalshi_fee(market_price, 1)
        cost_per_contract = market_price + fee_estimate
        
        contracts = int(position_size / cost_per_contract) if cost_per_contract > 0 else 0
        actual_position_size = contracts * cost_per_contract
        
        risk_pct = actual_position_size / current_capital if current_capital > 0 else 0
        
        return PositionSizeResult(
            method=PositionSizingMethod.FIXED_PERCENTAGE if self.sizing_type == "percentage" 
                   else PositionSizingMethod.FIXED_AMOUNT,
            recommended_size=actual_position_size,
            recommended_contracts=contracts,
            risk_percentage=risk_pct,
            rationale=rationale,
            confidence=confidence,
            warnings=warnings,
            metadata={
                'base_size': base_size,
                'confidence_scaling': self.confidence_scaling,
                'cost_per_contract': cost_per_contract
            }
        )


class VolatilitySizer(BaseSizer):
    """
    Volatility-based position sizing.
    
    Sizes positions inversely to volatility to maintain consistent
    risk across different market conditions and assets.
    """
    
    def __init__(
        self,
        target_volatility: float = 0.02,    # 2% daily volatility target
        lookback_period: int = 30,          # Days to calculate volatility
        max_position_size: float = 0.15,    # Maximum 15% of capital
        min_position_size: float = 0.01,    # Minimum 1% of capital
        volatility_floor: float = 0.005,    # 0.5% minimum volatility assumption
        volatility_cap: float = 0.10        # 10% maximum volatility assumption
    ):
        super().__init__("Volatility Targeting")
        self.target_volatility = target_volatility
        self.lookback_period = lookback_period
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.volatility_floor = volatility_floor
        self.volatility_cap = volatility_cap
    
    def calculate_size(
        self,
        signal: Signal,
        current_capital: float,
        current_positions: Dict[str, Any],
        market_data: Dict[str, Any] = None
    ) -> PositionSizeResult:
        """Calculate volatility-adjusted position size."""
        self._validate_inputs(signal, current_capital)
        
        warnings = []
        confidence = signal.confidence
        market_price = getattr(signal, 'market_price', 0.5)
        
        # Estimate volatility from market data or use default
        estimated_volatility = self._estimate_volatility(
            signal.market_id, market_data
        )
        
        # Apply volatility bounds
        if estimated_volatility < self.volatility_floor:
            estimated_volatility = self.volatility_floor
            warnings.append("Using minimum volatility assumption")
        elif estimated_volatility > self.volatility_cap:
            estimated_volatility = self.volatility_cap
            warnings.append("Using maximum volatility assumption")
        
        # Calculate position size inversely proportional to volatility
        volatility_ratio = self.target_volatility / estimated_volatility
        base_position_pct = self.target_volatility * volatility_ratio
        
        # Apply confidence scaling
        position_pct = base_position_pct * confidence
        
        # Apply bounds
        position_pct = max(self.min_position_size, position_pct)
        position_pct = min(self.max_position_size, position_pct)
        
        position_size = current_capital * position_pct
        
        # Calculate contracts
        fee_estimate = calculate_kalshi_fee(market_price, 1)
        cost_per_contract = market_price + fee_estimate
        
        contracts = int(position_size / cost_per_contract) if cost_per_contract > 0 else 0
        actual_position_size = contracts * cost_per_contract
        
        risk_pct = actual_position_size / current_capital if current_capital > 0 else 0
        
        rationale = (f"Vol targeting: estimated={estimated_volatility:.3f}, "
                    f"target={self.target_volatility:.3f}, "
                    f"allocation={position_pct:.3f}")
        
        return PositionSizeResult(
            method=PositionSizingMethod.VOLATILITY_TARGET,
            recommended_size=actual_position_size,
            recommended_contracts=contracts,
            risk_percentage=risk_pct,
            rationale=rationale,
            confidence=confidence,
            warnings=warnings,
            metadata={
                'estimated_volatility': estimated_volatility,
                'target_volatility': self.target_volatility,
                'volatility_ratio': volatility_ratio,
                'position_pct': position_pct
            }
        )
    
    def _estimate_volatility(
        self, 
        market_id: str, 
        market_data: Dict[str, Any] = None
    ) -> float:
        """Estimate market volatility from historical data."""
        if not market_data or 'price_history' not in market_data:
            # Default volatility for prediction markets (typically lower than stocks)
            return 0.03
        
        try:
            price_history = market_data['price_history']
            if isinstance(price_history, pd.Series) and len(price_history) >= 2:
                # Calculate returns and volatility
                returns = price_history.pct_change().dropna()
                if len(returns) >= 2:
                    volatility = returns.std() * np.sqrt(24)  # Assuming hourly data
                    return max(self.volatility_floor, min(volatility, self.volatility_cap))
        except Exception as e:
            logger.warning(f"Error estimating volatility for {market_id}: {e}")
        
        return 0.03  # Default volatility


class PositionSizer:
    """
    Main position sizing orchestrator that coordinates different sizing methods.
    
    This class provides a unified interface for all position sizing algorithms
    and can blend multiple methods or switch between them based on market conditions.
    """
    
    def __init__(
        self,
        primary_method: PositionSizingMethod = PositionSizingMethod.KELLY_CRITERION,
        fallback_method: PositionSizingMethod = PositionSizingMethod.FIXED_PERCENTAGE,
        blend_methods: bool = False,
        method_weights: Dict[PositionSizingMethod, float] = None
    ):
        """
        Initialize position sizer with primary and fallback methods.
        
        Args:
            primary_method: Main sizing method to use
            fallback_method: Method to use if primary fails/inappropriate
            blend_methods: Whether to blend multiple methods
            method_weights: Weights for blending methods
        """
        self.primary_method = primary_method
        self.fallback_method = fallback_method
        self.blend_methods = blend_methods
        self.method_weights = method_weights or {}
        
        # Initialize sizers
        self.sizers = {
            PositionSizingMethod.KELLY_CRITERION: KellySizer(),
            PositionSizingMethod.FIXED_PERCENTAGE: FixedSizer("percentage", 0.03),
            PositionSizingMethod.FIXED_AMOUNT: FixedSizer("amount", 100.0),
            PositionSizingMethod.VOLATILITY_TARGET: VolatilitySizer()
        }
        
        logger.info(f"Initialized PositionSizer with primary method: {primary_method.value}")
    
    def calculate_position_size(
        self,
        signal: Signal,
        current_capital: float,
        current_positions: Dict[str, Any],
        market_data: Dict[str, Any] = None,
        override_method: PositionSizingMethod = None
    ) -> PositionSizeResult:
        """
        Calculate optimal position size using configured methods.
        
        Args:
            signal: Trading signal with edge and confidence
            current_capital: Available capital
            current_positions: Current portfolio positions
            market_data: Market data for volatility estimation
            override_method: Override the default method
            
        Returns:
            PositionSizeResult with sizing recommendation
        """
        method_to_use = override_method or self.primary_method
        
        try:
            if method_to_use in self.sizers:
                result = self.sizers[method_to_use].calculate_size(
                    signal, current_capital, current_positions, market_data
                )
                
                # Validate result
                if self._validate_result(result, current_capital):
                    return result
                else:
                    logger.warning(f"{method_to_use.value} produced invalid result, using fallback")
                    method_to_use = self.fallback_method
            
            # Use fallback method
            if method_to_use != self.fallback_method and self.fallback_method in self.sizers:
                result = self.sizers[self.fallback_method].calculate_size(
                    signal, current_capital, current_positions, market_data
                )
                result.warnings.append(f"Used fallback method: {self.fallback_method.value}")
                return result
            
        except Exception as e:
            logger.error(f"Position sizing failed: {e}")
        
        # Emergency fallback: very conservative fixed sizing
        return self._emergency_fallback(signal, current_capital)
    
    def _validate_result(self, result: PositionSizeResult, current_capital: float) -> bool:
        """Validate position sizing result."""
        if result.recommended_size < 0:
            return False
        
        if result.recommended_contracts < 0:
            return False
        
        if result.risk_percentage > 0.5:  # Never risk more than 50%
            return False
        
        if result.recommended_size > current_capital * 0.5:  # Never exceed 50% of capital
            return False
        
        return True
    
    def _emergency_fallback(
        self, 
        signal: Signal, 
        current_capital: float
    ) -> PositionSizeResult:
        """Emergency fallback sizing when all methods fail."""
        emergency_size = min(50.0, current_capital * 0.01)  # 1% or $50, whichever is smaller
        market_price = getattr(signal, 'market_price', 0.5)
        
        fee_estimate = calculate_kalshi_fee(market_price, 1)
        cost_per_contract = market_price + fee_estimate
        contracts = int(emergency_size / cost_per_contract) if cost_per_contract > 0 else 0
        
        return PositionSizeResult(
            method=PositionSizingMethod.FIXED_PERCENTAGE,
            recommended_size=contracts * cost_per_contract,
            recommended_contracts=contracts,
            risk_percentage=emergency_size / current_capital,
            rationale="Emergency fallback: very conservative 1% allocation",
            confidence=0.1,
            warnings=["Using emergency fallback sizing"],
            metadata={'emergency_fallback': True}
        )
    
    def get_sizing_statistics(
        self, 
        signals: List[Signal], 
        current_capital: float
    ) -> Dict[str, Any]:
        """Get statistics about position sizing across multiple signals."""
        if not signals:
            return {}
        
        results = []
        for signal in signals:
            try:
                result = self.calculate_position_size(signal, current_capital, {})
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to size signal for {signal.market_id}: {e}")
        
        if not results:
            return {}
        
        total_allocation = sum(r.recommended_size for r in results)
        avg_risk_pct = np.mean([r.risk_percentage for r in results])
        
        return {
            'total_signals': len(signals),
            'successful_sizing': len(results),
            'total_allocation': total_allocation,
            'allocation_percentage': total_allocation / current_capital,
            'average_risk_per_position': avg_risk_pct,
            'method_distribution': {
                method.value: sum(1 for r in results if r.method == method)
                for method in PositionSizingMethod
            }
        }
