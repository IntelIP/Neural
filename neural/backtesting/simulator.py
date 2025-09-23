"""
Market Simulation for Realistic Backtesting

This module provides sophisticated market simulation capabilities that
model realistic trading conditions including:

- Fill simulation with market impact
- Slippage modeling based on liquidity
- Bid-ask spread simulation
- Partial fill handling
- Latency simulation
- Market microstructure effects

The simulator ensures backtests reflect realistic trading conditions
rather than assuming perfect execution at historical prices.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

from neural.strategy.base import Signal

logger = logging.getLogger(__name__)


class FillType(Enum):
    """Types of order fills."""
    FULL = "full"          # Order completely filled
    PARTIAL = "partial"    # Order partially filled
    REJECTED = "rejected"  # Order rejected
    PENDING = "pending"    # Order pending execution


class SlippageModel(Enum):
    """Slippage modeling approaches."""
    FIXED = "fixed"              # Fixed slippage per trade
    LINEAR = "linear"            # Linear with trade size
    SQRT = "sqrt"               # Square root of trade size  
    IMPACT = "impact"           # Market impact model
    HISTORICAL = "historical"   # Based on historical data


@dataclass
class MarketState:
    """Current market state for simulation."""
    market_id: str
    timestamp: datetime
    bid: float
    ask: float
    last: float
    volume: int
    open_interest: int
    spread: float
    depth: Dict[float, int] = field(default_factory=dict)  # Price -> Quantity
    volatility: float = 0.02  # Estimated volatility
    liquidity_score: float = 0.5  # 0-1 liquidity measure


@dataclass
class OrderRequest:
    """Order execution request."""
    order_id: str
    signal: Signal
    quantity: int
    side: str  # YES or NO
    order_type: str = "market"  # market, limit
    limit_price: Optional[float] = None
    time_in_force: str = "immediate"  # immediate, day, gtc
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class FillResult:
    """Result of order execution simulation."""
    order_id: str
    fill_type: FillType
    filled_quantity: int
    fill_price: float
    total_cost: float
    slippage: float
    market_impact: float
    fees: float
    latency_ms: float
    timestamp: datetime
    remaining_quantity: int = 0
    rejection_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SlippageCalculator(ABC):
    """Abstract base class for slippage calculation."""
    
    @abstractmethod
    def calculate_slippage(
        self, 
        quantity: int, 
        market_state: MarketState, 
        side: str
    ) -> float:
        """Calculate slippage for given trade parameters."""
        pass


class FixedSlippageCalculator(SlippageCalculator):
    """Fixed slippage per trade."""
    
    def __init__(self, slippage_bps: float = 10.0):
        self.slippage_bps = slippage_bps
    
    def calculate_slippage(self, quantity: int, market_state: MarketState, side: str) -> float:
        return self.slippage_bps / 10000  # Convert bps to decimal


class LinearSlippageCalculator(SlippageCalculator):
    """Linear slippage based on trade size."""
    
    def __init__(self, base_slippage_bps: float = 5.0, size_multiplier: float = 0.001):
        self.base_slippage_bps = base_slippage_bps
        self.size_multiplier = size_multiplier
    
    def calculate_slippage(self, quantity: int, market_state: MarketState, side: str) -> float:
        base_slippage = self.base_slippage_bps / 10000
        size_impact = quantity * self.size_multiplier / 10000
        return base_slippage + size_impact


class SqrtSlippageCalculator(SlippageCalculator):
    """Square root slippage model (common in equity markets)."""
    
    def __init__(self, coefficient: float = 0.01):
        self.coefficient = coefficient
    
    def calculate_slippage(self, quantity: int, market_state: MarketState, side: str) -> float:
        return self.coefficient * np.sqrt(quantity) / 10000


class MarketImpactSlippageCalculator(SlippageCalculator):
    """Sophisticated market impact model."""
    
    def __init__(self, temporary_impact: float = 0.1, permanent_impact: float = 0.05):
        self.temporary_impact = temporary_impact
        self.permanent_impact = permanent_impact
    
    def calculate_slippage(self, quantity: int, market_state: MarketState, side: str) -> float:
        # Estimate participation rate (quantity / expected volume)
        daily_volume = market_state.volume * 24  # Rough estimate
        if daily_volume <= 0:
            participation_rate = 0.1  # Default to 10%
        else:
            participation_rate = min(quantity / daily_volume, 0.5)  # Cap at 50%
        
        # Calculate impact based on liquidity and volatility
        liquidity_factor = 1.0 / max(market_state.liquidity_score, 0.1)
        volatility_factor = market_state.volatility / 0.02  # Normalize to 2% base volatility
        
        # Market impact
        temporary = self.temporary_impact * participation_rate * liquidity_factor
        permanent = self.permanent_impact * participation_rate * volatility_factor
        
        total_impact = (temporary + permanent) / 100  # Convert to decimal
        return min(total_impact, 0.05)  # Cap at 5%


class MarketSimulator:
    """
    Sophisticated market simulator for realistic backtesting.
    
    This simulator models realistic trading conditions including
    slippage, market impact, partial fills, and latency effects.
    """
    
    def __init__(
        self,
        slippage_model: SlippageModel = SlippageModel.LINEAR,
        slippage_params: Dict[str, Any] = None,
        min_fill_ratio: float = 0.8,  # Minimum fill ratio for market orders
        latency_ms: Tuple[float, float] = (10.0, 50.0),  # Min/max latency
        enable_partial_fills: bool = True,
        bid_ask_spread_model: str = "historical",  # historical, fixed, dynamic
        market_hours_only: bool = False
    ):
        """
        Initialize market simulator.
        
        Args:
            slippage_model: Type of slippage model to use
            slippage_params: Parameters for slippage model
            min_fill_ratio: Minimum fill ratio for market orders
            latency_ms: Tuple of (min, max) latency in milliseconds
            enable_partial_fills: Whether to allow partial fills
            bid_ask_spread_model: How to model bid-ask spreads
            market_hours_only: Whether to restrict trading to market hours
        """
        self.slippage_model = slippage_model
        self.min_fill_ratio = min_fill_ratio
        self.latency_range = latency_ms
        self.enable_partial_fills = enable_partial_fills
        self.bid_ask_spread_model = bid_ask_spread_model
        self.market_hours_only = market_hours_only
        
        # Initialize slippage calculator
        slippage_params = slippage_params or {}
        self.slippage_calculator = self._create_slippage_calculator(
            slippage_model, slippage_params
        )
        
        # Performance tracking
        self.fill_statistics = {
            'total_orders': 0,
            'full_fills': 0,
            'partial_fills': 0,
            'rejections': 0,
            'avg_slippage': 0.0,
            'avg_latency_ms': 0.0
        }
        
        logger.info(f"Initialized MarketSimulator with {slippage_model.value} slippage model")
    
    def _create_slippage_calculator(
        self, 
        model: SlippageModel, 
        params: Dict[str, Any]
    ) -> SlippageCalculator:
        """Create appropriate slippage calculator."""
        if model == SlippageModel.FIXED:
            return FixedSlippageCalculator(**params)
        elif model == SlippageModel.LINEAR:
            return LinearSlippageCalculator(**params)
        elif model == SlippageModel.SQRT:
            return SqrtSlippageCalculator(**params)
        elif model == SlippageModel.IMPACT:
            return MarketImpactSlippageCalculator(**params)
        else:
            logger.warning(f"Unsupported slippage model: {model}, using Linear")
            return LinearSlippageCalculator()
    
    def simulate_fill(
        self, 
        order_request: OrderRequest, 
        market_state: MarketState
    ) -> FillResult:
        """
        Simulate order execution in realistic market conditions.
        
        Args:
            order_request: Order to execute
            market_state: Current market state
            
        Returns:
            FillResult with execution details
        """
        self.fill_statistics['total_orders'] += 1
        
        # Check market hours (if enabled)
        if self.market_hours_only and not self._is_market_open(market_state.timestamp):
            return self._create_rejection(
                order_request, "Market closed", market_state.timestamp
            )
        
        # Check for obvious rejections
        rejection_reason = self._check_rejection_conditions(order_request, market_state)
        if rejection_reason:
            self.fill_statistics['rejections'] += 1
            return self._create_rejection(order_request, rejection_reason, market_state.timestamp)
        
        # Calculate execution details
        fill_quantity = self._calculate_fill_quantity(order_request, market_state)
        fill_price = self._calculate_fill_price(order_request, market_state, fill_quantity)
        slippage = self._calculate_slippage(order_request, market_state, fill_quantity)
        market_impact = self._calculate_market_impact(order_request, market_state, fill_quantity)
        latency = self._simulate_latency()
        
        # Apply slippage to fill price
        adjusted_fill_price = self._apply_slippage(fill_price, slippage, order_request.side)
        
        # Calculate fees (using Kalshi fee structure)
        from neural.kalshi.fees import calculate_kalshi_fee
        fees = calculate_kalshi_fee(adjusted_fill_price, fill_quantity)
        
        # Total cost
        total_cost = fill_quantity * adjusted_fill_price + fees
        
        # Determine fill type
        if fill_quantity == order_request.quantity:
            fill_type = FillType.FULL
            self.fill_statistics['full_fills'] += 1
        elif fill_quantity > 0:
            fill_type = FillType.PARTIAL
            self.fill_statistics['partial_fills'] += 1
        else:
            fill_type = FillType.REJECTED
            self.fill_statistics['rejections'] += 1
        
        # Update performance tracking
        self._update_statistics(slippage, latency)
        
        return FillResult(
            order_id=order_request.order_id,
            fill_type=fill_type,
            filled_quantity=fill_quantity,
            fill_price=adjusted_fill_price,
            total_cost=total_cost,
            slippage=slippage,
            market_impact=market_impact,
            fees=fees,
            latency_ms=latency,
            timestamp=market_state.timestamp + timedelta(milliseconds=latency),
            remaining_quantity=max(0, order_request.quantity - fill_quantity),
            metadata={
                'base_price': fill_price,
                'bid_ask_spread': market_state.spread,
                'liquidity_score': market_state.liquidity_score,
                'volatility': market_state.volatility
            }
        )
    
    def _check_rejection_conditions(
        self, 
        order_request: OrderRequest, 
        market_state: MarketState
    ) -> Optional[str]:
        """Check for conditions that would reject the order."""
        
        # Check for invalid prices
        if market_state.bid <= 0 or market_state.ask <= 0:
            return "Invalid market prices"
        
        # Check for excessive spread
        if market_state.spread > 0.20:  # 20 cent spread
            return "Excessive bid-ask spread"
        
        # Check for zero volume markets
        if market_state.volume <= 0 and market_state.open_interest <= 10:
            return "Insufficient liquidity"
        
        # Check quantity limits
        if order_request.quantity <= 0:
            return "Invalid quantity"
        
        if order_request.quantity > 10000:  # Reasonable limit
            return "Quantity too large"
        
        # Check for limit order conditions
        if order_request.order_type == "limit" and order_request.limit_price:
            if order_request.side == "YES":
                if order_request.limit_price < market_state.bid:
                    return "Limit price below bid"
            else:  # NO side
                if order_request.limit_price > market_state.ask:
                    return "Limit price above ask"
        
        return None
    
    def _calculate_fill_quantity(
        self, 
        order_request: OrderRequest, 
        market_state: MarketState
    ) -> int:
        """Calculate how much of the order gets filled."""
        
        # For market orders, usually get filled but might be partial in low liquidity
        if order_request.order_type == "market":
            if market_state.liquidity_score >= 0.7:  # High liquidity
                return order_request.quantity
            elif market_state.liquidity_score >= 0.3:  # Medium liquidity
                # Might get partial fill
                if self.enable_partial_fills and np.random.random() < 0.2:  # 20% chance
                    fill_ratio = np.random.uniform(self.min_fill_ratio, 1.0)
                    return int(order_request.quantity * fill_ratio)
                else:
                    return order_request.quantity
            else:  # Low liquidity
                # Higher chance of partial fill
                if self.enable_partial_fills and np.random.random() < 0.4:  # 40% chance
                    fill_ratio = np.random.uniform(0.5, 1.0)
                    return int(order_request.quantity * fill_ratio)
                else:
                    return order_request.quantity
        
        # For limit orders, more complex logic would be needed
        else:
            # Simplified: assume limit orders fill if price is favorable
            return order_request.quantity
    
    def _calculate_fill_price(
        self, 
        order_request: OrderRequest, 
        market_state: MarketState, 
        quantity: int
    ) -> float:
        """Calculate the base fill price before slippage."""
        
        if order_request.order_type == "market":
            # Market orders fill at bid/ask
            if order_request.side == "YES":
                return market_state.ask  # Buying YES at ask
            else:  # NO
                return market_state.bid  # Buying NO (selling YES) at bid
        
        elif order_request.order_type == "limit":
            # Limit orders fill at limit price if favorable
            return order_request.limit_price or market_state.last
        
        else:
            # Default to last price
            return market_state.last
    
    def _calculate_slippage(
        self, 
        order_request: OrderRequest, 
        market_state: MarketState, 
        quantity: int
    ) -> float:
        """Calculate slippage for the order."""
        return self.slippage_calculator.calculate_slippage(
            quantity, market_state, order_request.side
        )
    
    def _calculate_market_impact(
        self, 
        order_request: OrderRequest, 
        market_state: MarketState, 
        quantity: int
    ) -> float:
        """Calculate market impact of the trade."""
        # Simplified market impact model
        daily_volume = market_state.volume * 24
        if daily_volume <= 0:
            return 0.0
        
        participation_rate = quantity / daily_volume
        impact = participation_rate * market_state.volatility * 0.5
        
        return min(impact, 0.01)  # Cap at 1%
    
    def _apply_slippage(self, base_price: float, slippage: float, side: str) -> float:
        """Apply slippage to the base fill price."""
        if side == "YES":
            # Buying YES - slippage increases price
            adjusted_price = base_price * (1 + slippage)
        else:  # NO
            # Buying NO - slippage decreases the effective price
            adjusted_price = base_price * (1 - slippage)
        
        # Ensure price stays within valid bounds
        return max(0.01, min(adjusted_price, 0.99))
    
    def _simulate_latency(self) -> float:
        """Simulate execution latency."""
        return np.random.uniform(self.latency_range[0], self.latency_range[1])
    
    def _is_market_open(self, timestamp: datetime) -> bool:
        """Check if market is open (simplified)."""
        # For Kalshi, markets are generally open 24/7
        # This could be enhanced to handle specific market hours
        return True
    
    def _create_rejection(
        self, 
        order_request: OrderRequest, 
        reason: str, 
        timestamp: datetime
    ) -> FillResult:
        """Create a rejection result."""
        return FillResult(
            order_id=order_request.order_id,
            fill_type=FillType.REJECTED,
            filled_quantity=0,
            fill_price=0.0,
            total_cost=0.0,
            slippage=0.0,
            market_impact=0.0,
            fees=0.0,
            latency_ms=self._simulate_latency(),
            timestamp=timestamp,
            remaining_quantity=order_request.quantity,
            rejection_reason=reason
        )
    
    def _update_statistics(self, slippage: float, latency: float) -> None:
        """Update performance statistics."""
        total = self.fill_statistics['total_orders']
        
        # Update running averages
        current_avg_slippage = self.fill_statistics['avg_slippage']
        self.fill_statistics['avg_slippage'] = (
            (current_avg_slippage * (total - 1) + slippage) / total
        )
        
        current_avg_latency = self.fill_statistics['avg_latency_ms']
        self.fill_statistics['avg_latency_ms'] = (
            (current_avg_latency * (total - 1) + latency) / total
        )
    
    def create_market_state_from_data(self, market_data: pd.Series) -> MarketState:
        """Create MarketState from historical market data."""
        bid = market_data.get('bid', market_data.get('last', 0.5) - 0.01)
        ask = market_data.get('ask', market_data.get('last', 0.5) + 0.01)
        last = market_data.get('last', (bid + ask) / 2)
        
        # Estimate liquidity score based on volume and spread
        volume = market_data.get('volume', 0)
        spread = ask - bid
        
        # Simple liquidity scoring
        if volume > 5000 and spread < 0.02:
            liquidity_score = 0.9
        elif volume > 1000 and spread < 0.05:
            liquidity_score = 0.7
        elif volume > 100:
            liquidity_score = 0.5
        else:
            liquidity_score = 0.2
        
        return MarketState(
            market_id=market_data.get('market_id', 'UNKNOWN'),
            timestamp=datetime.now(),
            bid=bid,
            ask=ask,
            last=last,
            volume=volume,
            open_interest=market_data.get('open_interest', 0),
            spread=spread,
            liquidity_score=liquidity_score,
            volatility=0.02  # Default 2% volatility
        )
    
    def get_fill_statistics(self) -> Dict[str, Any]:
        """Get fill performance statistics."""
        total = self.fill_statistics['total_orders']
        if total == 0:
            return self.fill_statistics.copy()
        
        stats = self.fill_statistics.copy()
        stats.update({
            'fill_rate': (stats['full_fills'] + stats['partial_fills']) / total,
            'rejection_rate': stats['rejections'] / total,
            'avg_slippage_bps': stats['avg_slippage'] * 10000
        })
        
        return stats
    
    def reset_statistics(self) -> None:
        """Reset fill performance statistics."""
        self.fill_statistics = {
            'total_orders': 0,
            'full_fills': 0,
            'partial_fills': 0,
            'rejections': 0,
            'avg_slippage': 0.0,
            'avg_latency_ms': 0.0
        }


# Convenience class for different slippage models
class FillSimulation:
    """Factory class for creating different fill simulation setups."""
    
    @staticmethod
    def conservative_model() -> MarketSimulator:
        """Conservative fill simulation with higher slippage."""
        return MarketSimulator(
            slippage_model=SlippageModel.LINEAR,
            slippage_params={'base_slippage_bps': 15.0, 'size_multiplier': 0.002},
            min_fill_ratio=0.9,
            latency_ms=(15.0, 75.0)
        )
    
    @staticmethod
    def realistic_model() -> MarketSimulator:
        """Realistic fill simulation for typical conditions."""
        return MarketSimulator(
            slippage_model=SlippageModel.LINEAR,
            slippage_params={'base_slippage_bps': 8.0, 'size_multiplier': 0.001},
            min_fill_ratio=0.85,
            latency_ms=(10.0, 50.0)
        )
    
    @staticmethod
    def optimistic_model() -> MarketSimulator:
        """Optimistic fill simulation with lower costs."""
        return MarketSimulator(
            slippage_model=SlippageModel.FIXED,
            slippage_params={'slippage_bps': 5.0},
            min_fill_ratio=0.95,
            latency_ms=(5.0, 25.0)
        )
    
    @staticmethod
    def market_impact_model() -> MarketSimulator:
        """Sophisticated market impact model."""
        return MarketSimulator(
            slippage_model=SlippageModel.IMPACT,
            slippage_params={'temporary_impact': 0.15, 'permanent_impact': 0.08},
            min_fill_ratio=0.8,
            latency_ms=(10.0, 60.0),
            enable_partial_fills=True
        )
