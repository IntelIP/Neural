"""
Kalshi market data models and utilities.

This module provides the KalshiMarket class for representing
and working with Kalshi prediction market data.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum

from neural.kalshi.fees import calculate_kalshi_fee, calculate_expected_value


class MarketStatus(Enum):
    """Market status enumeration."""
    OPEN = "open"
    CLOSED = "closed"
    SETTLED = "settled"
    HALTED = "halted"


class OrderSide(Enum):
    """Order side for trading."""
    YES = "yes"
    NO = "no"


@dataclass
class KalshiMarket:
    """
    Represents a Kalshi prediction market.
    
    This class encapsulates all relevant data for a Kalshi market,
    including pricing, volume, and metadata.
    """
    
    # Required fields
    market_id: str
    ticker: str
    event_name: str
    
    # Pricing data
    yes_bid: Optional[float] = None
    yes_ask: Optional[float] = None
    no_bid: Optional[float] = None
    no_ask: Optional[float] = None
    last_price: Optional[float] = None
    
    # Volume and liquidity
    volume: int = 0
    volume_24h: int = 0
    open_interest: int = 0
    
    # Market metadata
    sport: Optional[str] = None
    league: Optional[str] = None
    game_id: Optional[str] = None
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    
    # Timing
    close_time: Optional[datetime] = None
    expiry_time: Optional[datetime] = None
    settled_time: Optional[datetime] = None
    
    # Status
    status: MarketStatus = MarketStatus.OPEN
    outcome: Optional[int] = None  # 1 for YES, 0 for NO
    
    # Additional data
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def mid_price(self) -> Optional[float]:
        """
        Calculate mid-market price.
        
        Returns:
            Average of best bid and ask, or None if not available
        """
        if self.yes_bid is not None and self.yes_ask is not None:
            return round((self.yes_bid + self.yes_ask) / 2, 3)
        elif self.last_price is not None:
            return self.last_price
        return None
    
    @property
    def spread(self) -> Optional[float]:
        """
        Calculate bid-ask spread.
        
        Returns:
            Spread in cents, or None if not available
        """
        if self.yes_bid is not None and self.yes_ask is not None:
            return round(self.yes_ask - self.yes_bid, 3)
        return None
    
    @property
    def implied_probability(self) -> Optional[float]:
        """
        Get market-implied probability.
        
        Returns:
            Probability implied by mid-market price
        """
        mid = self.mid_price
        if mid is not None:
            return mid
        return None
    
    @property
    def no_price(self) -> Optional[float]:
        """
        Calculate NO contract price from YES price.
        
        Returns:
            NO price (1 - YES price)
        """
        if self.mid_price is not None:
            return round(1.0 - self.mid_price, 3)
        return None
    
    @property
    def is_active(self) -> bool:
        """Check if market is currently tradeable."""
        return self.status == MarketStatus.OPEN
    
    @property
    def is_settled(self) -> bool:
        """Check if market has been settled."""
        return self.status == MarketStatus.SETTLED and self.outcome is not None
    
    def calculate_fees(self, side: OrderSide, price: float, quantity: int = 1) -> float:
        """
        Calculate trading fees for an order.
        
        Args:
            side: YES or NO
            price: Order price
            quantity: Number of contracts
            
        Returns:
            Total fees in dollars
        """
        # For NO orders, convert to equivalent YES price
        if side == OrderSide.NO:
            effective_price = 1.0 - price
        else:
            effective_price = price
        
        return calculate_kalshi_fee(effective_price, quantity)
    
    def calculate_ev(
        self, 
        your_probability: float, 
        side: OrderSide,
        quantity: int = 1
    ) -> float:
        """
        Calculate expected value for a trade.
        
        Args:
            your_probability: Your estimated probability of YES outcome
            side: YES or NO
            quantity: Number of contracts
            
        Returns:
            Expected value in dollars
        """
        if self.mid_price is None:
            raise ValueError("Cannot calculate EV without market price")
        
        if side == OrderSide.YES:
            return calculate_expected_value(
                your_probability, 
                self.mid_price, 
                quantity
            )
        else:  # NO side
            # For NO, we need probability of NO outcome
            no_probability = 1.0 - your_probability
            no_price = 1.0 - self.mid_price
            return calculate_expected_value(
                no_probability,
                no_price,
                quantity
            )
    
    def get_best_price(self, side: OrderSide, is_buy: bool) -> Optional[float]:
        """
        Get best available price for an order.
        
        Args:
            side: YES or NO
            is_buy: True for buy orders, False for sell
            
        Returns:
            Best price or None if not available
        """
        if side == OrderSide.YES:
            if is_buy:
                return self.yes_ask  # Buy at ask
            else:
                return self.yes_bid  # Sell at bid
        else:  # NO side
            if is_buy:
                return self.no_ask
            else:
                return self.no_bid
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert market to dictionary.
        
        Returns:
            Dictionary representation
        """
        data = asdict(self)
        
        # Convert datetime objects to timestamps
        if self.close_time:
            data['close_time'] = int(self.close_time.timestamp())
        if self.expiry_time:
            data['expiry_time'] = int(self.expiry_time.timestamp())
        if self.settled_time:
            data['settled_time'] = int(self.settled_time.timestamp())
        
        # Convert enum to string
        data['status'] = self.status.value
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KalshiMarket':
        """
        Create market from dictionary.
        
        Args:
            data: Dictionary with market data
            
        Returns:
            KalshiMarket instance
        """
        # Convert timestamps to datetime
        if 'close_time' in data and isinstance(data['close_time'], (int, float)):
            data['close_time'] = datetime.fromtimestamp(data['close_time'])
        if 'expiry_time' in data and isinstance(data['expiry_time'], (int, float)):
            data['expiry_time'] = datetime.fromtimestamp(data['expiry_time'])
        if 'settled_time' in data and isinstance(data['settled_time'], (int, float)):
            data['settled_time'] = datetime.fromtimestamp(data['settled_time'])
        
        # Convert status string to enum
        if 'status' in data and isinstance(data['status'], str):
            data['status'] = MarketStatus(data['status'])
        
        return cls(**data)
    
    def update_prices(
        self,
        yes_bid: Optional[float] = None,
        yes_ask: Optional[float] = None,
        no_bid: Optional[float] = None,
        no_ask: Optional[float] = None,
        last_price: Optional[float] = None,
        volume: Optional[int] = None
    ):
        """
        Update market prices and volume.
        
        Args:
            yes_bid: New YES bid price
            yes_ask: New YES ask price
            no_bid: New NO bid price
            no_ask: New NO ask price
            last_price: Last traded price
            volume: Updated volume
        """
        if yes_bid is not None:
            self.yes_bid = yes_bid
        if yes_ask is not None:
            self.yes_ask = yes_ask
        if no_bid is not None:
            self.no_bid = no_bid
        if no_ask is not None:
            self.no_ask = no_ask
        if last_price is not None:
            self.last_price = last_price
        if volume is not None:
            self.volume = volume
    
    def settle_market(self, outcome: int):
        """
        Settle the market with an outcome.
        
        Args:
            outcome: 1 for YES, 0 for NO
        """
        if outcome not in [0, 1]:
            raise ValueError(f"Outcome must be 0 or 1, got {outcome}")
        
        self.outcome = outcome
        self.status = MarketStatus.SETTLED
        self.settled_time = datetime.now()
    
    def __repr__(self) -> str:
        """String representation."""
        price_str = f"${self.mid_price:.2f}" if self.mid_price else "N/A"
        status_str = self.status.value.upper()
        return (
            f"KalshiMarket(ticker='{self.ticker}', "
            f"price={price_str}, status={status_str})"
        )


@dataclass
class MarketSnapshot:
    """
    Point-in-time snapshot of market state.
    
    Used for backtesting and historical analysis.
    """
    market: KalshiMarket
    timestamp: datetime
    sportsbook_consensus: Optional[float] = None
    public_money_pct: Optional[float] = None
    sharp_money_pct: Optional[float] = None
    news_sentiment: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'market': self.market.to_dict(),
            'timestamp': int(self.timestamp.timestamp()),
            'sportsbook_consensus': self.sportsbook_consensus,
            'public_money_pct': self.public_money_pct,
            'sharp_money_pct': self.sharp_money_pct,
            'news_sentiment': self.news_sentiment
        }