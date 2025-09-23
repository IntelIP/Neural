"""
Kalshi fee calculation utilities.

This module provides functions for calculating Kalshi trading fees
and expected values for trades.
"""

from typing import Optional


def calculate_kalshi_fee(
    price: float, 
    quantity: int = 1,
    fee_rate: float = 0.07
) -> float:
    """
    Calculate Kalshi trading fee.
    
    Kalshi charges fees based on the formula:
    Fee = fee_rate × P × (1 - P) × quantity
    
    Where P is the contract price. Fees are minimal at price extremes
    (near 0.01 or 0.99) and maximum around 0.50.
    
    Args:
        price: Contract price in dollars (0.01 to 0.99)
        quantity: Number of contracts
        fee_rate: Fee rate (default 0.07 for Kalshi)
        
    Returns:
        Total fee in dollars
        
    Example:
        >>> calculate_kalshi_fee(0.40, 10)
        0.168  # $0.168 fee for 10 contracts at $0.40
    """
    if not 0.01 <= price <= 0.99:
        raise ValueError(f"Price must be between 0.01 and 0.99, got {price}")
    
    if quantity <= 0:
        raise ValueError(f"Quantity must be positive, got {quantity}")
    
    # Kalshi fee formula
    fee_per_contract = fee_rate * price * (1 - price)
    total_fee = fee_per_contract * quantity
    
    return round(total_fee, 4)


def calculate_expected_value(
    your_probability: float,
    market_price: float,
    quantity: int = 1,
    fee_rate: float = 0.07
) -> float:
    """
    Calculate expected value of a Kalshi trade.
    
    EV = (Your_Probability × Profit_if_Win) - ((1 - Your_Probability) × Loss_if_Lose) - Fees
    
    Args:
        your_probability: Your estimated probability of the event (0 to 1)
        market_price: Current market price (0.01 to 0.99)
        quantity: Number of contracts
        fee_rate: Kalshi fee rate (default 0.07)
        
    Returns:
        Expected value in dollars (positive = profitable)
        
    Example:
        >>> calculate_expected_value(0.55, 0.40, 10)
        1.17  # $1.17 expected profit for this trade
    """
    if not 0 <= your_probability <= 1:
        raise ValueError(f"Probability must be between 0 and 1, got {your_probability}")
    
    if not 0.01 <= market_price <= 0.99:
        raise ValueError(f"Market price must be between 0.01 and 0.99, got {market_price}")
    
    # Calculate fees
    fees = calculate_kalshi_fee(market_price, quantity, fee_rate)
    
    # Calculate profit if the contract wins (pays $1)
    profit_if_win = (1.0 - market_price) * quantity - fees
    
    # Calculate loss if the contract loses (pays $0)
    loss_if_lose = market_price * quantity + fees
    
    # Calculate expected value
    ev = (your_probability * profit_if_win) - ((1 - your_probability) * loss_if_lose)
    
    return round(ev, 4)


def calculate_edge(
    your_probability: float,
    market_price: float
) -> float:
    """
    Calculate your edge over the market.
    
    Edge = Your_Probability - Market_Implied_Probability
    
    Args:
        your_probability: Your estimated probability (0 to 1)
        market_price: Market price (implies probability)
        
    Returns:
        Edge as a decimal (positive = you have edge)
        
    Example:
        >>> calculate_edge(0.65, 0.50)
        0.15  # 15% edge
    """
    return your_probability - market_price


def calculate_breakeven_probability(
    market_price: float,
    fee_rate: float = 0.07
) -> float:
    """
    Calculate the minimum probability needed to break even after fees.
    
    Args:
        market_price: Market price (0.01 to 0.99)
        fee_rate: Kalshi fee rate (default 0.07)
        
    Returns:
        Breakeven probability (0 to 1)
        
    Example:
        >>> calculate_breakeven_probability(0.40)
        0.428  # Need 42.8% probability to break even at 40¢
    """
    if not 0.01 <= market_price <= 0.99:
        raise ValueError(f"Market price must be between 0.01 and 0.99, got {market_price}")
    
    # Calculate fee per contract
    fee = fee_rate * market_price * (1 - market_price)
    
    # Breakeven probability calculation
    # You need probability >= (price + fee) to break even
    # because total cost is (price + fee) and payout is $1 if correct
    breakeven = market_price + fee
    
    return round(breakeven, 4)


def calculate_kelly_fraction(
    your_probability: float,
    market_price: float,
    max_fraction: float = 0.25
) -> float:
    """
    Calculate Kelly Criterion position size fraction.
    
    Kelly formula: f* = (bp - q) / b
    Where:
        f* = fraction of bankroll to bet
        b = net odds received (payout/cost - 1)
        p = probability of winning
        q = probability of losing (1 - p)
    
    Args:
        your_probability: Your estimated probability (0 to 1)
        market_price: Market price (0.01 to 0.99)
        max_fraction: Maximum fraction to bet (default 0.25 for safety)
        
    Returns:
        Fraction of bankroll to bet (0 to max_fraction)
        
    Example:
        >>> calculate_kelly_fraction(0.60, 0.40)
        0.25  # Bet 25% of bankroll (capped at max)
    """
    if not 0 <= your_probability <= 1:
        raise ValueError(f"Probability must be between 0 and 1, got {your_probability}")
    
    if not 0.01 <= market_price <= 0.99:
        raise ValueError(f"Market price must be between 0.01 and 0.99, got {market_price}")
    
    # Calculate edge
    edge = your_probability - market_price
    
    # If no edge, don't bet
    if edge <= 0:
        return 0.0
    
    # Calculate net odds (b)
    # If we win, we get $1 for a cost of market_price
    net_odds = (1.0 / market_price) - 1
    
    # Kelly formula
    kelly = edge / net_odds
    
    # Cap at maximum fraction for safety
    return min(kelly, max_fraction)


def calculate_position_value(
    quantity: int,
    entry_price: float,
    current_price: float
) -> float:
    """
    Calculate current value of a position.
    
    Args:
        quantity: Number of contracts (positive for YES, negative for NO)
        entry_price: Price at which position was entered
        current_price: Current market price
        
    Returns:
        Current P&L in dollars
        
    Example:
        >>> calculate_position_value(10, 0.40, 0.50)
        1.0  # $1.00 profit on 10 contracts
    """
    if quantity > 0:  # Long YES position
        pnl = (current_price - entry_price) * quantity
    else:  # Short YES / Long NO position
        pnl = (entry_price - current_price) * abs(quantity)
    
    return round(pnl, 4)