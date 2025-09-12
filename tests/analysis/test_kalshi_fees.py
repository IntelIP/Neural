"""
Unit tests for Kalshi fee calculations.

Tests fee formulas, expected value calculations, and position sizing.
"""

import pytest
from neural.kalshi.fees import (
    calculate_kalshi_fee,
    calculate_expected_value,
    calculate_edge,
    calculate_breakeven_probability,
    calculate_kelly_fraction,
    calculate_position_value
)


class TestKalshiFees:
    """Test suite for Kalshi fee calculations."""
    
    def test_calculate_kalshi_fee_basic(self):
        """Test basic fee calculation."""
        # Fee at 50 cents (maximum fee)
        fee = calculate_kalshi_fee(0.50, 1)
        assert fee == pytest.approx(0.0175, rel=1e-4)  # 0.07 * 0.5 * 0.5 = 0.0175
        
        # Fee at 40 cents
        fee = calculate_kalshi_fee(0.40, 1)
        assert fee == pytest.approx(0.0168, rel=1e-4)  # 0.07 * 0.4 * 0.6 = 0.0168
        
        # Fee at extremes (minimal)
        fee = calculate_kalshi_fee(0.01, 1)
        assert fee == pytest.approx(0.0007, rel=1e-4)  # 0.07 * 0.01 * 0.99 = 0.000693, rounded to 0.0007
        
        fee = calculate_kalshi_fee(0.99, 1)
        assert fee == pytest.approx(0.0007, rel=1e-4)  # 0.07 * 0.99 * 0.01 = 0.000693, rounded to 0.0007
    
    def test_calculate_kalshi_fee_quantity(self):
        """Test fee calculation with multiple contracts."""
        fee_10 = calculate_kalshi_fee(0.40, 10)
        fee_1 = calculate_kalshi_fee(0.40, 1)
        
        assert fee_10 == pytest.approx(fee_1 * 10, rel=1e-4)
    
    def test_calculate_kalshi_fee_invalid_inputs(self):
        """Test fee calculation with invalid inputs."""
        with pytest.raises(ValueError):
            calculate_kalshi_fee(0.00, 1)  # Price too low
        
        with pytest.raises(ValueError):
            calculate_kalshi_fee(1.00, 1)  # Price too high
        
        with pytest.raises(ValueError):
            calculate_kalshi_fee(0.50, 0)  # Invalid quantity
    
    def test_calculate_expected_value_positive(self):
        """Test EV calculation with positive edge."""
        # 55% probability, 40 cent market price
        ev = calculate_expected_value(0.55, 0.40, 1)
        
        # Manual calculation:
        # Fee = 0.07 * 0.4 * 0.6 = 0.0168
        # Profit if win = 0.60 - 0.0168 = 0.5832
        # Loss if lose = 0.40 + 0.0168 = 0.4168
        # EV = 0.55 * 0.5832 - 0.45 * 0.4168 = 0.3208 - 0.1876 = 0.1332
        
        assert ev == pytest.approx(0.1332, rel=1e-2)
    
    def test_calculate_expected_value_negative(self):
        """Test EV calculation with negative edge."""
        # 35% probability, 40 cent market price (market says 40%, we say 35%)
        ev = calculate_expected_value(0.35, 0.40, 1)
        
        assert ev < 0  # Should be negative EV
    
    def test_calculate_expected_value_multiple_contracts(self):
        """Test EV calculation with multiple contracts."""
        ev_1 = calculate_expected_value(0.55, 0.40, 1)
        ev_10 = calculate_expected_value(0.55, 0.40, 10)
        
        # EV should scale linearly with quantity
        assert ev_10 == pytest.approx(ev_1 * 10, rel=1e-2)
    
    def test_calculate_edge(self):
        """Test edge calculation."""
        edge = calculate_edge(0.65, 0.50)
        assert edge == pytest.approx(0.15, rel=1e-3)  # 15% edge
        
        edge = calculate_edge(0.45, 0.50)
        assert edge == pytest.approx(-0.05, rel=1e-3)  # -5% edge (disadvantage)
        
        edge = calculate_edge(0.50, 0.50)
        assert edge == pytest.approx(0.0, abs=1e-10)  # No edge
    
    def test_calculate_breakeven_probability(self):
        """Test breakeven probability calculation."""
        # At 50 cents (maximum fee)
        breakeven = calculate_breakeven_probability(0.50)
        assert breakeven > 0.50  # Should need > 50% to overcome fees
        
        # At 40 cents
        breakeven = calculate_breakeven_probability(0.40)
        assert breakeven > 0.40  # Should need higher than market price
        
        # At extremes (minimal fee)
        breakeven_low = calculate_breakeven_probability(0.01)
        assert breakeven_low == pytest.approx(0.01, rel=0.1)  # Close to market price
        
        breakeven_high = calculate_breakeven_probability(0.99)
        assert breakeven_high == pytest.approx(0.99, rel=0.1)  # Close to market price
    
    def test_calculate_kelly_fraction_with_edge(self):
        """Test Kelly fraction with positive edge."""
        # 60% probability, 40 cent market (20% edge)
        kelly = calculate_kelly_fraction(0.60, 0.40)
        
        # Manual calculation:
        # Edge = 0.60 - 0.40 = 0.20
        # Net odds = (1/0.40) - 1 = 1.5
        # Kelly = 0.20 / 1.5 = 0.1333
        
        assert kelly == pytest.approx(0.1333, rel=1e-2)
    
    def test_calculate_kelly_fraction_no_edge(self):
        """Test Kelly fraction with no edge."""
        kelly = calculate_kelly_fraction(0.40, 0.40)
        assert kelly == pytest.approx(0.0, abs=1e-6)  # No bet when no edge
        
        kelly = calculate_kelly_fraction(0.35, 0.40)
        assert kelly == pytest.approx(0.0, abs=1e-6)  # No bet when negative edge
    
    def test_calculate_kelly_fraction_capped(self):
        """Test Kelly fraction with cap."""
        # Scenario that exceeds the cap: probability=0.95, price=0.50
        # edge=0.45, net_odds=1, kelly=0.45 > 0.25, so should be capped
        kelly = calculate_kelly_fraction(0.95, 0.50, max_fraction=0.25)
        assert kelly == pytest.approx(0.25, rel=1e-3)  # Should be capped at 25%
        
        # Without cap would be much higher  
        kelly_uncapped = calculate_kelly_fraction(0.95, 0.50, max_fraction=1.0)
        assert kelly_uncapped == pytest.approx(0.45, rel=1e-3)  # Should be 0.45 without cap
    
    def test_calculate_position_value_long_yes(self):
        """Test position value for long YES position."""
        # Bought 10 YES at 40 cents, now at 50 cents
        pnl = calculate_position_value(10, 0.40, 0.50)
        assert pnl == 1.0  # $1 profit (10 * 0.10)
        
        # Price went down
        pnl = calculate_position_value(10, 0.40, 0.30)
        assert pnl == -1.0  # $1 loss
    
    def test_calculate_position_value_long_no(self):
        """Test position value for long NO position."""
        # Bought 10 NO at 60 cents (YES at 40), YES now at 50 cents (NO at 50)
        # This is represented as negative quantity
        pnl = calculate_position_value(-10, 0.40, 0.50)
        assert pnl == -1.0  # $1 loss (YES went up, NO went down)
        
        # YES went down to 30 (NO went up to 70)
        pnl = calculate_position_value(-10, 0.40, 0.30)
        assert pnl == 1.0  # $1 profit
    
    def test_fee_calculation_matches_kalshi_formula(self):
        """Test that fee calculation matches Kalshi's exact formula."""
        test_cases = [
            (0.10, 0.0063),   # 10 cents
            (0.25, 0.0131),   # 25 cents
            (0.40, 0.0168),   # 40 cents
            (0.50, 0.0175),   # 50 cents (maximum)
            (0.60, 0.0168),   # 60 cents
            (0.75, 0.0131),   # 75 cents
            (0.90, 0.0063),   # 90 cents
        ]
        
        for price, expected_fee in test_cases:
            fee = calculate_kalshi_fee(price, 1)
            assert fee == pytest.approx(expected_fee, rel=1e-3)
    
    def test_expected_value_with_real_scenario(self):
        """Test EV with realistic trading scenario."""
        # Scenario: NFL game, we think team has 65% chance, market at 50%
        your_prob = 0.65
        market_price = 0.50
        contracts = 100
        
        ev = calculate_expected_value(your_prob, market_price, contracts)
        
        # With 15% edge and 100 contracts, should be profitable
        assert ev > 0
        
        # Calculate components
        fee = calculate_kalshi_fee(market_price, contracts)
        profit_if_win = (1.0 - market_price) * contracts - fee
        loss_if_lose = market_price * contracts + fee
        
        expected = your_prob * profit_if_win - (1 - your_prob) * loss_if_lose
        assert ev == pytest.approx(expected, rel=1e-4)