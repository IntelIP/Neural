"""
Unit tests for Risk Management System

Tests cover risk rule validation, limit checking, and violation handling.
"""

import pytest
from unittest.mock import Mock
from datetime import datetime, timezone

from neural.trading.risk_manager import (
    TradingRiskManager, RiskViolationType, RiskViolation, RiskRule
)
from neural.strategy.base import Signal, SignalType


class TestRiskRule:
    """Test RiskRule data class."""
    
    def test_risk_rule_creation(self):
        """Test creating a RiskRule."""
        rule = RiskRule(
            rule_id="test_rule",
            name="Test Rule",
            limit_value=0.05,
            enabled=True,
            violation_type=RiskViolationType.POSITION_SIZE
        )
        
        assert rule.rule_id == "test_rule"
        assert rule.name == "Test Rule"
        assert rule.limit_value == 0.05
        assert rule.enabled is True
        assert rule.violation_type == RiskViolationType.POSITION_SIZE


class TestRiskViolation:
    """Test RiskViolation data class."""
    
    def test_risk_violation_creation(self):
        """Test creating a RiskViolation."""
        violation = RiskViolation(
            violation_type=RiskViolationType.POSITION_SIZE,
            limit_id="max_position",
            current_value=0.12,
            limit_value=0.10,
            message="Position size exceeds limit",
            severity="error"
        )
        
        assert violation.violation_type == RiskViolationType.POSITION_SIZE
        assert violation.current_value == 0.12
        assert violation.limit_value == 0.10
        assert violation.severity == "error"


class TestTradingRiskManager:
    """Test TradingRiskManager class."""
    
    @pytest.fixture
    def risk_manager(self):
        """Create TradingRiskManager instance."""
        return TradingRiskManager(initial_capital=100000.0)
    
    def test_initialization(self, risk_manager):
        """Test risk manager initialization."""
        assert risk_manager.initial_capital == 100000.0
        assert len(risk_manager.rules) > 0
        
        # Check default rules exist
        assert "max_position" in risk_manager.rules
        assert "daily_loss" in risk_manager.rules
        assert "total_exposure" in risk_manager.rules
    
    def test_default_rules_setup(self, risk_manager):
        """Test default risk rules are properly configured."""
        max_position_rule = risk_manager.rules["max_position"]
        assert max_position_rule.limit_value == 0.10  # 10%
        assert max_position_rule.violation_type == RiskViolationType.POSITION_SIZE
        
        daily_loss_rule = risk_manager.rules["daily_loss"]
        assert daily_loss_rule.limit_value == 0.05  # 5%
        assert daily_loss_rule.violation_type == RiskViolationType.DAILY_LOSS
        
        total_exposure_rule = risk_manager.rules["total_exposure"]
        assert total_exposure_rule.limit_value == 0.50  # 50%
        assert total_exposure_rule.violation_type == RiskViolationType.TOTAL_EXPOSURE
    
    def test_check_trade_allowed_success(self, risk_manager):
        """Test trade allowed with valid signal."""
        signal = Signal(
            signal_type=SignalType.BUY_YES,
            market_id="TEST-MARKET",
            timestamp=datetime.now(timezone.utc),
            confidence=0.8,
            edge=0.05,
            recommended_size=0.08  # 8% - below 10% limit
        )
        
        portfolio_state = {
            "daily_pnl": 0.0,  # No losses
            "total_exposure": 0.3  # Below 50% limit
        }
        
        allowed, reasons = risk_manager.check_trade_allowed(signal, portfolio_state)
        
        assert allowed is True
        assert len(reasons) == 0
    
    def test_check_trade_allowed_position_too_large(self, risk_manager):
        """Test trade rejected due to position size."""
        signal = Signal(
            signal_type=SignalType.BUY_YES,
            market_id="TEST-MARKET",
            timestamp=datetime.now(timezone.utc),
            confidence=0.8,
            edge=0.05,
            recommended_size=0.15  # 15% - above 10% limit
        )
        
        portfolio_state = {
            "daily_pnl": 0.0
        }
        
        allowed, reasons = risk_manager.check_trade_allowed(signal, portfolio_state)
        
        assert allowed is False
        assert len(reasons) == 1
        assert "Position size too large" in reasons[0]
    
    def test_check_trade_allowed_daily_loss_exceeded(self, risk_manager):
        """Test trade rejected due to daily loss limit."""
        signal = Signal(
            signal_type=SignalType.BUY_YES,
            market_id="TEST-MARKET",
            timestamp=datetime.now(timezone.utc),
            confidence=0.8,
            edge=0.05,
            recommended_size=0.08
        )
        
        # Daily loss of -6% (exceeds -5% limit)
        portfolio_state = {
            "daily_pnl": -6000.0  # -6% of 100k
        }
        
        allowed, reasons = risk_manager.check_trade_allowed(signal, portfolio_state)
        
        assert allowed is False
        assert len(reasons) == 1
        assert "Daily loss limit exceeded" in reasons[0]
    
    def test_check_trade_allowed_multiple_violations(self, risk_manager):
        """Test trade rejected due to multiple violations."""
        signal = Signal(
            signal_type=SignalType.BUY_YES,
            market_id="TEST-MARKET",
            timestamp=datetime.now(timezone.utc),
            confidence=0.8,
            edge=0.05,
            recommended_size=0.15  # Too large
        )
        
        portfolio_state = {
            "daily_pnl": -6000.0  # Daily loss exceeded
        }
        
        allowed, reasons = risk_manager.check_trade_allowed(signal, portfolio_state)
        
        assert allowed is False
        assert len(reasons) == 2  # Multiple violations
    
    def test_check_all_limits_no_violations(self, risk_manager):
        """Test checking all limits with no violations."""
        portfolio_state = {
            "daily_pnl": 1000.0,  # Positive
            "total_exposure": 0.3,  # Below limit
            "max_position_size": 0.08  # Below limit
        }
        
        passed, violations = risk_manager.check_all_limits(portfolio_state)
        
        assert passed is True
        assert len(violations) == 0
    
    def test_custom_configuration(self):
        """Test risk manager with custom configuration."""
        config = Mock()
        config.max_position_size = 0.08
        config.daily_loss_limit = 0.03
        
        risk_manager = TradingRiskManager(
            initial_capital=50000.0,
            config=config
        )
        
        assert risk_manager.initial_capital == 50000.0
        assert risk_manager.config == config
        # Rules should still be set up with defaults
        assert len(risk_manager.rules) > 0


class TestRiskManagerIntegration:
    """Integration tests for risk management system."""
    
    def test_realistic_trading_scenarios(self):
        """Test realistic trading scenarios."""
        risk_manager = TradingRiskManager(initial_capital=100000.0)
        
        # Scenario 1: Conservative trade - should pass
        conservative_signal = Signal(
            signal_type=SignalType.BUY_YES,
            market_id="CONSERVATIVE-TRADE",
            timestamp=datetime.now(timezone.utc),
            confidence=0.75,
            edge=0.04,
            recommended_size=0.03  # 3%
        )
        
        clean_portfolio = {
            "daily_pnl": 0.0,
            "total_exposure": 0.15
        }
        
        allowed, reasons = risk_manager.check_trade_allowed(conservative_signal, clean_portfolio)
        assert allowed is True
        
        # Scenario 2: Aggressive trade after losses - should fail
        aggressive_signal = Signal(
            signal_type=SignalType.BUY_YES,
            market_id="AGGRESSIVE-TRADE",
            timestamp=datetime.now(timezone.utc),
            confidence=0.65,
            edge=0.08,
            recommended_size=0.12  # 12% - too large
        )
        
        distressed_portfolio = {
            "daily_pnl": -4000.0,  # -4%
            "total_exposure": 0.45
        }
        
        allowed, reasons = risk_manager.check_trade_allowed(aggressive_signal, distressed_portfolio)
        assert allowed is False
        assert len(reasons) > 0
    
    def test_risk_limit_edge_cases(self):
        """Test edge cases in risk limit checking."""
        risk_manager = TradingRiskManager(initial_capital=100000.0)
        
        # Edge case: Exactly at the limit
        edge_signal = Signal(
            signal_type=SignalType.BUY_YES,
            market_id="EDGE-CASE",
            timestamp=datetime.now(timezone.utc),
            confidence=0.75,
            edge=0.04,
            recommended_size=0.10  # Exactly 10%
        )
        
        edge_portfolio = {
            "daily_pnl": -5000.0  # Exactly -5%
        }
        
        allowed, reasons = risk_manager.check_trade_allowed(edge_signal, edge_portfolio)
        # Should be allowed (not exceeding limits)
        assert allowed is True
        
        # Slightly over the limit
        over_limit_signal = Signal(
            signal_type=SignalType.BUY_YES,
            market_id="OVER-LIMIT",
            timestamp=datetime.now(timezone.utc),
            confidence=0.75,
            edge=0.04,
            recommended_size=0.101  # Just over 10%
        )
        
        allowed, reasons = risk_manager.check_trade_allowed(over_limit_signal, edge_portfolio)
        assert allowed is False
