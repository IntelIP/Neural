"""
Unit tests for Risk Limits and Constraint Enforcement.

Tests the risk limit management system including position limits,
loss limits, exposure limits, and violation handling.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from neural.risk.limits import (
    RiskLimitManager, RiskLimit, LimitType, LimitAction, LimitSeverity,
    LimitViolation, PositionLimit, ExposureLimit, LossLimit
)
from neural.strategy.base import Signal, SignalType


class TestRiskLimitManager:
    """Test RiskLimitManager functionality."""
    
    @pytest.fixture
    def limit_manager(self):
        """Create RiskLimitManager for testing."""
        return RiskLimitManager(initial_capital=50000.0, enable_enforcement=True)
    
    @pytest.fixture
    def sample_portfolio(self):
        """Create sample portfolio state."""
        return {
            'total_capital': 50000.0,
            'positions': {
                'MARKET_A': {'market_value': 5000.0},  # 10% position
                'MARKET_B': {'market_value': 3000.0},  # 6% position  
                'MARKET_C': {'market_value': 2000.0},  # 4% position
            },
            'daily_pnl': 500.0,  # +$500 today
            'peak_capital': 52000.0  # Was higher before
        }
    
    def test_limit_manager_initialization(self, limit_manager):
        """Test RiskLimitManager initialization."""
        assert limit_manager.initial_capital == 50000.0
        assert limit_manager.enable_enforcement is True
        assert limit_manager.trading_halted is False
        assert len(limit_manager.limits) == 5  # Default limits
        assert limit_manager.checks_performed == 0
        assert limit_manager.violations_detected == 0
    
    def test_add_custom_limit(self, limit_manager):
        """Test adding custom risk limits."""
        custom_limit = PositionLimit(
            limit_id="custom_position_limit",
            max_position_pct=0.08  # 8% max
        )
        
        initial_count = len(limit_manager.limits)
        limit_manager.add_limit(custom_limit)
        
        assert len(limit_manager.limits) == initial_count + 1
        assert "custom_position_limit" in limit_manager.limits
        assert limit_manager.limits["custom_position_limit"].limit_value == 0.08
    
    def test_enable_disable_limit(self, limit_manager):
        """Test enabling/disabling limits."""
        limit_id = "max_position_size"  # Default limit
        
        # Should be enabled by default
        assert limit_manager.limits[limit_id].enabled is True
        
        # Disable it
        success = limit_manager.enable_limit(limit_id, False)
        assert success is True
        assert limit_manager.limits[limit_id].enabled is False
        
        # Re-enable it
        success = limit_manager.enable_limit(limit_id, True)
        assert success is True
        assert limit_manager.limits[limit_id].enabled is True
        
        # Try non-existent limit
        success = limit_manager.enable_limit("non_existent", True)
        assert success is False
    
    def test_remove_limit(self, limit_manager):
        """Test removing limits."""
        limit_id = "max_position_size"
        
        # Should exist initially
        assert limit_id in limit_manager.limits
        
        # Remove it
        success = limit_manager.remove_limit(limit_id)
        assert success is True
        assert limit_id not in limit_manager.limits
        
        # Try to remove again (should fail)
        success = limit_manager.remove_limit(limit_id)
        assert success is False
    
    def test_check_all_limits_no_violations(self, limit_manager, sample_portfolio):
        """Test checking limits with no violations."""
        all_ok, violations = limit_manager.check_all_limits(sample_portfolio)
        
        assert all_ok is True
        assert len(violations) == 0
        assert limit_manager.checks_performed == 1
        assert limit_manager.violations_detected == 0
    
    def test_check_position_size_violation(self, limit_manager):
        """Test position size limit violation."""
        # Create portfolio with oversized position
        portfolio_with_violation = {
            'total_capital': 50000.0,
            'positions': {
                'HUGE_POSITION': {'market_value': 12000.0},  # 24% - exceeds 10% limit
                'NORMAL_POSITION': {'market_value': 2000.0}
            },
            'daily_pnl': 0.0,
            'peak_capital': 50000.0
        }
        
        all_ok, violations = limit_manager.check_all_limits(portfolio_with_violation)
        
        assert all_ok is False
        assert len(violations) > 0
        
        # Find the position size violation
        position_violations = [v for v in violations if v.limit_type == LimitType.POSITION_SIZE]
        assert len(position_violations) > 0
        
        violation = position_violations[0]
        assert violation.current_value > violation.limit_value
        assert 'HUGE_POSITION' in violation.affected_positions
        assert violation.severity in [LimitSeverity.WARNING, LimitSeverity.CRITICAL]
    
    def test_check_daily_loss_violation(self, limit_manager):
        """Test daily loss limit violation."""
        # Create portfolio with large daily loss
        portfolio_with_loss = {
            'total_capital': 50000.0,
            'positions': {
                'POSITION_A': {'market_value': 5000.0}
            },
            'daily_pnl': -3000.0,  # -$3000 (6% loss) - exceeds 5% limit
            'peak_capital': 50000.0
        }
        
        all_ok, violations = limit_manager.check_all_limits(portfolio_with_loss)
        
        assert all_ok is False
        
        # Find daily loss violation
        loss_violations = [v for v in violations if v.limit_type == LimitType.DAILY_LOSS]
        assert len(loss_violations) > 0
        
        violation = loss_violations[0]
        assert violation.current_value > violation.limit_value
        assert violation.severity == LimitSeverity.CRITICAL
    
    def test_check_total_loss_violation(self, limit_manager):
        """Test total loss (drawdown) limit violation."""
        # Create portfolio with large drawdown
        portfolio_with_drawdown = {
            'total_capital': 35000.0,  # Down from 50k to 35k = 30% drawdown
            'positions': {
                'POSITION_A': {'market_value': 5000.0}
            },
            'daily_pnl': -500.0,
            'peak_capital': 50000.0
        }
        
        all_ok, violations = limit_manager.check_all_limits(portfolio_with_drawdown)
        
        assert all_ok is False
        
        # Find drawdown violation (exceeds 20% limit)
        drawdown_violations = [v for v in violations if v.limit_type == LimitType.TOTAL_LOSS]
        assert len(drawdown_violations) > 0
        
        violation = drawdown_violations[0]
        assert violation.current_value > 0.20  # Should exceed 20% limit
        assert violation.severity == LimitSeverity.EMERGENCY
    
    def test_check_trade_allowed(self, limit_manager, sample_portfolio):
        """Test trade approval checking."""
        # Normal trade should be allowed
        normal_signal = Signal(
            signal_type=SignalType.BUY_YES,
            market_id="NEW_MARKET",
            timestamp=datetime.now(),
            confidence=0.75,
            edge=0.05,
            expected_value=25.0,
            recommended_size=0.05,
            max_contracts=100,
            metadata={'recommended_contracts': 100}
        )
        
        allowed, reasons = limit_manager.check_trade_allowed(normal_signal, sample_portfolio)
        
        assert allowed is True
        assert len(reasons) == 0
    
    def test_check_trade_rejected(self, limit_manager):
        """Test trade rejection due to limits."""
        # Portfolio that's already at position limits
        full_portfolio = {
            'total_capital': 10000.0,
            'positions': {f'POSITION_{i}': {'market_value': 400} for i in range(25)}  # 25 positions
        }
        
        # Try to add another position (would exceed position count limit of 20)
        new_signal = Signal(
            signal_type=SignalType.BUY_YES,
            market_id="NEW_POSITION",
            timestamp=datetime.now(),
            confidence=0.80,
            edge=0.06,
            expected_value=30.0,
            recommended_size=0.05,
            max_contracts=50,
            metadata={'recommended_contracts': 50}
        )
        
        allowed, reasons = limit_manager.check_trade_allowed(new_signal, full_portfolio)
        
        assert allowed is False
        assert len(reasons) > 0
        assert any('position' in reason.lower() for reason in reasons)
    
    def test_trading_halt_enforcement(self, limit_manager):
        """Test trading halt functionality."""
        # Simulate violation that triggers halt
        portfolio_emergency = {
            'total_capital': 30000.0,  # 40% drawdown from 50k
            'positions': {},
            'daily_pnl': -5000.0,  # Additional large daily loss
            'peak_capital': 50000.0
        }
        
        all_ok, violations = limit_manager.check_all_limits(portfolio_emergency)
        
        # Should trigger trading halt due to excessive drawdown
        assert limit_manager.trading_halted is True
        assert limit_manager.halt_reason != ""
        
        # Any subsequent trade should be rejected
        any_signal = Signal(
            signal_type=SignalType.BUY_YES,
            market_id="ANY_MARKET",
            timestamp=datetime.now(),
            confidence=0.90,
            edge=0.08,
            expected_value=40.0,
            recommended_size=0.03,
            max_contracts=25
        )
        
        allowed, reasons = limit_manager.check_trade_allowed(any_signal, {})
        
        assert allowed is False
        assert any('halt' in reason.lower() for reason in reasons)
    
    def test_reset_trading_halt(self, limit_manager):
        """Test resetting trading halt."""
        # First trigger a halt
        limit_manager.trading_halted = True
        limit_manager.halt_reason = "Test halt"
        
        # Reset it
        limit_manager.reset_trading_halt("Manual test reset")
        
        assert limit_manager.trading_halted is False
        assert limit_manager.halt_reason == ""
    
    def test_get_limit_status(self, limit_manager, sample_portfolio):
        """Test getting limit status summary."""
        # Run a few checks to populate statistics
        limit_manager.check_all_limits(sample_portfolio)
        limit_manager.check_all_limits(sample_portfolio)
        
        status = limit_manager.get_limit_status()
        
        assert 'trading_halted' in status
        assert 'total_limits' in status
        assert 'enabled_limits' in status
        assert 'checks_performed' in status
        assert 'violations_detected' in status
        assert 'recent_violations' in status
        
        assert status['trading_halted'] is False
        assert status['total_limits'] == len(limit_manager.limits)
        assert status['checks_performed'] == 2
    
    def test_generate_risk_report(self, limit_manager):
        """Test risk report generation."""
        report = limit_manager.generate_risk_report()
        
        assert isinstance(report, str)
        assert "RISK LIMITS STATUS REPORT" in report
        assert "System Status:" in report
        assert "Limits Overview:" in report
        assert "ACTIVE LIMITS:" in report
        
        # Should mention some default limits
        assert "max_position_size" in report or "position" in report.lower()


class TestRiskLimitTypes:
    """Test specific risk limit types."""
    
    def test_position_limit_creation(self):
        """Test PositionLimit creation."""
        limit = PositionLimit(
            limit_id="test_position",
            max_position_pct=0.15,
            action=LimitAction.REDUCE_POSITION
        )
        
        assert limit.limit_id == "test_position"
        assert limit.limit_type == LimitType.POSITION_SIZE
        assert limit.limit_value == 0.15
        assert limit.action == LimitAction.REDUCE_POSITION
        assert limit.enabled is True
        assert limit.warning_threshold == 0.15 * 0.8  # 80% of limit
    
    def test_exposure_limit_creation(self):
        """Test ExposureLimit creation."""
        limit = ExposureLimit(
            limit_id="sport_exposure",
            category="sport",
            max_exposure_pct=0.40,
            action=LimitAction.WARNING
        )
        
        assert limit.limit_id == "sport_exposure"
        assert limit.limit_type == LimitType.EXPOSURE
        assert limit.limit_value == 0.40
        assert limit.category == "sport"
        assert limit.action == LimitAction.WARNING
    
    def test_loss_limit_creation(self):
        """Test LossLimit creation."""
        daily_limit = LossLimit(
            limit_id="daily_loss",
            max_loss_pct=0.03,
            time_period="daily",
            action=LimitAction.HALT_TRADING
        )
        
        assert daily_limit.limit_id == "daily_loss"
        assert daily_limit.limit_type == LimitType.DAILY_LOSS
        assert daily_limit.limit_value == 0.03
        assert daily_limit.time_period == "daily"
        assert daily_limit.action == LimitAction.HALT_TRADING
        
        total_limit = LossLimit(
            limit_id="max_drawdown",
            max_loss_pct=0.25,
            time_period="total"
        )
        
        assert total_limit.limit_type == LimitType.TOTAL_LOSS


class TestLimitViolation:
    """Test LimitViolation data structure."""
    
    def test_limit_violation_creation(self):
        """Test LimitViolation creation."""
        violation = LimitViolation(
            limit_id="test_violation",
            limit_type=LimitType.POSITION_SIZE,
            current_value=0.15,
            limit_value=0.10,
            severity=LimitSeverity.WARNING,
            action_taken=LimitAction.REDUCE_POSITION,
            timestamp=datetime.now(),
            affected_positions=["MARKET_A", "MARKET_B"],
            message="Test violation message"
        )
        
        assert violation.limit_id == "test_violation"
        assert violation.limit_type == LimitType.POSITION_SIZE
        assert violation.current_value == 0.15
        assert violation.limit_value == 0.10
        assert violation.severity == LimitSeverity.WARNING
        assert violation.action_taken == LimitAction.REDUCE_POSITION
        assert violation.affected_positions == ["MARKET_A", "MARKET_B"]
        assert violation.message == "Test violation message"
        assert isinstance(violation.timestamp, datetime)
