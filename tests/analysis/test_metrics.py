"""
Unit tests for performance metrics and calculations.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch

from neural.analysis.metrics import (
    PerformanceCalculator,
    PerformanceMetrics,
    MetricType
)


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass."""
    
    def test_create_metrics(self):
        """Test creating performance metrics."""
        metrics = PerformanceMetrics(
            total_return=0.15,
            annual_return=0.12,
            volatility=0.08,
            sharpe_ratio=1.5,
            max_drawdown=0.05,
            win_rate=0.65,
            total_trades=100
        )
        
        assert metrics.total_return == pytest.approx(0.15, rel=1e-3)
        assert metrics.annual_return == pytest.approx(0.12, rel=1e-3)
        assert metrics.volatility == pytest.approx(0.08, rel=1e-3)
        assert metrics.sharpe_ratio == pytest.approx(1.5, rel=1e-3)
        assert metrics.max_drawdown == pytest.approx(0.05, rel=1e-3)
        assert metrics.win_rate == pytest.approx(0.65, rel=1e-3)
        assert metrics.total_trades == 100
    
    def test_metrics_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = PerformanceMetrics(
            total_return=0.10,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            monthly_returns=[0.01, 0.02, -0.005]
        )
        
        metrics_dict = metrics.to_dict()
        
        assert metrics_dict['total_return'] == pytest.approx(0.10, rel=1e-3)
        assert isinstance(metrics_dict['start_date'], str)  # Should be ISO format
        assert isinstance(metrics_dict['end_date'], str)
        assert metrics_dict['monthly_returns'] == [0.01, 0.02, -0.005]


class TestPerformanceCalculator:
    """Test PerformanceCalculator class."""
    
    @pytest.fixture
    def calculator(self):
        return PerformanceCalculator(risk_free_rate=0.02)
    
    @pytest.fixture
    def sample_pnl_series(self):
        """Create sample P&L series for testing."""
        dates = pd.date_range('2024-01-01', periods=252, freq='D')  # One year of daily data
        
        # Generate realistic P&L series with some volatility
        np.random.seed(42)  # For reproducible tests
        daily_returns = np.random.normal(0.0008, 0.02, 252)  # ~20% annual return, 20% volatility
        cumulative_returns = np.cumprod(1 + daily_returns)
        pnl_values = 1000 * cumulative_returns  # Start with $1000
        
        return pd.Series(pnl_values, index=dates)
    
    @pytest.fixture
    def sample_trades_data(self):
        """Create sample trade data for testing."""
        return [
            {'pnl': 50, 'entry_price': 0.40, 'exit_price': 0.50},
            {'pnl': -20, 'entry_price': 0.60, 'exit_price': 0.55},
            {'pnl': 30, 'entry_price': 0.35, 'exit_price': 0.42},
            {'pnl': 80, 'entry_price': 0.45, 'exit_price': 0.65},
            {'pnl': -15, 'entry_price': 0.50, 'exit_price': 0.47},
            {'pnl': 25, 'entry_price': 0.30, 'exit_price': 0.35},
            {'pnl': -10, 'entry_price': 0.70, 'exit_price': 0.68},
        ]
    
    def test_calculate_comprehensive_metrics(self, calculator, sample_pnl_series, sample_trades_data):
        """Test comprehensive metrics calculation."""
        metrics = calculator.calculate_comprehensive_metrics(
            sample_pnl_series,
            sample_trades_data,
            initial_capital=1000
        )
        
        # Basic checks
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.days_traded == 252
        assert metrics.total_trades == 7
        assert metrics.win_rate > 0  # Should have some winning trades
        
        # Return metrics
        assert metrics.total_return != 0
        assert metrics.annual_return != 0
        
        # Risk metrics
        assert metrics.volatility > 0
        assert metrics.annual_volatility > 0
        assert metrics.downside_deviation >= 0
        
        # Risk-adjusted metrics
        assert isinstance(metrics.sharpe_ratio, float)
        assert isinstance(metrics.sortino_ratio, float)
        
        # Drawdown metrics
        assert metrics.max_drawdown >= 0
        assert metrics.max_drawdown <= 1  # Should be percentage
        
        # Trading metrics
        assert 0 <= metrics.win_rate <= 1
        assert metrics.profit_factor >= 0
    
    def test_annualize_return(self, calculator):
        """Test annualized return calculation."""
        # Create 50% return over 6 months (should annualize to ~100%)
        returns = pd.Series([0.002] * 126)  # 126 trading days ≈ 6 months
        
        annual_return = calculator._annualize_return(returns)
        
        assert annual_return > 0.5  # Should be substantial due to compounding
        assert annual_return < 2.0  # Reasonable upper bound
    
    def test_calculate_downside_deviation(self, calculator):
        """Test downside deviation calculation."""
        # Series with positive and negative returns
        returns = pd.Series([0.01, -0.02, 0.015, -0.01, 0.005, -0.025])
        
        downside_dev = calculator._calculate_downside_deviation(returns, threshold=0.0)
        
        assert downside_dev > 0
        # Should only consider negative returns
        negative_returns = returns[returns < 0]
        expected = negative_returns.std()
        assert downside_dev == pytest.approx(expected, rel=1e-3)
    
    def test_calculate_var(self, calculator):
        """Test Value at Risk calculation."""
        # Normal distribution returns
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 1000))
        
        var_95 = calculator._calculate_var(returns, 0.95)
        
        assert var_95 < 0  # VaR should be negative (loss)
        # Should be approximately -1.645 * std for 95% confidence
        expected_var = returns.quantile(0.05)
        assert var_95 == pytest.approx(expected_var, rel=1e-2)
    
    def test_calculate_sharpe_ratio(self, calculator):
        """Test Sharpe ratio calculation."""
        # Positive returns series
        returns = pd.Series([0.001, 0.002, -0.0005, 0.0015, 0.0008] * 50)  # 250 days
        
        sharpe = calculator._calculate_sharpe_ratio(returns)
        
        assert isinstance(sharpe, float)
        assert sharpe > 0  # Should be positive for profitable strategy
        
        # Test with zero volatility (edge case)
        zero_vol_returns = pd.Series([0.001] * 100)  # Constant returns
        sharpe_zero = calculator._calculate_sharpe_ratio(zero_vol_returns)
        assert sharpe_zero == 0  # Should handle zero volatility
    
    def test_calculate_sortino_ratio(self, calculator):
        """Test Sortino ratio calculation."""
        returns = pd.Series([0.01, -0.005, 0.015, -0.002, 0.008])
        
        sortino = calculator._calculate_sortino_ratio(returns)
        
        assert isinstance(sortino, float)
        # Sortino should be higher than Sharpe for same returns (only penalizes downside)
    
    def test_calculate_calmar_ratio(self, calculator):
        """Test Calmar ratio calculation."""
        returns = pd.Series([0.001] * 252)  # Consistent positive returns
        max_drawdown = 0.05  # 5% max drawdown
        
        calmar = calculator._calculate_calmar_ratio(returns, max_drawdown)
        
        assert calmar > 0
        assert isinstance(calmar, float)
        
        # Test with zero drawdown (edge case)
        calmar_zero_dd = calculator._calculate_calmar_ratio(returns, 0)
        assert calmar_zero_dd == 0
    
    def test_calculate_omega_ratio(self, calculator):
        """Test Omega ratio calculation."""
        # Mix of positive and negative returns
        returns = pd.Series([0.02, -0.01, 0.015, -0.005, 0.03, -0.008])
        
        omega = calculator._calculate_omega_ratio(returns, threshold=0.0)
        
        assert omega > 0
        # Should be ratio of gains to losses
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns <= 0].sum())
        expected_omega = gains / losses
        assert omega == pytest.approx(expected_omega, rel=1e-3)
    
    def test_calculate_drawdown_metrics(self, calculator):
        """Test comprehensive drawdown analysis."""
        # Create series with known drawdown pattern
        pnl_values = [1000, 1100, 1050, 900, 850, 950, 1200, 1150, 1000, 1300]
        pnl_series = pd.Series(pnl_values)
        
        dd_metrics = calculator._calculate_drawdown_metrics(pnl_series)
        
        assert 'max_drawdown' in dd_metrics
        assert 'current_drawdown' in dd_metrics
        assert 'max_drawdown_duration' in dd_metrics
        assert 'drawdown_periods' in dd_metrics
        
        assert dd_metrics['max_drawdown'] > 0
        assert dd_metrics['max_drawdown'] <= 1  # Should be percentage
        assert dd_metrics['drawdown_periods'] > 0
    
    def test_calculate_trade_metrics(self, calculator, sample_trades_data):
        """Test trading-specific metrics calculation."""
        trade_metrics = calculator._calculate_trade_metrics(sample_trades_data)
        
        assert trade_metrics['total_trades'] == 7
        assert 0 <= trade_metrics['win_rate'] <= 1
        assert trade_metrics['avg_win'] > 0  # Should have positive wins
        assert trade_metrics['avg_loss'] > 0  # Should have positive losses (absolute)
        assert trade_metrics['profit_factor'] > 0
        assert 0 <= trade_metrics['kelly_criterion'] <= 0.25  # Capped at 25%
        
        # Calculate manually to verify
        pnls = [trade['pnl'] for trade in sample_trades_data]
        wins = [p for p in pnls if p > 0]
        losses = [abs(p) for p in pnls if p < 0]
        
        expected_win_rate = len(wins) / len(pnls)
        assert trade_metrics['win_rate'] == pytest.approx(expected_win_rate, rel=1e-3)
    
    def test_calculate_consistency_metrics(self, calculator):
        """Test consistency and distribution metrics."""
        # Consistent positive returns
        consistent_returns = pd.Series([0.001, 0.0012, 0.0008, 0.0011, 0.0009] * 20)
        
        consistency = calculator._calculate_consistency_metrics(consistent_returns)
        
        assert 'hit_ratio' in consistency
        assert 'consistency_score' in consistency
        assert 'tail_ratio' in consistency
        
        assert consistency['hit_ratio'] == 1.0  # All positive returns
        assert consistency['consistency_score'] > 0.5  # Should be consistent
        
        # Volatile returns
        volatile_returns = pd.Series([-0.05, 0.08, -0.03, 0.12, -0.02])
        volatile_consistency = calculator._calculate_consistency_metrics(volatile_returns)
        
        assert volatile_consistency['consistency_score'] < consistency['consistency_score']
    
    def test_compare_strategies(self, calculator):
        """Test strategy comparison functionality."""
        # Create mock strategies with different performance
        strategy1 = PerformanceMetrics(
            annual_return=0.15,
            sharpe_ratio=1.2,
            max_drawdown=0.08,
            win_rate=0.65,
            profit_factor=1.8,
            total_trades=100
        )
        
        strategy2 = PerformanceMetrics(
            annual_return=0.10,
            sharpe_ratio=1.5,
            max_drawdown=0.05,
            win_rate=0.70,
            profit_factor=2.2,
            total_trades=80
        )
        
        strategy_metrics = {
            'Strategy A': strategy1,
            'Strategy B': strategy2
        }
        
        comparison = calculator.compare_strategies(strategy_metrics)
        
        assert 'overall' in comparison
        assert 'annual_return' in comparison
        assert 'sharpe_ratio' in comparison
        assert 'max_drawdown' in comparison
        
        # Check rankings exist
        assert len(comparison['overall']['ranking']) == 2
        assert comparison['overall']['best'] in ['Strategy A', 'Strategy B']
        
        # Max drawdown ranking should be reversed (lower is better)
        dd_ranking = comparison['max_drawdown']['ranking']
        assert dd_ranking[0] == 'Strategy B'  # Lower drawdown should be ranked first
    
    def test_generate_performance_report(self, calculator):
        """Test performance report generation."""
        metrics = PerformanceMetrics(
            total_return=0.25,
            annual_return=0.18,
            annual_volatility=0.15,
            sharpe_ratio=1.2,
            max_drawdown=0.08,
            total_trades=150,
            win_rate=0.62,
            avg_win=45.5,
            avg_loss=28.2,
            profit_factor=1.6
        )
        
        report = calculator.generate_performance_report(metrics, "Test Strategy")
        
        assert "Test Strategy Performance Report" in report
        assert "25.00%" in report  # Total return
        assert "18.00%" in report  # Annual return
        assert "1.20" in report    # Sharpe ratio
        assert "8.00%" in report   # Max drawdown
        assert "150" in report     # Total trades
        assert "62.00%" in report  # Win rate
        assert "$45.50" in report  # Average win
        assert "$28.20" in report  # Average loss
    
    def test_edge_cases_empty_data(self, calculator):
        """Test handling of edge cases with empty/minimal data."""
        # Empty series
        empty_series = pd.Series([], dtype=float)
        metrics = calculator.calculate_comprehensive_metrics(empty_series)
        
        # Should return default metrics without crashing
        assert isinstance(metrics, PerformanceMetrics)
        
        # Single data point
        single_point = pd.Series([1000])
        metrics_single = calculator.calculate_comprehensive_metrics(single_point)
        assert isinstance(metrics_single, PerformanceMetrics)
    
    def test_zero_variance_returns(self, calculator):
        """Test handling of zero variance (constant) returns."""
        # Constant returns (no variance)
        constant_pnl = pd.Series([1000] * 100)
        
        metrics = calculator.calculate_comprehensive_metrics(constant_pnl)
        
        # Should handle gracefully
        assert metrics.volatility == 0
        assert metrics.annual_volatility == 0
        # Sharpe ratio should be 0 when volatility is 0
    
    def test_extreme_values(self, calculator):
        """Test handling of extreme values."""
        # Series with extreme drawdown
        extreme_pnl = [1000, 1100, 200, 220, 1200]  # ~80% drawdown
        extreme_series = pd.Series(extreme_pnl)
        
        metrics = calculator.calculate_comprehensive_metrics(extreme_series)
        
        assert 0 < metrics.max_drawdown < 1
        assert metrics.max_drawdown > 0.7  # Should capture large drawdown
    
    def test_negative_total_returns(self, calculator):
        """Test handling of strategies with negative total returns."""
        # Declining P&L series
        declining_pnl = pd.Series([1000, 950, 900, 850, 800])
        
        metrics = calculator.calculate_comprehensive_metrics(declining_pnl)
        
        assert metrics.total_return < 0
        assert metrics.annual_return < 0
        # Should still calculate other metrics meaningfully
