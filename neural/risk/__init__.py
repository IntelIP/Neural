"""
Neural SDK Risk Management Framework

This module provides comprehensive risk management capabilities for 
trading strategies, including:

- Advanced position sizing algorithms (Kelly, fixed, volatility-based)
- Portfolio optimization with correlation analysis
- Risk limit enforcement and constraint management
- Real-time risk monitoring and alerting
- Value-at-Risk (VaR) calculations
- Drawdown monitoring and circuit breakers

The risk management framework integrates seamlessly with the strategy
and backtesting frameworks to ensure safe and optimized trading.
"""

from .position_sizing import (
    PositionSizer, KellySizer, FixedSizer, VolatilitySizer,
    PositionSizingMethod, PositionSizeResult
)
from .portfolio import (
    PortfolioOptimizer, PortfolioAllocation, AllocationMethod,
    CorrelationAnalyzer, RiskParityOptimizer
)
from .limits import (
    RiskLimitManager, RiskLimit, LimitType, LimitViolation,
    PositionLimit, ExposureLimit, LossLimit
)
from .monitor import (
    RiskMonitor, RiskAlert, AlertSeverity, RiskMetrics,
    DrawdownMonitor, VaRMonitor
)

__all__ = [
    # Position sizing
    'PositionSizer',
    'KellySizer',
    'FixedSizer', 
    'VolatilitySizer',
    'PositionSizingMethod',
    'PositionSizeResult',
    
    # Portfolio optimization
    'PortfolioOptimizer',
    'PortfolioAllocation',
    'AllocationMethod',
    'CorrelationAnalyzer',
    'RiskParityOptimizer',
    
    # Risk limits
    'RiskLimitManager',
    'RiskLimit',
    'LimitType',
    'LimitViolation',
    'PositionLimit',
    'ExposureLimit', 
    'LossLimit',
    
    # Risk monitoring
    'RiskMonitor',
    'RiskAlert',
    'AlertSeverity',
    'RiskMetrics',
    'DrawdownMonitor',
    'VaRMonitor'
]
