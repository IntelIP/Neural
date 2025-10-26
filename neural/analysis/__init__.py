"""
Neural SDK Analysis Stack

A comprehensive framework for building, testing, and executing trading strategies
with seamless integration to Kalshi markets and ESPN data.
"""

from . import backtesting, execution, risk, sentiment, strategies

# Direct imports for commonly used classes
from .backtesting.engine import Backtester
from .execution.order_manager import OrderManager
from .risk.position_sizing import edge_proportional, fixed_percentage, kelly_criterion
from .strategies.base import Signal, Strategy

__all__ = [
    "backtesting",
    "execution",
    "risk",
    "sentiment",
    "strategies",
    "Strategy",
    "Signal",
    "Backtester",
    "OrderManager",
    "kelly_criterion",
    "fixed_percentage",
    "edge_proportional",
]
