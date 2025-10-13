"""
Neural SDK Analysis Stack

A comprehensive framework for building, testing, and executing trading strategies
with seamless integration to Kalshi markets and ESPN data.
"""

from .backtesting.engine import Backtester
from .execution.order_manager import OrderManager
from .risk.position_sizing import edge_proportional, fixed_percentage, kelly_criterion
from .strategies.base import Position, Signal, Strategy

__all__ = [
    "Strategy",
    "Signal",
    "Position",
    "Backtester",
    "OrderManager",
    "kelly_criterion",
    "fixed_percentage",
    "edge_proportional",
]
