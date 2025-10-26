"""
Execution Module for Neural Analysis Stack

Bridges analysis signals with trading execution.
"""

from .auto_executor import AutoExecutor, ExecutionConfig
from .order_manager import OrderManager

__all__ = ["OrderManager", "AutoExecutor", "ExecutionConfig"]
