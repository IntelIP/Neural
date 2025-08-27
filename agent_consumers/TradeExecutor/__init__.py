"""
Trade Executor Agent
Executes trades on Kalshi markets
"""

from .trade_executor import TradeExecutorAgent

# Create singleton instance
trade_executor_agent = TradeExecutorAgent()

__all__ = ['trade_executor_agent', 'TradeExecutorAgent']