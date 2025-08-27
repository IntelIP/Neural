"""
Market Engineer Agent
Analyzes sentiment and identifies opportunities
"""

from .market_engineer import MarketEngineerAgent

# Create singleton instance
market_engineer_agent = MarketEngineerAgent()

__all__ = ['market_engineer_agent', 'MarketEngineerAgent']