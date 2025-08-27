"""
Risk Manager Agent
Manages portfolio risk and monitors positions
"""

from .risk_manager import RiskManagerAgent

# Create singleton instance
risk_manager_agent = RiskManagerAgent()

__all__ = ['risk_manager_agent', 'RiskManagerAgent']