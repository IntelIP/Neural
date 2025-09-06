"""
Base classes for sports prediction strategy templates.

Provides common functionality for sports-specific trading strategies
including sentiment integration, weather data, and game-specific factors.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime

class SportsPredictionTemplate(ABC):
    """
    Base class for sports prediction market templates.

    Provides common functionality for:
    - Market data processing
    - Sentiment integration
    - Risk management
    - Position sizing
    """

    def __init__(self,
                 sport: str = 'nfl',
                 markets: Optional[List[str]] = None,
                 risk_management: Optional[Dict] = None):

        self.sport = sport
        self.markets = markets or self._get_default_markets()
        self.risk_management = risk_management or self._get_default_risk()

        # Template metadata
        self.name = self.__class__.__name__
        self.description = self._get_description()
        self.parameters = self._get_parameters()

    @abstractmethod
    def generate_signals(self, market_data: Dict, **kwargs) -> List[Dict]:
        """Generate trading signals from market data."""
        pass

    @abstractmethod
    def validate_parameters(self, params: Dict) -> bool:
        """Validate template parameters."""
        pass

    def _get_default_markets(self) -> List[str]:
        """Get default markets for this sport."""
        if self.sport == 'nfl':
            return ['KXNFL*']  # All NFL markets
        elif self.sport == 'cfb':
            return ['KXCFB*']  # All College Football
        elif self.sport == 'soccer':
            return ['KXSOCCER*']  # All Soccer
        return []

    def _get_default_risk(self) -> Dict:
        """Get default risk management settings."""
        return {
            'max_position_size': 0.05,
            'max_positions': 10,
            'stop_loss': 0.10,
            'take_profit': 0.20
        }

    def _get_description(self) -> str:
        """Get template description."""
        return f"Sports prediction template for {self.sport}"

    def _get_parameters(self) -> Dict:
        """Get template parameters with defaults."""
        return {}

    def calculate_position_size(self, confidence: float, market_volatility: float) -> float:
        """Calculate position size based on confidence and volatility."""
        base_size = self.risk_management['max_position_size']

        # Adjust for confidence
        confidence_multiplier = min(confidence * 2, 1.5)

        # Adjust for volatility (higher volatility = smaller size)
        volatility_adjustment = 1 / (1 + market_volatility)

        return base_size * confidence_multiplier * volatility_adjustment

    def apply_risk_management(self, signals: List[Dict]) -> List[Dict]:
        """Apply risk management filters to signals."""
        filtered_signals = []
        current_positions = 0  # This would come from portfolio

        for signal in signals:
            # Check position limits
            if current_positions >= self.risk_management['max_positions']:
                continue

            # Check position size limits
            if signal.get('size', 0) > self.risk_management['max_position_size']:
                signal['size'] = self.risk_management['max_position_size']

            filtered_signals.append(signal)
            current_positions += 1

        return filtered_signals