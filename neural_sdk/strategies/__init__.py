"""
Strategy Development Module

This module provides base classes and utilities for developing trading strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class StrategySignal:
    """Base class for strategy signals."""

    action: str  # "BUY", "SELL", "HOLD"
    symbol: str
    confidence: float
    quantity: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.

    All strategies should inherit from this class and implement
    the generate_signal method.
    """

    def __init__(self, name: str, **config):
        self.name = name
        self.config = config

    @abstractmethod
    def generate_signal(self, market_data: Dict[str, Any]) -> Optional[StrategySignal]:
        """
        Generate a trading signal based on market data.

        Args:
            market_data: Dictionary containing market information

        Returns:
            StrategySignal or None if no action should be taken
        """
        pass
