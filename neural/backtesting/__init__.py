"""
Neural SDK Backtesting Framework

This module provides comprehensive backtesting capabilities for validating
trading strategies with historical data, including:

- Event-driven backtesting engine
- Realistic market simulation
- Walk-forward validation
- Parameter optimization
- Performance analytics

The backtesting framework integrates seamlessly with the strategy framework
to provide robust validation and optimization capabilities.
"""

from .engine import BacktestEngine, BacktestResult, BacktestConfig
from .simulator import MarketSimulator, FillSimulation, SlippageModel
from .validator import BacktestValidator, ValidationConfig, ValidationResult, ValidationMethod
from .optimizer import ParameterOptimizer, OptimizationConfig, OptimizationResult, OptimizationMethod

__all__ = [
    'BacktestEngine',
    'BacktestResult', 
    'BacktestConfig',
    'MarketSimulator',
    'FillSimulation',
    'SlippageModel',
    'BacktestValidator',
    'ValidationConfig',
    'ValidationResult', 
    'ValidationMethod',
    'ParameterOptimizer',
    'OptimizationConfig',
    'OptimizationResult',
    'OptimizationMethod'
]
