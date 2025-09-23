"""
Strategy Library - Educational Examples

This library contains basic, educational strategy implementations that showcase
the Neural SDK's capabilities. These are designed to be:

1. Educational - Easy to understand and learn from
2. Demonstrative - Show framework features without revealing proprietary algorithms  
3. Modifiable - Templates for building custom strategies
4. Non-competitive - Basic implementations that protect our edge

For production trading, build sophisticated strategies using these as templates
while keeping your alpha-generating algorithms proprietary.
"""

from .mean_reversion import BasicMeanReversionStrategy
from .volume_anomaly import VolumeAnomalyStrategy  
from .simple_arbitrage import SimpleArbitrageStrategy
from .line_movement import LineMovementStrategy
from .news_reaction import NewsReactionFramework

__all__ = [
    'BasicMeanReversionStrategy',
    'VolumeAnomalyStrategy', 
    'SimpleArbitrageStrategy',
    'LineMovementStrategy',
    'NewsReactionFramework'
]
