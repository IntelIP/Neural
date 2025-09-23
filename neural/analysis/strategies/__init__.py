"""
Neural Analysis Strategies

Pre-built trading strategies for Kalshi sports markets.
"""

from .base import Strategy, Signal, SignalType, Position
from .mean_reversion import MeanReversionStrategy, SportsbookArbitrageStrategy
from .momentum import MomentumStrategy, GameMomentumStrategy
from .arbitrage import ArbitrageStrategy, HighSpeedArbitrageStrategy
from .news_based import NewsBasedStrategy, BreakingNewsStrategy

__all__ = [
    # Base classes
    "Strategy",
    "Signal",
    "SignalType",
    "Position",

    # Mean Reversion
    "MeanReversionStrategy",
    "SportsbookArbitrageStrategy",

    # Momentum
    "MomentumStrategy",
    "GameMomentumStrategy",

    # Arbitrage
    "ArbitrageStrategy",
    "HighSpeedArbitrageStrategy",

    # News Based
    "NewsBasedStrategy",
    "BreakingNewsStrategy",
]

# Strategy presets for quick initialization
STRATEGY_PRESETS = {
    "conservative": {
        "class": MeanReversionStrategy,
        "params": {
            "divergence_threshold": 0.08,
            "max_position_size": 0.05,
            "stop_loss": 0.2,
            "min_edge": 0.05
        }
    },
    "momentum": {
        "class": MomentumStrategy,
        "params": {
            "lookback_periods": 10,
            "momentum_threshold": 0.1,
            "use_rsi": True,
            "max_position_size": 0.1
        }
    },
    "arbitrage": {
        "class": ArbitrageStrategy,
        "params": {
            "min_arbitrage_profit": 0.01,
            "max_exposure_per_arb": 0.3,
            "speed_priority": True
        }
    },
    "news": {
        "class": NewsBasedStrategy,
        "params": {
            "sentiment_threshold": 0.65,
            "news_decay_minutes": 30,
            "min_social_volume": 100
        }
    },
    "aggressive": {
        "class": GameMomentumStrategy,
        "params": {
            "event_window": 5,
            "fade_blowouts": True,
            "max_position_size": 0.2,
            "min_edge": 0.02
        }
    },
    "high_frequency": {
        "class": HighSpeedArbitrageStrategy,
        "params": {
            "fixed_size": 100,
            "pre_calculate_size": True,
            "latency_threshold_ms": 50
        }
    }
}


def create_strategy(preset: str, **override_params) -> Strategy:
    """
    Create a strategy from a preset with optional parameter overrides.

    Args:
        preset: Name of preset ('conservative', 'momentum', etc.)
        **override_params: Parameters to override from preset

    Returns:
        Initialized strategy instance

    Example:
        >>> strategy = create_strategy('conservative', initial_capital=5000)
    """
    if preset not in STRATEGY_PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Choose from: {list(STRATEGY_PRESETS.keys())}")

    preset_config = STRATEGY_PRESETS[preset]
    strategy_class = preset_config["class"]
    params = preset_config["params"].copy()

    # Apply overrides
    params.update(override_params)

    return strategy_class(**params)