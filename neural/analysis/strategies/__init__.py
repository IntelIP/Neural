"""Neural Analysis strategies with lazy imports for optional strategy stacks."""

from __future__ import annotations

import importlib
from typing import Any

_ATTRIBUTE_EXPORTS = {
    "Strategy": (".base", "Strategy"),
    "Signal": (".base", "Signal"),
    "SignalType": (".base", "SignalType"),
    "Position": (".base", "Position"),
    "MeanReversionStrategy": (".mean_reversion", "MeanReversionStrategy"),
    "SportsbookArbitrageStrategy": (".mean_reversion", "SportsbookArbitrageStrategy"),
    "MomentumStrategy": (".momentum", "MomentumStrategy"),
    "GameMomentumStrategy": (".momentum", "GameMomentumStrategy"),
    "ArbitrageStrategy": (".arbitrage", "ArbitrageStrategy"),
    "HighSpeedArbitrageStrategy": (".arbitrage", "HighSpeedArbitrageStrategy"),
    "NewsBasedStrategy": (".news_based", "NewsBasedStrategy"),
    "BreakingNewsStrategy": (".news_based", "BreakingNewsStrategy"),
}


def __getattr__(name: str) -> object:
    if name == "STRATEGY_PRESETS":
        presets = _strategy_presets()
        globals()[name] = presets
        return presets

    target = _ATTRIBUTE_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = target
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_ATTRIBUTE_EXPORTS) | {"STRATEGY_PRESETS"})


def _strategy_presets() -> dict[str, dict[str, Any]]:
    return {
        "conservative": {
            "class": __getattr__("MeanReversionStrategy"),
            "params": {
                "divergence_threshold": 0.08,
                "max_position_size": 0.05,
                "stop_loss": 0.2,
                "min_edge": 0.05,
            },
        },
        "momentum": {
            "class": __getattr__("MomentumStrategy"),
            "params": {
                "lookback_periods": 10,
                "momentum_threshold": 0.1,
                "use_rsi": True,
                "max_position_size": 0.1,
            },
        },
        "arbitrage": {
            "class": __getattr__("ArbitrageStrategy"),
            "params": {
                "min_arbitrage_profit": 0.01,
                "max_exposure_per_arb": 0.3,
                "speed_priority": True,
            },
        },
        "news": {
            "class": __getattr__("NewsBasedStrategy"),
            "params": {
                "sentiment_threshold": 0.65,
                "news_decay_minutes": 30,
                "min_social_volume": 100,
            },
        },
        "aggressive": {
            "class": __getattr__("GameMomentumStrategy"),
            "params": {
                "event_window": 5,
                "fade_blowouts": True,
                "max_position_size": 0.2,
                "min_edge": 0.02,
            },
        },
        "high_frequency": {
            "class": __getattr__("HighSpeedArbitrageStrategy"),
            "params": {"fixed_size": 100, "pre_calculate_size": True, "latency_threshold_ms": 50},
        },
    }


def create_strategy(preset: str, **override_params: Any) -> object:
    presets = _strategy_presets()
    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Choose from: {list(presets.keys())}")

    preset_config = presets[preset]
    strategy_class = preset_config["class"]
    params = dict(preset_config["params"])
    params.update(override_params)
    return strategy_class(**params)


__all__ = [
    "Strategy",
    "Signal",
    "SignalType",
    "Position",
    "MeanReversionStrategy",
    "SportsbookArbitrageStrategy",
    "MomentumStrategy",
    "GameMomentumStrategy",
    "ArbitrageStrategy",
    "HighSpeedArbitrageStrategy",
    "NewsBasedStrategy",
    "BreakingNewsStrategy",
    "STRATEGY_PRESETS",
    "create_strategy",
]
