"""
Neural SDK Analysis Stack

A framework for building, testing, and executing trading strategies.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_SUBMODULE_EXPORTS = {"backtesting", "execution", "risk", "sentiment", "strategies"}
_SYMBOL_EXPORTS = {
    "Backtester": (".backtesting.engine", "Backtester"),
    "OrderManager": (".execution.order_manager", "OrderManager"),
    "Signal": (".strategies.base", "Signal"),
    "Strategy": (".strategies.base", "Strategy"),
    "edge_proportional": (".risk.position_sizing", "edge_proportional"),
    "fixed_percentage": (".risk.position_sizing", "fixed_percentage"),
    "kelly_criterion": (".risk.position_sizing", "kelly_criterion"),
}


def __getattr__(name: str) -> Any:
    if name in _SUBMODULE_EXPORTS:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module

    if name in _SYMBOL_EXPORTS:
        module_name, attr_name = _SYMBOL_EXPORTS[name]
        value = getattr(import_module(module_name, __name__), attr_name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _SUBMODULE_EXPORTS | set(_SYMBOL_EXPORTS))


__all__ = [
    "backtesting",
    "execution",
    "risk",
    "sentiment",
    "strategies",
    "Strategy",
    "Signal",
    "Backtester",
    "OrderManager",
    "kelly_criterion",
    "fixed_percentage",
    "edge_proportional",
]
