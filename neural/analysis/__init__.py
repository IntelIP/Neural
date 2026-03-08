"""Neural SDK analysis surface with lazy imports for optional modules."""

from __future__ import annotations

import importlib
from types import ModuleType

_MODULE_EXPORTS = {
    "backtesting": ".backtesting",
    "execution": ".execution",
    "risk": ".risk",
    "sentiment": ".sentiment",
    "strategies": ".strategies",
}

_ATTRIBUTE_EXPORTS = {
    "Backtester": (".backtesting.engine", "Backtester"),
    "OrderManager": (".execution.order_manager", "OrderManager"),
    "Strategy": (".strategies.base", "Strategy"),
    "Signal": (".strategies.base", "Signal"),
    "kelly_criterion": (".risk.position_sizing", "kelly_criterion"),
    "fixed_percentage": (".risk.position_sizing", "fixed_percentage"),
    "edge_proportional": (".risk.position_sizing", "edge_proportional"),
}


def __getattr__(name: str) -> ModuleType | object:
    module_name = _MODULE_EXPORTS.get(name)
    if module_name is not None:
        module = importlib.import_module(module_name, __name__)
        globals()[name] = module
        return module

    target = _ATTRIBUTE_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = target
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_MODULE_EXPORTS) | set(_ATTRIBUTE_EXPORTS))


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
