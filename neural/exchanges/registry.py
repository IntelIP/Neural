from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .base import ExchangeAdapter
from .types import ExchangeName

Factory = Callable[..., ExchangeAdapter]


class ExchangeRegistry:
    def __init__(self) -> None:
        self._factories: dict[ExchangeName, Factory] = {}

    def register(self, name: ExchangeName, factory: Factory) -> None:
        self._factories[name] = factory

    def create(self, name: ExchangeName, **kwargs: Any) -> ExchangeAdapter:
        if name not in self._factories:
            raise ValueError(f"Exchange adapter not registered: {name}")
        return self._factories[name](**kwargs)

    def names(self) -> list[str]:
        return sorted(self._factories.keys())


registry = ExchangeRegistry()
