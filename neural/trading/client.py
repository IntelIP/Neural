from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from neural.auth.env import get_api_key_id, get_base_url, get_private_key_material


class _KalshiClientFactory(Protocol):
    def __call__(self, **kwargs: Any) -> Any: ...


def _default_client_factory() -> _KalshiClientFactory:
    """Return a factory that builds the modern kalshi-python Client lazily.

    Keeping this import inside a function prevents import-time crashes if the
    optional dependency isn't installed. Errors surface only when constructing
    TradingClient, with a helpful message.
    """

    try:
        from kalshi_python import KalshiClient as Client  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise ImportError(
            "kalshi_python is required for neural.trading. Install with: pip install 'kalshi-python>=2'"
        ) from exc

    def _factory(**kwargs: Any) -> Any:
        # Create configuration for kalshi_python v2
        from kalshi_python.configuration import Configuration

        # Create configuration object (use default host with correct API path)
        config = Configuration()
        # Don't override host - Configuration default has correct /trade-api/v2 path

        # Set authentication attributes directly (v2 pattern)
        config.api_key_id = kwargs.get("api_key_id")
        config.private_key_pem = kwargs.get("private_key_pem")

        return Client(configuration=config)  # type: ignore[misc]

    return _factory


def _serialize(obj: Any) -> Any:
    # Normalize common pydantic-like models to dict for a stable SDK surface
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        ctor: Callable[[Any], Any] = list if isinstance(obj, list) else tuple
        return ctor(_serialize(v) for v in obj)
    return obj


class _ServiceProxy:
    """Thin proxy around a sub-API to provide stable call/return behavior."""

    def __init__(self, api: Any):
        self._api = api

    def __getattr__(self, name: str) -> Callable[..., Any]:
        target = getattr(self._api, name)
        if not callable(target):
            return target

        def call(*args: Any, **kwargs: Any) -> Any:
            result = target(*args, **kwargs)
            return _serialize(result)

        call.__name__ = getattr(target, "__name__", name)
        call.__doc__ = getattr(target, "__doc__", None)
        return call


@dataclass(slots=True)
class TradingClient:
    """High-level Kalshi trading client using the modern kalshi-python API.

    - Lazy optional dependency import
    - Explicit configuration via env/files
    - Stable facade for portfolio, markets, exchange
    - Dependency-injectable client factory for testing
    - Optional risk management integration
    """

    api_key_id: str | None = None
    private_key_pem: bytes | None = None
    env: str | None = None
    timeout: int = 15
    client_factory: _KalshiClientFactory | None = None
    risk_manager: Any = None  # Optional RiskManager instance

    _client: Any = field(init=False)
    portfolio: _ServiceProxy = field(init=False)
    markets: _ServiceProxy = field(init=False)
    exchange: _ServiceProxy = field(init=False)

    def __post_init__(self) -> None:
        api_key = self.api_key_id or get_api_key_id()
        priv_key = self.private_key_pem or get_private_key_material()
        # Keep key material as bytes; callers/tests and crypto tooling expect bytes
        priv_key_material = priv_key
        base_url = get_base_url(self.env)

        factory = self.client_factory or _default_client_factory()
        self._client = factory(
            base_url=base_url,
            api_key_id=api_key,
            private_key_pem=priv_key_material,
            timeout=self.timeout,
        )

        # Map common sub-APIs with graceful fallback
        self.portfolio = _ServiceProxy(getattr(self._client, "portfolio", self._client))
        self.markets = _ServiceProxy(getattr(self._client, "markets", self._client))
        self.exchange = _ServiceProxy(getattr(self._client, "exchange", self._client))

    def close(self) -> None:
        if hasattr(self._client, "close"):
            try:
                self._client.close()
            except Exception:
                pass

    def __enter__(self) -> TradingClient:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def place_order_with_risk(
        self, market_id: str, side: str, quantity: int, stop_loss_config: Any = None, **order_kwargs
    ) -> Any:
        """
        Place an order with optional risk management.

        Args:
            market_id: Market identifier
            side: "yes" or "no"
            quantity: Order quantity
            stop_loss_config: Optional stop-loss configuration
            **order_kwargs: Additional order parameters

        Returns:
            Order result
        """
        # Place the order first
        # Note: This assumes the exchange service has order methods
        # The actual implementation depends on kalshi-python API

        try:
            # Hypothetical order placement - adapt based on actual API
            order_result = self.exchange.create_order(
                market_id=market_id, side=side, quantity=quantity, **order_kwargs
            )

            # Register with risk manager if provided
            if self.risk_manager and stop_loss_config:
                try:
                    from neural.analysis.risk import Position
                    import time

                    position = Position(
                        market_id=market_id,
                        side=side,
                        quantity=quantity,
                        entry_price=order_kwargs.get("price", 0.5),  # Default assumption
                        entry_time=time.time(),
                        stop_loss=stop_loss_config,
                    )
                    self.risk_manager.add_position(position)
                except ImportError:
                    pass  # Risk module not available

            return order_result

        except Exception as e:
            # Log error and re-raise
            import logging

            logging.getLogger(__name__).error(f"Order placement failed: {e}")
            raise
