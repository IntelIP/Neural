from __future__ import annotations

import asyncio
from typing import Any

from neural.analysis.risk import Position
from neural.exchanges.registry import registry
from neural.exchanges.types import (
    ExchangeName,
    NormalizedOrderRequest,
    TradingPolicy,
)
from neural.trading.paper_client import PaperTradingClient

from .kalshi_adapter import KalshiAdapter, KalshiClientFactory, serialize_value
from .polymarket_us_adapter import PolymarketUSAdapter


def _default_client_factory() -> KalshiClientFactory:
    """Return a factory that builds the modern kalshi-python Client lazily."""

    try:
        from kalshi_python import KalshiClient as Client  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise ImportError(
            "kalshi_python is required for neural.trading. Install with: pip install 'kalshi-python>=2'"
        ) from exc

    def _factory(**kwargs: Any) -> Any:
        from kalshi_python.configuration import Configuration

        config = Configuration()
        config.api_key_id = kwargs.get("api_key_id")
        config.private_key_pem = kwargs.get("private_key_pem")
        return Client(configuration=config)  # type: ignore[misc]

    return _factory


# Backward-compatible alias used by existing tests and imports.
_KalshiClientFactory = KalshiClientFactory


class _CompatMarkets:
    def __init__(self, client: TradingClient) -> None:
        self._client = client

    def get_markets(self, **kwargs: Any) -> dict[str, Any]:
        sport = kwargs.get("sport") or kwargs.get("category")
        limit = int(kwargs.get("limit", 100))
        sports_only = bool(kwargs.get("sports_only", True))
        markets = self._client.list_markets(sport=sport, limit=limit, sports_only=sports_only)
        return {"markets": [serialize_value(m) for m in markets]}


class _CompatPortfolio:
    def __init__(self, client: TradingClient) -> None:
        self._client = client

    def get_positions(self) -> dict[str, Any]:
        rows = [serialize_value(p) for p in self._client.get_positions()]
        return {"positions": rows}


class _CompatExchange:
    def __init__(self, client: TradingClient) -> None:
        self._client = client

    def create_order(self, **kwargs: Any) -> dict[str, Any]:
        result = self._client.place_order(
            market_id=str(kwargs.get("market_id") or kwargs.get("ticker")),
            side=str(kwargs.get("side", "yes")),
            quantity=int(kwargs.get("quantity") or kwargs.get("count") or 0),
            price=kwargs.get("price"),
            order_type=str(kwargs.get("type") or kwargs.get("order_type") or "limit"),
            idempotency_key=kwargs.get("idempotency_key"),
            policy=kwargs.get("policy"),
            paper=bool(kwargs.get("paper", False)),
        )
        return {"order": result}


def _ensure_adapters_registered() -> None:
    if "kalshi" not in registry.names():
        registry.register("kalshi", lambda **kwargs: KalshiAdapter(**kwargs))
    if "polymarket_us" not in registry.names():
        registry.register("polymarket_us", lambda **kwargs: PolymarketUSAdapter(**kwargs))


class TradingClient:
    """Exchange-agnostic trading client with backward-compatible Kalshi defaults."""

    def __init__(
        self,
        api_key_id: str | None = None,
        private_key_pem: bytes | None = None,
        env: str | None = None,
        timeout: int = 15,
        client_factory: _KalshiClientFactory | None = None,
        risk_manager: Any = None,
        *,
        exchange: ExchangeName = "kalshi",
        paper_trading: bool = False,
        trading_policy: TradingPolicy | None = None,
        polymarket_us_api_key: str | None = None,
        polymarket_us_api_secret: bytes | None = None,
        polymarket_us_passphrase: str | None = None,
        polymarket_us_base_url: str | None = None,
        polymarket_us_session: Any | None = None,
    ) -> None:
        _ensure_adapters_registered()

        self.exchange_name: ExchangeName = exchange
        self.risk_manager = risk_manager
        self.paper_trading = paper_trading
        self.trading_policy = trading_policy or TradingPolicy()
        self._paper_client: PaperTradingClient | None = None

        if exchange == "kalshi":
            factory = client_factory or _default_client_factory()
            self._adapter = registry.create(
                "kalshi",
                api_key_id=api_key_id,
                private_key_pem=private_key_pem,
                env=env,
                timeout=timeout,
                client_factory=factory,
            )
        else:
            self._adapter = registry.create(
                "polymarket_us",
                api_key=polymarket_us_api_key,
                api_secret=polymarket_us_api_secret,
                passphrase=polymarket_us_passphrase,
                base_url=polymarket_us_base_url,
                timeout=timeout,
                session=polymarket_us_session,
            )

        self._client = getattr(self._adapter, "_client", self._adapter)
        self.portfolio = getattr(self._adapter, "portfolio", _CompatPortfolio(self))
        self.markets = getattr(self._adapter, "markets", _CompatMarkets(self))
        self.exchange = getattr(self._adapter, "exchange", _CompatExchange(self))

    def capabilities(self) -> dict[str, bool]:
        return self._adapter.capabilities().as_dict()

    def list_markets(
        self, *, sport: str | None = None, limit: int = 100, sports_only: bool = True
    ) -> list[dict[str, Any]]:
        rows = self._adapter.list_markets(sport=sport, limit=limit, sports_only=sports_only)
        return [serialize_value(row) for row in rows]

    def get_quote(self, market_id: str) -> dict[str, Any]:
        return serialize_value(self._adapter.get_quote(market_id))

    def get_positions(self) -> list[dict[str, Any]]:
        return [serialize_value(p) for p in self._adapter.get_positions()]

    def place_order(
        self,
        market_id: str,
        side: str,
        *,
        quantity: int | None = None,
        count: int | None = None,
        order_type: str = "market",
        price: float | None = None,
        idempotency_key: str | None = None,
        policy: TradingPolicy | None = None,
        paper: bool | None = None,
    ) -> dict[str, Any]:
        qty = quantity if quantity is not None else count
        if qty is None:
            raise ValueError("quantity or count is required")

        use_paper = self.paper_trading if paper is None else paper
        normalized_side = _normalize_order_side(side)

        if use_paper:
            return self._place_paper_order(
                market_id=market_id,
                side=normalized_side,
                quantity=qty,
                order_type=order_type,
                price=price,
            )

        order = NormalizedOrderRequest(
            market_id=market_id,
            side=normalized_side,
            quantity=qty,
            order_type="market" if order_type == "market" else "limit",
            price=price,
            idempotency_key=idempotency_key,
        )
        result = self._adapter.place_order(order, policy=policy or self.trading_policy)
        return serialize_value(result)

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        return serialize_value(self._adapter.cancel_order(order_id))

    def get_order_status(self, order_id: str) -> dict[str, Any]:
        return serialize_value(self._adapter.get_order_status(order_id))

    def close(self) -> None:
        self._adapter.close()

    def __enter__(self) -> TradingClient:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def place_order_with_risk(
        self,
        market_id: str,
        side: str,
        quantity: int,
        stop_loss_config: Any = None,
        **order_kwargs: Any,
    ) -> Any:
        result = self.place_order(
            market_id=market_id,
            side=side,
            quantity=quantity,
            order_type=str(order_kwargs.get("type") or order_kwargs.get("order_type") or "limit"),
            price=order_kwargs.get("price"),
            idempotency_key=order_kwargs.get("idempotency_key"),
            policy=order_kwargs.get("policy"),
            paper=order_kwargs.get("paper"),
        )

        if self.risk_manager and stop_loss_config:
            try:
                import time

                position = Position(
                    market_id=market_id,
                    side="yes" if _normalize_order_side(side).endswith("yes") else "no",
                    quantity=quantity,
                    entry_price=order_kwargs.get("price", 0.5),
                    entry_time=time.time(),
                    stop_loss=stop_loss_config,
                )
                self.risk_manager.add_position(position)
            except Exception:
                pass

        return result

    def _paper_client_instance(self) -> PaperTradingClient:
        if self._paper_client is None:
            self._paper_client = PaperTradingClient(save_trades=False)
        return self._paper_client

    def _place_paper_order(
        self,
        *,
        market_id: str,
        side: str,
        quantity: int,
        order_type: str,
        price: float | None,
    ) -> dict[str, Any]:
        paper_client = self._paper_client_instance()
        if side.startswith("sell_"):
            base_side = "yes" if side.endswith("yes") else "no"
            return serialize_value(
                paper_client.close_position(
                    market_id=market_id,
                    side=base_side,
                    quantity=quantity,
                )
            )

        base_side = "yes" if side.endswith("yes") else "no"
        return _run_coro_sync(
            paper_client.place_order(
                market_id=market_id,
                side=base_side,
                quantity=quantity,
                order_type=order_type,
                price=price,
            )
        )


def _normalize_order_side(side: str) -> str:
    normalized = side.lower().strip()
    if normalized in {"yes", "buy_yes"}:
        return "buy_yes"
    if normalized in {"no", "buy_no"}:
        return "buy_no"
    if normalized in {"sell_yes", "sell_no"}:
        return normalized
    raise ValueError(f"Unsupported side: {side}")


def _run_coro_sync(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError(
        "Cannot execute sync paper-trading call inside a running event loop. "
        "Use async PaperTradingClient APIs directly."
    )
