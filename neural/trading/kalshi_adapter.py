from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Protocol

from neural.auth.env import get_api_key_id, get_base_url, get_private_key_material
from neural.exchanges.base import BaseExchangeAdapter
from neural.exchanges.types import (
    ExchangeCapabilities,
    ExchangeName,
    NormalizedMarket,
    NormalizedOrderRequest,
    NormalizedOrderResult,
    NormalizedPosition,
    NormalizedQuote,
    TradingPolicy,
)


class KalshiClientFactory(Protocol):
    def __call__(self, **kwargs: Any) -> Any: ...


def serialize_value(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    if isinstance(obj, dict):
        return {k: serialize_value(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        ctor: Callable[[Any], Any] = list if isinstance(obj, list) else tuple
        return ctor(serialize_value(v) for v in obj)
    return obj


class ServiceProxy:
    """Thin proxy around a sub-API to provide stable call/return behavior."""

    def __init__(self, api: Any):
        self._api = api

    def __getattr__(self, name: str) -> Callable[..., Any]:
        target = getattr(self._api, name)
        if not callable(target):
            return target

        def call(*args: Any, **kwargs: Any) -> Any:
            return serialize_value(target(*args, **kwargs))

        call.__name__ = getattr(target, "__name__", name)
        call.__doc__ = getattr(target, "__doc__", None)
        return call


@dataclass(slots=True)
class KalshiAdapter(BaseExchangeAdapter):
    api_key_id: str | None = None
    private_key_pem: bytes | None = None
    env: str | None = None
    timeout: int = 15
    client_factory: KalshiClientFactory | None = None

    name: ExchangeName = "kalshi"

    def __post_init__(self) -> None:
        BaseExchangeAdapter.__init__(self)
        api_key = self.api_key_id or get_api_key_id()
        priv_key = self.private_key_pem or get_private_key_material()
        base_url = get_base_url(self.env)

        if self.client_factory is None:
            raise ImportError(
                "kalshi_python client factory is required. "
                "Install with: pip install 'kalshi-python>=2'"
            )

        self._client = self.client_factory(
            base_url=base_url,
            api_key_id=api_key,
            private_key_pem=priv_key,
            timeout=self.timeout,
        )
        self.portfolio = ServiceProxy(getattr(self._client, "portfolio", self._client))
        self.markets = ServiceProxy(getattr(self._client, "markets", self._client))
        self.exchange = ServiceProxy(getattr(self._client, "exchange", self._client))

    def capabilities(self) -> ExchangeCapabilities:
        return ExchangeCapabilities(read=True, paper=True, live=True, streaming=True)

    def list_markets(
        self, *, sport: str | None = None, limit: int = 100, sports_only: bool = True
    ) -> list[NormalizedMarket]:
        kwargs: dict[str, Any] = {"limit": limit}
        if sport:
            kwargs["category"] = sport
        elif sports_only:
            kwargs["category"] = "sports"

        raw = self._call_any(["get_markets", "list_markets"], self.markets, kwargs)
        rows = raw.get("markets", raw if isinstance(raw, list) else [])
        return [self._normalize_market(m) for m in rows]

    def get_quote(self, market_id: str) -> NormalizedQuote:
        raw = self._call_any(
            ["get_market", "get_event", "market"],
            self.markets,
            {"market_id": market_id, "ticker": market_id},
        )
        market = raw.get("market", raw)
        return NormalizedQuote(
            market_id=str(market.get("ticker") or market.get("id") or market_id),
            yes_bid=_to_prob(market.get("yes_bid")),
            yes_ask=_to_prob(market.get("yes_ask")),
            no_bid=_to_prob(market.get("no_bid")),
            no_ask=_to_prob(market.get("no_ask")),
            last_price=_to_prob(market.get("last_price") or market.get("yes_price")),
            volume=_to_float(market.get("volume")),
            metadata={"exchange": "kalshi", "raw": market},
        )

    def place_order(
        self,
        order: NormalizedOrderRequest,
        *,
        policy: TradingPolicy | None = None,
    ) -> NormalizedOrderResult:
        notional_to_record = 0.0
        if policy is not None:
            notional_to_record = self._enforce_policy(order, policy)

        side = "yes" if order.side.endswith("yes") else "no"
        payload: dict[str, Any] = {
            "market_id": order.market_id,
            "side": side,
            "quantity": order.quantity,
            "type": order.order_type,
        }
        if order.price is not None:
            payload["price"] = order.price

        raw = self._call_any(["create_order", "place_order"], self.exchange, payload)
        if policy is not None:
            self._record_daily_notional(notional_to_record)
        order_data = raw.get("order", raw)
        return NormalizedOrderResult(
            success=True,
            status=str(order_data.get("status", "submitted")),
            order_id=str(order_data.get("id") or order_data.get("order_id") or ""),
            metadata={"exchange": "kalshi", "raw": raw},
        )

    def cancel_order(self, order_id: str) -> NormalizedOrderResult:
        raw = self._call_any(
            ["cancel_order", "delete_order"], self.exchange, {"order_id": order_id}
        )
        status = "cancelled" if raw is not None else "unknown"
        return NormalizedOrderResult(
            success=True,
            status=status,
            order_id=order_id,
            metadata={"exchange": "kalshi", "raw": raw},
        )

    def get_order_status(self, order_id: str) -> NormalizedOrderResult:
        raw = self._call_any(["get_order", "order"], self.exchange, {"order_id": order_id})
        order_data = raw.get("order", raw)
        return NormalizedOrderResult(
            success=True,
            status=str(order_data.get("status", "unknown")),
            order_id=str(order_data.get("id") or order_data.get("order_id") or order_id),
            metadata={"exchange": "kalshi", "raw": raw},
        )

    def get_positions(self) -> list[NormalizedPosition]:
        raw = self._call_any(["get_positions", "positions"], self.portfolio, {})
        rows = raw.get("positions", raw if isinstance(raw, list) else [])
        out: list[NormalizedPosition] = []
        for row in rows:
            out.append(
                NormalizedPosition(
                    market_id=str(row.get("market_id") or row.get("ticker") or ""),
                    side="yes" if str(row.get("side", "yes")).lower() == "yes" else "no",
                    quantity=int(row.get("quantity") or row.get("count") or 0),
                    entry_price=_to_prob(row.get("avg_price") or row.get("entry_price")),
                    current_price=_to_prob(row.get("current_price")),
                    unrealized_pnl=_to_float(row.get("unrealized_pnl")),
                    metadata={"exchange": "kalshi", "raw": row},
                )
            )
        return out

    def close(self) -> None:
        if hasattr(self._client, "close"):
            try:
                self._client.close()
            except Exception:
                pass

    @staticmethod
    def _normalize_market(raw: dict[str, Any]) -> NormalizedMarket:
        market_id = str(raw.get("ticker") or raw.get("id") or "")
        return NormalizedMarket(
            market_id=market_id,
            ticker=market_id,
            title=str(raw.get("title") or raw.get("name") or market_id),
            status=str(raw.get("status") or "open"),
            yes_price=_to_prob(raw.get("yes_ask") or raw.get("yes_price")),
            no_price=_to_prob(raw.get("no_ask") or raw.get("no_price")),
            category=str(raw.get("category") or "sports"),
            sport=str(raw.get("sport") or raw.get("category") or "sports"),
            metadata={"exchange": "kalshi", "raw": raw},
        )

    @staticmethod
    def _call_any(method_names: list[str], target: Any, kwargs: dict[str, Any]) -> Any:
        cleaned_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        for name in method_names:
            method = getattr(target, name, None)
            if not callable(method):
                continue
            if KalshiAdapter._can_call_with_kwargs(method, cleaned_kwargs):
                return method(**cleaned_kwargs)
            if cleaned_kwargs and KalshiAdapter._can_call_without_kwargs(method):
                return method()
        raise AttributeError(f"None of methods {method_names} exist on target API")

    @staticmethod
    def _can_call_with_kwargs(method: Any, kwargs: dict[str, Any]) -> bool:
        if not kwargs:
            return True
        try:
            sig = inspect.signature(method)
        except (TypeError, ValueError):
            return True

        params = sig.parameters.values()
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params):
            return True

        allowed = {
            name
            for name, param in sig.parameters.items()
            if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        return set(kwargs).issubset(allowed)

    @staticmethod
    def _can_call_without_kwargs(method: Any) -> bool:
        try:
            sig = inspect.signature(method)
        except (TypeError, ValueError):
            return False

        for param in sig.parameters.values():
            if param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ) and param.default is inspect.Parameter.empty:
                return False
        return True


def _to_prob(value: Any) -> float | None:
    if value is None:
        return None
    f = float(value)
    if f > 1:
        return f / 100.0
    return f


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
